//! The interpreter: runs a graph node by node.
//!
//! Every node runs its precompiled kernel, or a unit compiled from just that node.
//! Memory:
//!
//! * `Caller` outputs (static shape) that are temporaries get fixed offsets in one arena
//!   per call state, reused by liveness;
//! * `Caller` outputs that are graph outputs get a fresh buffer per call (or the
//!   caller's memory when the program runs as a kernel);
//! * `Callee` outputs (dynamic shape) are allocated by the kernel and released after
//!   their last use;
//! * views share their input's memory, which stays alive as long as the view does.

use super::compile::{compile_units, CompileStats};
use super::graph::{ConstIdx, Graph, NodeId, Origin, ValueId};
use super::index::{Idx, IndexVec};
use super::op::{Kernel, KernelArgs, R};
use super::types::{c_strides, InputIdx, OutputAlloc, OutputIdx, Signature, TensorTy};
use super::unit::UnitSpec;
use crate::abi::{PtArray, PtBuffer, PtCtx};
use crate::define_index;
use std::sync::{Arc, Mutex};

define_index!(
    /// A step of a program.
    StepIdx
);

/// Byte offset into a call state's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaOffset(pub u64);

#[derive(Clone, Debug)]
pub struct Options {
    pub fastmath: bool,
    pub cache: bool,
    pub opt_level: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options { fastmath: true, cache: true, opt_level: 3 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Input(InputIdx),
    Const(ConstIdx),
    Arena(ArenaOffset),
    /// a graph output with a static shape
    Fresh,
    /// allocated by the producing kernel
    Callee,
    /// declared view; the kernel may also return a fresh buffer
    View(ValueId),
}

struct Step {
    node: NodeId,
    sig: Signature,
    kernel: Box<dyn Kernel>,
    /// values released after this step
    release: Vec<ValueId>,
}

pub struct Program {
    graph: Arc<Graph>,
    sig: Signature,
    slots: IndexVec<ValueId, Slot>,
    steps: IndexVec<StepIdx, Step>,
    arena_bytes: u64,
    /// the graph output whose memory a Fresh value is (first occurrence)
    fresh_output: IndexVec<ValueId, Option<OutputIdx>>,
    states: Mutex<Vec<CallState>>,
    pub stats: CompileStats,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// Per-call memory. Not shared between concurrent calls.
pub struct CallState {
    vals: IndexVec<ValueId, PtArray>,
    /// shape ++ strides of every value
    storage: IndexVec<ValueId, Box<[i64]>>,
    owned: IndexVec<ValueId, bool>,
    /// backs the arena offsets in `vals` (never read directly)
    _arena: Vec<Align64>,
    args: Vec<PtArray>,
    pub ctx: PtCtx,
}

// SAFETY: the raw pointers in a CallState point into its own storage, its arena, buffers
// it owns, or constants of the program; nothing is shared with another state.
unsafe impl Send for CallState {}

/// An output handed to the caller: one buffer reference, owned by the receiver.
pub struct OutArray {
    pub data: *mut u8,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub owner: *mut PtBuffer,
}

// SAFETY: an OutArray is one buffer reference, owned by whoever holds it.
unsafe impl Send for OutArray {}

impl Program {
    pub fn build(graph: Arc<Graph>, opts: &Options) -> R<Program> {
        let t0 = std::time::Instant::now();
        let g = &*graph;
        let sig = g.own_signature();

        // kernels, and units for the nodes that have none
        let mut kernels: IndexVec<NodeId, Option<Box<dyn Kernel>>> = IndexVec::new();
        let mut units = Vec::new();
        let mut unit_nodes = Vec::new();
        for (n, node) in g.nodes.iter_enumerated() {
            let nsig = g.signature(n);
            let k = node.op.kernel(&nsig).map_err(|m| format!("{:?}: {m}", node.op))?;
            if k.is_none() {
                if node.op.generate().is_none() {
                    return Err(format!("{:?} has no implementation", node.op));
                }
                units.push(UnitSpec { graph: Arc::new(single_node_graph(g, n)?), sig: nsig, fastmath: opts.fastmath });
                unit_nodes.push(n);
            }
            kernels.push(k);
        }
        let (compiled, mut stats) = compile_units(&units, opts)?;
        for (n, k) in unit_nodes.into_iter().zip(compiled) {
            kernels[n] = Some(Box::new(k));
        }

        // storage of every value
        let mut fresh_output: IndexVec<ValueId, Option<OutputIdx>> = g.values.iter().map(|_| None).collect();
        for (o, &v) in g.outputs.iter_enumerated() {
            if fresh_output[v].is_none() {
                fresh_output[v] = Some(o);
            }
        }
        let mut slots: IndexVec<ValueId, Slot> = IndexVec::new();
        for (v, val) in g.values.iter_enumerated() {
            slots.push(match val.origin {
                Origin::Input(i) => Slot::Input(i),
                Origin::Const(c) => Slot::Const(c),
                Origin::Node(n, o) => {
                    let nsig = g.signature(n);
                    match nsig.outputs[o].alloc {
                        OutputAlloc::ViewOf(i) => Slot::View(g.nodes[n].inputs[i]),
                        OutputAlloc::Callee => Slot::Callee,
                        OutputAlloc::Caller if fresh_output[v].is_some() => Slot::Fresh,
                        // placeholder offset, assigned below
                        OutputAlloc::Caller => Slot::Arena(ArenaOffset(u64::MAX)),
                    }
                }
            });
        }

        // lifetimes in steps; views keep what they view alive
        let n_steps = g.nodes.len();
        let mut def: IndexVec<ValueId, usize> = g.values.iter().map(|_| 0).collect();
        let mut end: IndexVec<ValueId, usize> = g.values.iter().map(|_| 0).collect();
        for (n, node) in g.nodes.iter_enumerated() {
            for &v in &node.outputs {
                def[v] = n.index();
                end[v] = end[v].max(n.index());
            }
            for &v in &node.inputs {
                end[v] = end[v].max(n.index());
            }
        }
        for &v in &g.outputs {
            end[v] = usize::MAX;
        }
        for v in g.values.indices().collect::<Vec<_>>().into_iter().rev() {
            if let Slot::View(src) = slots[v] {
                end[src] = end[src].max(end[v]);
            }
        }

        // arena: first fit over static-size temporaries
        let mut active: Vec<(u64, u64, usize)> = Vec::new(); // (offset, size, end)
        let mut arena_bytes = 0u64;
        for v in g.values.indices() {
            if !matches!(slots[v], Slot::Arena(_)) {
                continue;
            }
            let size = g.ty(v).static_nbytes().ok_or("arena value without static size")?.div_ceil(64).max(1) * 64;
            active.retain(|&(_, _, e)| e >= def[v]);
            active.sort_by_key(|&(o, _, _)| o);
            let mut off = 0u64;
            for &(o, s, _) in &active {
                if off + size <= o {
                    break;
                }
                off = off.max(o + s);
            }
            active.push((off, size, end[v]));
            arena_bytes = arena_bytes.max(off + size);
            slots[v] = Slot::Arena(ArenaOffset(off));
        }

        let mut release: Vec<Vec<ValueId>> = vec![vec![]; n_steps];
        for v in g.values.indices() {
            if matches!(slots[v], Slot::Callee | Slot::View(_) | Slot::Fresh) && end[v] != usize::MAX {
                release[end[v].max(def[v])].push(v);
            }
        }
        let steps: IndexVec<StepIdx, Step> = g
            .nodes
            .indices()
            .zip(release)
            .map(|(n, release)| Step { node: n, sig: g.signature(n), kernel: kernels[n].take().unwrap(), release })
            .collect();
        stats.build_s = t0.elapsed().as_secs_f64();
        Ok(Program { graph, sig, slots, steps, arena_bytes, fresh_output, states: Mutex::new(vec![]), stats })
    }

    pub fn sig(&self) -> &Signature {
        &self.sig
    }

    pub fn n_inputs(&self) -> usize {
        self.sig.inputs.len()
    }

    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    pub fn describe_steps(&self) -> Vec<String> {
        self.steps.iter().map(|s| format!("{:?}", self.graph.nodes[s.node].op)).collect()
    }

    pub fn new_state(&self) -> CallState {
        let g = &*self.graph;
        let mut storage: IndexVec<ValueId, Box<[i64]>> = IndexVec::new();
        for v in g.values.iter() {
            let nd = v.ty.ndim();
            let mut s = vec![0i64; 2 * nd].into_boxed_slice();
            if let Some(shape) = v.ty.static_shape() {
                let shape: Vec<i64> = shape.iter().map(|&n| n as i64).collect();
                let st = c_strides(&shape, v.ty.dtype.itemsize() as i64);
                s[..nd].copy_from_slice(&shape);
                s[nd..].copy_from_slice(&st);
            }
            storage.push(s);
        }
        let mut arena = vec![Align64([0; 64]); (self.arena_bytes / 64) as usize];
        let base = arena.as_mut_ptr() as *mut u8;
        let mut vals: IndexVec<ValueId, PtArray> = IndexVec::new();
        for (v, val) in g.values.iter_enumerated() {
            let nd = val.ty.ndim();
            let st = storage[v].as_mut_ptr();
            let data = match self.slots[v] {
                Slot::Const(c) => g.consts[c].data() as *mut u8,
                // SAFETY: offsets were assigned inside the arena.
                Slot::Arena(ArenaOffset(off)) => unsafe { base.add(off as usize) },
                _ => std::ptr::null_mut(),
            };
            if let Slot::Const(c) = self.slots[v] {
                let cd = &g.consts[c];
                storage[v][..nd].copy_from_slice(&cd.shape);
                storage[v][nd..].copy_from_slice(&cd.strides);
            }
            // SAFETY: the storage has 2 * ndim entries.
            let strides = unsafe { st.add(nd) };
            vals.push(PtArray { data, shape: st, strides, owner: std::ptr::null_mut(), ndim: nd as i64 });
        }
        let owned = g.values.iter().map(|_| false).collect();
        CallState { vals, storage, owned, _arena: arena, args: Vec::new(), ctx: PtCtx::new() }
    }

    pub fn take_state(&self) -> CallState {
        let s = self.states.lock().unwrap_or_else(|e| e.into_inner()).pop();
        s.unwrap_or_else(|| self.new_state())
    }

    pub fn put_state(&self, st: CallState) {
        self.states.lock().unwrap_or_else(|e| e.into_inner()).push(st);
    }

    /// Set input `i`. The memory must stay valid until `finish`.
    pub fn set_input(&self, st: &mut CallState, i: InputIdx, data: *mut u8, shape: &[i64], strides: &[i64]) -> R<()> {
        let v = self.graph.inputs[i];
        check_shape(self.graph.ty(v), shape).map_err(|m| format!("input {}: {m}", i.0))?;
        let isz = self.graph.ty(v).dtype.itemsize();
        crate::abi::validate_array(data, shape, strides, isz).map_err(|m| format!("input {}: {m}", i.0))?;
        let nd = shape.len();
        st.storage[v][..nd].copy_from_slice(shape);
        st.storage[v][nd..].copy_from_slice(strides);
        st.vals[v].data = data;
        Ok(())
    }

    /// Run all steps. `provided[o]` is C-contiguous memory for a static graph output.
    pub fn execute(&self, st: &mut CallState, ctx: &mut PtCtx, provided: &IndexVec<OutputIdx, Option<*mut u8>>) -> R<()> {
        let r = self.execute_inner(st, ctx, provided);
        if r.is_err() {
            self.release_all(st);
        }
        r
    }

    fn execute_inner(&self, st: &mut CallState, ctx: &mut PtCtx, provided: &IndexVec<OutputIdx, Option<*mut u8>>) -> R<()> {
        let g = &*self.graph;
        for step in self.steps.iter() {
            let node = &g.nodes[step.node];
            st.args.clear();
            for &v in &node.inputs {
                st.args.push(st.vals[v]);
            }
            for &v in &node.outputs {
                let mut a = st.vals[v];
                match self.slots[v] {
                    Slot::Fresh => match self.fresh_output[v].and_then(|o| provided.get(o).copied().flatten()) {
                        Some(p) => {
                            a.data = p;
                            a.owner = std::ptr::null_mut();
                        }
                        None => {
                            let n = g.ty(v).static_nbytes().unwrap_or(0) as usize;
                            // SAFETY: ctx is the live context.
                            let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
                            if data.is_null() {
                                return Err(format!("could not allocate {n} bytes"));
                            }
                            a.data = data;
                            a.owner = owner;
                            st.vals[v] = a;
                            st.owned[v] = true;
                        }
                    },
                    Slot::Callee | Slot::View(_) => {
                        a.data = std::ptr::null_mut();
                        a.owner = std::ptr::null_mut();
                    }
                    _ => {}
                }
                st.args.push(a);
            }
            // SAFETY: inputs are valid arrays of their types (checked at the program's
            // entry or produced by earlier steps and checked below), Caller outputs are
            // arena, fresh or provided memory of their static type, and every output
            // has shape/strides storage of its ndim.
            let mut ka = unsafe { KernelArgs::new(&step.sig, &mut st.args) };
            let res = step.kernel.call(ctx, &mut ka);
            ctx.end_kernel();
            let n_in = node.inputs.len();
            for (k, &v) in node.outputs.iter().enumerate() {
                let a = st.args[n_in + k];
                match self.slots[v] {
                    Slot::Callee => {
                        st.vals[v].data = a.data;
                        st.vals[v].owner = a.owner;
                        st.owned[v] = !a.owner.is_null();
                    }
                    Slot::View(src) => {
                        st.vals[v].data = a.data;
                        st.vals[v].owner = a.owner;
                        // a fresh buffer instead of a view is ours to release
                        st.owned[v] = !a.owner.is_null() && a.owner != st.vals[src].owner;
                    }
                    _ => {}
                }
            }
            res.map_err(|m| format!("{:?}: {m}", node.op))?;
            for (k, &v) in node.outputs.iter().enumerate() {
                if matches!(self.slots[v], Slot::Callee | Slot::View(_)) {
                    if st.args[n_in + k].data.is_null() && g.ty(v).static_size() != Some(0) {
                        return Err(format!("{:?}: output {k} was not set", node.op));
                    }
                    // SAFETY: the storage holds ndim entries, written by the kernel.
                    let shape = unsafe { st.vals[v].shape() };
                    check_shape(g.ty(v), shape).map_err(|m| format!("{:?}: output {k}: {m}", node.op))?;
                }
            }
            for &v in &step.release {
                self.release(st, v);
            }
        }
        Ok(())
    }

    fn release(&self, st: &mut CallState, v: ValueId) {
        if st.owned[v] {
            // SAFETY: an owned reference of this state, released once (flag cleared).
            unsafe { PtBuffer::decref(st.vals[v].owner) };
            st.owned[v] = false;
            st.vals[v].owner = std::ptr::null_mut();
        }
    }

    fn release_all(&self, st: &mut CallState) {
        for v in self.graph.values.indices() {
            self.release(st, v);
        }
    }

    /// Output `o` as a new reference (copied if its memory isn't a runtime buffer).
    pub fn take_output(&self, st: &CallState, ctx: &mut PtCtx, o: OutputIdx) -> R<OutArray> {
        let v = self.graph.outputs[o];
        let a = st.vals[v];
        // SAFETY: the value was computed by `execute` (or is an input or constant).
        let (shape, strides) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
        if !a.owner.is_null() {
            // SAFETY: a live buffer.
            unsafe { PtBuffer::incref(a.owner) };
            return Ok(OutArray { data: a.data, shape, strides, owner: a.owner });
        }
        let isz = self.graph.ty(v).dtype.itemsize();
        let n = crate::abi::checked_nbytes(&shape, isz).ok_or("output too large")?;
        // SAFETY: ctx is the live context.
        let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
        if data.is_null() {
            return Err(format!("could not allocate {n} bytes"));
        }
        let dst_strides = c_strides(&shape, isz as i64);
        let src = super::op::ArrayRef { data: a.data, shape: &shape, strides: &strides, owner: a.owner };
        // SAFETY: a valid array and a fresh buffer of its size.
        unsafe { super::kernels::copy_strided(src, data, &dst_strides, isz) };
        Ok(OutArray { data, shape, strides: dst_strides, owner })
    }

    /// Release what the call still holds (after the outputs were taken).
    pub fn finish(&self, st: &mut CallState) {
        self.release_all(st);
    }
}

/// Interpreted OpFromGraph: the program runs as a kernel.
impl Kernel for Program {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let mut st = self.take_state();
        let r = self.call_with(&mut st, ctx, args);
        if r.is_err() {
            self.release_all(&mut st);
        }
        self.put_state(st);
        r
    }
}

impl Program {
    fn call_with(&self, st: &mut CallState, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        for i in self.sig.inputs.indices() {
            let x = args.input(i);
            let v = self.graph.inputs[i];
            let nd = x.ndim();
            st.storage[v][..nd].copy_from_slice(x.shape);
            st.storage[v][nd..].copy_from_slice(x.strides);
            st.vals[v].data = x.data as *mut u8;
            st.vals[v].owner = std::ptr::null_mut();
        }
        let sig = args.sig().clone();
        let mut provided: IndexVec<OutputIdx, Option<*mut u8>> = IndexVec::new();
        for (o, os) in sig.outputs.iter_enumerated() {
            provided.push(match os.alloc {
                OutputAlloc::Caller => {
                    let shape: Vec<i64> = os.ty.static_shape().unwrap().iter().map(|&n| n as i64).collect();
                    Some(args.output(ctx, o, &shape)?.0)
                }
                _ => None,
            });
        }
        self.execute(st, ctx, &provided)?;
        for (o, os) in sig.outputs.iter_enumerated() {
            let v = self.graph.outputs[o];
            let in_place = self.slots[v] == Slot::Fresh && self.fresh_output[v] == Some(o);
            match os.alloc {
                OutputAlloc::Caller if in_place => {}
                OutputAlloc::Caller => {
                    let a = st.vals[v];
                    // SAFETY: a computed value.
                    let (shape, strides) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
                    let (dst, dst_strides) = args.output(ctx, o, &shape)?;
                    let src = super::op::ArrayRef { data: a.data, shape: &shape, strides: &strides, owner: a.owner };
                    // SAFETY: a valid array; the provided memory has its shape.
                    unsafe { super::kernels::copy_strided(src, dst, &dst_strides, os.ty.dtype.itemsize()) };
                }
                _ => {
                    let out = self.take_output(st, ctx, o)?;
                    args.set_output(o, out.data, &out.shape, &out.strides, out.owner);
                }
            }
        }
        self.finish(st);
        Ok(())
    }
}

fn check_shape(ty: &TensorTy, shape: &[i64]) -> R<()> {
    if shape.len() != ty.ndim() {
        return Err(format!("expected {} dimensions, got {}", ty.ndim(), shape.len()));
    }
    for (a, (&n, len)) in shape.iter().zip(&ty.shape).enumerate() {
        if n < 0 {
            return Err(format!("negative length on axis {a}"));
        }
        if let Some(s) = len.known() {
            if n as u64 != s {
                return Err(format!("length {n} on axis {a}, the type says {s}"));
            }
        }
    }
    Ok(())
}

/// A graph with just node `n` of `g` (its inputs become graph inputs, constants stay
/// constants).
fn single_node_graph(g: &Graph, n: NodeId) -> R<Graph> {
    let node = &g.nodes[n];
    let mut sg = Graph::default();
    let mut ins = Vec::new();
    for &v in &node.inputs {
        ins.push(sg.add_input(g.ty(v).clone()));
    }
    let tys = node.outputs.iter().map(|&v| g.ty(v).clone()).collect();
    let outs = sg.add_node(node.op.clone(), ins, tys, node.views.clone())?;
    sg.outputs = IndexVec::from_vec(outs);
    Ok(sg)
}
