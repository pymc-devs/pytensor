//! Buffer planning and execution.
//!
//! Planning happens once, when the function is compiled, and doesn't depend on shapes.
//! Every node output gets one of these storages:
//!
//! * `Slot`: a planner-owned buffer, reused across variables with disjoint lifetimes
//!   (linear scan over the schedule, like register allocation). Slots persist across
//!   calls and only grow. This is the only buffer reuse: it's static, the planner knows
//!   exactly which variable uses a slot next.
//! * `Fresh`: a new refcounted buffer allocated by the driver (graph outputs, which are
//!   handed to numpy without copying).
//! * `KernelAlloc`: the kernel allocates (output shape only known while running).
//! * `View`: a statically declared view of one input; it shares the root buffer.
//! * `Take`: an Elemwise writes in place into a kernel-allocated input that dies at this
//!   node and takes over its buffer (and ownership).
//!
//! Refcounted buffers are released right after their last use: plain malloc/free, no pool.
//! Layouts (C/F/Any) are tracked per variable; the planner chooses the layout of every
//! provided output so kernels can specialize on it.
//!
//! At runtime, shapes of provided outputs are inferred right before their node runs, so
//! anything downstream of a kernel-allocated output works without a separate pass.

use crate::abi::{Params, PtArray, PtBuffer, PtCtx, KernelFn, MAX_NDIM};
use crate::compile::{get_kernels, KernelStats, Options};
use crate::dtype::DType;
use crate::ir::{FusedOut, Graph, IndexEntry, Node, Op};
use crate::native::{self, *};
use crate::spec::{KernelSpec, Layout, Operand};
use std::alloc::{alloc, dealloc, Layout as MemLayout};

// ------------------------------------------------------------------ slot buffers

pub struct SlotBuf {
    ptr: *mut u8,
    cap: usize,
}

unsafe impl Send for SlotBuf {}
unsafe impl Sync for SlotBuf {}

impl SlotBuf {
    fn new() -> Self {
        SlotBuf { ptr: std::ptr::null_mut(), cap: 0 }
    }
    fn ensure(&mut self, bytes: usize) {
        let bytes = bytes.max(64);
        if bytes <= self.cap {
            return;
        }
        unsafe {
            if !self.ptr.is_null() {
                dealloc(self.ptr, MemLayout::from_size_align_unchecked(self.cap, 64));
            }
            self.ptr = alloc(MemLayout::from_size_align(bytes, 64).unwrap());
            assert!(!self.ptr.is_null(), "out of memory");
        }
        self.cap = bytes;
    }
}

impl Drop for SlotBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, MemLayout::from_size_align_unchecked(self.cap, 64)) };
        }
    }
}

/// Constant data, owned by the compiled function.
pub struct ConstData {
    pub buf: *mut PtBuffer,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

unsafe impl Send for ConstData {}
unsafe impl Sync for ConstData {}

impl ConstData {
    pub fn from_bytes(bytes: &[u8], shape: Vec<i64>, strides: Vec<i64>) -> Self {
        let buf = PtBuffer::new(bytes.len(), 64);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (*buf).data, bytes.len()) };
        ConstData { buf, shape, strides }
    }
}

impl Drop for ConstData {
    fn drop(&mut self) {
        unsafe { PtBuffer::decref(self.buf) };
    }
}

// ------------------------------------------------------------------ program

#[derive(Clone, Copy, Debug, PartialEq)]
enum Storage {
    Input(usize),
    Const(usize),
    Slot(usize),
    Fresh,
    KernelAlloc,
    /// declared view; `root` is the variable owning the memory
    View { root: usize },
    /// in-place into a kernel-allocated input, taking over its buffer
    Take { from: usize },
    /// lives inside a segment; never touched by the driver
    Internal,
}

enum Src {
    Jit(usize),
    Native(KernelFn),
}

struct Step {
    node: usize,
    src: Src,
    func: Option<KernelFn>,
    args: Vec<usize>,
    /// outputs (subset of args) by storage handling
    outs: Vec<usize>,
    params: Params,
    desc: String,
    /// for segments: member descriptions, indexed by (return code - 1)
    member_descs: Vec<String>,
    /// owned roots to release after this step
    release: Vec<usize>,
}

pub struct Program {
    graph: Graph,
    storage: Vec<Storage>,
    layout: Vec<Layout>,
    root: Vec<usize>,
    steps: Vec<Step>,
    /// graph output k comes from this var directly (zero-copy) or needs a copy
    out_direct: Vec<bool>,
    /// owned roots to release after the final output copies
    release_at_end: Vec<usize>,
    n_slots: usize,
    consts: Vec<ConstData>,
    dim_off: Vec<usize>,
    total_dims: usize,
    pub stats: KernelStats,
    pub n_inplace: usize,
    /// static shapes (-1 = unknown) of inputs and outputs, for the C API
    in_static: Vec<Vec<i64>>,
    out_static: Vec<Vec<i64>>,
    /// objects that own foreign kernel code (e.g. numba cfuncs); dropped with the program
    keepalive: Vec<Box<dyn std::any::Any + Send + Sync>>,
}

pub struct CallState {
    shapes: Vec<i64>,
    strides: Vec<i64>,
    ptrs: Vec<*mut u8>,
    owners: Vec<*mut PtBuffer>,
    slots: Vec<SlotBuf>,
    args: Vec<PtArray>,
    ctx: PtCtx,
    /// Per-step accumulated seconds, if profiling is on.
    pub profile: Option<Vec<f64>>,
    /// caller-provided output k for a variable, during a call
    prov_var: Vec<Option<usize>>,
    direct_written: Vec<bool>,
    /// outputs of the last successful call
    pub outs: Vec<OutArray>,
}

unsafe impl Send for CallState {}

/// A graph output of the last call. Unless `owner` is null (caller-provided memory), the
/// caller takes over one reference to `owner`. Fixed-size, so calls don't allocate.
#[derive(Clone, Copy)]
pub struct OutArray {
    pub owner: *mut PtBuffer,
    pub data: *mut u8,
    pub dtype: DType,
    pub ndim: usize,
    shape_buf: [i64; MAX_NDIM],
    strides_buf: [i64; MAX_NDIM],
}

impl OutArray {
    fn new(dtype: DType, ndim: usize) -> Self {
        OutArray { owner: std::ptr::null_mut(), data: std::ptr::null_mut(), dtype, ndim, shape_buf: [0; MAX_NDIM], strides_buf: [0; MAX_NDIM] }
    }
    pub fn shape(&self) -> &[i64] {
        &self.shape_buf[..self.ndim]
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides_buf[..self.ndim]
    }
}

unsafe impl Send for OutArray {}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Elemwise { .. } => "Elemwise",
        Op::Reduce { .. } => "Reduce",
        Op::DimShuffle { .. } => "DimShuffle",
        Op::Dot => "Dot",
        Op::Sort { .. } => "Sort",
        Op::Copy => "Copy",
        Op::ShapeI { .. } => "Shape_i",
        Op::Shape => "Shape",
        Op::MakeVector => "MakeVector",
        Op::Assert { .. } => "Assert",
        Op::Arange => "ARange",
        Op::Alloc => "Alloc",
        Op::Segment { .. } => "Segment",
        Op::Foreign { .. } => "Foreign",
        Op::Subtensor { .. } => "Subtensor",
        Op::IncSubtensor { .. } => "IncSubtensor",
        Op::AdvancedTake { .. } => "AdvancedSubtensor",
        Op::AdvancedIncTake { .. } => "AdvancedIncSubtensor",
        Op::Join { .. } => "Join",
        Op::Reshape => "Reshape",
        Op::FusedElemwise { .. } => "FusedElemwise",
    }
}

/// Where each loop dim of a FusedElemwise gets its length: (node input, dim), checked
/// against all other sources at runtime. Candidates for dim d, in order of preference.
fn fused_dim_candidates(g: &Graph, n: &Node, d: usize) -> Vec<(usize, usize)> {
    let Op::FusedElemwise { ndim, n_elem, reads, outs, .. } = &n.op else { unreachable!() };
    let var = |p: usize| &g.vars[n.inputs[p]];
    let mut c = Vec::new();
    for j in 0..*n_elem {
        match &reads[j] {
            None if var(j).ndim() == *ndim && !var(j).bcast[d] => c.push((j, d)),
            Some(r) if r.axis == d => c.push((r.idx, 0)),
            Some(_) if !var(j).bcast[d] => c.push((j, d)),
            _ => {}
        }
    }
    for o in outs {
        if let FusedOut::Write { at, target, .. } = o {
            if at.axis == d {
                c.push((at.idx, 0));
            } else if !var(*target).bcast[d] {
                c.push((*target, d));
            }
        }
    }
    c
}

fn subtensor_params(entries: &[IndexEntry], dtypes: &[DType]) -> Result<SubtensorParams, String> {
    if entries.len() > MAX_NDIM || dtypes.len() > MAX_ARGS {
        return Err("too many indices".into());
    }
    let mut prm = SubtensorParams { n: entries.len() as i64, entries: [IndexSpec::default(); MAX_NDIM], dtypes: [0; MAX_ARGS] };
    for (k, d) in dtypes.iter().enumerate() {
        prm.dtypes[k] = d.code();
    }
    let pos = |p: &Option<usize>| p.map(|x| x as i64).unwrap_or(-1);
    for (d, e) in entries.iter().enumerate() {
        prm.entries[d] = match e {
            IndexEntry::Int(a) => IndexSpec { kind: 1, a: *a as i64, b: -1, c: -1 },
            IndexEntry::Slice { start, stop, step } => IndexSpec { kind: 2, a: pos(start), b: pos(stop), c: pos(step) },
        };
    }
    Ok(prm)
}

#[derive(Clone, Copy)]
enum OutMode {
    Provided,
    KernelAlloc,
    View(usize), // input index
}

fn out_mode(op: &Op) -> OutMode {
    match op {
        Op::DimShuffle { .. } | Op::Assert { .. } | Op::Subtensor { .. } => OutMode::View(0),
        Op::Arange | Op::Alloc | Op::Foreign { .. } => OutMode::KernelAlloc,
        _ => OutMode::Provided,
    }
}

impl Program {
    pub fn new(graph: Graph, consts: Vec<ConstData>, opts: &Options) -> Result<Program, String> {
        let graph = if opts.segments {
            crate::segment::form_segments(graph, opts.segment_max_elems, opts.fastmath)
        } else {
            graph
        };
        let nv = graph.vars.len();
        let nn = graph.nodes.len();
        if consts.len() != graph.constants.len() {
            return Err("constant data count mismatch".into());
        }
        let ndim = |v: usize| graph.vars[v].ndim();
        let mut storage: Vec<Option<Storage>> = vec![None; nv];
        let mut layout = vec![Layout::Any; nv];
        let mut root: Vec<usize> = (0..nv).collect();
        for (k, &v) in graph.inputs.iter().enumerate() {
            storage[v] = Some(Storage::Input(k));
        }
        for (k, &v) in graph.constants.iter().enumerate() {
            storage[v] = Some(Storage::Const(k));
            layout[v] = Layout::C.normalize(ndim(v));
        }

        // ---- output modes and static layouts (forward pass; layouts don't depend on storage)
        let mut modes: Vec<OutMode> = Vec::with_capacity(nn);
        for n in &graph.nodes {
            let mode = match &n.op {
                // Reshape is a view if its input is known to be C-contiguous, else a copy
                Op::Reshape => {
                    if layout[n.inputs[0]] == Layout::C.normalize(ndim(n.inputs[0])) && layout[n.inputs[0]] != Layout::Any {
                        OutMode::View(0)
                    } else {
                        OutMode::KernelAlloc
                    }
                }
                op => out_mode(op),
            };
            match mode {
                OutMode::View(i) => {
                    let (x, o) = (n.inputs[i], n.outputs[0]);
                    layout[o] = match &n.op {
                        Op::DimShuffle { order } => {
                            let kept: Vec<usize> = order.iter().flatten().copied().collect();
                            let l = layout[x];
                            if kept.windows(2).all(|w| w[0] < w[1]) {
                                l
                            } else if kept.windows(2).all(|w| w[0] > w[1]) {
                                match l {
                                    Layout::C => Layout::F,
                                    Layout::F => Layout::C,
                                    Layout::Any => Layout::Any,
                                }
                            } else {
                                Layout::Any
                            }
                        }
                        // integer indices on leading dims of a C array keep it C
                        Op::Subtensor { entries } => {
                            if layout[x] == Layout::C && entries.iter().all(|e| matches!(e, IndexEntry::Int(_))) {
                                Layout::C
                            } else {
                                Layout::Any
                            }
                        }
                        Op::Reshape => Layout::C,
                        _ => layout[x],
                    }
                    .normalize(ndim(o));
                }
                OutMode::KernelAlloc => {
                    for &o in &n.outputs {
                        layout[o] = Layout::C.normalize(ndim(o)); // all of these allocate C arrays
                    }
                }
                OutMode::Provided => {
                    let l = match &n.op {
                        Op::Elemwise { .. } => {
                            let full: Vec<Layout> = n
                                .inputs
                                .iter()
                                .filter(|&&v| ndim(v) >= 2 && graph.vars[v].bcast.iter().all(|b| !b))
                                .map(|&v| layout[v])
                                .collect();
                            if full.contains(&Layout::F) && !full.contains(&Layout::C) {
                                Layout::F
                            } else {
                                Layout::C
                            }
                        }
                        // incremental updates keep x's layout so they can run in place
                        Op::IncSubtensor { .. } | Op::AdvancedIncTake { .. } if layout[n.inputs[0]] != Layout::Any => {
                            layout[n.inputs[0]]
                        }
                        _ => Layout::C,
                    };
                    for &o in &n.outputs {
                        layout[o] = l.normalize(ndim(o));
                    }
                }
            }
            modes.push(mode);
        }

        // ---- roots of declared views
        for (s, n) in graph.nodes.iter().enumerate() {
            if let OutMode::View(i) = modes[s] {
                root[n.outputs[0]] = root[n.inputs[i]];
            }
        }

        // ---- liveness by root (step index == node index)
        const END: usize = usize::MAX - 1;
        let mut last_use: Vec<usize> = vec![0; nv];
        for (s, n) in graph.nodes.iter().enumerate() {
            for &v in &n.inputs {
                last_use[root[v]] = last_use[root[v]].max(s);
            }
        }
        let mut def_step = vec![usize::MAX; nv];
        for (s, n) in graph.nodes.iter().enumerate() {
            for &o in &n.outputs {
                def_step[o] = s;
            }
        }

        // ---- graph outputs: direct if produced by a non-view node and first occurrence
        let mut out_direct = vec![false; graph.outputs.len()];
        let mut is_out = vec![false; nv];
        for (k, &v) in graph.outputs.iter().enumerate() {
            let produced = def_step[v] != usize::MAX && !matches!(modes[def_step[v]], OutMode::View(_));
            if produced && !is_out[v] {
                is_out[v] = true;
                out_direct[k] = true;
            } else {
                last_use[root[v]] = END; // copied after all steps
            }
        }

        // ---- storage, slots, in-place
        let mut free: Vec<usize> = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        let mut slot_last: Vec<usize> = Vec::new();
        let mut n_slots = 0usize;
        let mut inplace_of: Vec<Vec<Option<usize>>> = vec![Vec::new(); nn];
        let mut n_inplace = 0;
        for (s, n) in graph.nodes.iter().enumerate() {
            match modes[s] {
                OutMode::View(i) => {
                    storage[n.outputs[0]] = Some(Storage::View { root: root[n.inputs[i]] });
                    continue;
                }
                OutMode::KernelAlloc => {
                    for &o in &n.outputs {
                        storage[o] = Some(Storage::KernelAlloc);
                    }
                    continue;
                }
                OutMode::Provided => {}
            }
            // inputs whose buffer output k may overwrite
            let inplace_inputs = |k: usize| -> Vec<usize> {
                match &n.op {
                    Op::Elemwise { .. } => (0..n.inputs.len()).collect(),
                    Op::IncSubtensor { .. } | Op::AdvancedIncTake { .. } => vec![0],
                    Op::FusedElemwise { outs, .. } => match &outs[k] {
                        FusedOut::Write { target, .. } => vec![*target],
                        _ => vec![],
                    },
                    _ => vec![],
                }
            };
            let mut used: Vec<usize> = Vec::new();
            let mut ip = vec![None; n.outputs.len()];
            for (k, &o) in n.outputs.iter().enumerate() {
                let lu = last_use[o].max(s);
                let ov = &graph.vars[o];
                let cand = inplace_inputs(k).into_iter().map(|j| (j, n.inputs[j])).find(|&(j, v)| {
                    let st = storage[v].unwrap();
                    let dies_here = match st {
                        Storage::Slot(sl) => slot_last[sl] == s,
                        Storage::KernelAlloc | Storage::Take { .. } => last_use[v] == s && !is_out[v],
                        _ => false,
                    };
                    let iv = &graph.vars[v];
                    dies_here
                        && !used.contains(&v)
                        && iv.dtype == ov.dtype
                        && iv.bcast == ov.bcast
                        && layout[v] == layout[o]
                        // a slot can't escape as a graph output; an owned buffer can
                        && (!is_out[o] || !matches!(st, Storage::Slot(_)))
                        && n.inputs.iter().enumerate().all(|(j2, &v2)| j2 == j || v2 == v || root[v2] != root[v])
                });
                if let Some((j, v)) = cand {
                    used.push(v);
                    ip[k] = Some(j);
                    n_inplace += 1;
                    match storage[v].unwrap() {
                        Storage::Slot(sl) => {
                            slot_last[sl] = lu;
                            storage[o] = Some(Storage::Slot(sl));
                        }
                        _ => storage[o] = Some(Storage::Take { from: v }),
                    }
                    continue;
                }
                if is_out[o] {
                    storage[o] = Some(Storage::Fresh);
                    continue;
                }
                let sl = free.pop().unwrap_or_else(|| {
                    n_slots += 1;
                    slot_last.push(0);
                    n_slots - 1
                });
                slot_last[sl] = lu;
                active.push(sl);
                storage[o] = Some(Storage::Slot(sl));
            }
            inplace_of[s] = ip;
            active.retain(|&sl| {
                if slot_last[sl] <= s {
                    free.push(sl);
                    false
                } else {
                    true
                }
            });
        }
        let mut referenced = vec![false; nv];
        for n in &graph.nodes {
            for &v in n.inputs.iter().chain(&n.outputs) {
                referenced[v] = true;
            }
        }
        for &v in graph.outputs.iter().chain(&graph.inputs) {
            referenced[v] = true;
        }
        let storage: Vec<Storage> = storage
            .into_iter()
            .enumerate()
            .map(|(v, s)| match s {
                Some(s) => Ok(s),
                None if !referenced[v] => Ok(Storage::Internal),
                None => Err(format!("variable {v} is used but never defined")),
            })
            .collect::<Result<_, _>>()?;

        // ---- releases of owned (kernel-allocated) roots
        let mut release: Vec<Vec<usize>> = vec![Vec::new(); nn];
        let mut release_at_end = Vec::new();
        for v in 0..nv {
            let owned = matches!(storage[v], Storage::KernelAlloc | Storage::Take { .. });
            if !owned || is_out[v] {
                continue;
            }
            // a Take output inherits the buffer; the source must not release it
            let taken = storage.iter().any(|s| *s == Storage::Take { from: v });
            if taken {
                continue;
            }
            let lu = last_use[v].max(def_step[v]);
            if lu == END {
                release_at_end.push(v);
            } else {
                release[lu].push(v);
            }
        }

        // ---- kernels
        let mut specs: Vec<KernelSpec> = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        for (ni, n) in graph.nodes.iter().enumerate() {
            let vi = |v: usize| &graph.vars[v];
            let operand = |v: usize| Operand { dtype: vi(v).dtype, bcast: vi(v).bcast.clone(), layout: layout[v] };
            let mut args: Vec<usize> = n.inputs.clone();
            args.extend(&n.outputs);
            let mut member_descs = Vec::new();
            let (src, params, desc) = match &n.op {
                Op::FusedElemwise { scalar, ndim, n_elem, reads, outs } => {
                    let dim_src = (0..*ndim).map(|d| fused_dim_candidates(&graph, n, d).first().copied()).collect();
                    let spec = KernelSpec::Fused {
                        ndim: *ndim,
                        inputs: n.inputs.iter().map(|&v| operand(v)).collect(),
                        n_elem: *n_elem,
                        reads: reads.clone(),
                        outputs: n.outputs.iter().map(|&v| operand(v)).collect(),
                        outs: outs.clone(),
                        dim_src,
                        scalar: scalar.clone(),
                        fastmath: opts.fastmath,
                    };
                    let d = spec.describe();
                    specs.push(spec);
                    (Src::Jit(specs.len() - 1), Params::none(), d)
                }
                Op::Subtensor { entries } => {
                    let dts: Vec<DType> = n.inputs.iter().map(|&v| vi(v).dtype).collect();
                    let prm = subtensor_params(entries, &dts)?;
                    (Src::Native(native::ptl_subtensor), Params::of(prm), "view subtensor".into())
                }
                Op::IncSubtensor { entries, set } => {
                    let dts: Vec<DType> = n.inputs.iter().map(|&v| vi(v).dtype).collect();
                    if dts[1] != dts[0] {
                        return Err("IncSubtensor: y must have the dtype of x".into());
                    }
                    let prm = IncSubtensorParams { sub: subtensor_params(entries, &dts)?, set: *set as i64 };
                    let name = if *set { "set_subtensor" } else { "inc_subtensor" };
                    (Src::Native(native::ptl_inc_subtensor), Params::of(prm), format!("native {name}"))
                }
                Op::AdvancedTake { axis } => {
                    let prm = TakeParams { axis: *axis as i64, idx_dtype: vi(n.inputs[1]).dtype.code(), dtype: vi(n.inputs[0]).dtype.code(), set: 0 };
                    (Src::Native(native::ptl_take), Params::of(prm), "native take".into())
                }
                Op::AdvancedIncTake { axis, set } => {
                    if vi(n.inputs[1]).dtype != vi(n.inputs[0]).dtype {
                        return Err("AdvancedIncSubtensor: y must have the dtype of x".into());
                    }
                    let prm = TakeParams { axis: *axis as i64, idx_dtype: vi(n.inputs[2]).dtype.code(), dtype: vi(n.inputs[0]).dtype.code(), set: *set as i64 };
                    let name = if *set { "set" } else { "inc" };
                    (Src::Native(native::ptl_inc_take), Params::of(prm), format!("native {name}_take"))
                }
                Op::Join { axis } => {
                    let dt = vi(n.outputs[0]).dtype;
                    if n.inputs.iter().any(|&v| vi(v).dtype != dt) {
                        return Err("Join: inputs must have the output dtype".into());
                    }
                    let prm = JoinParams { axis: *axis as i64, n: n.inputs.len() as i64, itemsize: dt.itemsize() as i64 };
                    (Src::Native(native::ptl_join), Params::of(prm), "native join".into())
                }
                Op::Reshape => {
                    let view = matches!(modes[ni], OutMode::View(_));
                    let prm = ReshapeParams { shape_dtype: vi(n.inputs[1]).dtype.code(), itemsize: vi(n.inputs[0]).dtype.itemsize() as i32, view: view as i64 };
                    let d = if view { "view reshape" } else { "native reshape (allocates)" };
                    (Src::Native(native::ptl_reshape), Params::of(prm), d.into())
                }
                Op::Foreign { addr, desc } => {
                    let f = unsafe { std::mem::transmute::<usize, KernelFn>(*addr) };
                    (Src::Native(f), Params::none(), format!("foreign: {desc}"))
                }
                Op::Segment { spec, descs } => {
                    member_descs = descs.clone();
                    let d = format!("Segment[{}]", descs.iter().map(|d| d.chars().take(40).collect::<String>()).collect::<Vec<_>>().join("; "));
                    specs.push(KernelSpec::Segment(spec.clone()));
                    (Src::Jit(specs.len() - 1), Params::none(), d)
                }
                Op::Elemwise { scalar } => {
                    let spec = KernelSpec::Elemwise {
                        ndim: vi(n.outputs[0]).ndim(),
                        inputs: n.inputs.iter().map(|&v| operand(v)).collect(),
                        outputs: n.outputs.iter().map(|&v| operand(v)).collect(),
                        inplace: inplace_of[ni].clone(),
                        scalar: scalar.clone(),
                        fastmath: opts.fastmath,
                    };
                    let d = spec.describe();
                    specs.push(spec);
                    (Src::Jit(specs.len() - 1), Params::none(), d)
                }
                Op::Reduce { op, axes } => {
                    let spec = KernelSpec::Reduce {
                        op: *op,
                        axes: axes.clone(),
                        input: operand(n.inputs[0]),
                        output: operand(n.outputs[0]),
                        fastmath: opts.fastmath,
                    };
                    let d = spec.describe();
                    specs.push(spec);
                    (Src::Jit(specs.len() - 1), Params::none(), d)
                }
                Op::DimShuffle { order } => {
                    let mut o = [0i64; MAX_NDIM];
                    for (k, x) in order.iter().enumerate() {
                        o[k] = x.map(|d| d as i64).unwrap_or(-1);
                    }
                    (Src::Native(native::ptl_dimshuffle), Params::of(DimShuffleParams { order: o }), "view dimshuffle".into())
                }
                Op::Assert { msg } => {
                    for &c in &n.inputs[1..] {
                        if vi(c).ndim() != 0 || vi(c).dtype.itemsize() != 1 {
                            return Err("Assert only supports 0-d boolean conditions".into());
                        }
                    }
                    let prm = AssertParams { n_conds: (n.inputs.len() - 1) as i64 };
                    (Src::Native(native::ptl_assert), Params::of(prm), format!("Assert: {msg}"))
                }
                Op::Dot => {
                    let dt = vi(n.outputs[0]).dtype;
                    if n.inputs.iter().any(|&v| vi(v).dtype != dt) {
                        return Err("Dot with mixed dtypes not supported".into());
                    }
                    let f: KernelFn = match dt {
                        DType::Float64 => native::ptl_matmul_f64,
                        DType::Float32 => native::ptl_matmul_f32,
                        _ => return Err(format!("Dot for {dt:?} not supported")),
                    };
                    (Src::Native(f), Params::none(), "native matmul".into())
                }
                Op::Sort { axis, stable } => {
                    let f: KernelFn = match vi(n.inputs[0]).dtype {
                        DType::Float64 => native::ptl_sort_f64,
                        DType::Float32 => native::ptl_sort_f32,
                        DType::Int64 => native::ptl_sort_i64,
                        dt => return Err(format!("Sort for {dt:?} not supported")),
                    };
                    let prm = SortParams { axis: *axis as i64, stable: *stable as i64 };
                    (Src::Native(f), Params::of(prm), "native sort".into())
                }
                Op::Copy => {
                    let prm = CopyParams { itemsize: vi(n.inputs[0]).dtype.itemsize() as i64 };
                    (Src::Native(native::ptl_copy), Params::of(prm), "native copy".into())
                }
                Op::ShapeI { i } => (Src::Native(native::ptl_shape_i), Params::of(ShapeIParams { i: *i as i64 }), "native shape_i".into()),
                Op::Shape => (Src::Native(native::ptl_shape), Params::none(), "native shape".into()),
                Op::MakeVector => {
                    let dt = vi(n.outputs[0]).dtype;
                    if n.inputs.iter().any(|&v| vi(v).dtype != dt || vi(v).ndim() != 0) {
                        return Err("MakeVector needs 0-d inputs of the output dtype".into());
                    }
                    let prm = MakeVectorParams { n: n.inputs.len() as i64, itemsize: dt.itemsize() as i64 };
                    (Src::Native(native::ptl_make_vector), Params::of(prm), "native make_vector".into())
                }
                Op::Arange => {
                    let d = |v: usize| vi(v).dtype.code();
                    let prm = ArangeParams { dtypes: [d(n.inputs[0]), d(n.inputs[1]), d(n.inputs[2]), d(n.outputs[0])] };
                    (Src::Native(native::ptl_arange), Params::of(prm), "native arange (allocates)".into())
                }
                Op::Alloc => {
                    let mut sd = [0i32; MAX_NDIM];
                    for (k, &v) in n.inputs[1..].iter().enumerate() {
                        sd[k] = vi(v).dtype.code();
                    }
                    let prm = AllocParams {
                        n_shape: (n.inputs.len() - 1) as i64,
                        itemsize: vi(n.outputs[0]).dtype.itemsize() as i64,
                        shape_dtypes: sd,
                    };
                    (Src::Native(native::ptl_alloc), Params::of(prm), "native alloc (allocates)".into())
                }
            };
            let release = std::mem::take(&mut release[ni]);
            steps.push(Step { node: ni, src, func: None, args, outs: n.outputs.clone(), params, desc, member_descs, release });
        }

        let (addrs, _names, stats) = get_kernels(&specs, opts)?;
        for st in &mut steps {
            st.func = Some(match st.src {
                Src::Jit(i) => unsafe { std::mem::transmute::<usize, KernelFn>(addrs[i]) },
                Src::Native(f) => f,
            });
        }

        let mut dim_off = Vec::with_capacity(nv);
        let mut total = 0;
        for v in &graph.vars {
            dim_off.push(total);
            total += v.ndim();
        }
        let static_of = |v: usize| -> Vec<i64> {
            let info = &graph.vars[v];
            (0..info.ndim())
                .map(|d| if info.bcast[d] { 1 } else { info.static_shape.get(d).copied().flatten().unwrap_or(-1) })
                .collect()
        };
        let in_static = graph.inputs.iter().map(|&v| static_of(v)).collect();
        let out_static = graph.outputs.iter().map(|&v| static_of(v)).collect();
        Ok(Program {
            keepalive: Vec::new(),
            in_static,
            out_static,
            graph,
            storage,
            layout,
            root,
            steps,
            out_direct,
            release_at_end,
            n_slots,
            consts,
            dim_off,
            total_dims: total,
            stats,
            n_inplace,
        })
    }

    pub fn keep_alive(&mut self, obj: Box<dyn std::any::Any + Send + Sync>) {
        self.keepalive.push(obj);
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn n_inputs(&self) -> usize {
        self.graph.inputs.len()
    }

    pub fn n_outputs(&self) -> usize {
        self.graph.outputs.len()
    }

    pub fn input_type(&self, k: usize) -> (DType, &[i64]) {
        (self.graph.vars[self.graph.inputs[k]].dtype, &self.in_static[k])
    }

    pub fn output_type(&self, k: usize) -> (DType, &[i64]) {
        (self.graph.vars[self.graph.outputs[k]].dtype, &self.out_static[k])
    }

    pub fn kernel_descriptions(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.desc.clone()).collect()
    }

    /// (variable, storage kind, layout) for every node output, for inspection.
    pub fn plan_description(&self) -> Vec<(String, String, String)> {
        let mut v = Vec::new();
        for n in &self.graph.nodes {
            for &o in &n.outputs {
                let st = match self.storage[o] {
                    Storage::Slot(s) => format!("slot {s}"),
                    Storage::Fresh => "fresh (output)".into(),
                    Storage::KernelAlloc => "kernel-allocated".into(),
                    Storage::View { root } => format!("view of v{root}"),
                    Storage::Take { from } => format!("in-place, takes v{from}"),
                    Storage::Input(_) | Storage::Const(_) | Storage::Internal => unreachable!(),
                };
                v.push((format!("v{o} ({})", op_name(&n.op)), st, format!("{:?}", self.layout[o])));
            }
        }
        v
    }

    pub fn new_state(&self) -> CallState {
        CallState {
            shapes: vec![0; self.total_dims],
            strides: vec![0; self.total_dims],
            ptrs: vec![std::ptr::null_mut(); self.graph.vars.len()],
            owners: vec![std::ptr::null_mut(); self.graph.vars.len()],
            slots: (0..self.n_slots).map(|_| SlotBuf::new()).collect(),
            args: Vec::with_capacity(16),
            ctx: PtCtx::new(),
            profile: None,
            prov_var: vec![None; self.graph.vars.len()],
            direct_written: vec![false; self.graph.outputs.len()],
            outs: self.graph.outputs.iter().map(|&v| OutArray::new(self.graph.vars[v].dtype, self.graph.vars[v].ndim())).collect(),
        }
    }

    fn shape<'a>(&self, st: &'a CallState, v: usize) -> &'a [i64] {
        let o = self.dim_off[v];
        &st.shapes[o..o + self.graph.vars[v].ndim()]
    }

    /// Bind input k. `strides` are in bytes.
    pub fn set_input(&self, st: &mut CallState, k: usize, data: *mut u8, dtype: Option<DType>, shape: &[usize], strides: &[isize]) -> Result<(), String> {
        let v = self.graph.inputs[k];
        let info = &self.graph.vars[v];
        if dtype != Some(info.dtype) {
            return Err(format!("input {k}: expected dtype {}, got {:?}", info.dtype.numpy_name(), dtype));
        }
        self.bind(st, k, data, shape.len(), |d| (shape[d] as i64, strides[d] as i64))
    }

    /// Bind input k from raw int64 shape/strides (C API; dtype is trusted).
    pub fn set_input_raw(&self, st: &mut CallState, k: usize, data: *mut u8, shape: &[i64], strides: &[i64]) -> Result<(), String> {
        self.bind(st, k, data, shape.len(), |d| (shape[d], strides[d]))
    }

    fn bind(&self, st: &mut CallState, k: usize, data: *mut u8, nd: usize, dim: impl Fn(usize) -> (i64, i64)) -> Result<(), String> {
        let v = self.graph.inputs[k];
        let info = &self.graph.vars[v];
        if nd != info.ndim() {
            return Err(format!("input {k}: expected {} dims, got {}", info.ndim(), nd));
        }
        let o = self.dim_off[v];
        for d in 0..nd {
            let (s, t) = dim(d);
            st.shapes[o + d] = s;
            st.strides[o + d] = t;
        }
        let shape = &st.shapes[o..o + nd];
        for d in 0..shape.len() {
            if info.bcast[d] && shape[d] != 1 {
                return Err(format!("input {k}: dim {d} is broadcastable but has length {}", shape[d]));
            }
            if let Some(Some(s)) = info.static_shape.get(d) {
                if *s != shape[d] {
                    return Err(format!("input {k}: dim {d} has static length {s}, got {}", shape[d]));
                }
            }
        }
        st.ptrs[v] = data;
        Ok(())
    }

    /// Shapes of the provided outputs of node `n`, from its inputs' (already known) shapes.
    fn infer_shapes(&self, st: &mut CallState, ni: usize) -> Result<(), String> {
        let n = &self.graph.nodes[ni];
        let off = &self.dim_off;
        let sh = &mut st.shapes;
        let o0 = off[n.outputs[0]];
        match &n.op {
            Op::Elemwise { .. } => {
                let ov = &self.graph.vars[n.outputs[0]];
                for d in 0..ov.ndim() {
                    let mut val: i64 = -1;
                    if !ov.bcast[d] {
                        for &i in &n.inputs {
                            if self.graph.vars[i].bcast[d] {
                                continue;
                            }
                            let s = sh[off[i] + d];
                            if val < 0 {
                                val = s;
                            } else if val != s {
                                return Err(format!("Elemwise shape mismatch in dim {d}: {val} vs {s} (runtime broadcasting is not allowed)"));
                            }
                        }
                    }
                    let val = if val < 0 { 1 } else { val };
                    for &o in &n.outputs {
                        sh[off[o] + d] = val;
                    }
                }
            }
            Op::Reduce { axes, .. } => {
                let (i0, nd) = (off[n.inputs[0]], self.graph.vars[n.inputs[0]].ndim());
                let mut k = 0;
                for d in 0..nd {
                    if !axes.contains(&d) {
                        sh[o0 + k] = sh[i0 + d];
                        k += 1;
                    }
                }
            }
            Op::Dot => {
                let (a, b) = (n.inputs[0], n.inputs[1]);
                let (na, nb) = (self.graph.vars[a].ndim(), self.graph.vars[b].ndim());
                let (oa, ob) = (off[a], off[b]);
                if sh[oa + na - 1] != sh[ob] {
                    return Err(format!("Dot shape mismatch: {:?} @ {:?}", &sh[oa..oa + na], &sh[ob..ob + nb]));
                }
                let mut k = 0;
                if na == 2 {
                    sh[o0] = sh[oa];
                    k = 1;
                }
                if nb == 2 {
                    sh[o0 + k] = sh[ob + 1];
                }
            }
            Op::Sort { .. } | Op::Copy => {
                let i0 = off[n.inputs[0]];
                for d in 0..self.graph.vars[n.inputs[0]].ndim() {
                    sh[o0 + d] = sh[i0 + d];
                }
            }
            Op::ShapeI { .. } => {}
            Op::Shape => sh[o0] = self.graph.vars[n.inputs[0]].ndim() as i64,
            Op::MakeVector => sh[o0] = n.inputs.len() as i64,
            Op::DimShuffle { .. } | Op::Assert { .. } | Op::Arange | Op::Alloc | Op::Foreign { .. } | Op::Subtensor { .. } | Op::Reshape => {} // kernel fills shapes
            Op::FusedElemwise { ndim, outs, .. } => {
                let mut loop_shape = [1i64; MAX_NDIM];
                for d in 0..*ndim {
                    let mut val: Option<i64> = None;
                    for (p, pd) in fused_dim_candidates(&self.graph, n, d) {
                        let s = sh[off[n.inputs[p]] + pd];
                        match val {
                            None => val = Some(s),
                            Some(x) if x != s => {
                                return Err(format!("FusedElemwise shape mismatch in loop dim {d}: {x} vs {s}"));
                            }
                            _ => {}
                        }
                    }
                    loop_shape[d] = val.unwrap_or(1);
                }
                for (k, o) in outs.iter().enumerate() {
                    let ob = off[n.outputs[k]];
                    match o {
                        FusedOut::Plain => sh[ob..ob + ndim].copy_from_slice(&loop_shape[..*ndim]),
                        FusedOut::Reduce { axes, .. } => {
                            let mut q = 0;
                            for d in 0..*ndim {
                                if !axes.contains(&d) {
                                    sh[ob + q] = loop_shape[d];
                                    q += 1;
                                }
                            }
                        }
                        FusedOut::Write { target, .. } => {
                            let t = off[n.inputs[*target]];
                            for d in 0..*ndim {
                                sh[ob + d] = sh[t + d];
                            }
                        }
                    }
                }
            }
            Op::IncSubtensor { .. } | Op::AdvancedIncTake { .. } => {
                let i0 = off[n.inputs[0]];
                for d in 0..self.graph.vars[n.inputs[0]].ndim() {
                    sh[o0 + d] = sh[i0 + d];
                }
            }
            Op::AdvancedTake { axis } => {
                let (x, idx) = (n.inputs[0], n.inputs[1]);
                let (xn, kn) = (self.graph.vars[x].ndim(), self.graph.vars[idx].ndim());
                let mut k = 0;
                for d in 0..xn {
                    if d == *axis {
                        for j in 0..kn {
                            sh[o0 + k] = sh[off[idx] + j];
                            k += 1;
                        }
                    } else {
                        sh[o0 + k] = sh[off[x] + d];
                        k += 1;
                    }
                }
            }
            Op::Join { axis } => {
                let first = n.inputs[0];
                let nd = self.graph.vars[first].ndim();
                let mut total = 0;
                for &i in &n.inputs {
                    for d in 0..nd {
                        if d != *axis && sh[off[i] + d] != sh[off[first] + d] {
                            return Err(format!(
                                "Join: all inputs must have the same shape except along axis {axis}: {:?} vs {:?}",
                                &sh[off[first]..off[first] + nd],
                                &sh[off[i]..off[i] + nd]
                            ));
                        }
                    }
                    total += sh[off[i] + axis];
                }
                for d in 0..nd {
                    sh[o0 + d] = if d == *axis { total } else { sh[off[first] + d] };
                }
            }
            Op::Segment { spec, .. } => {
                for (k, &o) in n.outputs.iter().enumerate() {
                    let s = spec.out_shape(k);
                    sh[off[o]..off[o] + s.len()].copy_from_slice(s);
                }
            }
        }
        Ok(())
    }

    fn set_contiguous_strides(&self, st: &mut CallState, v: usize) {
        let o = self.dim_off[v];
        let nd = self.graph.vars[v].ndim();
        let isz = self.graph.vars[v].dtype.itemsize() as i64;
        let (sh, strd) = (&st.shapes[o..o + nd], &mut st.strides[o..o + nd]);
        self.layout[v].strides(sh, isz, strd);
    }

    fn nbytes(&self, st: &CallState, v: usize) -> usize {
        self.shape(st, v).iter().product::<i64>() as usize * self.graph.vars[v].dtype.itemsize()
    }

    /// Release every owned buffer still held (error path, or leftovers).
    fn release_all(&self, st: &mut CallState) {
        for v in 0..st.owners.len() {
            let is_const = matches!(self.storage[v], Storage::Const(_));
            if !st.owners[v].is_null() && !is_const {
                unsafe { PtBuffer::decref(st.owners[v]) };
            }
            st.owners[v] = std::ptr::null_mut();
        }
    }

    /// Run the program. Inputs must have been bound with `set_input`.
    ///
    /// `provided[k]` with non-null data is caller memory for output k (see the C header).
    /// Outputs written there are returned with a null owner.
    /// On success the outputs are in `st.outs`.
    pub fn execute(&self, st: &mut CallState, provided: Option<&[PtArray]>) -> Result<(), String> {
        let res = self.execute_inner(st, provided);
        if res.is_err() {
            self.release_all(st);
        }
        res
    }

    fn execute_inner(&self, st: &mut CallState, provided: Option<&[PtArray]>) -> Result<(), String> {
        st.prov_var.iter_mut().for_each(|x| *x = None);
        st.direct_written.iter_mut().for_each(|x| *x = false);
        if let Some(p) = provided {
            for (k, &v) in self.graph.outputs.iter().enumerate() {
                if !p[k].data.is_null() && self.out_direct[k] && self.storage[v] == Storage::Fresh {
                    st.prov_var[v] = Some(k);
                }
            }
        }
        for (k, &v) in self.graph.constants.iter().enumerate() {
            let c = &self.consts[k];
            let o = self.dim_off[v];
            let nd = c.shape.len();
            st.shapes[o..o + nd].copy_from_slice(&c.shape);
            st.strides[o..o + nd].copy_from_slice(&c.strides);
            st.ptrs[v] = unsafe { (*c.buf).data };
            st.owners[v] = c.buf; // borrowed; never released by the driver
        }
        let mut args = std::mem::take(&mut st.args);
        for (si, step) in self.steps.iter().enumerate() {
            // memory for provided outputs
            self.infer_shapes(st, step.node)?;
            for &o in &step.outs {
                match self.storage[o] {
                    Storage::Slot(sl) => {
                        let nb = self.nbytes(st, o);
                        st.slots[sl].ensure(nb);
                        st.ptrs[o] = st.slots[sl].ptr;
                        self.set_contiguous_strides(st, o);
                    }
                    Storage::Fresh if st.prov_var[o].is_some() && {
                        // write straight into caller memory if its strides are the planned ones
                        let k = st.prov_var[o].unwrap();
                        let p = &provided.unwrap()[k];
                        self.set_contiguous_strides(st, o);
                        let (off, nd) = (self.dim_off[o], self.graph.vars[o].ndim());
                        let (sh, sd) = unsafe { (p.shape(), p.strides()) };
                        p.ndim as usize == nd
                            && (0..nd).all(|d| sh[d] == st.shapes[off + d] && (sh[d] == 1 || sd[d] == st.strides[off + d]))
                    } =>
                    {
                        let k = st.prov_var[o].unwrap();
                        st.ptrs[o] = provided.unwrap()[k].data;
                        st.direct_written[k] = true;
                    }
                    Storage::Fresh => {
                        let b = PtBuffer::new(self.nbytes(st, o), 64);
                        st.owners[o] = b;
                        st.ptrs[o] = unsafe { (*b).data };
                        self.set_contiguous_strides(st, o);
                    }
                    Storage::Take { from } => {
                        st.ptrs[o] = st.ptrs[from];
                        st.owners[o] = std::mem::replace(&mut st.owners[from], std::ptr::null_mut());
                        let (a, b) = (self.dim_off[from], self.dim_off[o]);
                        for d in 0..self.graph.vars[o].ndim() {
                            st.strides[b + d] = st.strides[a + d];
                        }
                    }
                    Storage::KernelAlloc | Storage::View { .. } => {}
                    Storage::Input(_) | Storage::Const(_) | Storage::Internal => unreachable!(),
                }
            }
            args.clear();
            for &v in &step.args {
                let o = self.dim_off[v];
                args.push(PtArray {
                    data: st.ptrs[v],
                    shape: unsafe { st.shapes.as_mut_ptr().add(o) },
                    strides: unsafe { st.strides.as_mut_ptr().add(o) },
                    owner: match self.storage[v] {
                        Storage::View { root } => st.owners[root],
                        _ => st.owners[v],
                    },
                    ndim: self.graph.vars[v].ndim() as i64,
                });
            }
            let f = step.func.unwrap();
            let t = st.profile.as_ref().map(|_| std::time::Instant::now());
            let ctx: *mut PtCtx = &mut st.ctx;
            let rc = unsafe { f(ctx, args.as_mut_ptr(), step.params.as_ptr()) };
            st.ctx.end_kernel();
            if let (Some(t), Some(p)) = (t, st.profile.as_mut()) {
                p[si] += t.elapsed().as_secs_f64();
            }
            // collect what the kernel wrote for kernel-allocated outputs and views
            let ni = self.graph.nodes[step.node].inputs.len();
            for (k, &o) in step.outs.iter().enumerate() {
                match self.storage[o] {
                    Storage::KernelAlloc => {
                        st.ptrs[o] = args[ni + k].data;
                        st.owners[o] = args[ni + k].owner;
                    }
                    Storage::View { .. } => st.ptrs[o] = args[ni + k].data,
                    _ => {}
                }
            }
            if rc != 0 {
                st.args = args;
                let msg = st.ctx.error.take().map(|(_, m)| m);
                let desc = match step.member_descs.get((rc as usize).wrapping_sub(1)) {
                    Some(d) => d.as_str(),
                    None => step.desc.as_str(),
                };
                return Err(match (msg, desc.strip_prefix("Assert: ")) {
                    (Some(m), _) => m,
                    (None, Some(m)) => m.to_string(),
                    (None, None) if step.desc.starts_with("FusedElemwise") => format!("index out of bounds in {}", step.desc),
                    (None, None) if step.desc.starts_with("foreign: ") => {
                        format!("{} failed (an exception was raised in its numba implementation)", &step.desc[9..])
                    }
                    (None, None) => format!("kernel {} (node {}) failed with code {rc}", step.desc, step.node),
                });
            }
            for &v in &step.release {
                let b = std::mem::replace(&mut st.owners[v], std::ptr::null_mut());
                if !b.is_null() {
                    unsafe { PtBuffer::decref(b) };
                }
            }
        }
        st.args = args;

        // graph outputs
        for (k, &v) in self.graph.outputs.iter().enumerate() {
            let info = &self.graph.vars[v];
            let nd = info.ndim();
            let o = self.dim_off[v];
            let isz = info.dtype.itemsize() as i64;
            let mut out = OutArray::new(info.dtype, nd);
            out.shape_buf[..nd].copy_from_slice(&st.shapes[o..o + nd]);
            let src = PtArray {
                data: st.ptrs[v],
                shape: st.shapes.as_mut_ptr().wrapping_add(o),
                strides: st.strides.as_mut_ptr().wrapping_add(o),
                owner: std::ptr::null_mut(),
                ndim: nd as i64,
            };
            let prm = CopyParams { itemsize: isz };
            let prm_ptr = &prm as *const CopyParams as *const std::ffi::c_void;
            if let Some(p) = provided.map(|p| p[k]).filter(|p| !p.data.is_null()) {
                let (psh, pst) = unsafe { (p.shape(), p.strides()) };
                if p.ndim as usize != nd || (0..nd).any(|d| psh[d] != out.shape_buf[d]) {
                    return Err(format!("provided output {k} has shape {psh:?}, result has shape {:?}", out.shape()));
                }
                if !st.direct_written[k] {
                    let mut a = [src, p];
                    unsafe { native::ptl_copy(&mut st.ctx, a.as_mut_ptr(), prm_ptr) };
                    if self.out_direct[k] {
                        let b = std::mem::replace(&mut st.owners[v], std::ptr::null_mut());
                        if !b.is_null() {
                            unsafe { PtBuffer::decref(b) };
                        }
                    }
                }
                out.data = p.data;
                out.strides_buf[..nd].copy_from_slice(pst);
            } else if self.out_direct[k] {
                out.owner = std::mem::replace(&mut st.owners[v], std::ptr::null_mut());
                out.data = st.ptrs[v];
                out.strides_buf[..nd].copy_from_slice(&st.strides[o..o + nd]);
            } else {
                // inputs, constants, views or repeated outputs: copy into a fresh C array
                Layout::C.strides(&out.shape_buf[..nd], isz, &mut out.strides_buf[..nd]);
                let n: i64 = out.shape().iter().product();
                let b = PtBuffer::new(n as usize * isz as usize, 64);
                out.owner = b;
                out.data = unsafe { (*b).data };
                let dst = PtArray { data: out.data, shape: out.shape_buf.as_mut_ptr(), strides: out.strides_buf.as_mut_ptr(), owner: b, ndim: nd as i64 };
                let mut a = [src, dst];
                unsafe { native::ptl_copy(&mut st.ctx, a.as_mut_ptr(), prm_ptr) };
            }
            st.outs[k] = out;
        }
        for &v in &self.release_at_end {
            let b = std::mem::replace(&mut st.owners[v], std::ptr::null_mut());
            if !b.is_null() {
                unsafe { PtBuffer::decref(b) };
            }
        }
        // constants are borrowed; clear so release_all never touches them
        for &v in &self.graph.constants {
            st.owners[v] = std::ptr::null_mut();
        }
        debug_assert!(st.owners.iter().all(|p| p.is_null()), "leaked buffer");
        let _ = &self.root;
        Ok(())
    }
}
