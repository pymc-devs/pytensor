//! The interpreter: runs a task list planned in Python.
//!
//! Every task calls a kernel with the kernel ABI
//! `int32 f(PtCtx*, PtArray* args /* inputs, outputs */, const void* config)`:
//! a generated kernel, a Rust kernel, a subprogram, or a numba
//! kernel. The program doesn't know which.
//!
//! Python decides everything else: which values live where (arena offsets for static
//! temporaries, fresh buffers for graph outputs, buffers the kernels
//! allocate, views), when values are released, and how outputs are produced. The
//! interpreter checks what it can cheaply (shapes of kernel-set outputs against the
//! static types) and cleans up on errors.

use super::consts::ConstData;
use crate::abi::{validate_array, PtArray, PtBuffer};
use crate::runtime::{PtCtx};
use crate::dtype::DType;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

pub type R<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    fn i(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug)]
pub enum ValueKind {
    /// graph input `k`
    Input(usize),
    Const(Arc<ConstData>),
    /// a static-shape temporary at this arena offset
    Arena(u64),
    /// a static-shape graph output: a fresh buffer per call (or the caller's memory)
    Fresh,
    /// Returned by the producing task: a buffer it allocated (it hands over one
    /// reference), or a view into the memory of one of these values (the planner keeps
    /// all of them alive as long as this one).
    Returned(Vec<ValueId>),
}

#[derive(Clone, Debug)]
pub struct ValueSpec {
    pub dtype: DType,
    pub shape: Vec<Option<u64>>,
    pub kind: ValueKind,
    /// memory order of memory the program provides (arena, fresh), outermost axis
    /// first; empty: C order
    pub order: Vec<usize>,
}

impl ValueSpec {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.shape.iter().map(|s| s.map(|n| n as i64)).collect()
    }

    pub fn static_nbytes(&self) -> Option<usize> {
        let s = self.static_shape()?;
        crate::abi::checked_nbytes(&s, self.dtype.itemsize())
    }
}

pub use crate::abi::OutputMode;
use crate::abi::KernelFn;

pub struct Task {
    pub func: usize,
    pub config: usize,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub release: Vec<ValueId>,
    pub desc: String,
}


pub struct ProgramSpec {
    pub values: Vec<ValueSpec>,
    pub n_inputs: usize,
    pub tasks: Vec<Task>,
    pub outputs: Vec<ValueId>,
    /// how each graph output is produced when the program runs as a task of another
    /// program: provided or allocated (never a view)
    pub output_modes: Vec<OutputMode>,
    pub arena_bytes: u64,
    /// inputs the generated code assumes C-contiguous (checked by `set_input`)
    pub contiguous_inputs: Vec<bool>,
}

pub struct Program {
    spec: ProgramSpec,
    /// the first graph output each Fresh value is
    fresh_output: Vec<Option<usize>>,
    memories: Mutex<Vec<CallMemory>>,
    profile: Profile,
}

/// Wall time per task, summed over all calls (from any thread) while profiling is on.
pub struct Profile {
    on: AtomicBool,
    calls: AtomicU64,
    task_ns: Box<[AtomicU64]>,
}

impl Profile {
    fn new(n_tasks: usize) -> Profile {
        Profile { on: AtomicBool::new(false), calls: AtomicU64::new(0), task_ns: (0..n_tasks).map(|_| AtomicU64::new(0)).collect() }
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// Per-call memory of one program; not shared between concurrent calls. (The context,
/// `PtCtx`, belongs to the caller's session: subprograms run with their
/// caller's.)
pub struct CallMemory {
    vals: Vec<PtArray>,
    /// shape ++ strides of every value
    storage: Vec<Box<[i64]>>,
    owned: Vec<bool>,
    // (owns the memory arena values point into)
    #[allow(dead_code)]
    arena: Vec<Align64>,
    args: Vec<PtArray>,
}

// SAFETY: the raw pointers in a CallMemory point into its own storage and arena, into
// buffers it owns, or into constants of its program; nothing is shared between memories.
unsafe impl Send for CallMemory {}

/// An output handed to the caller: one buffer reference.
pub struct OutArray {
    pub data: *mut u8,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub owner: *mut PtBuffer,
}

// SAFETY: an OutArray is one buffer reference, owned by whoever holds it.
unsafe impl Send for OutArray {}

/// numpy's C-contiguity: length-1 axes and empty arrays don't count.
pub fn is_c_contiguous(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut acc = itemsize as i64;
    for (&n, &s) in shape.iter().zip(strides).rev() {
        if n != 1 && s != acc {
            return false;
        }
        acc *= n;
    }
    true
}

/// Strides of an array packed in axis `order` (outermost first; empty: C order).
pub fn packed_strides(shape: &[i64], itemsize: i64, order: &[usize]) -> Vec<i64> {
    if order.is_empty() {
        return c_strides(shape, itemsize);
    }
    let mut st = vec![0i64; shape.len()];
    let mut acc = itemsize;
    for &a in order.iter().rev() {
        st[a] = acc;
        acc = acc.saturating_mul(shape[a].max(1));
    }
    st
}

pub fn c_strides(shape: &[i64], itemsize: i64) -> Vec<i64> {
    let mut st = vec![0i64; shape.len()];
    let mut acc = itemsize;
    for d in (0..shape.len()).rev() {
        st[d] = acc;
        acc = acc.saturating_mul(shape[d].max(1));
    }
    st
}

fn check_shape(v: &ValueSpec, shape: &[i64]) -> R<()> {
    if shape.len() != v.ndim() {
        return Err(format!("expected {} dimensions, got {}", v.ndim(), shape.len()));
    }
    for (a, (&n, s)) in shape.iter().zip(&v.shape).enumerate() {
        if n < 0 {
            return Err(format!("negative length on axis {a}"));
        }
        if let Some(s) = s {
            if n as u64 != *s {
                return Err(format!("length {n} on axis {a}, the type says {s}"));
            }
        }
    }
    Ok(())
}

impl Program {
    /// Check a spec for consistency (ids in range, tasks produce their outputs once,
    /// arena placement inside the arena) and build the program.
    pub fn new(spec: ProgramSpec) -> R<Program> {
        let n = spec.values.len();
        let ok = |v: ValueId| -> R<()> {
            if v.i() < n {
                Ok(())
            } else {
                Err(format!("value {} out of range", v.0))
            }
        };
        let mut produced = vec![false; n];
        for (k, v) in spec.values.iter().enumerate() {
            if !v.order.is_empty() {
                let mut seen = v.order.clone();
                seen.sort_unstable();
                if seen != (0..v.ndim()).collect::<Vec<_>>() {
                    return Err(format!("value {k}: memory order {:?} isn't a permutation of the axes", v.order));
                }
                if !matches!(v.kind, ValueKind::Arena(_)) {
                    return Err(format!("value {k}: only arena values have a memory order"));
                }
            }
            match &v.kind {
                ValueKind::Input(i) => {
                    if *i >= spec.n_inputs {
                        return Err(format!("value {k}: input {i} out of range"));
                    }
                    produced[k] = true;
                }
                ValueKind::Const(c) => {
                    if c.dtype != v.dtype || c.shape.len() != v.ndim() {
                        return Err(format!("value {k}: constant doesn't match its type"));
                    }
                    produced[k] = true;
                }
                ValueKind::Arena(off) => {
                    let nb = v.static_nbytes().ok_or(format!("value {k}: arena value without static size"))? as u64;
                    if off % 64 != 0 || off + nb > spec.arena_bytes {
                        return Err(format!("value {k}: arena placement {off}+{nb} outside the arena"));
                    }
                }
                ValueKind::Fresh => {
                    v.static_nbytes().ok_or(format!("value {k}: fresh value without static size"))?;
                }
                ValueKind::Returned(sources) => {
                    for src in sources {
                        ok(*src)?;
                    }
                }
            }
        }
        for s in &spec.tasks {
            if s.func == 0 {
                return Err(format!("task {}: null function", s.desc));
            }
            for &v in s.inputs.iter().chain(&s.release) {
                ok(v)?;
            }
            for &v in &s.inputs {
                if !produced[v.i()] {
                    return Err(format!("task {}: value {} used before it is produced", s.desc, v.0));
                }
            }
            for &v in &s.outputs {
                ok(v)?;
                // (an output that is also an input: the task works in place)
                let in_place = s.inputs.contains(&v);
                if in_place && matches!(spec.values[v.i()].kind, ValueKind::Const(_)) {
                    return Err(format!("task {}: a constant can't be overwritten", s.desc));
                }
                if !in_place && (produced[v.i()] || matches!(spec.values[v.i()].kind, ValueKind::Input(_) | ValueKind::Const(_))) {
                    return Err(format!("task {}: value {} produced twice", s.desc, v.0));
                }
                produced[v.i()] = true;
            }
        }
        for &v in &spec.outputs {
            ok(v)?;
            if !produced[v.i()] {
                return Err(format!("output value {} is never produced", v.0));
            }
        }
        if spec.output_modes.len() != spec.outputs.len() {
            return Err("one output mode per output".into());
        }
        let mut fresh_output = vec![None; n];
        for (o, &v) in spec.outputs.iter().enumerate() {
            if fresh_output[v.i()].is_none() {
                fresh_output[v.i()] = Some(o);
            }
        }
        let profile = Profile::new(spec.tasks.len());
        Ok(Program { spec, fresh_output, memories: Mutex::new(vec![]), profile })
    }

    pub fn spec(&self) -> &ProgramSpec {
        &self.spec
    }

    fn new_memory(&self) -> CallMemory {
        let mut storage: Vec<Box<[i64]>> = Vec::with_capacity(self.spec.values.len());
        for v in &self.spec.values {
            let nd = v.ndim();
            let mut s = vec![0i64; 2 * nd].into_boxed_slice();
            if let Some(shape) = v.static_shape() {
                let st = packed_strides(&shape, v.dtype.itemsize() as i64, &v.order);
                s[..nd].copy_from_slice(&shape);
                s[nd..].copy_from_slice(&st);
            }
            if let ValueKind::Const(c) = &v.kind {
                s[..nd].copy_from_slice(&c.shape);
                s[nd..].copy_from_slice(&c.strides);
            }
            storage.push(s);
        }
        let mut arena = vec![Align64([0; 64]); self.spec.arena_bytes.div_ceil(64) as usize];
        let base = arena.as_mut_ptr() as *mut u8;
        let mut vals = Vec::with_capacity(self.spec.values.len());
        for (k, v) in self.spec.values.iter().enumerate() {
            let nd = v.ndim();
            let st = storage[k].as_mut_ptr();
            let data = match &v.kind {
                ValueKind::Const(c) => c.data() as *mut u8,
                // SAFETY: arena placements were checked to lie inside the arena.
                ValueKind::Arena(off) => unsafe { base.add(*off as usize) },
                _ => std::ptr::null_mut(),
            };
            // SAFETY: the storage has 2 * ndim entries.
            let strides = unsafe { st.add(nd) };
            vals.push(PtArray { data, shape: st, strides, owner: std::ptr::null_mut(), ndim: nd as i64 });
        }
        CallMemory { vals, storage, owned: vec![false; self.spec.values.len()], arena, args: vec![] }
    }

    pub fn take_memory(&self) -> CallMemory {
        let s = self.memories.lock().unwrap_or_else(|e| e.into_inner()).pop();
        s.unwrap_or_else(|| self.new_memory())
    }

    pub fn put_memory(&self, st: CallMemory) {
        self.memories.lock().unwrap_or_else(|e| e.into_inner()).push(st);
    }

    fn input_value(&self, k: usize) -> ValueId {
        let i = self.spec.values.iter().position(|v| matches!(v.kind, ValueKind::Input(j) if j == k));
        ValueId(i.expect("every input has a value") as u32)
    }

    /// Set graph input `k`; returns the byte range it covers. The memory must stay
    /// valid until `finish`.
    pub fn set_input(&self, st: &mut CallMemory, k: usize, data: *mut u8, shape: &[i64], strides: &[i64]) -> R<(usize, usize)> {
        let v = self.input_value(k);
        let spec = &self.spec.values[v.i()];
        check_shape(spec, shape).map_err(|m| format!("input {k}: {m}"))?;
        let range = validate_array(data, shape, strides, spec.dtype.itemsize()).map_err(|m| format!("input {k}: {m}"))?;
        if self.input_contiguous(k) && !is_c_contiguous(shape, strides, spec.dtype.itemsize()) {
            return Err(format!("input {k}: must be C-contiguous (the function was compiled for C-contiguous inputs)"));
        }
        let nd = shape.len();
        st.storage[v.i()][..nd].copy_from_slice(shape);
        st.storage[v.i()][nd..].copy_from_slice(strides);
        st.vals[v.i()].data = data;
        st.vals[v.i()].owner = std::ptr::null_mut();
        Ok(range)
    }

    /// Whether input `k` must be C-contiguous.
    pub fn input_contiguous(&self, k: usize) -> bool {
        self.spec.contiguous_inputs.get(k).copied().unwrap_or(false)
    }

    pub fn input_ndim(&self, k: usize) -> usize {
        self.spec.values[self.input_value(k).i()].ndim()
    }

    /// Static lengths of input `k`, -1 where only known at run time.
    pub fn input_shape(&self, k: usize) -> Vec<i64> {
        let v = &self.spec.values[self.input_value(k).i()];
        v.shape.iter().map(|s| s.map_or(-1, |n| n as i64)).collect()
    }

    /// Static lengths of output `o`, -1 where only known at run time.
    pub fn output_shape(&self, o: usize) -> Vec<i64> {
        let v = &self.spec.values[self.spec.outputs[o].i()];
        v.shape.iter().map(|s| s.map_or(-1, |n| n as i64)).collect()
    }

    /// Whether memory provided for output `o` (C-contiguous) is written directly by the
    /// task producing it (otherwise the result is copied there).
    pub fn writes_in_place(&self, o: usize) -> bool {
        let v = self.spec.outputs[o];
        self.spec.values[v.i()].kind.is_fresh() && self.fresh_output[v.i()] == Some(o)
    }

    /// Run all tasks. `provided[o]` is C-contiguous memory for static graph output `o`.
    pub fn execute(&self, st: &mut CallMemory, ctx: &mut PtCtx, provided: &[Option<*mut u8>]) -> R<()> {
        // whatever happens, the call's scratch is released (a panic skips the tasks' own
        // releases)
        let base = ctx.scratch_mark();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute_inner(st, ctx, provided)))
            .unwrap_or_else(|p| {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
                Err(format!("internal error (panic): {}", msg.unwrap_or_default()))
            });
        ctx.scratch_release(base);
        if r.is_err() {
            self.release_all(st);
        }
        r
    }

    fn execute_inner(&self, st: &mut CallMemory, ctx: &mut PtCtx, provided: &[Option<*mut u8>]) -> R<()> {
        let values = &self.spec.values;
        let profiling = self.profile.on.load(Relaxed);
        if profiling {
            self.profile.calls.fetch_add(1, Relaxed);
        }
        for (k_task, task) in self.spec.tasks.iter().enumerate() {
            st.args.clear();
            for &v in &task.inputs {
                st.args.push(st.vals[v.i()]);
            }
            for &v in &task.outputs {
                let mut a = st.vals[v.i()];
                if task.inputs.contains(&v) {
                    // in place: the output is the input's memory, as it is now
                    st.args.push(a);
                    continue;
                }
                match values[v.i()].kind {
                    ValueKind::Fresh => match self.fresh_output[v.i()].and_then(|o| provided.get(o).copied().flatten()) {
                        Some(p) => {
                            a.data = p;
                            a.owner = std::ptr::null_mut();
                        }
                        None => {
                            let nb = values[v.i()].static_nbytes().unwrap_or(0);
                            // SAFETY: ctx is the live context.
                            let (data, owner) = unsafe { PtCtx::alloc(ctx, nb) };
                            if data.is_null() {
                                return Err(format!("{}: could not allocate {nb} bytes", task.desc));
                            }
                            a.data = data;
                            a.owner = owner;
                            st.owned[v.i()] = true;
                        }
                    },
                    ValueKind::Returned(_) => {
                        a.data = std::ptr::null_mut();
                        a.owner = std::ptr::null_mut();
                    }
                    _ => {}
                }
                st.vals[v.i()].data = a.data;
                st.vals[v.i()].owner = a.owner;
                st.args.push(a);
            }
            // SAFETY: the arguments follow the task's declared modes: inputs are valid
            // arrays of their types, provided outputs are arena/fresh/caller memory of
            // their static types, other outputs have writable shape/strides storage;
            // `func` has the declared ABI (Python built the task from a compiled
            // function or a kernel handle).
            let t0 = profiling.then(std::time::Instant::now);
            // the task's scratch lives until it returns (also inside a subprogram:
            // marks, not a reset, so the caller's scratch survives)
            let mark = ctx.scratch_mark();
            let rc = unsafe {
                let f: KernelFn = std::mem::transmute::<usize, KernelFn>(task.func);
                f(ctx, st.args.as_mut_ptr(), task.config as *const u8)
            };
            if let Some(t0) = t0 {
                self.profile.task_ns[k_task].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            }
            ctx.scratch_release(mark);
            let n_in = task.inputs.len();
            for (k, &v) in task.outputs.iter().enumerate() {
                let a = st.args[n_in + k];
                if task.inputs.contains(&v) {
                    continue; // in place: the value is what it was
                }
                if let ValueKind::Returned(sources) = &values[v.i()].kind {
                    st.vals[v.i()].data = a.data;
                    st.vals[v.i()].owner = a.owner;
                    // A buffer the task allocated is ours to release. A view comes back
                    // with the owner of the value it lies in, or with none.
                    st.owned[v.i()] = !a.owner.is_null() && sources.iter().all(|src| a.owner != st.vals[src.i()].owner);
                }
            }
            if rc != 0 {
                let msg = ctx.error.take().map(|(_, m)| m).unwrap_or_else(|| format!("error {rc}"));
                return Err(format!("{}: {msg}", task.desc));
            }
            for (k, &v) in task.outputs.iter().enumerate() {
                if matches!(values[v.i()].kind, ValueKind::Returned(_)) {
                    // SAFETY: the storage holds ndim entries, written by the kernel.
                    let shape = unsafe { st.vals[v.i()].shape() };
                    check_shape(&values[v.i()], shape).map_err(|m| format!("{}: output {k}: {m}", task.desc))?;
                    if st.vals[v.i()].data.is_null() && shape.iter().product::<i64>() != 0 {
                        return Err(format!("{}: output {k} was not set", task.desc));
                    }
                }
            }
            for &v in &task.release {
                release(st, v);
            }
        }
        Ok(())
    }

    fn release_all(&self, st: &mut CallMemory) {
        for k in 0..self.spec.values.len() {
            release(st, ValueId(k as u32));
        }
    }

    /// Output `o` as a new reference (copied if its memory isn't a runtime buffer).
    pub fn take_output(&self, st: &CallMemory, ctx: &mut PtCtx, o: usize) -> R<OutArray> {
        let v = self.spec.outputs[o];
        let a = st.vals[v.i()];
        // SAFETY: the value was produced by `execute` (or is an input or a constant).
        let (shape, strides) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
        if !a.owner.is_null() {
            // SAFETY: a live buffer.
            unsafe { PtBuffer::incref(a.owner) };
            return Ok(OutArray { data: a.data, shape, strides, owner: a.owner });
        }
        let isz = self.spec.values[v.i()].dtype.itemsize();
        let n = crate::abi::checked_nbytes(&shape, isz).ok_or("output too large")?;
        // SAFETY: ctx is the live context.
        let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
        if data.is_null() {
            return Err(format!("could not allocate {n} bytes"));
        }
        let dst_strides = c_strides(&shape, isz as i64);
        // SAFETY: a valid array and a fresh buffer of its size.
        unsafe { copy_strided(a.data, &shape, &strides, data, &dst_strides, isz) };
        Ok(OutArray { data, shape, strides: dst_strides, owner })
    }

    /// Release what the call still holds (after the outputs were taken).
    pub fn finish(&self, st: &mut CallMemory) {
        self.release_all(st);
    }

    /// Run as a task of another program (the kernel ABI, see `ptl_program_call`).
    fn call_as_kernel(&self, ctx: &mut PtCtx, args: *mut PtArray) -> R<()> {
        let n_in = self.spec.n_inputs;
        let n_out = self.spec.outputs.len();
        // SAFETY: the caller passes n_in + n_out arrays (the contract of the task).
        let args = unsafe { std::slice::from_raw_parts_mut(args, n_in + n_out) };
        let mut st = self.take_memory();
        let r = (|| -> R<()> {
            for k in 0..n_in {
                let a = args[k];
                // SAFETY: inputs are valid arrays of their declared types.
                let (shape, strides) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
                self.set_input(&mut st, k, a.data, &shape, &strides)?;
            }
            let provided: Vec<Option<*mut u8>> = (0..n_out)
                .map(|o| match self.spec.output_modes[o] {
                    OutputMode::Provided => Some(args[n_in + o].data),
                    OutputMode::Allocated | OutputMode::View => None,
                })
                .collect();
            self.execute(&mut st, ctx, &provided)?;
            for o in 0..n_out {
                let v = self.spec.outputs[o];
                let dst = &mut args[n_in + o];
                match self.spec.output_modes[o] {
                    OutputMode::Provided => {
                        if !self.writes_in_place(o) {
                            let a = st.vals[v.i()];
                            // SAFETY: a computed value and the caller's memory of the same
                            // static shape (with the caller's strides).
                            unsafe {
                                let (shape, strides) = (a.shape().to_vec(), a.strides().to_vec());
                                copy_strided(a.data, &shape, &strides, dst.data, dst.strides(), self.spec.values[v.i()].dtype.itemsize());
                            }
                        }
                    }
                    OutputMode::Allocated | OutputMode::View => {
                        let out = self.take_output(&st, ctx, o)?;
                        // SAFETY: the output's shape/strides storage has ndim entries.
                        unsafe {
                            dst.shape_mut().copy_from_slice(&out.shape);
                            dst.strides_mut().copy_from_slice(&out.strides);
                        }
                        dst.data = out.data;
                        dst.owner = out.owner;
                    }
                }
            }
            Ok(())
        })();
        self.finish(&mut st);
        self.put_memory(st);
        r
    }

    pub fn n_inputs(&self) -> usize {
        self.spec.n_inputs
    }

    /// Turn per-task timing on or off (for this program only, not its subprograms).
    pub fn set_profiling(&self, on: bool) {
        self.profile.on.store(on, Relaxed);
    }

    pub fn reset_profile(&self) {
        self.profile.calls.store(0, Relaxed);
        self.profile.task_ns.iter().for_each(|t| t.store(0, Relaxed));
    }

    /// (number of calls, seconds per task) since the last reset.
    pub fn profile(&self) -> (u64, Vec<f64>) {
        (self.profile.calls.load(Relaxed), self.profile.task_ns.iter().map(|t| t.load(Relaxed) as f64 * 1e-9).collect())
    }

    /// The program as text: values with their types and storage, then the tasks.
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let sp = &self.spec;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "program: {} inputs, {} outputs, {} tasks, arena {} bytes",
            sp.n_inputs,
            sp.outputs.len(),
            sp.tasks.len(),
            sp.arena_bytes
        );
        let _ = writeln!(s, "values:");
        for (k, v) in sp.values.iter().enumerate() {
            let shape: Vec<String> = v.shape.iter().map(|d| d.map_or("?".to_string(), |n| n.to_string())).collect();
            let ty = format!("{}[{}]", v.dtype.numpy_name(), shape.join(", "));
            let kind = match &v.kind {
                ValueKind::Input(i) if self.input_contiguous(*i) => format!("input {i} (C-contiguous)"),
                ValueKind::Input(i) => format!("input {i}"),
                ValueKind::Const(_) => "constant".to_string(),
                ValueKind::Arena(off) if !v.order.is_empty() => format!("arena +{off}, order {:?}", v.order),
                ValueKind::Arena(off) => format!("arena +{off}"),
                ValueKind::Fresh => "fresh (output)".to_string(),
                ValueKind::Returned(sources) if sources.is_empty() => "allocated by its task".to_string(),
                ValueKind::Returned(sources) => {
                    let names: Vec<String> = sources.iter().map(|s| format!("%{}", s.0)).collect();
                    format!("returned by its task, may be a view of {}", names.join(", "))
                }
            };
            let _ = writeln!(s, "  %{k:<4} {ty:<24} {kind}");
        }
        let _ = writeln!(s, "tasks:");
        let ids = |v: &[ValueId]| v.iter().map(|x| format!("%{}", x.0)).collect::<Vec<_>>().join(", ");
        for (k, st) in sp.tasks.iter().enumerate() {
            let mut line = format!("  {k:<4} ({}) -> ({})  {}", ids(&st.inputs), ids(&st.outputs), st.desc);
            if !st.release.is_empty() {
                let _ = write!(line, "  [release {}]", ids(&st.release));
            }
            let _ = writeln!(s, "{line}");
        }
        let outs: Vec<String> = sp
            .outputs
            .iter()
            .zip(&sp.output_modes)
            .map(|(v, m)| format!("%{} ({})", v.0, if *m == OutputMode::Provided { "provided" } else { "allocated" }))
            .collect();
        let _ = writeln!(s, "outputs: {}", outs.join(", "));
        s
    }

    /// Output `o` after `execute`, as a borrowed array (valid until `finish`).
    pub fn output_view(&self, st: &CallMemory, o: usize) -> PtArray {
        let mut a = st.vals[self.spec.outputs[o].i()];
        a.owner = std::ptr::null_mut();
        a
    }

    pub fn output_ndim(&self, o: usize) -> usize {
        self.spec.values[self.spec.outputs[o].i()].ndim()
    }

    /// The static shape of output `o`, if its type has one.
    pub fn output_static_shape(&self, o: usize) -> Option<Vec<i64>> {
        self.spec.values[self.spec.outputs[o].i()].static_shape()
    }

    pub fn n_outputs(&self) -> usize {
        self.spec.outputs.len()
    }

    pub fn output_dtype(&self, o: usize) -> DType {
        self.spec.values[self.spec.outputs[o].i()].dtype
    }

    pub fn input_dtype(&self, k: usize) -> DType {
        self.spec.values[self.input_value(k).i()].dtype
    }
}

impl ValueKind {
    fn is_fresh(&self) -> bool {
        matches!(self, ValueKind::Fresh)
    }
}

fn release(st: &mut CallMemory, v: ValueId) {
    if st.owned[v.i()] {
        // SAFETY: an owned reference of this memory, released once (the flag is cleared).
        unsafe { PtBuffer::decref(st.vals[v.i()].owner) };
        st.owned[v.i()] = false;
        st.vals[v.i()].owner = std::ptr::null_mut();
    }
}

/// A program as a task: `config` is the `Program`.
///
/// # Safety
/// `config` points to a live `Program` (kept alive by the outer program) and `args` holds
/// its inputs and outputs as the outer task declared them.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_program_call(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8) -> i32 {
    // SAFETY: the function's contract.
    unsafe {
        let p = &*(config as *const Program);
        match p.call_as_kernel(&mut *ctx, args) {
            Ok(()) => 0,
            Err(m) => PtCtx::fail(ctx, 1, &m),
        }
    }
}

/// Copy elements between two arrays of the same shape.
///
/// # Safety
/// Both describe valid memory of `shape` with their strides; they don't overlap.
pub unsafe fn copy_strided(src: *const u8, shape: &[i64], src_strides: &[i64], dst: *mut u8, dst_strides: &[i64], itemsize: usize) {
    let nd = shape.len();
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; nd];
    loop {
        let (mut so, mut d) = (0isize, 0isize);
        for a in 0..nd {
            so += (idx[a] * src_strides[a]) as isize;
            d += (idx[a] * dst_strides[a]) as isize;
        }
        // SAFETY: the offsets stay inside both arrays (index < shape on every axis).
        unsafe { std::ptr::copy_nonoverlapping(src.offset(so), dst.offset(d), itemsize) };
        let mut a = nd;
        loop {
            if a == 0 {
                return;
            }
            a -= 1;
            idx[a] += 1;
            if idx[a] < shape[a] {
                break;
            }
            idx[a] = 0;
        }
    }
}
