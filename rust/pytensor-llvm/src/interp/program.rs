//! The interpreter: runs a task list planned in Python.
//!
//! Every task calls a kernel with the kernel ABI
//! `int32 f(PtCtx*, PtArray* args /* inputs, outputs */, int32 n_inputs, int32 n_outputs,
//! const void* config)`: a generated kernel, a Rust kernel, a subprogram, or a numba
//! kernel. The program doesn't know which.
//!
//! Python decides everything else: which values live where (arena offsets for static
//! temporaries, fresh buffers for graph outputs, buffers the kernels
//! allocate, views), when values are released, and how outputs are produced. The
//! interpreter checks what it can cheaply (shapes of kernel-set outputs against the
//! static types) and cleans up on errors.
//!
//! A call goes `begin` -> `Call` (set the inputs, provide memory for outputs) -> `run`
//! -> `Done` (read or take the outputs). The arrays of a call are borrowed until the
//! `Done` is dropped, which is also what releases the memory the call holds.
//!
//! The raw `PtArray`s of all values live in a `CallMemory`, and the interpreter is who
//! fills them in: what it knows about them (from the plan) is what makes its calls of
//! the kernels keep the kernel contract. That reasoning is in `execute_inner`; what
//! comes in from outside and goes out arrives and leaves as the views of
//! `crate::array`.

use super::consts::ConstData;
use crate::abi::{KernelFn, OwnedBuf, PtArray};
pub use crate::abi::{c_strides, is_c_contiguous};
use crate::array::{OutSlot, OwnedArray, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use crate::kernel::{Args, Kernel};
use crate::runtime::PtCtx;
use std::sync::atomic::AtomicUsize;
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

/// A kernel to call: a function with the kernel ABI, and its config.
#[derive(Clone, Copy)]
pub struct KernelHandle {
    func: KernelFn,
    config: *const u8,
}

// SAFETY: a function and the config it reads, both usable from any thread
// (`from_raw`'s contract).
unsafe impl Send for KernelHandle {}
// SAFETY: as above.
unsafe impl Sync for KernelHandle {}

impl KernelHandle {
    /// The kernel at `address` with `config` (None if the address is null).
    ///
    /// # Safety
    /// `address` is a function with the kernel ABI (`KernelFn`) and `config` is the
    /// config it expects. Both stay valid for as long as the handle is used, from any
    /// thread. Called with arguments that keep the kernel contract for the node it
    /// was made for, the function keeps its side of it: it reads and writes only what
    /// the arguments describe, and returns allocated outputs and views as the ABI
    /// says.
    pub unsafe fn from_raw(address: usize, config: usize) -> Option<KernelHandle> {
        // SAFETY: a null or a function pointer of this type (the function's contract).
        let func = unsafe { std::mem::transmute::<usize, Option<KernelFn>>(address) }?;
        Some(KernelHandle {
            func,
            config: config as *const u8,
        })
    }
}

pub struct Task {
    pub kernel: KernelHandle,
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
    /// which program this is (its call memories carry it)
    id: usize,
    spec: ProgramSpec,
    /// the value of each graph input
    input_values: Vec<ValueId>,
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
        Profile {
            on: AtomicBool::new(false),
            calls: AtomicU64::new(0),
            task_ns: (0..n_tasks).map(|_| AtomicU64::new(0)).collect(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// Per-call memory of one program; not shared between concurrent calls. (The context,
/// `PtCtx`, belongs to the caller's session: subprograms run with their
/// caller's.)
///
/// Invariants, between calls and at every step of one:
/// * `vals[v]` has the ndim and dtype of value `v` and its `shape` and `strides` point
///   to the two halves of `storage[v]`.
/// * The memory of constants and arena values is there for good. The data of inputs,
///   of graph outputs and of what tasks return is set during a call, and only used
///   after it was set in that call; `Call`'s drop clears it.
/// * `owned[v]` says that `vals[v].owner` is a buffer reference this memory holds.
pub struct CallMemory {
    /// the program this is the memory of
    program: usize,
    vals: Vec<PtArray>,
    /// shape ++ strides of every value
    storage: Vec<Box<[i64]>>,
    owned: Vec<bool>,
    // (owns the memory arena values point into)
    #[allow(dead_code)]
    arena: Vec<Align64>,
    args: Vec<PtArray>,
    /// which graph inputs the current call has set
    input_set: Vec<bool>,
    /// memory the caller of the current call provides, per graph output (null: none)
    provided: Vec<*mut u8>,
}

// SAFETY: the raw pointers in a CallMemory point into its own storage and arena, into
// buffers it owns, or into constants of its program; nothing is shared between memories.
unsafe impl Send for CallMemory {}

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

fn check_shape(v: &ValueSpec, shape: &[i64]) -> R<()> {
    if shape.len() != v.ndim() {
        return Err(format!(
            "expected {} dimensions, got {}",
            v.ndim(),
            shape.len()
        ));
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
                    return Err(format!(
                        "value {k}: memory order {:?} isn't a permutation of the axes",
                        v.order
                    ));
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
                    let nb = v
                        .static_nbytes()
                        .ok_or(format!("value {k}: arena value without static size"))?
                        as u64;
                    if off % 64 != 0 || off + nb > spec.arena_bytes {
                        return Err(format!(
                            "value {k}: arena placement {off}+{nb} outside the arena"
                        ));
                    }
                }
                ValueKind::Fresh => {
                    v.static_nbytes()
                        .ok_or(format!("value {k}: fresh value without static size"))?;
                }
                ValueKind::Returned(sources) => {
                    for src in sources {
                        ok(*src)?;
                    }
                }
            }
        }
        for s in &spec.tasks {
            for &v in s.inputs.iter().chain(&s.release) {
                ok(v)?;
            }
            for &v in &s.inputs {
                if !produced[v.i()] {
                    return Err(format!(
                        "task {}: value {} used before it is produced",
                        s.desc, v.0
                    ));
                }
            }
            for &v in &s.outputs {
                ok(v)?;
                // (an output that is also an input: the task works in place)
                let in_place = s.inputs.contains(&v);
                if in_place && matches!(spec.values[v.i()].kind, ValueKind::Const(_)) {
                    return Err(format!("task {}: a constant can't be overwritten", s.desc));
                }
                if !in_place
                    && (produced[v.i()]
                        || matches!(
                            spec.values[v.i()].kind,
                            ValueKind::Input(_) | ValueKind::Const(_)
                        ))
                {
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
        let mut input_values = vec![None; spec.n_inputs];
        for (k, v) in spec.values.iter().enumerate() {
            if let ValueKind::Input(i) = v.kind {
                input_values[i] = Some(ValueId(k as u32));
            }
        }
        let input_values = input_values
            .into_iter()
            .enumerate()
            .map(|(i, v)| v.ok_or(format!("input {i} has no value")))
            .collect::<R<Vec<_>>>()?;
        static PROGRAMS: AtomicUsize = AtomicUsize::new(0);
        Ok(Program {
            id: PROGRAMS.fetch_add(1, Relaxed),
            spec,
            input_values,
            fresh_output,
            memories: Mutex::new(vec![]),
            profile,
        })
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
            vals.push(PtArray {
                data,
                shape: st,
                strides,
                owner: std::ptr::null_mut(),
                ndim: nd as i32,
                dtype: v.dtype.code(),
            });
        }
        CallMemory {
            program: self.id,
            vals,
            storage,
            owned: vec![false; self.spec.values.len()],
            arena,
            args: vec![],
            input_set: vec![false; self.spec.n_inputs],
            provided: vec![std::ptr::null_mut(); self.spec.outputs.len()],
        }
    }

    pub fn take_memory(&self) -> CallMemory {
        let s = self
            .memories
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop();
        s.unwrap_or_else(|| self.new_memory())
    }

    pub fn put_memory(&self, st: CallMemory) {
        self.memories
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(st);
    }

    fn input_value(&self, k: usize) -> ValueId {
        self.input_values[k]
    }

    /// Start a call in `st`, a memory of this program (`take_memory`).
    pub fn begin<'c>(&'c self, st: &'c mut CallMemory) -> R<Call<'c>> {
        if st.program != self.id {
            return Err("this call memory belongs to another program".into());
        }
        Ok(Call { prog: self, st })
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

    /// Run all tasks of the call prepared in `st`.
    fn execute(&self, st: &mut CallMemory, ctx: &mut PtCtx) -> R<()> {
        if let Some(k) = st.input_set.iter().position(|&set| !set) {
            return Err(format!("input {k} was not set"));
        }
        // whatever happens, the call's scratch is released (a panic skips the tasks' own
        // releases)
        let base = ctx.scratch_mark();
        let r =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute_inner(st, ctx)))
                .unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
                    Err(format!(
                        "internal error (panic): {}",
                        msg.unwrap_or_default()
                    ))
                });
        ctx.scratch_release(base);
        if r.is_err() {
            self.release_all(st);
        }
        r
    }

    fn execute_inner(&self, st: &mut CallMemory, ctx: &mut PtCtx) -> R<()> {
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
                    ValueKind::Fresh => match self.fresh_output[v.i()]
                        .map(|o| st.provided[o])
                        .filter(|p| !p.is_null())
                    {
                        Some(p) => {
                            a.data = p;
                            a.owner = std::ptr::null_mut();
                        }
                        None => {
                            let nb = values[v.i()].static_nbytes().unwrap_or(0);
                            let buf = ctx.alloc(nb).ok_or_else(|| {
                                format!("{}: could not allocate {nb} bytes", task.desc)
                            })?;
                            a.data = buf.data();
                            a.owner = buf.into_raw();
                            st.owned[v.i()] = true;
                        }
                    },
                    ValueKind::Returned(_) => {
                        a.data = std::ptr::null_mut();
                        a.owner = std::ptr::null_mut();
                        // unset until the kernel writes it (checked after the call)
                        st.storage[v.i()][..values[v.i()].ndim()].fill(-1);
                    }
                    _ => {}
                }
                st.vals[v.i()].data = a.data;
                st.vals[v.i()].owner = a.owner;
                st.args.push(a);
            }
            let t0 = profiling.then(std::time::Instant::now);
            // the task's scratch lives until it returns (also inside a subprogram:
            // marks, not a reset, so the caller's scratch survives)
            let mark = ctx.scratch_mark();
            let kernel = task.kernel;
            // SAFETY: the kernel contract, by what the plan says about the values
            // (checked by `Program::new` as far as it goes, and otherwise the
            // planner's promise) and the invariants of `CallMemory`:
            // * `ctx` is ours for the call; `kernel` and its config are those of the
            //   task (`KernelHandle::from_raw`).
            // * `st.args` holds the task's inputs, then its outputs, each with the
            //   ndim and dtype of its value and its own shape and strides storage.
            // * Inputs were produced before (`Program::new` checks the order): graph
            //   inputs came in as valid arrays (`Call::set_input`), constants and arena
            //   values have their memory, and what a task returned is that task's to
            //   get right (`KernelHandle::from_raw`; its shape is checked below).
            //   The planner releases no value a later task still reads, or views.
            // * Outputs with memory (arena, fresh or the caller's, of the static shape
            //   in their storage) overlap nothing the task reads, unless the task
            //   works in place, which the planner only lets generated kernels do.
            //   The other outputs have no data: the kernel returns them.
            let rc = unsafe {
                (kernel.func)(
                    ctx,
                    st.args.as_mut_ptr(),
                    task.inputs.len() as i32,
                    task.outputs.len() as i32,
                    kernel.config,
                )
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
                    st.owned[v.i()] = !a.owner.is_null()
                        && sources.iter().all(|src| a.owner != st.vals[src.i()].owner);
                }
            }
            if rc != 0 {
                let msg = ctx
                    .error
                    .take()
                    .map(|(_, m)| m)
                    .unwrap_or_else(|| format!("error {rc}"));
                return Err(format!("{}: {msg}", task.desc));
            }
            for (k, &v) in task.outputs.iter().enumerate() {
                if matches!(values[v.i()].kind, ValueKind::Returned(_)) {
                    // (the kernel wrote the shape there, through `vals[v].shape`)
                    let shape = &st.storage[v.i()][..values[v.i()].ndim()];
                    let null = st.vals[v.i()].data.is_null();
                    if null && (shape.iter().any(|&n| n < 0) || shape.iter().product::<i64>() != 0)
                    {
                        return Err(format!("{}: output {k} was not set", task.desc));
                    }
                    check_shape(&values[v.i()], shape)
                        .map_err(|m| format!("{}: output {k}: {m}", task.desc))?;
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

    /// Release what the call still holds, and forget its arrays.
    fn finish(&self, st: &mut CallMemory) {
        self.release_all(st);
        for (k, set) in st.input_set.iter_mut().enumerate() {
            if std::mem::take(set) {
                st.vals[self.input_values[k].i()].data = std::ptr::null_mut();
            }
        }
        st.provided.fill(std::ptr::null_mut());
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
        self.profile
            .task_ns
            .iter()
            .for_each(|t| t.store(0, Relaxed));
    }

    /// (number of calls, seconds per task) since the last reset.
    pub fn profile(&self) -> (u64, Vec<f64>) {
        (
            self.profile.calls.load(Relaxed),
            self.profile
                .task_ns
                .iter()
                .map(|t| t.load(Relaxed) as f64 * 1e-9)
                .collect(),
        )
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
            let shape: Vec<String> = v
                .shape
                .iter()
                .map(|d| d.map_or("?".to_string(), |n| n.to_string()))
                .collect();
            let ty = format!("{}[{}]", v.dtype.numpy_name(), shape.join(", "));
            let kind = match &v.kind {
                ValueKind::Input(i) if self.input_contiguous(*i) => {
                    format!("input {i} (C-contiguous)")
                }
                ValueKind::Input(i) => format!("input {i}"),
                ValueKind::Const(c) => format!("constant at {:p}", c.data()),
                ValueKind::Arena(off) if !v.order.is_empty() => {
                    format!("arena +{off}, order {:?}", v.order)
                }
                ValueKind::Arena(off) => format!("arena +{off}"),
                ValueKind::Fresh => "fresh (output)".to_string(),
                ValueKind::Returned(sources) if sources.is_empty() => {
                    "allocated by its task".to_string()
                }
                ValueKind::Returned(sources) => {
                    let names: Vec<String> = sources.iter().map(|s| format!("%{}", s.0)).collect();
                    format!(
                        "returned by its task, may be a view of {}",
                        names.join(", ")
                    )
                }
            };
            let _ = writeln!(s, "  %{k:<4} {ty:<24} {kind}");
        }
        let _ = writeln!(s, "tasks:");
        let ids = |v: &[ValueId]| {
            v.iter()
                .map(|x| format!("%{}", x.0))
                .collect::<Vec<_>>()
                .join(", ")
        };
        for (k, st) in sp.tasks.iter().enumerate() {
            let mut line = format!(
                "  {k:<4} ({}) -> ({})  {}",
                ids(&st.inputs),
                ids(&st.outputs),
                st.desc
            );
            if !st.release.is_empty() {
                let _ = write!(line, "  [release {}]", ids(&st.release));
            }
            let _ = writeln!(s, "{line}");
        }
        let outs: Vec<String> = sp
            .outputs
            .iter()
            .zip(&sp.output_modes)
            .map(|(v, m)| {
                format!(
                    "%{} ({})",
                    v.0,
                    if *m == OutputMode::Provided {
                        "provided"
                    } else {
                        "allocated"
                    }
                )
            })
            .collect();
        let _ = writeln!(s, "outputs: {}", outs.join(", "));
        s
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
        drop(unsafe { OwnedBuf::from_raw(st.vals[v.i()].owner) });
        st.owned[v.i()] = false;
        st.vals[v.i()].owner = std::ptr::null_mut();
    }
}

// ------------------------------------------------------------------ calls

/// A call being prepared: `set_input` every input, `provide` memory for outputs if
/// you have it, then `run`.
///
/// The arrays the call is given are borrowed for `'c`: until the call, and the `Done`
/// it becomes, are dropped.
pub struct Call<'c> {
    prog: &'c Program,
    st: &'c mut CallMemory,
}

impl<'c> Call<'c> {
    /// Graph input `k` is `a`. Checked: its dtype, its shape against the type, and that
    /// it is C-contiguous if the function was compiled for that.
    pub fn set_input(&mut self, k: usize, a: PtArrayRef<'c>) -> R<()> {
        let prog = self.prog;
        if k >= prog.n_inputs() {
            return Err(format!(
                "no input {k}: the function has {}",
                prog.n_inputs()
            ));
        }
        let v = prog.input_value(k);
        let spec = &prog.spec.values[v.i()];
        if a.dtype() != spec.dtype {
            return Err(format!(
                "input {k}: an array of {} is expected, got one of {}",
                spec.dtype.numpy_name(),
                a.dtype().numpy_name()
            ));
        }
        check_shape(spec, a.shape()).map_err(|m| format!("input {k}: {m}"))?;
        if prog.input_contiguous(k) && !a.is_c_contiguous() {
            return Err(format!(
                "input {k}: must be C-contiguous (the function was compiled for C-contiguous inputs)"
            ));
        }
        let nd = a.ndim();
        let st = &mut *self.st;
        st.storage[v.i()][..nd].copy_from_slice(a.shape());
        st.storage[v.i()][nd..].copy_from_slice(a.strides());
        // (never written through: nothing overwrites a graph input)
        st.vals[v.i()].data = a.as_ptr() as *mut u8;
        st.vals[v.i()].owner = std::ptr::null_mut();
        st.input_set[k] = true;
        Ok(())
    }

    /// Memory for graph output `o`. Checked: its dtype, and its shape against the
    /// type. If the result can be computed straight into it, it is (None); otherwise
    /// the memory comes back, for the caller to copy `Done::output(o)` into.
    pub fn provide(&mut self, o: usize, mut a: PtArrayMut<'c>) -> R<Option<PtArrayMut<'c>>> {
        let prog = self.prog;
        if o >= prog.n_outputs() {
            return Err(format!(
                "no output {o}: the function has {}",
                prog.n_outputs()
            ));
        }
        let spec = &prog.spec.values[prog.spec.outputs[o].i()];
        if a.dtype() != spec.dtype {
            return Err(format!(
                "provided output {o}: an array of {} is expected, got one of {}",
                spec.dtype.numpy_name(),
                a.dtype().numpy_name()
            ));
        }
        check_shape(spec, a.shape()).map_err(|m| format!("provided output {o}: {m}"))?;
        // the producing task writes C-contiguous memory of the static shape
        if prog.writes_in_place(o) && a.is_c_contiguous() {
            self.st.provided[o] = a.as_mut_ptr();
            return Ok(None);
        }
        Ok(Some(a))
    }

    /// Run the program.
    pub fn run(self, ctx: &mut PtCtx) -> R<Done<'c>> {
        self.prog.execute(self.st, ctx)?;
        Ok(Done(self))
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        self.prog.finish(self.st);
    }
}

/// A call that ran: its outputs can be read until this is dropped.
pub struct Done<'c>(Call<'c>);

impl Done<'_> {
    /// The header of graph output `o`.
    fn header(&self, o: usize) -> R<&PtArray> {
        let Call { prog, st } = &self.0;
        let v = *prog
            .spec
            .outputs
            .get(o)
            .ok_or_else(|| format!("no output {o}: the function has {}", prog.n_outputs()))?;
        Ok(&st.vals[v.i()])
    }

    /// Graph output `o`, borrowed.
    pub fn output(&self, o: usize) -> R<PtArrayRef<'_>> {
        // SAFETY: a value of a call that ran is an input of the call (an array for
        // `'c`), a constant, memory of `st`, memory the caller provided for `'c`, or
        // what a task returned, in a buffer `st` holds; and its header keeps the
        // invariants of `CallMemory`. Nothing writes any of them before the call is
        // dropped, which can't happen while the result borrows it. What tasks
        // returned is checked here as far as that goes.
        unsafe { PtArrayRef::from_raw_checked(self.header(o)?) }
            .map_err(|e| format!("output {o}: {e}"))
    }

    /// Graph output `o`, to keep: a reference to its buffer, or a copy if its memory
    /// isn't a runtime buffer (an input, a constant, provided memory).
    pub fn take_output(&self, o: usize) -> R<OwnedArray> {
        let raw = self.header(o)?;
        if raw.owner.is_null() {
            return OwnedArray::copy_of(&self.output(o)?);
        }
        // SAFETY: as in `output`. (Only the shape and strides are read here; that
        // they describe an array inside the buffer is checked by `in_buffer`.)
        let a = unsafe { PtArrayRef::from_raw(raw) }.map_err(|e| format!("output {o}: {e}"))?;
        // SAFETY: the buffer the output is in, which the call's memory holds.
        let buf = unsafe { OwnedBuf::from_borrowed(raw.owner) };
        OwnedArray::in_buffer(
            buf,
            raw.data,
            a.shape().to_vec(),
            a.strides().to_vec(),
            a.dtype(),
        )
        .map_err(|e| format!("output {o}: {e}"))
    }
}

/// A program as a kernel: a task of another program, or the body of a loop.
impl Kernel for Program {
    fn arity(&self) -> (usize, usize) {
        (self.n_inputs(), self.n_outputs())
    }

    fn run(&self, ctx: &mut PtCtx, args: Args) -> R<()> {
        let mut st = self.take_memory();
        let result = self.run_as_kernel(&mut st, ctx, args);
        self.put_memory(st);
        result
    }
}

impl Program {
    fn run_as_kernel(&self, st: &mut CallMemory, ctx: &mut PtCtx, args: Args) -> R<()> {
        /// What is left to do for an output after the call.
        enum Pending<'a> {
            Written,
            Copy(PtArrayMut<'a>),
            Take(OutSlot<'a>),
        }
        let mut call = self.begin(st)?;
        for k in 0..self.n_inputs() {
            call.set_input(k, args.inputs.get(k)?)?;
        }
        let mut pending = Vec::with_capacity(self.n_outputs());
        for (o, slot) in args.outputs.into_slots().enumerate() {
            let slot = slot?;
            pending.push(match self.spec.output_modes[o] {
                OutputMode::Provided => match call.provide(o, slot.provided()?)? {
                    None => Pending::Written,
                    Some(memory) => Pending::Copy(memory),
                },
                OutputMode::Allocated | OutputMode::View => Pending::Take(slot),
            });
        }
        let done = call.run(ctx)?;
        for (o, pending) in pending.into_iter().enumerate() {
            match pending {
                Pending::Written => {}
                Pending::Copy(mut memory) => memory.copy_from(&done.output(o)?)?,
                Pending::Take(slot) => slot.set(done.take_output(o)?)?,
            }
        }
        Ok(())
    }
}
