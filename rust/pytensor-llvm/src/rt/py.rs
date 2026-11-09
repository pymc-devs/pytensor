//! `pytensor_llvm.rt`: programs and Rust kernels for Python.

use super::consts::ConstData;
use super::kernels::{self, OutMode};
use super::program::{Abi as RAbi, Kind, OutputMode as ROutMode, Program as RProgram, ProgramSpec, Step, ValueId, ValueSpec};
use crate::dtype::DType;
use numpy::npyffi::{npy_intp, NpyTypes, NPY_ARRAY_WRITEABLE, PY_ARRAY_API};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyClassInitializer;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use std::sync::Arc;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn dtype_of(d: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    DType::from_numpy(d.kind(), d.itemsize()).ok_or_else(|| PyTypeError::new_err(format!("dtype {d} is not supported")))
}

fn descr<'py>(py: Python<'py>, dt: DType) -> Bound<'py, PyArrayDescr> {
    match dt {
        DType::Float64 => numpy::dtype::<f64>(py),
        DType::Float32 => numpy::dtype::<f32>(py),
        DType::Int64 => numpy::dtype::<i64>(py),
        DType::Int32 => numpy::dtype::<i32>(py),
        DType::Int16 => numpy::dtype::<i16>(py),
        DType::Int8 => numpy::dtype::<i8>(py),
        DType::Uint8 => numpy::dtype::<u8>(py),
        DType::Bool => numpy::dtype::<bool>(py),
    }
}

#[pyclass(module = "pytensor_llvm.rt", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Abi {
    /// `int32 f(ctx, args, env, ws)`
    Standard,
    /// `int32 f(ctx, args, params)`, every output allocated (the numba fallback)
    Foreign,
}

#[pyclass(module = "pytensor_llvm.rt", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Provided,
    Allocated,
    View,
}

impl OutputMode {
    fn kernel(self) -> OutMode {
        match self {
            OutputMode::Provided => OutMode::Provided,
            OutputMode::Allocated => OutMode::Allocated,
            OutputMode::View => OutMode::View,
        }
    }
}

// ------------------------------------------------------------------ kernels

/// A precompiled kernel: a function address and its environment (owned by this object).
#[pyclass(module = "pytensor_llvm.rt", subclass, frozen)]
pub struct Kernel {
    address: usize,
    env: Box<dyn std::any::Any + Send + Sync>,
    env_ptr: usize,
    desc: String,
    /// programs the kernel runs (Scan's step), for profiling
    programs: Vec<Py<Program>>,
}

impl Kernel {
    fn new<T: Send + Sync + 'static>(f: usize, env: T, desc: String) -> Self {
        let env: Box<T> = Box::new(env);
        let env_ptr = &*env as *const T as usize;
        Kernel { address: f, env, env_ptr, desc, programs: vec![] }
    }
}

#[pymethods]
impl Kernel {
    #[getter]
    fn address(&self) -> usize {
        self.address
    }

    /// Pointer to the kernel's environment, valid while this object lives.
    #[getter]
    fn env(&self) -> usize {
        let _ = &self.env;
        self.env_ptr
    }

    fn __repr__(&self) -> String {
        self.desc.clone()
    }

    /// Programs this kernel runs (e.g. Scan's step program).
    #[getter]
    fn programs(&self, py: Python<'_>) -> Vec<Py<Program>> {
        self.programs.iter().map(|p| p.clone_ref(py)).collect()
    }
}

#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct DimShuffle;

#[pymethods]
impl DimShuffle {
    /// `order[k]`: input axis of output axis k, or None for a new axis. Its output is a
    /// view of the input.
    #[new]
    fn new(order: Vec<Option<usize>>) -> PyClassInitializer<Self> {
        let desc = format!("DimShuffle{order:?}");
        let k = Kernel::new(kernels::k_dimshuffle as *const () as usize, kernels::DimShuffleEnv { order }, desc);
        PyClassInitializer::from(k).add_subclass(DimShuffle)
    }
}

#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct Copy;

#[pymethods]
impl Copy {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>) -> PyResult<PyClassInitializer<Self>> {
        let env = kernels::CopyEnv { mode: mode.kernel(), dtype: dtype_of(&dtype)? };
        let k = Kernel::new(kernels::k_copy as *const () as usize, env, "Copy".into());
        Ok(PyClassInitializer::from(k).add_subclass(Copy))
    }
}

#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct Dot;

#[pymethods]
impl Dot {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>) -> PyResult<PyClassInitializer<Self>> {
        let dt = dtype_of(&dtype)?;
        if !matches!(dt, DType::Float64 | DType::Float32) {
            return err(format!("Dot of {dt:?}"));
        }
        let env = kernels::DotEnv { mode: mode.kernel(), dtype: dt };
        let k = Kernel::new(kernels::k_dot as *const () as usize, env, "Dot".into());
        Ok(PyClassInitializer::from(k).add_subclass(Dot))
    }
}

fn legacy(f: kernels::LegacyFn, params: Vec<i64>, n_in: usize, shape_from: usize, mode: OutputMode, dtype: DType, desc: String) -> PyResult<Kernel> {
    if mode == OutputMode::View {
        return err("this kernel can't produce a view");
    }
    let env = kernels::LegacyEnv { f, params, n_in, shape_from, mode: mode.kernel(), dtype };
    Ok(Kernel::new(kernels::k_legacy as *const () as usize, env, desc))
}

/// Sort along `axis` (float64, float32, int64), NaNs last.
#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct Sort;

#[pymethods]
impl Sort {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>, axis: i64, stable: bool) -> PyResult<PyClassInitializer<Self>> {
        let dt = dtype_of(&dtype)?;
        let f: kernels::LegacyFn = match dt {
            DType::Float64 => crate::native::ptl_sort_f64,
            DType::Float32 => crate::native::ptl_sort_f32,
            DType::Int64 => crate::native::ptl_sort_i64,
            _ => return err(format!("Sort of {dt:?}")),
        };
        let k = legacy(f, vec![axis, stable as i64], 1, 0, mode, dt, format!("Sort(axis={axis})"))?;
        Ok(PyClassInitializer::from(k).add_subclass(Sort))
    }
}

/// Cholesky factor (faer), LAPACK potrf semantics: NaN if not positive definite.
#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct Cholesky;

#[pymethods]
impl Cholesky {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>, lower: bool) -> PyResult<PyClassInitializer<Self>> {
        let dt = dtype_of(&dtype)?;
        let f: kernels::LegacyFn = match dt {
            DType::Float64 => crate::linalg::ptl_cholesky_f64,
            DType::Float32 => crate::linalg::ptl_cholesky_f32,
            _ => return err(format!("Cholesky of {dt:?}")),
        };
        let k = legacy(f, vec![lower as i64], 1, 0, mode, dt, format!("Cholesky(lower={lower})"))?;
        Ok(PyClassInitializer::from(k).add_subclass(Cholesky))
    }
}

/// Triangular solve (faer), LAPACK trtrs semantics: NaN if the diagonal has a zero.
#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct SolveTriangular;

#[pymethods]
impl SolveTriangular {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>, lower: bool, unit_diagonal: bool) -> PyResult<PyClassInitializer<Self>> {
        let dt = dtype_of(&dtype)?;
        let f: kernels::LegacyFn = match dt {
            DType::Float64 => crate::linalg::ptl_solve_triangular_f64,
            DType::Float32 => crate::linalg::ptl_solve_triangular_f32,
            _ => return err(format!("SolveTriangular of {dt:?}")),
        };
        let desc = format!("SolveTriangular(lower={lower}, unit_diagonal={unit_diagonal})");
        let k = legacy(f, vec![lower as i64, unit_diagonal as i64], 2, 1, mode, dt, desc)?;
        Ok(PyClassInitializer::from(k).add_subclass(SolveTriangular))
    }
}

/// Scan: a loop around a step program. Every output is allocated by the kernel.
#[pyclass(module = "pytensor_llvm.rt", extends = Kernel, frozen)]
pub struct Scan;

#[pymethods]
impl Scan {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_seqs, mit_mot_in, mit_mot_out, mit_sot_in, sit_sot_in, n_untraced, n_nit_sot, n_non_seqs, as_while, step, outer_in_dtypes, inner_out_dtypes))]
    fn new(
        py: Python<'_>,
        n_seqs: usize,
        mit_mot_in: Vec<Vec<i64>>,
        mit_mot_out: Vec<Vec<i64>>,
        mit_sot_in: Vec<Vec<i64>>,
        sit_sot_in: Vec<Vec<i64>>,
        n_untraced: usize,
        n_nit_sot: usize,
        n_non_seqs: usize,
        as_while: bool,
        step: Bound<'_, Program>,
        outer_in_dtypes: Vec<Bound<'_, PyArrayDescr>>,
        inner_out_dtypes: Vec<Bound<'_, PyArrayDescr>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let info = super::scan::ScanInfo { n_seqs, mit_mot_in, mit_mot_out, mit_sot_in, sit_sot_in, n_untraced, n_nit_sot, n_non_seqs, as_while };
        let n_in = 1 + n_seqs + info.mit_mot_in.len() + info.mit_sot_in.len() + info.sit_sot_in.len() + n_untraced + n_nit_sot + n_non_seqs;
        if outer_in_dtypes.len() != n_in {
            return err(format!("Scan: {n_in} outer inputs, got {} dtypes", outer_in_dtypes.len()));
        }
        let prog = step.get().prog.clone();
        let n_step_in = n_seqs
            + info.mit_mot_in.iter().map(|t| t.len()).sum::<usize>()
            + info.mit_sot_in.iter().map(|t| t.len()).sum::<usize>()
            + info.sit_sot_in.len()
            + n_untraced
            + n_non_seqs;
        let n_step_out = info.mit_mot_out.iter().map(|t| t.len()).sum::<usize>()
            + info.mit_sot_in.len()
            + info.sit_sot_in.len()
            + n_nit_sot
            + n_untraced
            + as_while as usize;
        if prog.n_inputs() != n_step_in || prog.n_outputs() != n_step_out || inner_out_dtypes.len() != n_step_out {
            return err("Scan: the step program doesn't match the scan's layout");
        }
        let env = super::scan::ScanEnv {
            info,
            step: prog,
            outer_in_dtypes: outer_in_dtypes.iter().map(dtype_of).collect::<PyResult<_>>()?,
            inner_out_dtypes: inner_out_dtypes.iter().map(dtype_of).collect::<PyResult<_>>()?,
            keepalive: step.clone().into_any().unbind(),
        };
        let mut k = Kernel::new(super::scan::k_scan as *const () as usize, env, "Scan".into());
        k.programs.push(step.clone().unbind());
        let _ = py;
        Ok(PyClassInitializer::from(k).add_subclass(Scan))
    }
}

// ------------------------------------------------------------------ programs

/// Builds a program: values, then steps in execution order, then outputs.
#[pyclass(module = "pytensor_llvm.rt")]
pub struct ProgramBuilder {
    values: Vec<ValueSpec>,
    n_inputs: usize,
    steps: Vec<Step>,
    keepalive: Vec<Py<PyAny>>,
    arena_bytes: u64,
    contiguous_inputs: Vec<bool>,
}

impl ProgramBuilder {
    fn add(&mut self, dtype: &Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>, kind: Kind) -> PyResult<u32> {
        self.values.push(ValueSpec { dtype: dtype_of(dtype)?, shape, kind });
        Ok((self.values.len() - 1) as u32)
    }
}

#[pymethods]
impl ProgramBuilder {
    #[new]
    fn new() -> Self {
        ProgramBuilder { values: vec![], n_inputs: 0, steps: vec![], keepalive: vec![], arena_bytes: 0, contiguous_inputs: vec![] }
    }

    /// The next graph input. Returns its value id.
    /// A graph input; `contiguous`: the generated code assumes it is C-contiguous
    /// (Python calls copy other arrays, the C API rejects them).
    #[pyo3(signature = (dtype, shape, contiguous = false))]
    fn input(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>, contiguous: bool) -> PyResult<u32> {
        let k = self.n_inputs;
        self.n_inputs += 1;
        self.contiguous_inputs.push(contiguous);
        self.add(&dtype, shape, Kind::Input(k))
    }

    fn constant(&mut self, array: &Bound<'_, PyUntypedArray>) -> PyResult<u32> {
        let dt = dtype_of(&array.dtype())?;
        let np = array.py().import("numpy")?;
        let native = np.call_method1("ascontiguousarray", (array,))?;
        let native = native.call_method1("astype", (array.dtype().call_method1("newbyteorder", ("=",))?,))?;
        let bytes: Vec<u8> = native.call_method0("tobytes")?.extract()?;
        let shape: Vec<i64> = array.shape().iter().map(|&n| n as i64).collect();
        let c = ConstData::new(dt, shape.clone(), &bytes).map_err(PyValueError::new_err)?;
        self.values.push(ValueSpec { dtype: dt, shape: shape.iter().map(|&n| Some(n as u64)).collect(), kind: Kind::Const(Arc::new(c)) });
        Ok((self.values.len() - 1) as u32)
    }

    /// A static temporary at `offset` in the arena (provided to its producer).
    fn arena(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<u64>, offset: u64) -> PyResult<u32> {
        self.add(&dtype, shape.into_iter().map(Some).collect(), Kind::Arena(offset))
    }

    /// A static graph output: a fresh buffer per call, provided to its producer.
    fn fresh(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<u64>) -> PyResult<u32> {
        self.add(&dtype, shape.into_iter().map(Some).collect(), Kind::Fresh)
    }

    /// A value its producer allocates.
    fn allocated(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>) -> PyResult<u32> {
        self.add(&dtype, shape, Kind::Allocated)
    }

    /// A value its producer sets as a view into the memory of `source`.
    fn view(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>, source: u32) -> PyResult<u32> {
        self.add(&dtype, shape, Kind::View(ValueId(source)))
    }

    /// Append a step. `keepalive` is kept alive as long as the program (the compiled
    /// module or kernel object the address and env belong to).
    #[pyo3(signature = (address, env, abi, inputs, outputs, release, workspace_offset, description, keepalive))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        address: usize,
        env: usize,
        abi: Abi,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
        release: Vec<u32>,
        workspace_offset: Option<u64>,
        description: String,
        keepalive: Py<PyAny>,
    ) {
        self.steps.push(Step {
            func: address,
            abi: match abi {
                Abi::Standard => RAbi::Standard,
                Abi::Foreign => RAbi::Foreign,
            },
            env,
            ws_offset: workspace_offset,
            inputs: inputs.into_iter().map(ValueId).collect(),
            outputs: outputs.into_iter().map(ValueId).collect(),
            release: release.into_iter().map(ValueId).collect(),
            desc: description,
        });
        self.keepalive.push(keepalive);
    }

    fn set_arena_bytes(&mut self, n: u64) {
        self.arena_bytes = n;
    }

    /// Finish with these outputs. `modes` say how each output is produced when the
    /// program runs as a step of another program (Provided or Allocated).
    fn build(&mut self, outputs: Vec<u32>, modes: Vec<OutputMode>) -> PyResult<Program> {
        let output_modes = modes
            .into_iter()
            .map(|m| match m {
                OutputMode::Provided => Ok(ROutMode::Provided),
                OutputMode::Allocated => Ok(ROutMode::Allocated),
                OutputMode::View => Err(PyValueError::new_err("program outputs can't be views")),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let spec = ProgramSpec {
            values: std::mem::take(&mut self.values),
            n_inputs: self.n_inputs,
            steps: std::mem::take(&mut self.steps),
            outputs: outputs.into_iter().map(ValueId).collect(),
            output_modes,
            arena_bytes: self.arena_bytes,
            contiguous_inputs: std::mem::take(&mut self.contiguous_inputs),
        };
        let prog = RProgram::new(spec).map_err(PyValueError::new_err)?;
        Ok(Program { prog: Arc::new(prog), keepalive: std::mem::take(&mut self.keepalive) })
    }
}

struct SendPtr(*mut crate::abi::PtBuffer);
// SAFETY: moves one buffer reference into a capsule destructor; refcounts are atomic.
unsafe impl Send for SendPtr {}

fn wrap_output<'py>(py: Python<'py>, dt: DType, o: &super::program::OutArray) -> PyResult<Bound<'py, PyAny>> {
    let owner = SendPtr(o.owner);
    // SAFETY: the capsule holds the one reference of `o`; numpy calls the destructor
    // once, when the array (its base) is freed.
    let capsule = PyCapsule::new_with_value_and_destructor(py, owner, c"pytensor_llvm.buffer", |b: SendPtr, _| unsafe {
        crate::abi::PtBuffer::decref(b.0)
    })?;
    let mut dims: Vec<npy_intp> = o.shape.iter().map(|&s| s as npy_intp).collect();
    let mut strides: Vec<npy_intp> = o.strides.iter().map(|&s| s as npy_intp).collect();
    // SAFETY: `o` describes a valid array in the buffer the capsule keeps alive; the
    // descr and capsule references are stolen by the numpy calls.
    unsafe {
        let ty = numpy::npyffi::get_type_object(py, NpyTypes::PyArray_Type);
        let arr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            ty,
            descr(py, dt).into_ptr() as *mut _,
            dims.len() as i32,
            dims.as_mut_ptr(),
            strides.as_mut_ptr(),
            o.data as *mut _,
            NPY_ARRAY_WRITEABLE,
            std::ptr::null_mut(),
        );
        let arr = Bound::from_owned_ptr_or_err(py, arr)?;
        if PY_ARRAY_API.PyArray_SetBaseObject(py, arr.as_ptr() as *mut _, capsule.into_ptr()) != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(arr)
    }
}

fn runtime_error(msg: String) -> PyErr {
    if msg.contains("out of bounds") {
        PyIndexError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// A runnable program: call it with arrays, or use it as a step of another program
/// (`address` / `env`).
#[pyclass(module = "pytensor_llvm.rt", frozen)]
pub struct Program {
    prog: Arc<RProgram>,
    keepalive: Vec<Py<PyAny>>,
}

#[pymethods]
impl Program {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyList>> {
        let prog = &*self.prog;
        if args.len() != prog.n_inputs() {
            return err(format!("expected {} inputs, got {}", prog.n_inputs(), args.len()));
        }
        let np = py.import("numpy")?;
        let mut st = prog.take_state();
        let mut keep = Vec::with_capacity(args.len());
        let mut work = 0usize;
        let res = (|| -> PyResult<Vec<super::program::OutArray>> {
            for (k, a) in args.iter().enumerate() {
                let want = prog.input_dtype(k);
                let exact = match a.cast::<PyUntypedArray>() {
                    Ok(arr) => {
                        let d = arr.dtype();
                        DType::from_numpy(d.kind(), d.itemsize()) == Some(want) && d.is_native_byteorder() != Some(false)
                    }
                    Err(_) => false,
                };
                let a = if exact {
                    a
                } else {
                    let kw = pyo3::types::PyDict::new(py);
                    kw.set_item("dtype", descr(py, want))?;
                    np.call_method("asarray", (a,), Some(&kw))?
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: reading the flags of a live array.
                let flags = unsafe { (*arr.as_array_ptr()).fields.flags };
                let aligned = flags & numpy::npyffi::NPY_ARRAY_ALIGNED != 0;
                let a = if !aligned {
                    np.call_method1("ascontiguousarray", (a,))?.call_method0("copy")?
                } else if prog.input_contiguous(k) && flags & numpy::npyffi::NPY_ARRAY_C_CONTIGUOUS == 0 {
                    np.call_method1("ascontiguousarray", (a,))?
                } else {
                    a
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: the data pointer of a live array, kept alive by `keep` until the
                // call is finished.
                let data = unsafe { (*arr.as_array_ptr()).fields.data as *mut u8 };
                let shape: Vec<i64> = arr.shape().iter().map(|&n| n as i64).collect();
                let strides: Vec<i64> = arr.strides().iter().map(|&s| s as i64).collect();
                prog.set_input(&mut st, k, data, &shape, &strides).map_err(PyValueError::new_err)?;
                work += arr.len();
                keep.push(a);
            }
            let mut ctx = std::mem::take(&mut st.ctx);
            let run = |st: &mut super::program::CallState, ctx: &mut crate::abi::PtCtx| -> Result<Vec<super::program::OutArray>, String> {
                let none = vec![None; prog.n_outputs()];
                prog.execute(st, ctx, &none)?;
                let outs = (0..prog.n_outputs()).map(|o| prog.take_output(st, ctx, o)).collect::<Result<Vec<_>, _>>();
                prog.finish(st);
                outs
            };
            // releasing the GIL costs ~100ns: only worth it with real work
            let r = if work > 20_000 {
                struct S<'a>(&'a mut super::program::CallState, &'a mut crate::abi::PtCtx);
                // SAFETY: the state and context are exclusively ours for this call.
                unsafe impl Send for S<'_> {}
                let s = S(&mut st, &mut ctx);
                py.detach(move || {
                    let s = s;
                    run(s.0, s.1)
                })
            } else {
                run(&mut st, &mut ctx)
            };
            st.ctx = ctx;
            r.map_err(runtime_error)
        })();
        drop(keep);
        prog.put_state(st);
        let outs = res?;
        let arrays: Vec<Bound<'_, PyAny>> =
            outs.iter().enumerate().map(|(o, a)| wrap_output(py, prog.output_dtype(o), a)).collect::<PyResult<_>>()?;
        Ok(PyList::new(py, arrays)?.unbind())
    }

    /// The C ABI entry for running this program as a step.
    #[getter]
    fn address(&self) -> usize {
        super::program::ptl_program_call as *const () as usize
    }

    /// The environment for `address`: this program (valid while this object lives).
    #[getter]
    fn env(&self) -> usize {
        Arc::as_ptr(&self.prog) as usize
    }

    /// The program as text (values, storage, steps, releases).
    fn dump(&self) -> String {
        self.prog.dump()
    }

    fn __str__(&self) -> String {
        self.prog.dump()
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.prog.spec().steps.len()
    }

    #[getter]
    fn arena_bytes(&self) -> u64 {
        self.prog.spec().arena_bytes
    }

    /// Turn per-step timing on or off, here and in nested programs (a step that is a
    /// program, Scan's step program). Timing costs two clock reads per step.
    fn set_profiling(&self, py: Python<'_>, on: bool) {
        self.prog.set_profiling(on);
        for c in self.children(py).into_iter().flatten() {
            c.get().set_profiling(py, on);
        }
    }

    fn reset_profile(&self, py: Python<'_>) {
        self.prog.reset_profile();
        for c in self.children(py).into_iter().flatten() {
            c.get().reset_profile(py);
        }
    }

    /// Rows `(depth, description, calls, seconds)`: each step with its total time,
    /// followed by the steps of the programs it runs at depth + 1; `calls` is the number
    /// of runs of the program the step belongs to.
    fn profile(&self, py: Python<'_>) -> Vec<(usize, String, u64, f64)> {
        let mut rows = vec![];
        self.profile_rows(py, 0, &mut rows);
        rows
    }

    /// The profile as a table, slowest first within each program.
    fn profile_report(&self, py: Python<'_>) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let (calls, _) = self.prog.profile();
        let total: f64 = self.prog.profile().1.iter().sum();
        let _ = writeln!(s, "{calls} calls, {:.3} ms in steps ({:.2} us per call)", total * 1e3, if calls > 0 { total / calls as f64 * 1e6 } else { 0.0 });
        let _ = writeln!(s, "{:>7} {:>10} {:>10}  step (nested steps indented)", "%", "total ms", "us/call");
        self.report_into(py, 0, total, calls, &mut s);
        s
    }

    /// A new reference to this program for the C API, as the address of a
    /// `PtFunction` (release it with `function_decref` from `c_api_table()`). It keeps
    /// the compiled code alive on its own.
    fn c_function(&self, py: Python<'_>) -> usize {
        let keep = self.keepalive.iter().map(|o| o.clone_ref(py)).collect();
        super::capi::Function::new(self.prog.clone(), keep).into_handle() as usize
    }

    /// The address of the `PtFunctionApi` table for functions from `c_function()`.
    #[staticmethod]
    fn c_api_table() -> usize {
        &super::capi::API as *const _ as usize
    }

    #[getter]
    fn steps(&self) -> Vec<String> {
        let _ = &self.keepalive;
        self.prog.spec().steps.iter().map(|s| s.desc.clone()).collect()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Abi>()?;
    m.add_class::<OutputMode>()?;
    m.add_class::<Kernel>()?;
    m.add_class::<DimShuffle>()?;
    m.add_class::<Copy>()?;
    m.add_class::<Dot>()?;
    m.add_class::<Sort>()?;
    m.add_class::<Cholesky>()?;
    m.add_class::<SolveTriangular>()?;
    m.add_class::<Scan>()?;
    m.add_class::<ProgramBuilder>()?;
    m.add_class::<Program>()?;
    m.add_function(wrap_pyfunction!(c_api_table, m)?)?;
    Ok(())
}

/// The address of the `PtFunctionApi` table for functions from `Program.c_function()`
/// (header: `pytensor_llvm.c_header()`).
#[pyfunction]
fn c_api_table() -> usize {
    &super::capi::API as *const _ as usize
}

impl Program {
    /// For each step, the programs it runs.
    fn children(&self, py: Python<'_>) -> Vec<Vec<Py<Program>>> {
        self.keepalive
            .iter()
            .map(|k| {
                let k = k.bind(py);
                if let Ok(p) = k.cast::<Program>() {
                    vec![p.clone().unbind()]
                } else if let Ok(kern) = k.cast::<Kernel>() {
                    kern.get().programs.iter().map(|p| p.clone_ref(py)).collect()
                } else {
                    vec![]
                }
            })
            .collect()
    }

    fn profile_rows(&self, py: Python<'_>, depth: usize, rows: &mut Vec<(usize, String, u64, f64)>) {
        let (calls, secs) = self.prog.profile();
        let children = self.children(py);
        for (k, step) in self.prog.spec().steps.iter().enumerate() {
            rows.push((depth, step.desc.clone(), calls, secs[k]));
            for c in children.get(k).into_iter().flatten() {
                c.get().profile_rows(py, depth + 1, rows);
            }
        }
    }

    /// `calls`: of the top-level program (us/call is per top-level call).
    fn report_into(&self, py: Python<'_>, depth: usize, total: f64, calls: u64, s: &mut String) {
        use std::fmt::Write;
        let (_, secs) = self.prog.profile();
        let children = self.children(py);
        let mut order: Vec<usize> = (0..secs.len()).collect();
        order.sort_by(|&a, &b| secs[b].total_cmp(&secs[a]));
        for k in order {
            let pct = if total > 0.0 { 100.0 * secs[k] / total } else { 0.0 };
            let per = if calls > 0 { secs[k] / calls as f64 * 1e6 } else { 0.0 };
            let desc: String = self.prog.spec().steps[k].desc.chars().take(100).collect();
            let _ = writeln!(s, "{pct:>6.1}% {:>10.3} {per:>10.2}  {}{desc}", secs[k] * 1e3, "  ".repeat(depth));
            for c in children.get(k).into_iter().flatten() {
                c.get().report_into(py, depth + 1, total, calls, s);
            }
        }
    }
}
