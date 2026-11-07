//! Python bindings: the compiled function and module definition.

use crate::abi::PtBuffer;
use crate::builder::{GraphBuilder, ScalarGraph, ScalarRef};
use crate::dtype::DType;
use crate::runtime::{CallState, OutArray, Program};
use numpy::npyffi::{npy_intp, NpyTypes, NPY_ARRAY_WRITEABLE, PY_ARRAY_API};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use std::sync::Mutex;

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

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

/// Wrap an owned output buffer as a numpy array without copying. The array's base is a
/// capsule that releases our reference when numpy is done with the memory.
fn wrap_output<'py>(py: Python<'py>, o: &OutArray) -> PyResult<Bound<'py, PyAny>> {
    let owner = SendPtr(o.owner);
    let capsule = PyCapsule::new_with_value_and_destructor(py, owner, c"pytensor_llvm.buffer", |b: SendPtr<PtBuffer>, _| unsafe {
        PtBuffer::decref(b.0)
    })?;
    let mut dims: Vec<npy_intp> = o.shape().iter().map(|&s| s as npy_intp).collect();
    let mut strides: Vec<npy_intp> = o.strides().iter().map(|&s| s as npy_intp).collect();
    unsafe {
        let ty = numpy::npyffi::get_type_object(py, NpyTypes::PyArray_Type);
        let arr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            ty,
            descr(py, o.dtype).into_ptr() as *mut _, // stolen
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

#[pyclass(module = "pytensor_llvm")]
pub struct CompiledFunction {
    prog: std::sync::Arc<Program>,
    state: Mutex<CallState>,
    np_asarray: Py<PyAny>,
}

impl CompiledFunction {
    pub fn new(py: Python<'_>, prog: Program) -> PyResult<Self> {
        let state = Mutex::new(prog.new_state());
        let np = py.import("numpy")?;
        Ok(CompiledFunction { prog: std::sync::Arc::new(prog), state, np_asarray: np.getattr("asarray")?.unbind() })
    }
}

#[pymethods]
impl CompiledFunction {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyList>> {
        let prog: &Program = &self.prog;
        if args.len() != prog.n_inputs() {
            return Err(PyValueError::new_err(format!("expected {} inputs, got {}", prog.n_inputs(), args.len())));
        }
        let mut st = self.state.lock().unwrap();
        let mut keep: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
        let mut work = 0usize;
        for (k, a) in args.iter().enumerate() {
            let a = if a.cast::<PyUntypedArray>().is_ok() { a } else { self.np_asarray.bind(py).call1((a,))? };
            let arr = a.cast::<PyUntypedArray>()?;
            let dt = arr.dtype();
            let dtype = DType::from_numpy(dt.kind(), dt.itemsize());
            let data = unsafe { (*arr.as_array_ptr()).fields.data as *mut u8 };
            prog.set_input(&mut st, k, data, dtype, arr.shape(), arr.strides()).map_err(PyValueError::new_err)?;
            work += arr.len();
            keep.push(a);
        }
        // Releasing the GIL costs ~100ns; only worth it when there's real work.
        let res = if work > 20_000 {
            let sp = SendPtr(&mut *st as *mut CallState);
            let pp = SendPtr(prog as *const Program as *mut Program);
            py.detach(move || {
                let (sp, pp) = (sp, pp);
                unsafe { (*pp.0).execute(&mut *sp.0, None) }
            })
        } else {
            prog.execute(&mut st, None)
        };
        drop(keep);
        res.map_err(PyRuntimeError::new_err)?;
        let arrays: Vec<Bound<'_, PyAny>> = st.outs.iter().map(|o| wrap_output(py, o)).collect::<PyResult<_>>()?;
        drop(st);
        Ok(PyList::new(py, arrays)?.unbind())
    }

    /// Address of a new `PtFunction` reference for the C API (release with
    /// `function_decref` from the table returned by `pytensor_llvm.c_api_table()`).
    fn c_function(&self) -> usize {
        crate::capi::function_handle(&self.prog) as usize
    }

    /// Turn per-kernel timing on or off. `profile_report()` returns (description, seconds).
    fn set_profiling(&self, on: bool) {
        let mut st = self.state.lock().unwrap();
        st.profile = if on { Some(vec![0.0; self.prog.n_steps()]) } else { None };
    }

    fn profile_report(&self) -> Vec<(String, f64)> {
        let st = self.state.lock().unwrap();
        match &st.profile {
            Some(p) => self.prog.kernel_descriptions().into_iter().zip(p.iter().copied()).collect(),
            None => vec![],
        }
    }

    /// Storage and layout chosen for every node output.
    fn plan(&self) -> Vec<(String, String, String)> {
        self.prog.plan_description()
    }

    /// Compile statistics and the kernel schedule.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let s = &self.prog.stats;
        d.set_item("n_kernels", s.n_kernels)?;
        d.set_item("n_unique", s.n_unique)?;
        d.set_item("n_in_process", s.n_in_process)?;
        d.set_item("n_disk_cache", s.n_disk_cache)?;
        d.set_item("n_compiled", s.n_compiled)?;
        d.set_item("llvm_cpu_s", s.llvm_cpu_s)?;
        d.set_item("kernel_wall_s", s.wall_s)?;
        d.set_item("link_s", s.link_s)?;
        d.set_item("n_slots", self.prog.n_slots())?;
        d.set_item("n_inplace", self.prog.n_inplace)?;
        d.set_item("schedule", self.prog.kernel_descriptions())?;
        Ok(d)
    }
}

/// Address of the static `PtFunctionApi` table (see include/pytensor_llvm.h).
#[pyfunction]
fn c_api_table() -> usize {
    &crate::capi::API as *const _ as usize
}

/// Contents of the C header (include/pytensor_llvm.h).
#[pyfunction]
fn c_header() -> &'static str {
    include_str!("../include/pytensor_llvm.h")
}

#[pyfunction]
fn llvm_version() -> String {
    let (mut a, mut b, mut c) = (0, 0, 0);
    unsafe { llvm_sys::core::LLVMGetVersion(&mut a, &mut b, &mut c) };
    format!("{a}.{b}.{c}")
}

#[pymodule]
fn pytensor_llvm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(llvm_version, m)?)?;
    m.add_function(wrap_pyfunction!(c_api_table, m)?)?;
    m.add_function(wrap_pyfunction!(c_header, m)?)?;
    m.add_class::<GraphBuilder>()?;
    m.add_class::<ScalarGraph>()?;
    m.add_class::<ScalarRef>()?;
    m.add_class::<CompiledFunction>()?;
    m.add("ABI_VERSION", crate::abi::ABI_VERSION)?;
    Ok(())
}
