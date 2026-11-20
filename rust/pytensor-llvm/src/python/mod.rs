//! `pytensor_llvm.runtime`: compiled functions (`function`) and Rust kernels
//! (`kernels`) for Python, the threads of the runtime, and the addresses C callers and
//! numba kernels need.

mod function;
mod kernels;
pub mod layout;

use self::function::Function;
use self::kernels::*;
use crate::abi::OutputMode;
use crate::dtype::DType;
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

pub(crate) fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

pub(crate) fn dtype_of(d: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    DType::from_numpy(d.kind(), d.itemsize())
        .ok_or_else(|| PyTypeError::new_err(format!("dtype {d} is not supported")))
}

pub(crate) fn descr<'py>(py: Python<'py>, dt: DType) -> Bound<'py, PyArrayDescr> {
    match dt {
        DType::Float64 => numpy::dtype::<f64>(py),
        DType::Float32 => numpy::dtype::<f32>(py),
        DType::Int64 => numpy::dtype::<i64>(py),
        DType::Int32 => numpy::dtype::<i32>(py),
        DType::Int16 => numpy::dtype::<i16>(py),
        DType::Int8 => numpy::dtype::<i8>(py),
        DType::Uint8 => numpy::dtype::<u8>(py),
        DType::Bool => numpy::dtype::<bool>(py),
        DType::Uint16 => numpy::dtype::<u16>(py),
        DType::Uint32 => numpy::dtype::<u32>(py),
        DType::Uint64 => numpy::dtype::<u64>(py),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputMode>()?;
    m.add_class::<RustKernel>()?;
    m.add_class::<Dot>()?;
    m.add_class::<Sort>()?;
    m.add_class::<Cholesky>()?;
    m.add_class::<SolveTriangular>()?;
    m.add_class::<Solve>()?;
    m.add_class::<CholeskySolve>()?;
    m.add_class::<MatrixInverse>()?;
    m.add_class::<Det>()?;
    m.add_class::<SLogDet>()?;
    m.add_class::<LUFactor>()?;
    m.add_class::<LU>()?;
    m.add_class::<PivotToPermutations>()?;
    m.add_class::<Eigh>()?;
    m.add_class::<SVD>()?;
    m.add_class::<QR>()?;
    m.add_class::<MatrixPinv>()?;
    m.add_class::<Function>()?;
    m.add_function(wrap_pyfunction!(c_api_table, m)?)?;
    m.add_function(wrap_pyfunction!(numba_entry, m)?)?;
    Ok(())
}

/// The address of the kernel that runs a numba kernel: the task's config is the address
/// of the numba cfunc (`kernel::numba_entry`).
#[pyfunction]
fn numba_entry() -> usize {
    crate::kernels::numba_entry as *const () as usize
}

/// The address of the `PtFunctionApi` table for functions from `Function.c_function()`
/// (header: `pytensor_llvm.c_header()`).
#[pyfunction]
fn c_api_table() -> usize {
    &crate::function::capi::API as *const _ as usize
}

/// Choose the number of threads parallel loops use (process-wide; only before the
/// first parallel loop runs). Default: `PYTENSOR_LLVM_THREADS`, else one per CPU.
#[pyfunction]
pub fn set_threads(n: usize) -> PyResult<()> {
    crate::runtime::runtime()
        .set_threads(n)
        .map_err(PyRuntimeError::new_err)
}

/// The number of threads parallel loops use.
#[pyfunction]
pub fn threads() -> usize {
    crate::runtime::runtime().threads()
}
