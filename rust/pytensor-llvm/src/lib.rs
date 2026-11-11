#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
//! Native code generation and runtime for PyTensor's LLVM backend (`pytensor/link/llvm`).
//!
//! * `ir` (Python: `pytensor_llvm.ir`): the IR builder Python generates code with,
//!   modules, compilation;
//! * `interp` (Python: `pytensor_llvm.interp`): the program interpreter, Rust kernels,
//!   the C API;
//! * `abi`: the kernel ABI (arrays, buffers);
//! * `runtime`: the process-wide runtime (runtime functions, JIT, thread pool) and the
//!   context of a call;
//! * `jit`, `target`: the JIT and LLVM setup; `perf`: perf maps;
//! * `math`, `random`, `parallel`: what generated code calls;
//! * `kernel`: what Rust kernels are written with; `linalg`, `sort`, `interp::kernels`,
//!   `interp::scan`: the kernels.

use pyo3::prelude::*;

pub mod abi;
pub mod dtype;
pub mod interp;
pub mod ir;
pub mod jit;
pub mod kernel;
pub mod linalg;
pub mod math;
pub mod parallel;
pub mod perf;
pub mod random;
pub mod runtime;
pub mod sort;
pub mod target;

/// Contents of the C header (include/pytensor_llvm.h).
#[pyfunction]
fn c_header() -> &'static str {
    include_str!("../include/pytensor_llvm.h")
}

#[pyfunction]
fn llvm_version() -> String {
    let (mut a, mut b, mut c) = (0, 0, 0);
    // SAFETY: plain out-parameters.
    unsafe { llvm_sys::core::LLVMGetVersion(&mut a, &mut b, &mut c) };
    format!("{a}.{b}.{c}")
}

#[pymodule]
fn pytensor_llvm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(llvm_version, m)?)?;
    m.add_function(wrap_pyfunction!(c_header, m)?)?;
    m.add("ABI_VERSION", abi::ABI_VERSION)?;
    let ir_mod = PyModule::new(m.py(), "ir")?;
    ir::register(&ir_mod)?;
    m.add_submodule(&ir_mod)?;
    m.py().import("sys")?.getattr("modules")?.set_item("pytensor_llvm.ir", &ir_mod)?;
    let interp_mod = PyModule::new(m.py(), "interp")?;
    interp::py::register(&interp_mod)?;
    m.add_submodule(&interp_mod)?;
    m.py().import("sys")?.getattr("modules")?.set_item("pytensor_llvm.interp", &interp_mod)?;
    // the process-wide runtime
    m.add_function(wrap_pyfunction!(interp::py::set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(interp::py::threads, m)?)?;
    Ok(())
}
