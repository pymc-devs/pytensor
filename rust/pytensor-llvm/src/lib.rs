#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
//! Native code generation and runtime for PyTensor's LLVM backend (`pytensor/link/llvm`).
//!
//! * `llvmir`: the IR builder Python generates code with, modules, compilation;
//! * `rt`: the program interpreter, Rust kernels, the C API;
//! * `abi`: the kernel ABI, buffers, the runtime functions and the context;
//! * `jit`, `target`: the JIT and LLVM setup; `perf`: perf maps;
//! * `linalg`, `native`, `random`, `parallel`: what kernels and generated code call.

use pyo3::prelude::*;

pub mod abi;
pub mod dtype;
pub mod jit;
pub mod linalg;
pub mod llvmir;
pub mod native;
pub mod parallel;
pub mod perf;
pub mod random;
pub mod rt;
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
    let ir = PyModule::new(m.py(), "ir")?;
    llvmir::register(&ir)?;
    m.add_submodule(&ir)?;
    m.py().import("sys")?.getattr("modules")?.set_item("pytensor_llvm.ir", &ir)?;
    let rt = PyModule::new(m.py(), "rt")?;
    rt::py::register(&rt)?;
    m.add_submodule(&rt)?;
    m.py().import("sys")?.getattr("modules")?.set_item("pytensor_llvm.rt", &rt)?;
    Ok(())
}
