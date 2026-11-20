#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
//! Native code generation and runtime for PyTensor's LLVM backend (`pytensor/link/llvm`).
//!
//! * `abi`, `array`, `dtype`, `layout`: the data: the kernel ABI (arrays, buffers),
//!   arrays as Rust uses them, dtypes, what is known about strides;
//! * `ir` (Python: `pytensor_llvm.ir`): the IR builder Python generates code with,
//!   modules, compilation;
//! * `jit`: the JIT and LLVM setup (`target`), perf maps (`perf`);
//! * `runtime`: the process-wide runtime (runtime functions, JIT, thread pool) and the
//!   context of a call, and what generated code calls (`math`, `random`, `parallel`);
//! * `kernels`: writing Rust kernels, and the kernels (`linalg`, `sort`);
//! * `function`: calling a compiled program (and from C: `capi`);
//! * `python` (Python: `pytensor_llvm.runtime`): compiled functions and Rust kernels
//!   for Python;
//! * `testing`: calling kernels in tests.

use pyo3::prelude::*;

pub mod abi;
pub mod array;
pub mod dtype;
pub mod function;
pub mod ir;
pub mod jit;
pub mod kernels;
pub mod layout;
pub mod python;
pub mod runtime;
#[cfg(test)]
pub mod testing;

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
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("pytensor_llvm.ir", &ir_mod)?;
    let runtime_mod = PyModule::new(m.py(), "runtime")?;
    python::register(&runtime_mod)?;
    m.add_submodule(&runtime_mod)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("pytensor_llvm.runtime", &runtime_mod)?;
    // the process-wide runtime
    m.add_function(wrap_pyfunction!(python::set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(python::threads, m)?)?;
    Ok(())
}
