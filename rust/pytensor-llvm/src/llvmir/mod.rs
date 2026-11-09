//! `pytensor_llvm.ir`: a small LLVM IR builder for Python.
//!
//! Code generation happens in Python (one dispatch function per PyTensor op); this
//! module gives it modules, functions, values and a checked instruction builder, and
//! compiles finished modules into the JIT. See pytensor/link/llvm_v2/DESIGN.md.

pub mod builder;
pub mod module;
pub mod runtime;
pub mod ty;

use pyo3::prelude::*;
use ty::{PyType, Ty};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyType>()?;
    m.add_class::<module::Module>()?;
    m.add_class::<module::Function>()?;
    m.add_class::<module::Value>()?;
    m.add_class::<module::Compiled>()?;
    m.add_class::<module::Inline>()?;
    m.add_class::<builder::Builder>()?;
    m.add_class::<builder::Loop>()?;
    m.add_class::<builder::If>()?;
    m.add_class::<builder::ICmp>()?;
    m.add_class::<builder::FCmp>()?;
    m.add_class::<builder::Intrinsic>()?;
    m.add_class::<builder::Libm>()?;
    m.add_class::<builder::RustMath>()?;
    for (name, t) in [
        ("void", Ty::Void),
        ("i1", Ty::Int(1)),
        ("i8", Ty::Int(8)),
        ("i16", Ty::Int(16)),
        ("i32", Ty::Int(32)),
        ("i64", Ty::Int(64)),
        ("f32", Ty::F32),
        ("f64", Ty::F64),
        ("ptr", Ty::Ptr),
    ] {
        m.add(name, PyType(t))?;
    }
    m.add_function(wrap_pyfunction!(runtime::call_abi, m)?)?;
    m.add_function(wrap_pyfunction!(module::compile_many, m)?)?;
    Ok(())
}
