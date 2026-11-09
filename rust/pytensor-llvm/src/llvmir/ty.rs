//! LLVM types as plain data (independent of any module).

use llvm_sys::core as c;
use llvm_sys::prelude::{LLVMContextRef, LLVMTypeRef};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Void,
    Int(u32),
    F32,
    F64,
    Ptr,
    Struct(Arc<[Ty]>),
    Array(Arc<Ty>, u64),
}

impl Ty {
    pub fn is_int(&self) -> bool {
        matches!(self, Ty::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }

    pub fn is_bool(&self) -> bool {
        *self == Ty::Int(1)
    }

    pub fn int_bits(&self) -> Option<u32> {
        match self {
            Ty::Int(b) => Some(*b),
            _ => None,
        }
    }

    pub fn float_bits(&self) -> Option<u32> {
        match self {
            Ty::F32 => Some(32),
            Ty::F64 => Some(64),
            _ => None,
        }
    }

    /// Can be held in a value (an SSA register).
    pub fn is_first_class(&self) -> bool {
        !matches!(self, Ty::Void)
    }

    /// The type of the element at `index` of an aggregate.
    pub fn member(&self, index: u32) -> Option<&Ty> {
        match self {
            Ty::Struct(fields) => fields.get(index as usize),
            Ty::Array(elem, n) if (index as u64) < *n => Some(elem),
            _ => None,
        }
    }

    pub fn to_llvm(&self, ctx: LLVMContextRef) -> LLVMTypeRef {
        // SAFETY: `ctx` is a live context; the type constructors only read their inputs.
        unsafe {
            match self {
                Ty::Void => c::LLVMVoidTypeInContext(ctx),
                Ty::Int(bits) => c::LLVMIntTypeInContext(ctx, *bits),
                Ty::F32 => c::LLVMFloatTypeInContext(ctx),
                Ty::F64 => c::LLVMDoubleTypeInContext(ctx),
                Ty::Ptr => c::LLVMPointerTypeInContext(ctx, 0),
                Ty::Struct(fields) => {
                    let mut f: Vec<LLVMTypeRef> = fields.iter().map(|t| t.to_llvm(ctx)).collect();
                    c::LLVMStructTypeInContext(ctx, f.as_mut_ptr(), f.len() as u32, 0)
                }
                Ty::Array(elem, n) => c::LLVMArrayType2(elem.to_llvm(ctx), *n),
            }
        }
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Void => write!(f, "void"),
            Ty::Int(b) => write!(f, "i{b}"),
            Ty::F32 => write!(f, "float"),
            Ty::F64 => write!(f, "double"),
            Ty::Ptr => write!(f, "ptr"),
            Ty::Struct(fields) => {
                write!(f, "{{ ")?;
                for (k, t) in fields.iter().enumerate() {
                    if k > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, " }}")
            }
            Ty::Array(elem, n) => write!(f, "[{n} x {elem}]"),
        }
    }
}

/// An LLVM type.
#[pyclass(module = "pytensor_llvm.ir", name = "Type", frozen, eq, hash, from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyType(pub Ty);

#[pymethods]
impl PyType {
    fn __repr__(&self) -> String {
        format!("ir.Type({})", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// An integer type of `bits` bits (1, 8, 16, 32 or 64).
    #[staticmethod]
    fn int(bits: u32) -> PyResult<Self> {
        if ![1, 8, 16, 32, 64].contains(&bits) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("unsupported integer width {bits}")));
        }
        Ok(PyType(Ty::Int(bits)))
    }

    /// A struct of the given field types.
    #[staticmethod]
    fn struct_(fields: Vec<PyType>) -> Self {
        PyType(Ty::Struct(fields.into_iter().map(|t| t.0).collect()))
    }

    /// An array of `n` elements.
    #[staticmethod]
    fn array(elem: PyType, n: u64) -> PyResult<Self> {
        if !elem.0.is_first_class() {
            return Err(pyo3::exceptions::PyValueError::new_err("array of void"));
        }
        Ok(PyType(Ty::Array(Arc::new(elem.0), n)))
    }

    #[getter]
    fn is_int(&self) -> bool {
        self.0.is_int()
    }

    #[getter]
    fn is_float(&self) -> bool {
        self.0.is_float()
    }

    /// Bit width of integer and float types.
    #[getter]
    fn bits(&self) -> Option<u32> {
        self.0.int_bits().or(self.0.float_bits())
    }

    /// Field types of a struct.
    #[getter]
    fn fields(&self) -> Option<Vec<PyType>> {
        match &self.0 {
            Ty::Struct(f) => Some(f.iter().cloned().map(PyType).collect()),
            _ => None,
        }
    }
}
