//! The instruction builder.
//!
//! Every method checks its operands before calling LLVM: values must come from this
//! module and function, have the right types, and the current block must be open.
//! LLVM's own checks are assertions, which release builds don't have.

use super::module::{add_fn_attr, cstring, err, ModuleInner, Value};
use super::ty::{PyType, Ty};
use llvm_sys::core as c;
use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMTypeRef, LLVMValueRef};
use llvm_sys::{LLVMIntPredicate as IP, LLVMRealPredicate as RP};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cell::RefCell;
use std::rc::Rc;

type BuildFn = unsafe extern "C" fn(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, *const std::ffi::c_char) -> LLVMValueRef;
type CastFn = unsafe extern "C" fn(LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, *const std::ffi::c_char) -> LLVMValueRef;

const NONAME: *const std::ffi::c_char = c"".as_ptr();

const FAST: u32 = llvm_sys::LLVMFastMathAllowReassoc
    | llvm_sys::LLVMFastMathNoSignedZeros
    | llvm_sys::LLVMFastMathAllowReciprocal
    | llvm_sys::LLVMFastMathAllowContract
    | llvm_sys::LLVMFastMathApproxFunc;

#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ICmp {
    EQ,
    NE,
    SLT,
    SLE,
    SGT,
    SGE,
    ULT,
    ULE,
    UGT,
    UGE,
}

#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FCmp {
    /// ordered: false if either operand is NaN
    OEQ,
    ONE,
    OLT,
    OLE,
    OGT,
    OGE,
    /// unordered: true if either operand is NaN
    UEQ,
    UNE,
    ULT,
    ULE,
    UGT,
    UGE,
    /// neither is NaN
    ORD,
    /// either is NaN
    UNO,
}

/// LLVM intrinsics on floats (overloaded on the operand type).
#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Intrinsic {
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Sqrt,
    Sin,
    Cos,
    Floor,
    Ceil,
    Trunc,
    RoundEven,
    Round,
    Fabs,
    Pow,
    Maximum,
    Minimum,
    Copysign,
    Fma,
}

impl Intrinsic {
    fn name(self) -> &'static str {
        match self {
            Intrinsic::Exp => "llvm.exp",
            Intrinsic::Exp2 => "llvm.exp2",
            Intrinsic::Log => "llvm.log",
            Intrinsic::Log2 => "llvm.log2",
            Intrinsic::Log10 => "llvm.log10",
            Intrinsic::Sqrt => "llvm.sqrt",
            Intrinsic::Sin => "llvm.sin",
            Intrinsic::Cos => "llvm.cos",
            Intrinsic::Floor => "llvm.floor",
            Intrinsic::Ceil => "llvm.ceil",
            Intrinsic::Trunc => "llvm.trunc",
            Intrinsic::RoundEven => "llvm.roundeven",
            Intrinsic::Round => "llvm.round",
            Intrinsic::Fabs => "llvm.fabs",
            Intrinsic::Pow => "llvm.pow",
            Intrinsic::Maximum => "llvm.maximum",
            Intrinsic::Minimum => "llvm.minimum",
            Intrinsic::Copysign => "llvm.copysign",
            Intrinsic::Fma => "llvm.fma",
        }
    }

    fn arity(self) -> usize {
        match self {
            Intrinsic::Pow | Intrinsic::Maximum | Intrinsic::Minimum | Intrinsic::Copysign => 2,
            Intrinsic::Fma => 3,
            _ => 1,
        }
    }
}

/// Random numbers from the call context (the runtime functions `ptl_rt_rng_*`,
/// `crate::random`).
#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rng {
    /// () -> i64: 64 random bits
    U64,
    /// () -> f64 on [0, 1)
    Uniform,
    /// () -> f64, standard normal
    Normal,
    /// (ptr out, i64 n): n uniforms on [0, 1)
    FillUniform,
    /// (ptr out, i64 n): n standard normals
    FillNormal,
    /// (f64 shape) -> f64, scale 1
    Gamma,
    /// (f64 a, f64 b) -> f64
    Beta,
    /// (f64 rate) -> i64
    Poisson,
    /// (i64 n, f64 p) -> i64
    Binomial,
}

impl Rng {
    fn signature(self) -> (&'static str, Vec<Ty>, Ty) {
        use Ty::*;
        match self {
            Rng::U64 => ("ptl_rt_rng_u64", vec![], Int(64)),
            Rng::Uniform => ("ptl_rt_rng_uniform", vec![], F64),
            Rng::Normal => ("ptl_rt_rng_normal", vec![], F64),
            Rng::FillUniform => ("ptl_rt_rng_fill_uniform", vec![Ptr, Int(64)], Void),
            Rng::FillNormal => ("ptl_rt_rng_fill_normal", vec![Ptr, Int(64)], Void),
            Rng::Gamma => ("ptl_rt_rng_gamma", vec![F64], F64),
            Rng::Beta => ("ptl_rt_rng_beta", vec![F64, F64], F64),
            Rng::Poisson => ("ptl_rt_rng_poisson", vec![F64], Int(64)),
            Rng::Binomial => ("ptl_rt_rng_binomial", vec![Int(64), F64], Int(64)),
        }
    }
}

/// libm functions (glibc). All have SIMD variants in libmvec, which we advertise.
#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Libm {
    Expm1,
    Log1p,
    Tan,
    Tanh,
    Sinh,
    Cosh,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
    Atan2,
    Erf,
    Erfc,
}

impl Libm {
    fn c_name(self) -> &'static str {
        match self {
            Libm::Expm1 => "expm1",
            Libm::Log1p => "log1p",
            Libm::Tan => "tan",
            Libm::Tanh => "tanh",
            Libm::Sinh => "sinh",
            Libm::Cosh => "cosh",
            Libm::Asin => "asin",
            Libm::Acos => "acos",
            Libm::Atan => "atan",
            Libm::Asinh => "asinh",
            Libm::Acosh => "acosh",
            Libm::Atanh => "atanh",
            Libm::Atan2 => "atan2",
            Libm::Erf => "erf",
            Libm::Erfc => "erfc",
        }
    }

    fn arity(self) -> usize {
        if self == Libm::Atan2 {
            2
        } else {
            1
        }
    }

    fn has_vector_variant(self) -> bool {
        !matches!(self, Libm::Erf | Libm::Erfc)
    }
}

/// Scalar functions precompiled in Rust.
#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RustMath {
    GammaLn,
    Gamma,
    Psi,
}

impl RustMath {
    fn symbol(self, f32: bool) -> &'static str {
        match (self, f32) {
            (RustMath::GammaLn, false) => "ptl_gammaln_f64",
            (RustMath::GammaLn, true) => "ptl_gammaln_f32",
            (RustMath::Gamma, false) => "ptl_gamma_f64",
            (RustMath::Gamma, true) => "ptl_gamma_f32",
            (RustMath::Psi, false) => "ptl_psi_f64",
            (RustMath::Psi, true) => "ptl_psi_f32",
        }
    }
}

/// Builds the body of one function.
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Builder {
    m: Rc<ModuleInner>,
    b: LLVMBuilderRef,
    /// positioned before the entry block's terminator: allocas go there
    allocas: LLVMBuilderRef,
    func: LLVMValueRef,
    ret: Ty,
    /// blocks of open `with` contexts, innermost last (for error messages)
    depth: RefCell<usize>,
}

impl Drop for Builder {
    fn drop(&mut self) {
        // SAFETY: we own both builders.
        unsafe {
            c::LLVMDisposeBuilder(self.b);
            c::LLVMDisposeBuilder(self.allocas);
        }
    }
}

impl Builder {
    pub fn new(m: Rc<ModuleInner>, func: LLVMValueRef, ret: Ty) -> PyResult<Self> {
        // SAFETY: `func` is a function of `m` without blocks; both builders belong to
        // `m`'s context.
        unsafe {
            let entry = c::LLVMAppendBasicBlockInContext(m.ctx, func, c"entry".as_ptr());
            let body = c::LLVMAppendBasicBlockInContext(m.ctx, func, c"body".as_ptr());
            let b = c::LLVMCreateBuilderInContext(m.ctx);
            c::LLVMPositionBuilderAtEnd(b, entry);
            let br = c::LLVMBuildBr(b, body);
            let allocas = c::LLVMCreateBuilderInContext(m.ctx);
            c::LLVMPositionBuilderBefore(allocas, br);
            c::LLVMPositionBuilderAtEnd(b, body);
            Ok(Builder { m, b, allocas, func, ret, depth: RefCell::new(0) })
        }
    }

    fn block(&self) -> LLVMBasicBlockRef {
        // SAFETY: the builder is always positioned in a block of `func`.
        unsafe { c::LLVMGetInsertBlock(self.b) }
    }

    fn is_terminated(&self) -> bool {
        // SAFETY: a live block.
        unsafe { !c::LLVMGetBasicBlockTerminator(self.block()).is_null() }
    }

    /// Ready to add an instruction?
    fn open(&self) -> PyResult<()> {
        self.m.check_open()?;
        if self.is_terminated() {
            return err("the current block already ended (after ret); no more instructions can follow");
        }
        Ok(())
    }

    /// The raw value, if it may be used here.
    fn v(&self, v: &Value) -> PyResult<LLVMValueRef> {
        if !Rc::ptr_eq(&v.m, &self.m) {
            return err("value from a different module");
        }
        if let Some(f) = v.func {
            if f != self.func {
                return err("value from a different function");
            }
        }
        Ok(v.raw)
    }

    fn mk(&self, raw: LLVMValueRef, ty: Ty) -> Value {
        Value { m: self.m.clone(), raw, ty, func: Some(self.func) }
    }

    fn konst(&self, raw: LLVMValueRef, ty: Ty) -> Value {
        Value { m: self.m.clone(), raw, ty, func: None }
    }

    fn lt(&self, t: &Ty) -> LLVMTypeRef {
        self.m.llvm_type(t)
    }

    fn fast(&self, v: LLVMValueRef) {
        if self.m.fastmath {
            // SAFETY: `v` is a floating-point instruction (callers ensure it).
            unsafe {
                if c::LLVMCanValueUseFastMathFlags(v) != 0 {
                    c::LLVMSetFastMathFlags(v, FAST);
                }
            }
        }
    }

    fn same_type(&self, what: &str, a: &Value, b: &Value) -> PyResult<()> {
        if a.ty != b.ty {
            return err(format!("{what}: operand types differ ({} and {})", a.ty, b.ty));
        }
        Ok(())
    }

    fn int_binop(&self, what: &str, f: BuildFn, a: &Value, b: &Value) -> PyResult<Value> {
        self.open()?;
        self.same_type(what, a, b)?;
        if !a.ty.is_int() {
            return err(format!("{what} needs integers, got {}", a.ty));
        }
        // SAFETY: two integer values of the same type in this function.
        let r = unsafe { f(self.b, self.v(a)?, self.v(b)?, NONAME) };
        Ok(self.mk(r, a.ty.clone()))
    }

    fn float_binop(&self, what: &str, f: BuildFn, a: &Value, b: &Value) -> PyResult<Value> {
        self.open()?;
        self.same_type(what, a, b)?;
        if !a.ty.is_float() {
            return err(format!("{what} needs floats, got {}", a.ty));
        }
        // SAFETY: two float values of the same type in this function.
        let r = unsafe { f(self.b, self.v(a)?, self.v(b)?, NONAME) };
        self.fast(r);
        Ok(self.mk(r, a.ty.clone()))
    }

    fn cast(&self, what: &str, f: CastFn, v: &Value, to: &Ty) -> PyResult<Value> {
        // SAFETY: the caller checked that this cast is valid between the two types.
        let r = unsafe { f(self.b, self.v(v)?, self.lt(to), NONAME) };
        let _ = what;
        Ok(self.mk(r, to.clone()))
    }

    fn int_cast(&self, what: &str, f: CastFn, v: &Value, to: &Ty, widen: bool) -> PyResult<Value> {
        self.open()?;
        match (v.ty.int_bits(), to.int_bits()) {
            (Some(a), Some(b)) if (widen && b > a) || (!widen && b < a) => self.cast(what, f, v, to),
            _ => err(format!("{what} from {} to {}", v.ty, to)),
        }
    }

    fn call_raw(&self, fnty: LLVMTypeRef, f: LLVMValueRef, args: &mut [LLVMValueRef], ret: &Ty) -> Option<Value> {
        // SAFETY: `f` has type `fnty` and the arguments were checked against it.
        let r = unsafe { c::LLVMBuildCall2(self.b, fnty, f, args.as_mut_ptr(), args.len() as u32, NONAME) };
        if *ret == Ty::Void {
            None
        } else {
            Some(self.mk(r, ret.clone()))
        }
    }

    /// Declare (once) an external function.
    fn declare(&self, name: &str, params: &[Ty], ret: &Ty, pure: bool) -> (LLVMValueRef, LLVMTypeRef) {
        let n = cstring(name);
        // SAFETY: looking up / adding a declaration in this module.
        unsafe {
            let existing = c::LLVMGetNamedFunction(self.m.module, n.as_ptr());
            let f = if existing.is_null() {
                let mut ps: Vec<LLVMTypeRef> = params.iter().map(|t| self.lt(t)).collect();
                let fnty = c::LLVMFunctionType(self.lt(ret), ps.as_mut_ptr(), ps.len() as u32, 0);
                let f = c::LLVMAddFunction(self.m.module, n.as_ptr(), fnty);
                if pure {
                    for a in ["nounwind", "willreturn", "nosync", "nofree"] {
                        add_fn_attr(&self.m, f, a);
                    }
                    // memory(none): we never read errno (like -fno-math-errno)
                    add_fn_attr(&self.m, f, "memory");
                }
                f
            } else {
                existing
            };
            (f, c::LLVMGlobalGetValueType(f))
        }
    }

    fn phi(&self, ty: &Ty, incoming: &[(LLVMValueRef, LLVMBasicBlockRef)]) -> LLVMValueRef {
        // SAFETY: builds a phi at the start of the current block (callers position the
        // builder there) with incoming values of `ty`.
        unsafe {
            let p = c::LLVMBuildPhi(self.b, self.lt(ty), NONAME);
            for &(v, bb) in incoming {
                let (mut v, mut bb) = (v, bb);
                c::LLVMAddIncoming(p, &mut v, &mut bb, 1);
            }
            p
        }
    }

    fn new_block(&self, name: &std::ffi::CStr) -> LLVMBasicBlockRef {
        // SAFETY: appending a block to this function.
        unsafe { c::LLVMAppendBasicBlockInContext(self.m.ctx, self.func, name.as_ptr()) }
    }

    fn position(&self, bb: LLVMBasicBlockRef) {
        // SAFETY: `bb` is a block of this function.
        unsafe { c::LLVMPositionBuilderAtEnd(self.b, bb) }
    }

    fn br(&self, bb: LLVMBasicBlockRef) {
        if !self.is_terminated() {
            // SAFETY: the current block is open; `bb` is a block of this function.
            unsafe { c::LLVMBuildBr(self.b, bb) };
        }
    }

    fn cond_br(&self, cond: LLVMValueRef, t: LLVMBasicBlockRef, f: LLVMBasicBlockRef) {
        // SAFETY: `cond` is an i1 in this function; the blocks belong to it.
        unsafe { c::LLVMBuildCondBr(self.b, cond, t, f) };
    }

    fn check_values(&self, what: &str, vals: &[Value], want: &[Ty]) -> PyResult<()> {
        if vals.len() != want.len() {
            return err(format!("{what}: expected {} values, got {}", want.len(), vals.len()));
        }
        for (k, (v, t)) in vals.iter().zip(want).enumerate() {
            self.v(v)?;
            if &v.ty != t {
                return err(format!("{what}: value {k} has type {}, expected {t}", v.ty));
            }
        }
        Ok(())
    }
}

#[pymethods]
impl Builder {
    /// The current block has ended (with `ret`).
    #[getter]
    fn terminated(&self) -> bool {
        self.is_terminated()
    }

    // ------------------------------------------------------------ constants

    fn const_int(&self, ty: PyType, value: i128) -> PyResult<Value> {
        let bits = ty.0.int_bits().ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("const_int of {}", ty.0)))?;
        let (lo, hi): (i128, i128) = if bits == 1 { (0, 1) } else { (-(1i128 << (bits - 1)), (1i128 << bits) - 1) };
        if value < lo || value > hi {
            return err(format!("{value} doesn't fit in i{bits}"));
        }
        // SAFETY: an integer constant of a valid integer type.
        let r = unsafe { c::LLVMConstInt(self.lt(&ty.0), value as i64 as u64, 1) };
        Ok(self.konst(r, ty.0))
    }

    fn const_float(&self, ty: PyType, value: f64) -> PyResult<Value> {
        if !ty.0.is_float() {
            return err(format!("const_float of {}", ty.0));
        }
        // SAFETY: a float constant of a float type.
        let r = unsafe { c::LLVMConstReal(self.lt(&ty.0), value) };
        Ok(self.konst(r, ty.0))
    }

    fn const_bool(&self, value: bool) -> Value {
        // SAFETY: an i1 constant.
        let r = unsafe { c::LLVMConstInt(self.lt(&Ty::Int(1)), value as u64, 0) };
        self.konst(r, Ty::Int(1))
    }

    /// The null pointer.
    fn null(&self) -> Value {
        // SAFETY: the null constant of ptr.
        let r = unsafe { c::LLVMConstNull(self.lt(&Ty::Ptr)) };
        self.konst(r, Ty::Ptr)
    }

    /// The all-zero value of a type (also structs and arrays).
    fn zero(&self, ty: PyType) -> PyResult<Value> {
        if !ty.0.is_first_class() {
            return err("zero of void");
        }
        // SAFETY: the null constant of a first-class type.
        let r = unsafe { c::LLVMConstNull(self.lt(&ty.0)) };
        Ok(self.konst(r, ty.0))
    }

    fn undef(&self, ty: PyType) -> PyResult<Value> {
        if !ty.0.is_first_class() {
            return err("undef of void");
        }
        // SAFETY: the undef constant of a first-class type.
        let r = unsafe { c::LLVMGetUndef(self.lt(&ty.0)) };
        Ok(self.konst(r, ty.0))
    }

    /// Pointer to a constant NUL-terminated string.
    fn cstr(&self, s: &str) -> PyResult<Value> {
        self.open()?;
        let s = cstring(s);
        // SAFETY: creates a private global in this module.
        let r = unsafe { c::LLVMBuildGlobalStringPtr(self.b, s.as_ptr(), c"str".as_ptr()) };
        Ok(self.konst(r, Ty::Ptr))
    }

    // ------------------------------------------------------------ integer arithmetic

    /// `nsw`: the result doesn't overflow as a signed integer (lets LLVM simplify).
    #[pyo3(signature = (a, b, nsw = false))]
    fn add(&self, a: Value, b: Value, nsw: bool) -> PyResult<Value> {
        self.int_binop("add", if nsw { c::LLVMBuildNSWAdd } else { c::LLVMBuildAdd }, &a, &b)
    }

    #[pyo3(signature = (a, b, nsw = false))]
    fn sub(&self, a: Value, b: Value, nsw: bool) -> PyResult<Value> {
        self.int_binop("sub", if nsw { c::LLVMBuildNSWSub } else { c::LLVMBuildSub }, &a, &b)
    }

    #[pyo3(signature = (a, b, nsw = false))]
    fn mul(&self, a: Value, b: Value, nsw: bool) -> PyResult<Value> {
        self.int_binop("mul", if nsw { c::LLVMBuildNSWMul } else { c::LLVMBuildMul }, &a, &b)
    }

    /// Signed division. Division by zero (or MIN / -1) is undefined: check first.
    fn sdiv(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("sdiv", c::LLVMBuildSDiv, &a, &b)
    }

    fn udiv(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("udiv", c::LLVMBuildUDiv, &a, &b)
    }

    fn srem(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("srem", c::LLVMBuildSRem, &a, &b)
    }

    fn urem(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("urem", c::LLVMBuildURem, &a, &b)
    }

    fn shl(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("shl", c::LLVMBuildShl, &a, &b)
    }

    fn lshr(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("lshr", c::LLVMBuildLShr, &a, &b)
    }

    fn ashr(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("ashr", c::LLVMBuildAShr, &a, &b)
    }

    fn and_(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("and", c::LLVMBuildAnd, &a, &b)
    }

    fn or_(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("or", c::LLVMBuildOr, &a, &b)
    }

    fn xor(&self, a: Value, b: Value) -> PyResult<Value> {
        self.int_binop("xor", c::LLVMBuildXor, &a, &b)
    }

    fn not_(&self, a: Value) -> PyResult<Value> {
        self.open()?;
        if !a.ty.is_int() {
            return err(format!("not of {}", a.ty));
        }
        // SAFETY: an integer value of this function.
        let r = unsafe { c::LLVMBuildNot(self.b, self.v(&a)?, NONAME) };
        Ok(self.mk(r, a.ty))
    }

    fn neg(&self, a: Value) -> PyResult<Value> {
        self.open()?;
        if !a.ty.is_int() {
            return err(format!("neg of {}", a.ty));
        }
        // SAFETY: an integer value of this function.
        let r = unsafe { c::LLVMBuildNeg(self.b, self.v(&a)?, NONAME) };
        Ok(self.mk(r, a.ty))
    }

    // ------------------------------------------------------------ float arithmetic

    fn fadd(&self, a: Value, b: Value) -> PyResult<Value> {
        self.float_binop("fadd", c::LLVMBuildFAdd, &a, &b)
    }

    fn fsub(&self, a: Value, b: Value) -> PyResult<Value> {
        self.float_binop("fsub", c::LLVMBuildFSub, &a, &b)
    }

    fn fmul(&self, a: Value, b: Value) -> PyResult<Value> {
        self.float_binop("fmul", c::LLVMBuildFMul, &a, &b)
    }

    fn fdiv(&self, a: Value, b: Value) -> PyResult<Value> {
        self.float_binop("fdiv", c::LLVMBuildFDiv, &a, &b)
    }

    fn frem(&self, a: Value, b: Value) -> PyResult<Value> {
        self.float_binop("frem", c::LLVMBuildFRem, &a, &b)
    }

    fn fneg(&self, a: Value) -> PyResult<Value> {
        self.open()?;
        if !a.ty.is_float() {
            return err(format!("fneg of {}", a.ty));
        }
        // SAFETY: a float value of this function.
        let r = unsafe { c::LLVMBuildFNeg(self.b, self.v(&a)?, NONAME) };
        self.fast(r);
        Ok(self.mk(r, a.ty))
    }

    // ------------------------------------------------------------ comparisons

    fn icmp(&self, pred: ICmp, a: Value, b: Value) -> PyResult<Value> {
        self.open()?;
        self.same_type("icmp", &a, &b)?;
        if !(a.ty.is_int() || a.ty == Ty::Ptr) {
            return err(format!("icmp of {}", a.ty));
        }
        let p = match pred {
            ICmp::EQ => IP::LLVMIntEQ,
            ICmp::NE => IP::LLVMIntNE,
            ICmp::SLT => IP::LLVMIntSLT,
            ICmp::SLE => IP::LLVMIntSLE,
            ICmp::SGT => IP::LLVMIntSGT,
            ICmp::SGE => IP::LLVMIntSGE,
            ICmp::ULT => IP::LLVMIntULT,
            ICmp::ULE => IP::LLVMIntULE,
            ICmp::UGT => IP::LLVMIntUGT,
            ICmp::UGE => IP::LLVMIntUGE,
        };
        // SAFETY: two integer (or pointer) values of the same type.
        let r = unsafe { c::LLVMBuildICmp(self.b, p, self.v(&a)?, self.v(&b)?, NONAME) };
        Ok(self.mk(r, Ty::Int(1)))
    }

    fn fcmp(&self, pred: FCmp, a: Value, b: Value) -> PyResult<Value> {
        self.open()?;
        self.same_type("fcmp", &a, &b)?;
        if !a.ty.is_float() {
            return err(format!("fcmp of {}", a.ty));
        }
        let p = match pred {
            FCmp::OEQ => RP::LLVMRealOEQ,
            FCmp::ONE => RP::LLVMRealONE,
            FCmp::OLT => RP::LLVMRealOLT,
            FCmp::OLE => RP::LLVMRealOLE,
            FCmp::OGT => RP::LLVMRealOGT,
            FCmp::OGE => RP::LLVMRealOGE,
            FCmp::UEQ => RP::LLVMRealUEQ,
            FCmp::UNE => RP::LLVMRealUNE,
            FCmp::ULT => RP::LLVMRealULT,
            FCmp::ULE => RP::LLVMRealULE,
            FCmp::UGT => RP::LLVMRealUGT,
            FCmp::UGE => RP::LLVMRealUGE,
            FCmp::ORD => RP::LLVMRealORD,
            FCmp::UNO => RP::LLVMRealUNO,
        };
        // SAFETY: two float values of the same type.
        let r = unsafe { c::LLVMBuildFCmp(self.b, p, self.v(&a)?, self.v(&b)?, NONAME) };
        Ok(self.mk(r, Ty::Int(1)))
    }

    fn select(&self, cond: Value, a: Value, b: Value) -> PyResult<Value> {
        self.open()?;
        if !cond.ty.is_bool() {
            return err(format!("select condition of type {}", cond.ty));
        }
        self.same_type("select", &a, &b)?;
        // SAFETY: an i1 condition and two values of the same type.
        let r = unsafe { c::LLVMBuildSelect(self.b, self.v(&cond)?, self.v(&a)?, self.v(&b)?, NONAME) };
        Ok(self.mk(r, a.ty))
    }

    // ------------------------------------------------------------ casts

    fn trunc(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.int_cast("trunc", c::LLVMBuildTrunc, &v, &to.0, false)
    }

    fn zext(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.int_cast("zext", c::LLVMBuildZExt, &v, &to.0, true)
    }

    fn sext(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.int_cast("sext", c::LLVMBuildSExt, &v, &to.0, true)
    }

    fn fptrunc(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        match (v.ty.float_bits(), to.0.float_bits()) {
            (Some(a), Some(b)) if b < a => self.cast("fptrunc", c::LLVMBuildFPTrunc, &v, &to.0),
            _ => err(format!("fptrunc from {} to {}", v.ty, to.0)),
        }
    }

    fn fpext(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        match (v.ty.float_bits(), to.0.float_bits()) {
            (Some(a), Some(b)) if b > a => self.cast("fpext", c::LLVMBuildFPExt, &v, &to.0),
            _ => err(format!("fpext from {} to {}", v.ty, to.0)),
        }
    }

    /// Float to signed integer. Out-of-range values give poison: clamp first if needed.
    fn fptosi(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        if !v.ty.is_float() || !to.0.is_int() {
            return err(format!("fptosi from {} to {}", v.ty, to.0));
        }
        self.cast("fptosi", c::LLVMBuildFPToSI, &v, &to.0)
    }

    fn fptoui(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        if !v.ty.is_float() || !to.0.is_int() {
            return err(format!("fptoui from {} to {}", v.ty, to.0));
        }
        self.cast("fptoui", c::LLVMBuildFPToUI, &v, &to.0)
    }

    fn sitofp(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        if !v.ty.is_int() || !to.0.is_float() {
            return err(format!("sitofp from {} to {}", v.ty, to.0));
        }
        self.cast("sitofp", c::LLVMBuildSIToFP, &v, &to.0)
    }

    fn uitofp(&self, v: Value, to: PyType) -> PyResult<Value> {
        self.open()?;
        if !v.ty.is_int() || !to.0.is_float() {
            return err(format!("uitofp from {} to {}", v.ty, to.0));
        }
        self.cast("uitofp", c::LLVMBuildUIToFP, &v, &to.0)
    }

    // ------------------------------------------------------------ memory

    /// Load a value of type `ty`. `align` in bytes (default: the type's ABI alignment).
    #[pyo3(signature = (ty, ptr, align = None))]
    fn load(&self, ty: PyType, ptr: Value, align: Option<u32>) -> PyResult<Value> {
        self.open()?;
        if ptr.ty != Ty::Ptr || !ty.0.is_first_class() {
            return err(format!("load of {} through {}", ty.0, ptr.ty));
        }
        // SAFETY: a load of a first-class type through a pointer value.
        let r = unsafe { c::LLVMBuildLoad2(self.b, self.lt(&ty.0), self.v(&ptr)?, NONAME) };
        if let Some(a) = align {
            // SAFETY: setting the alignment of the load just built.
            unsafe { c::LLVMSetAlignment(r, a) };
        }
        Ok(self.mk(r, ty.0))
    }

    #[pyo3(signature = (value, ptr, align = None))]
    fn store(&self, value: Value, ptr: Value, align: Option<u32>) -> PyResult<()> {
        self.open()?;
        if ptr.ty != Ty::Ptr {
            return err(format!("store through {}", ptr.ty));
        }
        // SAFETY: a store of a first-class value through a pointer.
        let r = unsafe { c::LLVMBuildStore(self.b, self.v(&value)?, self.v(&ptr)?) };
        if let Some(a) = align {
            // SAFETY: setting the alignment of the store just built.
            unsafe { c::LLVMSetAlignment(r, a) };
        }
        Ok(())
    }

    /// Address of element `index` of an array of `elem` starting at `ptr` (inbounds).
    fn gep(&self, elem: PyType, ptr: Value, index: Value) -> PyResult<Value> {
        self.open()?;
        if ptr.ty != Ty::Ptr || !index.ty.is_int() || !elem.0.is_first_class() {
            return err(format!("gep of {} with {} index", ptr.ty, index.ty));
        }
        let mut idx = [self.v(&index)?];
        // SAFETY: an inbounds GEP over a first-class element type.
        let r = unsafe { c::LLVMBuildInBoundsGEP2(self.b, self.lt(&elem.0), self.v(&ptr)?, idx.as_mut_ptr(), 1, NONAME) };
        Ok(self.mk(r, Ty::Ptr))
    }

    /// `ptr + offset` bytes (inbounds).
    fn gep_bytes(&self, ptr: Value, offset: Value) -> PyResult<Value> {
        self.gep(PyType(Ty::Int(8)), ptr, offset)
    }

    /// Address of field `index` of a struct of type `ty` at `ptr`.
    fn field_ptr(&self, ty: PyType, ptr: Value, index: u32) -> PyResult<Value> {
        self.open()?;
        if ptr.ty != Ty::Ptr || !matches!(ty.0, Ty::Struct(_)) || ty.0.member(index).is_none() {
            return err(format!("field {index} of {} through {}", ty.0, ptr.ty));
        }
        // SAFETY: a struct GEP with a valid field index.
        let r = unsafe { c::LLVMBuildStructGEP2(self.b, self.lt(&ty.0), self.v(&ptr)?, index, NONAME) };
        Ok(self.mk(r, Ty::Ptr))
    }

    /// Stack memory for `count` values of `ty`, in the entry block (64-byte aligned).
    #[pyo3(signature = (ty, count = 1))]
    fn alloca(&self, ty: PyType, count: u32) -> PyResult<Value> {
        self.m.check_open()?;
        if !ty.0.is_first_class() || count == 0 {
            return err(format!("alloca of {count} x {}", ty.0));
        }
        // SAFETY: the alloca builder sits before the entry block's terminator.
        let r = unsafe {
            let r = if count == 1 {
                c::LLVMBuildAlloca(self.allocas, self.lt(&ty.0), NONAME)
            } else {
                let n = c::LLVMConstInt(self.lt(&Ty::Int(64)), count as u64, 0);
                c::LLVMBuildArrayAlloca(self.allocas, self.lt(&ty.0), n, NONAME)
            };
            c::LLVMSetAlignment(r, 64);
            r
        };
        Ok(self.mk(r, Ty::Ptr))
    }

    // ------------------------------------------------------------ aggregates

    fn extract(&self, agg: Value, index: u32) -> PyResult<Value> {
        self.open()?;
        let t = agg.ty.member(index).cloned().ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no member {index} in {}", agg.ty)))?;
        // SAFETY: extracting an existing member.
        let r = unsafe { c::LLVMBuildExtractValue(self.b, self.v(&agg)?, index, NONAME) };
        Ok(self.mk(r, t))
    }

    fn insert(&self, agg: Value, value: Value, index: u32) -> PyResult<Value> {
        self.open()?;
        let t = agg.ty.member(index).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no member {index} in {}", agg.ty)))?;
        if *t != value.ty {
            return err(format!("member {index} of {} has type {t}, got {}", agg.ty, value.ty));
        }
        // SAFETY: inserting a value of the member's type.
        let r = unsafe { c::LLVMBuildInsertValue(self.b, self.v(&agg)?, self.v(&value)?, index, NONAME) };
        Ok(self.mk(r, agg.ty))
    }

    /// A struct or array value from its members.
    fn aggregate(&self, ty: PyType, members: Vec<Value>) -> PyResult<Value> {
        let n = match &ty.0 {
            Ty::Struct(f) => f.len(),
            Ty::Array(_, n) => *n as usize,
            t => return err(format!("aggregate of {t}")),
        };
        if members.len() != n {
            return err(format!("{} has {n} members, got {}", ty.0, members.len()));
        }
        let mut v = self.undef(ty)?;
        for (k, m) in members.into_iter().enumerate() {
            v = self.insert(v, m, k as u32)?;
        }
        Ok(v)
    }

    // ------------------------------------------------------------ calls

    /// Call a function of this module.
    fn call(&self, f: PyRef<'_, super::module::Function>, args: Vec<Value>) -> PyResult<Option<Value>> {
        self.open()?;
        if !Rc::ptr_eq(&f.m, &self.m) {
            return err("function from a different module");
        }
        f.signature_matches(&args)?;
        let mut raw: Vec<LLVMValueRef> = args.iter().map(|a| self.v(a)).collect::<PyResult<_>>()?;
        Ok(self.call_raw(f.fnty, f.raw, &mut raw, &f.ret))
    }

    /// A float intrinsic, overloaded on the operands' type.
    fn intrinsic(&self, which: Intrinsic, args: Vec<Value>) -> PyResult<Value> {
        self.open()?;
        if args.len() != which.arity() {
            return err(format!("{} takes {} arguments", which.name(), which.arity()));
        }
        let t = args[0].ty.clone();
        if !t.is_float() || args.iter().any(|a| a.ty != t) {
            return err(format!("{} needs float arguments of one type", which.name()));
        }
        let name = which.name();
        // SAFETY: looking up a known intrinsic and declaring it for a float type.
        let (f, fnty) = unsafe {
            let id = c::LLVMLookupIntrinsicID(name.as_ptr() as *const _, name.len());
            let mut tys = [self.lt(&t)];
            let f = c::LLVMGetIntrinsicDeclaration(self.m.module, id, tys.as_mut_ptr(), 1);
            (f, c::LLVMGlobalGetValueType(f))
        };
        let mut raw: Vec<LLVMValueRef> = args.iter().map(|a| self.v(a)).collect::<PyResult<_>>()?;
        let r = self.call_raw(fnty, f, &mut raw, &t).unwrap();
        self.fast(r.raw);
        Ok(r)
    }

    /// A libm function (`f`-suffixed for float). Advertises libmvec SIMD variants so the
    /// loop vectorizer can use them.
    /// Draw from the call context's random state: `b.rng(ir.Rng.Normal, ctx)`,
    /// `b.rng(ir.Rng.Gamma, ctx, [shape])`, ... Returns the value (None for the fills).
    #[pyo3(signature = (which, ctx, args = vec![]))]
    fn rng(&self, which: Rng, ctx: Value, args: Vec<Value>) -> PyResult<Option<Value>> {
        self.open()?;
        let (name, params, ret) = which.signature();
        if ctx.ty != Ty::Ptr || args.len() != params.len() || args.iter().zip(&params).any(|(a, t)| a.ty != *t) {
            let want: Vec<String> = params.iter().map(|t| t.to_string()).collect();
            return err(format!("{name}(ctx: ptr, {})", want.join(", ")));
        }
        let mut all = vec![Ty::Ptr];
        all.extend(params);
        let (f, fnty) = self.declare(name, &all, &ret, false);
        let mut raw = vec![self.v(&ctx)?];
        for a in &args {
            raw.push(self.v(a)?);
        }
        Ok(self.call_raw(fnty, f, &mut raw, &ret))
    }

    fn libm(&self, which: Libm, args: Vec<Value>) -> PyResult<Value> {
        self.open()?;
        if args.len() != which.arity() {
            return err(format!("{} takes {} arguments", which.c_name(), which.arity()));
        }
        let t = args[0].ty.clone();
        if !t.is_float() || args.iter().any(|a| a.ty != t) {
            return err(format!("{} needs float arguments of one type", which.c_name()));
        }
        let f32 = t == Ty::F32;
        let name = if f32 { format!("{}f", which.c_name()) } else { which.c_name().to_string() };
        let params = vec![t.clone(); args.len()];
        let (f, fnty) = self.declare(&name, &params, &t, true);
        let mut raw: Vec<LLVMValueRef> = args.iter().map(|a| self.v(a)).collect::<PyResult<_>>()?;
        let r = self.call_raw(fnty, f, &mut raw, &t).unwrap();
        self.fast(r.raw);
        let variants = super::module::vector_variants();
        if which.has_vector_variant() && !variants.is_empty() {
            let vparams = "v".repeat(args.len());
            let mut names = Vec::new();
            for (isa, lanes64) in variants {
                let lanes = if f32 { lanes64 * 2 } else { lanes64 };
                let vname = format!("_ZGV{isa}N{lanes}{vparams}_{name}");
                names.push(format!("{vname}({vname})"));
                let n = cstring(&vname);
                // SAFETY: declaring the vector variant in this module (once).
                unsafe {
                    if c::LLVMGetNamedFunction(self.m.module, n.as_ptr()).is_null() {
                        let vt = c::LLVMVectorType(self.lt(&t), lanes);
                        let mut ps = vec![vt; args.len()];
                        let fty = c::LLVMFunctionType(vt, ps.as_mut_ptr(), ps.len() as u32, 0);
                        let vf = c::LLVMAddFunction(self.m.module, n.as_ptr(), fty);
                        self.m.vector_decls.borrow_mut().push(vf);
                    }
                }
            }
            let k = "vector-function-abi-variant";
            let v = names.join(",");
            // SAFETY: adding a string attribute to the call just built.
            unsafe {
                let a = c::LLVMCreateStringAttribute(self.m.ctx, k.as_ptr() as *const _, k.len() as u32, v.as_ptr() as *const _, v.len() as u32);
                c::LLVMAddCallSiteAttribute(r.raw, llvm_sys::LLVMAttributeFunctionIndex, a);
            }
        }
        Ok(r)
    }

    /// A scalar function precompiled in Rust.
    fn rust_math(&self, which: RustMath, x: Value) -> PyResult<Value> {
        self.open()?;
        if !x.ty.is_float() {
            return err(format!("rust_math of {}", x.ty));
        }
        let t = x.ty.clone();
        let (f, fnty) = self.declare(which.symbol(t == Ty::F32), std::slice::from_ref(&t), &t, true);
        let mut raw = [self.v(&x)?];
        let r = self.call_raw(fnty, f, &mut raw, &t).unwrap();
        self.fast(r.raw);
        Ok(r)
    }

    // ------------------------------------------------------------ runtime

    /// Report an error on `ctx` (the function should then return nonzero).
    fn rt_fail(&self, ctx: Value, message: &str) -> PyResult<()> {
        self.open()?;
        if ctx.ty != Ty::Ptr {
            return err("ctx must be a ptr");
        }
        let msg = self.cstr(message)?;
        // SAFETY: an integer constant of this module's context.
        let code = self.mk(unsafe { c::LLVMConstInt(self.lt(&Ty::Int(32)), 1, 0) }, Ty::Int(32));
        let (f, fnty) = self.declare("ptl_rt_set_error", &[Ty::Ptr, Ty::Int(32), Ty::Ptr], &Ty::Void, false);
        let mut raw = [self.v(&ctx)?, self.v(&code)?, self.v(&msg)?];
        self.call_raw(fnty, f, &mut raw, &Ty::Void);
        Ok(())
    }

    /// A new buffer of `nbytes` (i64), 64-byte aligned. Writes the owning reference to
    /// `owner_slot` and returns the data pointer (null on failure, with the error set).
    fn rt_alloc(&self, ctx: Value, nbytes: Value, owner_slot: Value) -> PyResult<Value> {
        self.open()?;
        if ctx.ty != Ty::Ptr || nbytes.ty != Ty::Int(64) || owner_slot.ty != Ty::Ptr {
            return err("rt_alloc(ctx: ptr, nbytes: i64, owner_slot: ptr)");
        }
        // SAFETY: an integer constant of this module's context.
        let align = self.mk(unsafe { c::LLVMConstInt(self.lt(&Ty::Int(64)), 64, 0) }, Ty::Int(64));
        let (f, fnty) = self.declare("ptl_rt_alloc", &[Ty::Ptr, Ty::Int(64), Ty::Int(64), Ty::Ptr], &Ty::Ptr, false);
        let mut raw = [self.v(&ctx)?, self.v(&nbytes)?, self.v(&align)?, self.v(&owner_slot)?];
        Ok(self.call_raw(fnty, f, &mut raw, &Ty::Ptr).unwrap())
    }

    /// Scratch memory: `nbytes` (i64), 64-byte aligned, valid until the step ends or an
    /// earlier mark is released. Null on failure, with the error set.
    fn rt_scratch(&self, ctx: Value, nbytes: Value) -> PyResult<Value> {
        self.open()?;
        if ctx.ty != Ty::Ptr || nbytes.ty != Ty::Int(64) {
            return err("rt_scratch(ctx: ptr, nbytes: i64)");
        }
        let (f, fnty) = self.declare("ptl_rt_scratch", &[Ty::Ptr, Ty::Int(64)], &Ty::Ptr, false);
        let mut raw = [self.v(&ctx)?, self.v(&nbytes)?];
        Ok(self.call_raw(fnty, f, &mut raw, &Ty::Ptr).unwrap())
    }

    /// Mark the scratch position (an i64 token for `rt_scratch_release`).
    fn rt_scratch_mark(&self, ctx: Value) -> PyResult<Value> {
        self.open()?;
        if ctx.ty != Ty::Ptr {
            return err("ctx must be a ptr");
        }
        let (f, fnty) = self.declare("ptl_rt_scratch_mark", &[Ty::Ptr], &Ty::Int(64), false);
        let mut raw = [self.v(&ctx)?];
        Ok(self.call_raw(fnty, f, &mut raw, &Ty::Int(64)).unwrap())
    }

    /// Release the scratch handed out since `mark`.
    fn rt_scratch_release(&self, ctx: Value, mark: Value) -> PyResult<()> {
        self.open()?;
        if ctx.ty != Ty::Ptr || mark.ty != Ty::Int(64) {
            return err("rt_scratch_release(ctx: ptr, mark: i64)");
        }
        let (f, fnty) = self.declare("ptl_rt_scratch_release", &[Ty::Ptr, Ty::Int(64)], &Ty::Void, false);
        let mut raw = [self.v(&ctx)?, self.v(&mark)?];
        self.call_raw(fnty, f, &mut raw, &Ty::Void);
        Ok(())
    }

    // ------------------------------------------------------------ control flow

    /// Return from the function. Ends the current block.
    #[pyo3(signature = (value = None))]
    fn ret(&self, value: Option<Value>) -> PyResult<()> {
        self.open()?;
        match (value, &self.ret) {
            (None, Ty::Void) => {
                // SAFETY: the function returns void.
                unsafe { c::LLVMBuildRetVoid(self.b) };
            }
            (Some(v), t) if &v.ty == t => {
                // SAFETY: the value has the function's return type.
                unsafe { c::LLVMBuildRet(self.b, self.v(&v)?) };
            }
            (v, t) => return err(format!("ret of {} in a function returning {t}", v.map_or("nothing".to_string(), |v| v.ty.to_string()))),
        }
        Ok(())
    }

    /// `with b.loop(n) as i:` runs the body for i = 0 .. n-1 (i64).
    #[pyo3(name = "loop")]
    fn loop_(slf: Py<Self>, py: Python<'_>, n: Value) -> PyResult<Loop> {
        {
            let b = slf.borrow(py);
            b.open()?;
            b.v(&n)?;
            if n.ty != Ty::Int(64) {
                return err(format!("loop count of type {}", n.ty));
            }
        }
        Ok(Loop { b: slf, n, state: None })
    }

    /// `with b.if_(cond):` runs the body if `cond` (i1) is true.
    #[pyo3(name = "if_")]
    fn if_(slf: Py<Self>, py: Python<'_>, cond: Value) -> PyResult<If> {
        {
            let b = slf.borrow(py);
            b.open()?;
            b.v(&cond)?;
            if !cond.ty.is_bool() {
                return err(format!("if condition of type {}", cond.ty));
            }
        }
        Ok(If { b: slf, cond, done: None })
    }

    /// Loop with carried values: `acc = init; for i in 0..n: acc = body(i, *acc)`;
    /// returns the final values. `body` returns a list of values of the init types.
    ///
    /// With `until=True` the body returns one more value, an i1 stop flag; the loop
    /// ends after the first step that sets it, and the flag is returned last (false
    /// when no step ran).
    #[pyo3(signature = (n, init, body, until=false))]
    fn fold(&self, py: Python<'_>, n: Value, init: Vec<Value>, body: Py<PyAny>, until: bool) -> PyResult<Vec<Value>> {
        self.open()?;
        if n.ty != Ty::Int(64) {
            return err(format!("fold count of type {}", n.ty));
        }
        let nr = self.v(&n)?;
        let tys: Vec<Ty> = init.iter().map(|v| v.ty.clone()).collect();
        let init_raw: Vec<LLVMValueRef> = init.iter().map(|v| self.v(v)).collect::<PyResult<_>>()?;
        let pre = self.block();
        let header = self.new_block(c"fold");
        let exit = self.new_block(c"fold.exit");
        let i64t = Ty::Int(64);
        // SAFETY: integer constants and a compare of i64 values.
        let (zero, one, nz) = unsafe {
            let zero = c::LLVMConstInt(self.lt(&i64t), 0, 0);
            let one = c::LLVMConstInt(self.lt(&i64t), 1, 0);
            (zero, one, c::LLVMBuildICmp(self.b, IP::LLVMIntSGT, nr, zero, NONAME))
        };
        self.cond_br(nz, header, exit);
        self.position(header);
        let iphi = self.phi(&i64t, &[(zero, pre)]);
        let phis: Vec<LLVMValueRef> = tys.iter().zip(&init_raw).map(|(t, &v)| self.phi(t, &[(v, pre)])).collect();
        let i = self.mk(iphi, i64t.clone());
        let mut args = vec![i.into_pyobject(py)?.into_any()];
        for (p, t) in phis.iter().zip(&tys) {
            args.push(self.mk(*p, t.clone()).into_pyobject(py)?.into_any());
        }
        *self.depth.borrow_mut() += 1;
        let res = body.call1(py, PyTuple::new(py, args)?);
        *self.depth.borrow_mut() -= 1;
        let mut new: Vec<Value> = res?.extract(py)?;
        let stop = if until {
            match new.pop() {
                Some(f) if f.ty.is_bool() => Some(self.v(&f)?),
                Some(f) => return err(format!("fold stop flag of type {}", f.ty)),
                None => return err("fold body with until must return the stop flag last"),
            }
        } else {
            None
        };
        self.check_values("fold body result", &new, &tys)?;
        if self.is_terminated() {
            return err("fold body must not end its block");
        }
        let latch = self.block();
        // SAFETY: i64 arithmetic and compares in the open latch block.
        let (next, more) = unsafe {
            let next = c::LLVMBuildNSWAdd(self.b, iphi, one, NONAME);
            let mut more = c::LLVMBuildICmp(self.b, IP::LLVMIntSLT, next, nr, NONAME);
            if let Some(f) = stop {
                more = c::LLVMBuildAnd(self.b, more, c::LLVMBuildNot(self.b, f, NONAME), NONAME);
            }
            (next, more)
        };
        self.cond_br(more, header, exit);
        // SAFETY: completing the phis with the latch's values.
        unsafe {
            let (mut nv, mut lb) = (next, latch);
            c::LLVMAddIncoming(iphi, &mut nv, &mut lb, 1);
            for (p, v) in phis.iter().zip(&new) {
                let (mut vv, mut lb) = (v.raw, latch);
                c::LLVMAddIncoming(*p, &mut vv, &mut lb, 1);
            }
        }
        self.position(exit);
        let mut out = Vec::new();
        for ((t, &iv), v) in tys.iter().zip(&init_raw).zip(&new) {
            out.push(self.mk(self.phi(t, &[(iv, pre), (v.raw, latch)]), t.clone()));
        }
        if let Some(f) = stop {
            let bt = Ty::Int(1);
            // SAFETY: an i1 constant.
            let no = unsafe { c::LLVMConstInt(self.lt(&bt), 0, 0) };
            out.push(self.mk(self.phi(&bt, &[(no, pre), (f, latch)]), bt));
        }
        Ok(out)
    }

    /// `then()` if `cond` else `else_()`; both return lists of values of the same types,
    /// which are merged and returned.
    fn if_else(&self, py: Python<'_>, cond: Value, then: Py<PyAny>, else_: Py<PyAny>) -> PyResult<Vec<Value>> {
        self.open()?;
        if !cond.ty.is_bool() {
            return err(format!("if_else condition of type {}", cond.ty));
        }
        let cr = self.v(&cond)?;
        let tb = self.new_block(c"then");
        let eb = self.new_block(c"else");
        let done = self.new_block(c"endif");
        self.cond_br(cr, tb, eb);
        let arm = |bb: LLVMBasicBlockRef, f: &Py<PyAny>| -> PyResult<(Vec<Value>, LLVMBasicBlockRef)> {
            self.position(bb);
            *self.depth.borrow_mut() += 1;
            let r = f.call0(py);
            *self.depth.borrow_mut() -= 1;
            let vals: Vec<Value> = r?.extract(py)?;
            for v in &vals {
                self.v(v)?;
            }
            if self.is_terminated() {
                return err("if_else branches must not end their block");
            }
            let end = self.block();
            self.br(done);
            Ok((vals, end))
        };
        let (tv, tend) = arm(tb, &then)?;
        let (ev, eend) = arm(eb, &else_)?;
        let tys: Vec<Ty> = tv.iter().map(|v| v.ty.clone()).collect();
        self.check_values("else branch result", &ev, &tys)?;
        self.position(done);
        Ok(tv
            .iter()
            .zip(&ev)
            .map(|(a, b)| self.mk(self.phi(&a.ty, &[(a.raw, tend), (b.raw, eend)]), a.ty.clone()))
            .collect())
    }
}

/// `with b.loop(n) as i:`
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Loop {
    b: Py<Builder>,
    n: Value,
    /// (phi of i, header block, exit block) once entered
    state: Option<(LLVMValueRef, LLVMBasicBlockRef, LLVMBasicBlockRef)>,
}

#[pymethods]
impl Loop {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Value> {
        let b = self.b.borrow(py);
        if self.state.is_some() {
            return err("a loop can be entered once");
        }
        b.open()?;
        let pre = b.block();
        let header = b.new_block(c"loop");
        let exit = b.new_block(c"loop.exit");
        let nr = b.v(&self.n)?;
        // SAFETY: i64 constant and compare in the open block.
        let (zero, nz) = unsafe {
            let zero = c::LLVMConstInt(b.lt(&Ty::Int(64)), 0, 0);
            (zero, c::LLVMBuildICmp(b.b, IP::LLVMIntSGT, nr, zero, NONAME))
        };
        b.cond_br(nz, header, exit);
        b.position(header);
        let iphi = b.phi(&Ty::Int(64), &[(zero, pre)]);
        self.state = Some((iphi, header, exit));
        *b.depth.borrow_mut() += 1;
        Ok(b.mk(iphi, Ty::Int(64)))
    }

    fn __exit__(&mut self, py: Python<'_>, exc_type: Option<Py<PyAny>>, _v: Option<Py<PyAny>>, _tb: Option<Py<PyAny>>) -> PyResult<bool> {
        let b = self.b.borrow(py);
        *b.depth.borrow_mut() -= 1;
        let Some((iphi, _header, exit)) = self.state.take() else {
            return Ok(false);
        };
        if exc_type.is_some() {
            return Ok(false); // propagate; the module is abandoned
        }
        if !b.is_terminated() {
            let latch = b.block();
            let nr = b.v(&self.n)?;
            // SAFETY: i64 arithmetic and compare in the open latch block; completing the
            // phi with the latch's value.
            unsafe {
                let one = c::LLVMConstInt(b.lt(&Ty::Int(64)), 1, 0);
                let next = c::LLVMBuildNSWAdd(b.b, iphi, one, NONAME);
                let more = c::LLVMBuildICmp(b.b, IP::LLVMIntSLT, next, nr, NONAME);
                b.cond_br(more, _header, exit);
                let (mut nv, mut lb) = (next, latch);
                c::LLVMAddIncoming(iphi, &mut nv, &mut lb, 1);
            }
        }
        b.position(exit);
        Ok(false)
    }
}

/// `with b.if_(cond):`
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct If {
    b: Py<Builder>,
    cond: Value,
    done: Option<LLVMBasicBlockRef>,
}

#[pymethods]
impl If {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        let b = self.b.borrow(py);
        if self.done.is_some() {
            return err("an if can be entered once");
        }
        b.open()?;
        let then = b.new_block(c"then");
        let done = b.new_block(c"endif");
        b.cond_br(b.v(&self.cond)?, then, done);
        b.position(then);
        self.done = Some(done);
        *b.depth.borrow_mut() += 1;
        Ok(())
    }

    fn __exit__(&mut self, py: Python<'_>, exc_type: Option<Py<PyAny>>, _v: Option<Py<PyAny>>, _tb: Option<Py<PyAny>>) -> PyResult<bool> {
        let b = self.b.borrow(py);
        *b.depth.borrow_mut() -= 1;
        let Some(done) = self.done.take() else {
            return Ok(false);
        };
        if exc_type.is_some() {
            return Ok(false);
        }
        b.br(done);
        b.position(done);
        Ok(false)
    }
}
