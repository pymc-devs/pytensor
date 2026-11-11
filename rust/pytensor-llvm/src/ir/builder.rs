//! The instruction builder.
//!
//! Every method checks its operands before calling LLVM: values must come from this
//! module and function, have the right types, and the current block must be open.
//! LLVM's own checks are assertions, which release builds don't have.
//!
//! Operands may be Python literals (`b.add(i, 1)`, `b.select(c, x, 0.0)`): a literal
//! takes the type of the other operand, or the type the method asks for (`b.loop(4)`).
//!
//! Integer arithmetic simplifies what it can see: constants fold (LLVM's builder does
//! that), and so do `x + 0`, `x * 1`, `x * 0`, `x & 0`, a select on a constant
//! condition, ... The result of such a call may be one of its operands or a constant
//! (`value.const_int`), which is how emitters keep static shapes static.

use super::module::{add_fn_attr, cstring, err, ModuleInner, Value};
use super::ty::{PyType, Ty};
use llvm_sys::core as c;
use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMMetadataRef, LLVMTypeRef, LLVMValueRef};
use llvm_sys::{LLVMIntPredicate as IP, LLVMRealPredicate as RP};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

type BuildFn = unsafe extern "C" fn(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, *const std::ffi::c_char) -> LLVMValueRef;
type CastFn = unsafe extern "C" fn(LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, *const std::ffi::c_char) -> LLVMValueRef;

const NONAME: *const std::ffi::c_char = c"".as_ptr();

/// An operand: a value, or a Python literal that becomes a constant of the type the
/// operation needs (the other operand's type, or the one the method asks for).
#[derive(FromPyObject)]
pub enum Operand {
    Value(Value),
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl Operand {
    fn ty(&self) -> Option<&Ty> {
        match self {
            Operand::Value(v) => Some(&v.ty),
            _ => None,
        }
    }
}

/// The values a body returned: None (no values), one value, or a sequence of values.
fn values_of(py: Python<'_>, r: Py<PyAny>) -> PyResult<Vec<Value>> {
    let r = r.bind(py);
    if r.is_none() {
        return Ok(vec![]);
    }
    if let Ok(v) = r.extract::<Value>() {
        return Ok(vec![v]);
    }
    r.extract()
}

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
    /// positioned before the entry block's terminator of the current function: allocas
    /// go there
    allocas: Cell<LLVMBuilderRef>,
    /// the function code goes into: the builder's own, or the body of the innermost open
    /// parallel region (`b.parallel`)
    func: Cell<LLVMValueRef>,
    ret: RefCell<Ty>,
    /// blocks of open `with` contexts, innermost last (for error messages)
    depth: RefCell<usize>,
    /// open parallel regions, innermost last
    regions: RefCell<Vec<Region>>,
    /// alias information for this function's array accesses (`alias_domain`)
    alias: RefCell<Option<AliasInfo>>,
    /// the `if` that just ended: its conditional branch and the block after it
    /// (`else_` may follow while nothing was added there)
    last_if: Cell<Option<(LLVMValueRef, LLVMBasicBlockRef)>>,
}

/// Alias information of a node function (DESIGN.md, "Aliasing"): pointers into arrays
/// (`mark_array`, and everything derived from them by GEPs or captured into parallel
/// bodies) carry an alias scope, the scopes they don't alias, and type-based (TBAA)
/// tags per accessed type. Domain and TBAA root are per function: claims never cross
/// node functions, which may share memory (inlined into one unit) in other dtypes.
struct AliasInfo {
    domain: LLVMMetadataRef,
    tbaa_root: LLVMMetadataRef,
    prefix: String,
    scopes: std::collections::HashMap<String, LLVMMetadataRef>,
    tbaa: std::collections::HashMap<Ty, LLVMValueRef>,
    /// raw pointer -> (alias.scope list, noalias list)
    arrays: std::collections::HashMap<usize, (LLVMValueRef, LLVMValueRef)>,
}

/// An open parallel region: its body function, and how to get back to its parent.
struct Region {
    parent_func: LLVMValueRef,
    parent_allocas: LLVMBuilderRef,
    parent_ret: Ty,
    body: LLVMValueRef,
    body_allocas: LLVMBuilderRef,
    /// the body's `captures` parameter (the captured values, 8-byte slots)
    captures_param: LLVMValueRef,
    /// parent value -> the body's (the parent's ctx maps to the body's own)
    map: std::collections::HashMap<usize, LLVMValueRef>,
    /// captured parent values, in slot order
    captures: Vec<(LLVMValueRef, Ty)>,
}

impl Drop for Builder {
    fn drop(&mut self) {
        // SAFETY: we own these builders (regions left open by an error included).
        unsafe {
            c::LLVMDisposeBuilder(self.b);
            c::LLVMDisposeBuilder(self.allocas.get());
            for f in self.regions.borrow().iter() {
                c::LLVMDisposeBuilder(f.parent_allocas);
            }
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
            Ok(Builder {
                m,
                b,
                allocas: Cell::new(allocas),
                func: Cell::new(func),
                ret: RefCell::new(ret),
                depth: RefCell::new(0),
                regions: RefCell::new(vec![]),
                alias: RefCell::new(None),
                last_if: Cell::new(None),
            })
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

    /// The raw value, if it may be used here. Inside a parallel region, values of the
    /// enclosing functions are captured (passed to the body through its `captures`).
    fn v(&self, v: &Value) -> PyResult<LLVMValueRef> {
        if !Rc::ptr_eq(&v.m, &self.m) {
            return err("value from a different module");
        }
        match v.func {
            Some(f) if f != self.func.get() => self.capture(v.raw, &v.ty, f),
            _ => Ok(v.raw),
        }
    }

    /// `raw` (of type `ty`) from function `owner`, an enclosing function of an open
    /// parallel region, as a value of the current function: loaded from the captures slot of
    /// each region in between (in its body's entry block).
    fn capture(&self, raw: LLVMValueRef, ty: &Ty, owner: LLVMValueRef) -> PyResult<LLVMValueRef> {
        let mut regions = self.regions.borrow_mut();
        let Some(first) = regions.iter().position(|f| f.parent_func == owner) else {
            return err("value from a different function");
        };
        if !matches!(ty, Ty::Int(_) | Ty::F32 | Ty::F64 | Ty::Ptr) {
            return err(format!("a parallel region can't use a value of type {ty} from outside"));
        }
        let mut raw = raw;
        for region in regions[first..].iter_mut() {
            if let Some(&r) = region.map.get(&(raw as usize)) {
                raw = r;
                continue;
            }
            let slot = region.captures.len();
            // SAFETY: a GEP and a load in the body's entry block (before its terminator);
            // `captures` holds a slot per capture, filled by the parent before the call.
            let loaded = unsafe {
                let i8t = c::LLVMInt8TypeInContext(self.m.ctx);
                let mut off = c::LLVMConstInt(self.lt(&Ty::Int(64)), 8 * slot as u64, 0);
                let p = c::LLVMBuildInBoundsGEP2(region.body_allocas, i8t, region.captures_param, &mut off, 1, NONAME);
                c::LLVMBuildLoad2(region.body_allocas, self.lt(ty), p, NONAME)
            };
            region.captures.push((raw, ty.clone()));
            region.map.insert(raw as usize, loaded);
            self.derive(raw, loaded);
            raw = loaded;
        }
        Ok(raw)
    }

    fn mk(&self, raw: LLVMValueRef, ty: Ty) -> Value {
        // SAFETY: `raw` is a live value of this module. A folded result is a constant:
        // it belongs to no function (so parallel regions don't capture it).
        let konst = unsafe { !c::LLVMIsAConstantInt(raw).is_null() || !c::LLVMIsAConstantFP(raw).is_null() };
        Value { m: self.m.clone(), raw, ty, func: if konst { None } else { Some(self.func.get()) } }
    }

    /// `x` as a value of type `ty`: itself, or the literal as a constant of `ty`.
    fn lit(&self, what: &str, x: Operand, ty: &Ty) -> PyResult<Value> {
        match x {
            Operand::Value(v) => Ok(v),
            Operand::Bool(v) if ty.is_int() => self.const_int(PyType(ty.clone()), v as i128),
            Operand::Int(v) if ty.is_int() => self.const_int(PyType(ty.clone()), v),
            Operand::Int(v) if ty.is_float() => self.const_float(PyType(ty.clone()), v as f64),
            Operand::Float(v) if ty.is_float() => self.const_float(PyType(ty.clone()), v),
            Operand::Bool(v) => err(format!("{what}: the literal {v} can't be a {ty}")),
            Operand::Int(v) => err(format!("{what}: the literal {v} can't be a {ty}")),
            Operand::Float(v) => err(format!("{what}: the literal {v} can't be a {ty} (convert it explicitly)")),
        }
    }

    /// Two operands of one type: a literal takes the type of the other operand.
    fn pair(&self, what: &str, a: Operand, b: Operand) -> PyResult<(Value, Value)> {
        let Some(ty) = a.ty().or(b.ty()).cloned() else {
            return err(format!("{what}: one operand must be a value (a literal takes the other operand's type)"));
        };
        Ok((self.lit(what, a, &ty)?, self.lit(what, b, &ty)?))
    }

    /// Operands of one type: literals take the type of the first value.
    fn many(&self, what: &str, args: Vec<Operand>) -> PyResult<Vec<Value>> {
        let Some(ty) = args.iter().find_map(|a| a.ty()).cloned() else {
            return err(format!("{what}: one argument must be a value (literals take its type)"));
        };
        args.into_iter().map(|a| self.lit(what, a, &ty)).collect()
    }

    /// The integer constant `v` is (booleans: 0 or 1; others sign-extended).
    fn ci(v: &Value) -> Option<i64> {
        v.const_int_value()
    }

    /// All bits set?
    fn all_ones(v: &Value) -> bool {
        match Self::ci(v) {
            Some(1) if v.ty.is_bool() => true,
            Some(-1) if !v.ty.is_bool() => true,
            _ => false,
        }
    }

    /// Checked operands of an integer operation.
    fn int_pair(&self, what: &str, a: Operand, b: Operand) -> PyResult<(Value, Value)> {
        self.open()?;
        let (a, b) = self.pair(what, a, b)?;
        self.same_type(what, &a, &b)?;
        if !a.ty.is_int() {
            return err(format!("{what} needs integers, got {}", a.ty));
        }
        self.v(&a)?;
        self.v(&b)?;
        Ok((a, b))
    }

    fn int_op(&self, f: BuildFn, a: &Value, b: &Value) -> PyResult<Value> {
        // SAFETY: two integer values of the same type in this function (`int_pair`).
        let r = unsafe { f(self.b, self.v(a)?, self.v(b)?, NONAME) };
        Ok(self.mk(r, a.ty.clone()))
    }

    fn compare(&self, pred: ICmp, a: Operand, b: Operand) -> PyResult<Value> {
        self.open()?;
        let (a, b) = self.pair("icmp", a, b)?;
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

    fn pick(&self, cond: &Value, a: Value, b: Value) -> PyResult<Value> {
        self.same_type("select", &a, &b)?;
        let (cr, ar, br) = (self.v(cond)?, self.v(&a)?, self.v(&b)?);
        match Self::ci(cond) {
            Some(0) => return Ok(b),
            Some(_) => return Ok(a),
            None if a.raw == b.raw => return Ok(a),
            None => {}
        }
        // SAFETY: an i1 condition and two values of the same type.
        let r = unsafe { c::LLVMBuildSelect(self.b, cr, ar, br, NONAME) };
        Ok(self.mk(r, a.ty))
    }

    fn min_max(&self, what: &str, pred: ICmp, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair(what, a, b)?;
        let c = self.compare(pred, Operand::Value(a.clone()), Operand::Value(b.clone()))?;
        self.pick(&c, a, b)
    }

    /// An i64 operand (a count, a size, an index).
    fn i64_of(&self, what: &str, x: Operand) -> PyResult<Value> {
        let v = self.lit(what, x, &Ty::Int(64))?;
        if v.ty != Ty::Int(64) {
            return err(format!("{what} must be i64, got {}", v.ty));
        }
        Ok(v)
    }

    fn konst(&self, raw: LLVMValueRef, ty: Ty) -> Value {
        Value { m: self.m.clone(), raw, ty, func: None }
    }

    fn lt(&self, t: &Ty) -> LLVMTypeRef {
        self.m.llvm_type(t)
    }

    /// `derived` points into the same array as `base` (a GEP of it, or its capture).
    fn derive(&self, base: LLVMValueRef, derived: LLVMValueRef) {
        if let Some(a) = self.alias.borrow_mut().as_mut() {
            if let Some(&tags) = a.arrays.get(&(base as usize)) {
                a.arrays.insert(derived as usize, tags);
            }
        }
    }

    fn md_string(&self, s: &str) -> LLVMMetadataRef {
        // SAFETY: an MDString in the module's context (copied by LLVM).
        unsafe { c::LLVMMDStringInContext2(self.m.ctx, s.as_ptr() as *const _, s.len()) }
    }

    fn md_node(&self, ops: &mut [LLVMMetadataRef]) -> LLVMMetadataRef {
        // SAFETY: a uniqued metadata tuple of valid operands.
        unsafe { c::LLVMMDNodeInContext2(self.m.ctx, ops.as_mut_ptr(), ops.len()) }
    }

    fn md_value(&self, md: LLVMMetadataRef) -> LLVMValueRef {
        // SAFETY: wrapping metadata of the module's context as a value.
        unsafe { c::LLVMMetadataAsValue(self.m.ctx, md) }
    }

    fn md_kind(&self, name: &str) -> u32 {
        // SAFETY: looking up (or registering) a metadata kind by name.
        unsafe { c::LLVMGetMDKindIDInContext(self.m.ctx, name.as_ptr() as *const _, name.len() as u32) }
    }

    /// Scope metadata for scope `name` of the function's domain.
    fn scope(&self, a: &mut AliasInfo, name: &str) -> LLVMMetadataRef {
        if let Some(&s) = a.scopes.get(name) {
            return s;
        }
        let id = self.md_string(&format!("{}.{name}", a.prefix));
        let s = self.md_node(&mut [id, a.domain]);
        a.scopes.insert(name.to_string(), s);
        s
    }

    /// TBAA access tag for a scalar access of type `ty` (None: no tag, e.g. vectors).
    fn tbaa_tag(&self, a: &mut AliasInfo, ty: &Ty) -> Option<LLVMValueRef> {
        if !matches!(ty, Ty::Int(_) | Ty::F32 | Ty::F64 | Ty::Ptr) {
            return None;
        }
        if let Some(&t) = a.tbaa.get(ty) {
            return Some(t);
        }
        // SAFETY: an i64 constant as metadata.
        let zero = unsafe { c::LLVMValueAsMetadata(c::LLVMConstInt(self.lt(&Ty::Int(64)), 0, 0)) };
        let name = self.md_string(&format!("{ty}"));
        let node = self.md_node(&mut [name, a.tbaa_root, zero]);
        let tag = self.md_node(&mut [node, node, zero]);
        let v = self.md_value(tag);
        a.tbaa.insert(ty.clone(), v);
        Some(v)
    }

    /// Alias metadata on load or store `inst` through `ptr` (accessing a `ty`), if `ptr`
    /// points into a marked array.
    fn annotate(&self, inst: LLVMValueRef, ptr: LLVMValueRef, ty: &Ty) {
        let mut guard = self.alias.borrow_mut();
        let Some(a) = guard.as_mut() else { return };
        let Some(&(scope, noalias)) = a.arrays.get(&(ptr as usize)) else { return };
        let tag = self.tbaa_tag(a, ty);
        // SAFETY: attaching metadata of the module's context to an instruction.
        unsafe {
            c::LLVMSetMetadata(inst, self.md_kind("alias.scope"), scope);
            c::LLVMSetMetadata(inst, self.md_kind("noalias"), noalias);
            if let Some(t) = tag {
                c::LLVMSetMetadata(inst, self.md_kind("tbaa"), t);
            }
        }
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

    fn float_binop(&self, what: &str, f: BuildFn, a: Operand, b: Operand) -> PyResult<Value> {
        self.open()?;
        let (a, b) = self.pair(what, a, b)?;
        let (a, b) = (&a, &b);
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
        unsafe { c::LLVMAppendBasicBlockInContext(self.m.ctx, self.func.get(), name.as_ptr()) }
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

    /// An i64 constant.
    fn i64(&self, value: i128) -> PyResult<Value> {
        self.const_int(PyType(Ty::Int(64)), value)
    }

    /// An i32 constant.
    fn i32(&self, value: i128) -> PyResult<Value> {
        self.const_int(PyType(Ty::Int(32)), value)
    }

    /// An i8 constant.
    fn i8(&self, value: i128) -> PyResult<Value> {
        self.const_int(PyType(Ty::Int(8)), value)
    }

    /// A double constant.
    fn f64(&self, value: f64) -> PyResult<Value> {
        self.const_float(PyType(Ty::F64), value)
    }

    /// A float constant.
    fn f32(&self, value: f64) -> PyResult<Value> {
        self.const_float(PyType(Ty::F32), value)
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
    fn add(&self, a: Operand, b: Operand, nsw: bool) -> PyResult<Value> {
        let (a, b) = self.int_pair("add", a, b)?;
        match (Self::ci(&a), Self::ci(&b)) {
            (Some(0), _) => Ok(b),
            (_, Some(0)) => Ok(a),
            _ => self.int_op(if nsw { c::LLVMBuildNSWAdd } else { c::LLVMBuildAdd }, &a, &b),
        }
    }

    #[pyo3(signature = (a, b, nsw = false))]
    fn sub(&self, a: Operand, b: Operand, nsw: bool) -> PyResult<Value> {
        let (a, b) = self.int_pair("sub", a, b)?;
        match Self::ci(&b) {
            Some(0) => Ok(a),
            _ => self.int_op(if nsw { c::LLVMBuildNSWSub } else { c::LLVMBuildSub }, &a, &b),
        }
    }

    #[pyo3(signature = (a, b, nsw = false))]
    fn mul(&self, a: Operand, b: Operand, nsw: bool) -> PyResult<Value> {
        let (a, b) = self.int_pair("mul", a, b)?;
        let (ca, cb) = (Self::ci(&a), Self::ci(&b));
        if ca == Some(0) || cb == Some(1) {
            return Ok(a);
        }
        if cb == Some(0) || ca == Some(1) {
            return Ok(b);
        }
        self.int_op(if nsw { c::LLVMBuildNSWMul } else { c::LLVMBuildMul }, &a, &b)
    }

    /// Signed division. Division by zero (or MIN / -1) is undefined: check first.
    fn sdiv(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("sdiv", a, b)?;
        match Self::ci(&b) {
            Some(1) if !a.ty.is_bool() => Ok(a),
            _ => self.int_op(c::LLVMBuildSDiv, &a, &b),
        }
    }

    fn udiv(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("udiv", a, b)?;
        match Self::ci(&b) {
            Some(1) => Ok(a),
            _ => self.int_op(c::LLVMBuildUDiv, &a, &b),
        }
    }

    fn srem(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("srem", a, b)?;
        self.int_op(c::LLVMBuildSRem, &a, &b)
    }

    fn urem(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("urem", a, b)?;
        self.int_op(c::LLVMBuildURem, &a, &b)
    }

    fn shl(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("shl", a, b)?;
        match Self::ci(&b) {
            Some(0) => Ok(a),
            _ => self.int_op(c::LLVMBuildShl, &a, &b),
        }
    }

    fn lshr(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("lshr", a, b)?;
        match Self::ci(&b) {
            Some(0) => Ok(a),
            _ => self.int_op(c::LLVMBuildLShr, &a, &b),
        }
    }

    fn ashr(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("ashr", a, b)?;
        match Self::ci(&b) {
            Some(0) => Ok(a),
            _ => self.int_op(c::LLVMBuildAShr, &a, &b),
        }
    }

    fn and_(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("and", a, b)?;
        if Self::ci(&a) == Some(0) || Self::all_ones(&b) || a.raw == b.raw {
            return Ok(a);
        }
        if Self::ci(&b) == Some(0) || Self::all_ones(&a) {
            return Ok(b);
        }
        self.int_op(c::LLVMBuildAnd, &a, &b)
    }

    fn or_(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("or", a, b)?;
        if Self::ci(&b) == Some(0) || Self::all_ones(&a) || a.raw == b.raw {
            return Ok(a);
        }
        if Self::ci(&a) == Some(0) || Self::all_ones(&b) {
            return Ok(b);
        }
        self.int_op(c::LLVMBuildOr, &a, &b)
    }

    fn xor(&self, a: Operand, b: Operand) -> PyResult<Value> {
        let (a, b) = self.int_pair("xor", a, b)?;
        match (Self::ci(&a), Self::ci(&b)) {
            (Some(0), _) => Ok(b),
            (_, Some(0)) => Ok(a),
            _ => self.int_op(c::LLVMBuildXor, &a, &b),
        }
    }

    /// The smaller of two signed integers.
    fn smin(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.min_max("smin", ICmp::SLT, a, b)
    }

    /// The larger of two signed integers.
    fn smax(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.min_max("smax", ICmp::SGT, a, b)
    }

    /// The smaller of two unsigned integers.
    fn umin(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.min_max("umin", ICmp::ULT, a, b)
    }

    /// The larger of two unsigned integers.
    fn umax(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.min_max("umax", ICmp::UGT, a, b)
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

    fn fadd(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.float_binop("fadd", c::LLVMBuildFAdd, a, b)
    }

    fn fsub(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.float_binop("fsub", c::LLVMBuildFSub, a, b)
    }

    fn fmul(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.float_binop("fmul", c::LLVMBuildFMul, a, b)
    }

    fn fdiv(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.float_binop("fdiv", c::LLVMBuildFDiv, a, b)
    }

    fn frem(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.float_binop("frem", c::LLVMBuildFRem, a, b)
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

    fn icmp(&self, pred: ICmp, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(pred, a, b)
    }

    /// `a == b` (integers or pointers).
    fn eq(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::EQ, a, b)
    }

    /// `a != b` (integers or pointers).
    fn ne(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::NE, a, b)
    }

    /// `a < b`, signed.
    fn slt(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::SLT, a, b)
    }

    /// `a <= b`, signed.
    fn sle(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::SLE, a, b)
    }

    /// `a > b`, signed.
    fn sgt(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::SGT, a, b)
    }

    /// `a >= b`, signed.
    fn sge(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::SGE, a, b)
    }

    /// `a < b`, unsigned.
    fn ult(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::ULT, a, b)
    }

    /// `a <= b`, unsigned.
    fn ule(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::ULE, a, b)
    }

    /// `a > b`, unsigned.
    fn ugt(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::UGT, a, b)
    }

    /// `a >= b`, unsigned.
    fn uge(&self, a: Operand, b: Operand) -> PyResult<Value> {
        self.compare(ICmp::UGE, a, b)
    }

    fn fcmp(&self, pred: FCmp, a: Operand, b: Operand) -> PyResult<Value> {
        self.open()?;
        let (a, b) = self.pair("fcmp", a, b)?;
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

    /// `a` if `cond` else `b`. A constant condition picks its operand.
    fn select(&self, cond: Value, a: Operand, b: Operand) -> PyResult<Value> {
        self.open()?;
        if !cond.ty.is_bool() {
            return err(format!("select condition of type {}", cond.ty));
        }
        let (a, b) = self.pair("select", a, b)?;
        self.pick(&cond, a, b)
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

    /// Start alias information for this function (DESIGN.md, "Aliasing"): a scope
    /// domain and a TBAA root named after `name` (unique in the module).
    fn alias_domain(&self, name: &str) -> PyResult<()> {
        self.open()?;
        let prefix = format!("pt.{}.{name}", self.m.name);
        let d = self.md_string(&prefix);
        let domain = self.md_node(&mut [d]);
        let r = self.md_string(&format!("{prefix}.tbaa"));
        let tbaa_root = self.md_node(&mut [r]);
        *self.alias.borrow_mut() = Some(AliasInfo {
            domain,
            tbaa_root,
            prefix,
            scopes: Default::default(),
            tbaa: Default::default(),
            arrays: Default::default(),
        });
        Ok(())
    }

    /// `ptr` points into an array in scope `scope` that doesn't overlap the arrays of
    /// scopes `noalias`. Loads and stores through it (and through pointers derived from
    /// it) get scoped-alias and type-based alias metadata. The caller guarantees the
    /// claim (written outputs of a task never overlap its inputs or each other).
    fn mark_array(&self, ptr: Value, scope: &str, noalias: Vec<String>) -> PyResult<()> {
        let p = self.v(&ptr)?;
        let mut guard = self.alias.borrow_mut();
        let Some(a) = guard.as_mut() else {
            return err("mark_array before alias_domain");
        };
        let s = self.scope(a, scope);
        let mut others: Vec<_> = noalias.iter().map(|n| self.scope(a, n)).collect();
        let scope_list = self.md_value(self.md_node(&mut [s]));
        let noalias_list = self.md_value(self.md_node(&mut others));
        a.arrays.insert(p as usize, (scope_list, noalias_list));
        Ok(())
    }

    /// Load a value of type `ty`. `align` in bytes (default: the type's ABI alignment).
    #[pyo3(signature = (ty, ptr, align = None))]
    fn load(&self, ty: PyType, ptr: Value, align: Option<u32>) -> PyResult<Value> {
        self.open()?;
        if ptr.ty != Ty::Ptr || !ty.0.is_first_class() {
            return err(format!("load of {} through {}", ty.0, ptr.ty));
        }
        // SAFETY: a load of a first-class type through a pointer value.
        let p = self.v(&ptr)?;
        let r = unsafe { c::LLVMBuildLoad2(self.b, self.lt(&ty.0), p, NONAME) };
        if let Some(a) = align {
            // SAFETY: setting the alignment of the load just built.
            unsafe { c::LLVMSetAlignment(r, a) };
        }
        self.annotate(r, p, &ty.0);
        Ok(self.mk(r, ty.0))
    }

    #[pyo3(signature = (value, ptr, align = None))]
    fn store(&self, value: Value, ptr: Value, align: Option<u32>) -> PyResult<()> {
        self.open()?;
        if ptr.ty != Ty::Ptr {
            return err(format!("store through {}", ptr.ty));
        }
        // SAFETY: a store of a first-class value through a pointer.
        let p = self.v(&ptr)?;
        let r = unsafe { c::LLVMBuildStore(self.b, self.v(&value)?, p) };
        if let Some(a) = align {
            // SAFETY: setting the alignment of the store just built.
            unsafe { c::LLVMSetAlignment(r, a) };
        }
        self.annotate(r, p, &value.ty);
        Ok(())
    }

    /// Address of element `index` of an array of `elem` starting at `ptr` (inbounds).
    fn gep(&self, elem: PyType, ptr: Value, index: Operand) -> PyResult<Value> {
        self.open()?;
        let index = self.lit("gep", index, &Ty::Int(64))?;
        if ptr.ty != Ty::Ptr || !index.ty.is_int() || !elem.0.is_first_class() {
            return err(format!("gep of {} with {} index", ptr.ty, index.ty));
        }
        let mut idx = [self.v(&index)?];
        // SAFETY: an inbounds GEP over a first-class element type.
        let base = self.v(&ptr)?;
        let r = unsafe { c::LLVMBuildInBoundsGEP2(self.b, self.lt(&elem.0), base, idx.as_mut_ptr(), 1, NONAME) };
        self.derive(base, r);
        Ok(self.mk(r, Ty::Ptr))
    }

    /// `ptr + offset` bytes (inbounds).
    fn gep_bytes(&self, ptr: Value, offset: Operand) -> PyResult<Value> {
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
                c::LLVMBuildAlloca(self.allocas.get(), self.lt(&ty.0), NONAME)
            } else {
                let n = c::LLVMConstInt(self.lt(&Ty::Int(64)), count as u64, 0);
                c::LLVMBuildArrayAlloca(self.allocas.get(), self.lt(&ty.0), n, NONAME)
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
    fn call(&self, f: PyRef<'_, super::module::Function>, args: Vec<Operand>) -> PyResult<Option<Value>> {
        self.open()?;
        if !Rc::ptr_eq(&f.m, &self.m) {
            return err("function from a different module");
        }
        if args.len() != f.params.len() {
            return err(format!("{} takes {} arguments, got {}", f.name, f.params.len(), args.len()));
        }
        let args: Vec<Value> = args.into_iter().zip(&f.params).map(|(a, t)| self.lit("call", a, t)).collect::<PyResult<_>>()?;
        f.signature_matches(&args)?;
        let mut raw: Vec<LLVMValueRef> = args.iter().map(|a| self.v(a)).collect::<PyResult<_>>()?;
        Ok(self.call_raw(f.fnty, f.raw, &mut raw, &f.ret))
    }

    /// Call the function at address `f` (a ptr value) with `args`, returning `ret`
    /// (None: void). For functions outside the module (kernels), whose address is only
    /// known at run time (loaded from the link table, not part of the IR).
    #[pyo3(signature = (f, args, ret = None))]
    fn call_ptr(&self, f: Value, args: Vec<Value>, ret: Option<PyType>) -> PyResult<Option<Value>> {
        self.open()?;
        if f.ty != Ty::Ptr {
            return err(format!("call through {}", f.ty));
        }
        let ret = ret.map(|t| t.0).unwrap_or(Ty::Void);
        let mut ps: Vec<LLVMTypeRef> = args.iter().map(|a| self.lt(&a.ty)).collect();
        // SAFETY: a function type of first-class parameter types.
        let fnty = unsafe { c::LLVMFunctionType(self.lt(&ret), ps.as_mut_ptr(), ps.len() as u32, 0) };
        let mut raw: Vec<LLVMValueRef> = args.iter().map(|a| self.v(a)).collect::<PyResult<_>>()?;
        let fp = self.v(&f)?;
        Ok(self.call_raw(fnty, fp, &mut raw, &ret))
    }

    /// A float intrinsic, overloaded on the operands' type.
    fn intrinsic(&self, which: Intrinsic, args: Vec<Operand>) -> PyResult<Value> {
        self.open()?;
        let args = self.many(which.name(), args)?;
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

    fn libm(&self, which: Libm, args: Vec<Operand>) -> PyResult<Value> {
        self.open()?;
        let args = self.many(which.c_name(), args)?;
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
    fn rt_alloc(&self, ctx: Value, nbytes: Operand, owner_slot: Value) -> PyResult<Value> {
        self.open()?;
        let nbytes = self.i64_of("rt_alloc nbytes", nbytes)?;
        if ctx.ty != Ty::Ptr || nbytes.ty != Ty::Int(64) || owner_slot.ty != Ty::Ptr {
            return err("rt_alloc(ctx: ptr, nbytes: i64, owner_slot: ptr)");
        }
        // SAFETY: an integer constant of this module's context.
        let align = self.mk(unsafe { c::LLVMConstInt(self.lt(&Ty::Int(64)), 64, 0) }, Ty::Int(64));
        let (f, fnty) = self.declare("ptl_rt_alloc", &[Ty::Ptr, Ty::Int(64), Ty::Int(64), Ty::Ptr], &Ty::Ptr, false);
        let mut raw = [self.v(&ctx)?, self.v(&nbytes)?, self.v(&align)?, self.v(&owner_slot)?];
        Ok(self.call_raw(fnty, f, &mut raw, &Ty::Ptr).unwrap())
    }

    /// Scratch memory: `nbytes` (i64), 64-byte aligned, valid until the task ends or an
    /// earlier mark is released. Null on failure, with the error set.
    fn rt_scratch(&self, ctx: Value, nbytes: Operand) -> PyResult<Value> {
        self.open()?;
        let nbytes = self.i64_of("rt_scratch nbytes", nbytes)?;
        if ctx.ty != Ty::Ptr || nbytes.ty != Ty::Int(64) {
            return err("rt_scratch(ctx: ptr, nbytes: i64)");
        }
        let (f, fnty) = self.declare("ptl_rt_scratch", &[Ty::Ptr, Ty::Int(64)], &Ty::Ptr, false);
        // Like malloc: the memory returned is not reachable through any other pointer
        // (earlier scratch is disjoint; released scratch is dead by contract).
        // SAFETY: a return attribute on the declaration just looked up.
        unsafe {
            let kind = c::LLVMGetEnumAttributeKindForName(c"noalias".as_ptr(), 7);
            c::LLVMAddAttributeAtIndex(f, llvm_sys::LLVMAttributeReturnIndex, c::LLVMCreateEnumAttribute(self.m.ctx, kind, 0));
        }
        let mut raw = [self.v(&ctx)?, self.v(&nbytes)?];
        Ok(self.call_raw(fnty, f, &mut raw, &Ty::Ptr).unwrap())
    }

    /// Give up a buffer reference (`buffer`: a PtBuffer pointer, may be null).
    fn rt_decref(&self, ctx: Value, buffer: Value) -> PyResult<()> {
        self.open()?;
        if ctx.ty != Ty::Ptr || buffer.ty != Ty::Ptr {
            return err("rt_decref(ctx: ptr, buffer: ptr)");
        }
        let (f, fnty) = self.declare("ptl_rt_buffer_decref", &[Ty::Ptr, Ty::Ptr], &Ty::Void, false);
        let mut raw = [self.v(&ctx)?, self.v(&buffer)?];
        self.call_raw(fnty, f, &mut raw, &Ty::Void);
        Ok(())
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
    fn ret(&self, value: Option<Operand>) -> PyResult<()> {
        self.open()?;
        let ret_ty = self.ret.borrow().clone();
        let value = match value {
            Some(x) if ret_ty != Ty::Void => Some(self.lit("ret", x, &ret_ty)?),
            Some(Operand::Value(v)) => Some(v),
            Some(_) => return err("ret of a literal in a function returning void"),
            None => None,
        };
        match (value, &ret_ty) {
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
    fn loop_(slf: Py<Self>, py: Python<'_>, n: Operand) -> PyResult<Loop> {
        let n = {
            let b = slf.borrow(py);
            b.open()?;
            let n = b.i64_of("loop count", n)?;
            b.v(&n)?;
            n
        };
        Ok(Loop { b: slf, n, state: None })
    }

    /// `with b.parallel(ctx, n, grain) as r:` a loop over `0..n` split into chunks of
    /// `grain` (i64s) that may run in parallel (the runtime's `parallel_for`). Inside the
    /// block, code goes into an outlined body function, called per chunk with its own
    /// context; `r.start`, `r.end` are the chunk's bounds. Values of the enclosing
    /// function used inside are captured automatically, and `ctx` stands for the
    /// chunk's context there. After the block, `r.result` (i32) is the runtime's result:
    /// nonzero if a chunk failed (the error is set on `ctx`).
    ///
    /// Chunks must not write memory another chunk reads or writes. Captured values must
    /// be scalars or pointers.
    fn parallel(slf: Py<Self>, py: Python<'_>, ctx: Value, n: Operand, grain: Operand) -> PyResult<Parallel> {
        let (n, grain) = {
            let b = slf.borrow(py);
            b.open()?;
            let n = b.i64_of("parallel: n", n)?;
            let grain = b.i64_of("parallel: grain", grain)?;
            for (what, v, t) in [("ctx", &ctx, Ty::Ptr), ("n", &n, Ty::Int(64)), ("grain", &grain, Ty::Int(64))] {
                b.v(v)?;
                if v.ty != t {
                    return err(format!("parallel: {what} must be {t}"));
                }
            }
            (n, grain)
        };
        Ok(Parallel { b: slf, ctx, n, grain, parent_block: None, start: None, end: None, result: None })
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
        Ok(If { b: slf, cond, done: None, branch: None })
    }

    /// `with b.else_():` right after a `with b.if_(cond):` block runs the body if `cond`
    /// was false.
    #[pyo3(name = "else_")]
    fn else_(slf: Py<Self>, py: Python<'_>) -> PyResult<Else> {
        let done = {
            let b = slf.borrow(py);
            b.m.check_open()?;
            let Some((branch, done)) = b.last_if.take() else {
                return err("else_ must directly follow a `with b.if_(...)` block");
            };
            // SAFETY: `done` is a live block of this function, `branch` the conditional
            // branch of the if that ended in it.
            let fresh = unsafe { b.block() == done && c::LLVMGetFirstInstruction(done).is_null() };
            if !fresh {
                return err("else_ must directly follow a `with b.if_(...)` block");
            }
            let other = b.new_block(c"else");
            // SAFETY: successor 1 of a conditional branch is its false target.
            unsafe { c::LLVMSetSuccessor(branch, 1, other) };
            b.position(other);
            done
        };
        Ok(Else { b: slf, done: Some(done) })
    }

    /// Loop with carried values: `acc = init; for i in 0..n: acc = body(i, *acc)`;
    /// returns the final values (a list). `body` returns the new values: a sequence of
    /// the init types, or the value itself if there is one.
    ///
    /// With `until=True` the body returns one more value, an i1 stop flag; the loop
    /// ends after the first step that sets it, and the flag is returned last (false
    /// when no step ran).
    #[pyo3(signature = (n, init, body, until=false))]
    fn fold(&self, py: Python<'_>, n: Operand, init: Vec<Value>, body: Py<PyAny>, until: bool) -> PyResult<Vec<Value>> {
        self.open()?;
        let n = self.i64_of("fold count", n)?;
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
        let mut new: Vec<Value> = values_of(py, res?)?;
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

    /// `then()` if `cond` else `else_()`; both return values of the same types (a
    /// sequence, one value, or None for none), which are merged and returned as a list.
    /// For branches without values, `with b.if_(cond): ...` and `with b.else_(): ...`
    /// read better.
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
            let vals: Vec<Value> = values_of(py, r?)?;
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

/// `with b.parallel(ctx, n, grain) as r:`
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Parallel {
    b: Py<Builder>,
    ctx: Value,
    n: Value,
    grain: Value,
    /// where the parent's code continues after the region
    parent_block: Option<LLVMBasicBlockRef>,
    start: Option<Value>,
    end: Option<Value>,
    result: Option<Value>,
}

#[pymethods]
impl Parallel {
    fn __enter__(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<Self>> {
        {
            let mut me = slf.borrow_mut(py);
            if me.start.is_some() || me.result.is_some() {
                return err("a parallel region can be entered once");
            }
            let b = me.b.clone_ref(py);
            let b = b.borrow(py);
            let ctx_raw = b.v(&me.ctx)?;
            me.parent_block = Some(b.block());
            let m = &b.m;
            // SAFETY: a new internal function i32(ptr ctx, ptr captures, i64 start, i64 end) in
            // this module, with its entry and body blocks like `Builder::new`; the
            // builder is then positioned in it (the parent's position is saved).
            let (body, body_allocas, params) = unsafe {
                let (i32t, ptr, i64t) = (b.lt(&Ty::Int(32)), b.lt(&Ty::Ptr), b.lt(&Ty::Int(64)));
                let mut ps = [ptr, ptr, i64t, i64t];
                let fnty = c::LLVMFunctionType(i32t, ps.as_mut_ptr(), 4, 0);
                let parent_name = std::ffi::CStr::from_ptr(c::LLVMGetValueName2(b.func.get(), &mut 0)).to_string_lossy().into_owned();
                let name = cstring(&format!("{parent_name}.par"));
                let f = c::LLVMAddFunction(m.module, name.as_ptr(), fnty);
                c::LLVMSetLinkage(f, llvm_sys::LLVMLinkage::LLVMInternalLinkage);
                add_fn_attr(m, f, "noinline");
                let entry = c::LLVMAppendBasicBlockInContext(m.ctx, f, c"entry".as_ptr());
                let blk = c::LLVMAppendBasicBlockInContext(m.ctx, f, c"body".as_ptr());
                let tmp = c::LLVMCreateBuilderInContext(m.ctx);
                c::LLVMPositionBuilderAtEnd(tmp, entry);
                let br = c::LLVMBuildBr(tmp, blk);
                c::LLVMPositionBuilderBefore(tmp, br);
                let params: Vec<LLVMValueRef> = (0..4).map(|k| c::LLVMGetParam(f, k)).collect();
                c::LLVMPositionBuilderAtEnd(b.b, blk);
                (f, tmp, params)
            };
            let region = Region {
                parent_func: b.func.get(),
                parent_allocas: b.allocas.get(),
                parent_ret: b.ret.borrow().clone(),
                body,
                body_allocas,
                captures_param: params[1],
                map: std::collections::HashMap::from([(ctx_raw as usize, params[0])]),
                captures: vec![],
            };
            b.regions.borrow_mut().push(region);
            b.func.set(body);
            b.allocas.set(body_allocas);
            *b.ret.borrow_mut() = Ty::Int(32);
            me.start = Some(b.mk(params[2], Ty::Int(64)));
            me.end = Some(b.mk(params[3], Ty::Int(64)));
            *b.depth.borrow_mut() += 1;
        }
        Ok(slf)
    }

    #[getter]
    fn start(&self) -> PyResult<Value> {
        self.start.clone().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("enter the region first"))
    }

    #[getter]
    fn end(&self) -> PyResult<Value> {
        self.end.clone().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("enter the region first"))
    }

    /// The runtime's result (i32, 0 on success), after the block.
    #[getter]
    fn result(&self) -> PyResult<Value> {
        self.result.clone().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("leave the region first"))
    }

    fn __exit__(&mut self, py: Python<'_>, exc_type: Option<Py<PyAny>>, _v: Option<Py<PyAny>>, _tb: Option<Py<PyAny>>) -> PyResult<bool> {
        let b = self.b.borrow(py);
        *b.depth.borrow_mut() -= 1;
        // the body ends with success unless it already returned
        if exc_type.is_none() && !b.is_terminated() {
            // SAFETY: i32 0 in the body's open block.
            unsafe { c::LLVMBuildRet(b.b, c::LLVMConstInt(b.lt(&Ty::Int(32)), 0, 0)) };
        }
        let region = b.regions.borrow_mut().pop().expect("an open region");
        // SAFETY: the body's alloca builder isn't used any more.
        unsafe { c::LLVMDisposeBuilder(region.body_allocas) };
        b.func.set(region.parent_func);
        b.allocas.set(region.parent_allocas);
        *b.ret.borrow_mut() = region.parent_ret;
        b.position(self.parent_block.take().expect("saved at entry"));
        if exc_type.is_some() {
            return Ok(false);
        }
        // in the parent: the captures (one 8-byte slot per capture), then the runtime call
        let k = region.captures.len().max(1);
        // SAFETY: an array alloca in the parent's entry, stores of the captured parent
        // values into their slots, and the call of the runtime's parallel_for with the
        // body's address, all in the parent's open block.
        let rc = unsafe {
            let i8t = c::LLVMInt8TypeInContext(b.m.ctx);
            let i64t = b.lt(&Ty::Int(64));
            let cnt = c::LLVMConstInt(i64t, k as u64, 0);
            let captures = c::LLVMBuildArrayAlloca(b.allocas.get(), i64t, cnt, NONAME);
            c::LLVMSetAlignment(captures, 64);
            for (slot, (raw, _ty)) in region.captures.iter().enumerate() {
                let mut off = c::LLVMConstInt(i64t, 8 * slot as u64, 0);
                let p = c::LLVMBuildInBoundsGEP2(b.b, i8t, captures, &mut off, 1, NONAME);
                c::LLVMBuildStore(b.b, *raw, p);
            }
            let (f, fnty) = b.declare("ptl_rt_parallel_for", &[Ty::Ptr, Ty::Int(64), Ty::Int(64), Ty::Ptr, Ty::Ptr], &Ty::Int(32), false);
            let mut args = [b.v(&self.ctx)?, b.v(&self.n)?, b.v(&self.grain)?, region.body, captures];
            c::LLVMBuildCall2(b.b, fnty, f, args.as_mut_ptr(), 5, NONAME)
        };
        self.result = Some(b.mk(rc, Ty::Int(32)));
        Ok(false)
    }
}

/// `with b.if_(cond):`
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct If {
    b: Py<Builder>,
    cond: Value,
    done: Option<LLVMBasicBlockRef>,
    /// the conditional branch into the body
    branch: Option<LLVMValueRef>,
}

/// `with b.else_():`
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Else {
    b: Py<Builder>,
    done: Option<LLVMBasicBlockRef>,
}

#[pymethods]
impl Else {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.done.is_none() {
            return err("an else can be entered once");
        }
        *self.b.borrow(py).depth.borrow_mut() += 1;
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
        // SAFETY: `cond` is an i1 of this function; the blocks belong to it.
        self.branch = Some(unsafe { c::LLVMBuildCondBr(b.b, b.v(&self.cond)?, then, done) });
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
        b.last_if.set(self.branch.take().map(|br| (br, done)));
        Ok(false)
    }
}
