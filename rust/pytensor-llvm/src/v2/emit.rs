//! LLVM IR building blocks shared by scalar ops and tensor ops.

use super::op::R;
use crate::dtype::DType;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType, PointerType};
use inkwell::values::{AsValueRef, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FastMathFlags, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::cell::RefCell;

pub type V<'c> = BasicValueEnum<'c>;

pub fn e<E: std::fmt::Debug>(err: E) -> String {
    format!("{err:?}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

/// LLVM float intrinsics we use, by their LLVM names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FIntr {
    Exp,
    Log,
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
}

impl FIntr {
    fn llvm_name(self) -> &'static str {
        match self {
            FIntr::Exp => "llvm.exp",
            FIntr::Log => "llvm.log",
            FIntr::Sqrt => "llvm.sqrt",
            FIntr::Sin => "llvm.sin",
            FIntr::Cos => "llvm.cos",
            FIntr::Floor => "llvm.floor",
            FIntr::Ceil => "llvm.ceil",
            FIntr::Trunc => "llvm.trunc",
            FIntr::RoundEven => "llvm.roundeven",
            FIntr::Round => "llvm.round",
            FIntr::Fabs => "llvm.fabs",
            FIntr::Pow => "llvm.pow",
            FIntr::Maximum => "llvm.maximum",
            FIntr::Minimum => "llvm.minimum",
        }
    }
}

/// libm functions (C names of the double version; float adds `f`). All of them have
/// SIMD variants in glibc's libmvec (>= 2.35) that LLVM 18's vector library table lacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Libm {
    Exp2,
    Log2,
    Log10,
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
}

impl Libm {
    fn c_name(self) -> &'static str {
        match self {
            Libm::Exp2 => "exp2",
            Libm::Log2 => "log2",
            Libm::Log10 => "log10",
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
        }
    }
}

/// Precompiled Rust scalar functions (`native::scalar_symbols`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustMath {
    Erf,
    Erfc,
    GammaLn,
    Gamma,
    Psi,
}

impl RustMath {
    fn symbol(self, dt: DType) -> &'static str {
        let f32 = dt == DType::Float32;
        match (self, f32) {
            (RustMath::Erf, false) => "ptl_erf_f64",
            (RustMath::Erf, true) => "ptl_erf_f32",
            (RustMath::Erfc, false) => "ptl_erfc_f64",
            (RustMath::Erfc, true) => "ptl_erfc_f32",
            (RustMath::GammaLn, false) => "ptl_gammaln_f64",
            (RustMath::GammaLn, true) => "ptl_gammaln_f32",
            (RustMath::Gamma, false) => "ptl_gamma_f64",
            (RustMath::Gamma, true) => "ptl_gamma_f32",
            (RustMath::Psi, false) => "ptl_psi_f64",
            (RustMath::Psi, true) => "ptl_psi_f32",
        }
    }
}

pub struct Emitter<'c> {
    pub ctx: &'c Context,
    pub module: Module<'c>,
    pub b: Builder<'c>,
    fm: Option<FastMathFlags>,
    /// glibc libmvec variants to advertise on libm calls: (ISA letter, f64 lanes).
    vfabi: Vec<(char, u32)>,
    used: RefCell<Vec<FunctionValue<'c>>>,
}

impl<'c> Emitter<'c> {
    pub fn new(ctx: &'c Context, name: &str, fastmath: bool, vfabi: Vec<(char, u32)>) -> Self {
        let fm = fastmath.then_some(
            FastMathFlags::AllowReciprocal
                | FastMathFlags::AllowContract
                | FastMathFlags::ApproxFunc
                | FastMathFlags::AllowReassoc
                | FastMathFlags::NoSignedZeros,
        );
        Emitter { ctx, module: ctx.create_module(name), b: ctx.create_builder(), fm, vfabi, used: Default::default() }
    }

    /// Keep vector-variant declarations alive until the loop vectorizer has seen them.
    pub fn finish(&self) {
        let used = self.used.borrow();
        if used.is_empty() {
            return;
        }
        let ptrs: Vec<PointerValue> = used.iter().map(|f| f.as_global_value().as_pointer_value()).collect();
        let arr = self.ptrt().const_array(&ptrs);
        let g = self.module.add_global(arr.get_type(), None, "llvm.compiler.used");
        g.set_linkage(Linkage::Appending);
        g.set_section(Some("llvm.metadata"));
        g.set_initializer(&arr);
    }

    // ------------------------------------------------------------ types & constants

    pub fn i64t(&self) -> IntType<'c> {
        self.ctx.i64_type()
    }

    pub fn i32t(&self) -> IntType<'c> {
        self.ctx.i32_type()
    }

    pub fn ptrt(&self) -> PointerType<'c> {
        self.ctx.ptr_type(AddressSpace::default())
    }

    /// In-memory element type (bool is a byte).
    pub fn memty(&self, dt: DType) -> BasicTypeEnum<'c> {
        match dt {
            DType::Float64 => self.ctx.f64_type().into(),
            DType::Float32 => self.ctx.f32_type().into(),
            DType::Int64 => self.ctx.i64_type().into(),
            DType::Int32 => self.ctx.i32_type().into(),
            DType::Int16 => self.ctx.i16_type().into(),
            DType::Int8 | DType::Uint8 | DType::Bool => self.ctx.i8_type().into(),
        }
    }

    /// In-register type (bool is i1).
    pub fn regty(&self, dt: DType) -> BasicTypeEnum<'c> {
        match dt {
            DType::Bool => self.ctx.bool_type().into(),
            _ => self.memty(dt),
        }
    }

    pub fn i64c(&self, v: i64) -> IntValue<'c> {
        self.i64t().const_int(v as u64, true)
    }

    pub fn constant(&self, dt: DType, f: f64, i: i64) -> V<'c> {
        match dt {
            DType::Float64 => self.ctx.f64_type().const_float(f).into(),
            DType::Float32 => self.ctx.f32_type().const_float(f).into(),
            DType::Bool => self.ctx.bool_type().const_int((i != 0) as u64, false).into(),
            _ => self.regty(dt).into_int_type().const_int(i as u64, true).into(),
        }
    }

    pub fn fconst(&self, dt: DType, f: f64) -> V<'c> {
        self.constant(dt, f, f as i64)
    }

    pub fn fm(&self, v: V<'c>) -> V<'c> {
        if let (Some(flags), Some(inst)) = (self.fm, v.as_instruction_value()) {
            let _ = inst.set_fast_math_flags(flags);
        }
        v
    }

    pub fn cur_fn(&self) -> FunctionValue<'c> {
        self.b.get_insert_block().unwrap().get_parent().unwrap()
    }

    pub fn attr(&self, name: &str) -> Attribute {
        self.ctx.create_enum_attribute(Attribute::get_named_enum_kind_id(name), 0)
    }

    // ------------------------------------------------------------ memory

    pub fn load_elem(&self, ptr: PointerValue<'c>, dt: DType) -> R<V<'c>> {
        let v = self.b.build_load(self.memty(dt), ptr, "ld").map_err(e)?;
        if dt == DType::Bool {
            let z = self.ctx.i8_type().const_zero();
            Ok(self.b.build_int_compare(IntPredicate::NE, v.into_int_value(), z, "b").map_err(e)?.into())
        } else {
            Ok(v)
        }
    }

    pub fn store_elem(&self, ptr: PointerValue<'c>, v: V<'c>, dt: DType) -> R<()> {
        let v: V = if dt == DType::Bool {
            self.b.build_int_z_extend(v.into_int_value(), self.ctx.i8_type(), "zb").map_err(e)?.into()
        } else {
            v
        };
        self.b.build_store(ptr, v).map_err(e)?;
        Ok(())
    }

    pub fn gep_elem(&self, dt: DType, base: PointerValue<'c>, idx: IntValue<'c>) -> R<PointerValue<'c>> {
        // SAFETY: builds IR only; generated accesses stay inside the arrays (shapes are
        // checked or static).
        unsafe { self.b.build_in_bounds_gep(self.memty(dt), base, &[idx], "p").map_err(e) }
    }

    pub fn gep_bytes(&self, base: PointerValue<'c>, off: IntValue<'c>) -> R<PointerValue<'c>> {
        // SAFETY: as in gep_elem.
        unsafe { self.b.build_in_bounds_gep(self.ctx.i8_type(), base, &[off], "pb").map_err(e) }
    }

    pub fn load_i64(&self, base: PointerValue<'c>, i: usize) -> R<IntValue<'c>> {
        let p = self.gep_elem(DType::Int64, base, self.i64c(i as i64))?;
        Ok(self.b.build_load(self.i64t(), p, "i").map_err(e)?.into_int_value())
    }

    pub fn store_i64(&self, base: PointerValue<'c>, i: usize, v: IntValue<'c>) -> R<()> {
        let p = self.gep_elem(DType::Int64, base, self.i64c(i as i64))?;
        self.b.build_store(p, v).map_err(e)?;
        Ok(())
    }

    pub fn mul(&self, a: IntValue<'c>, b: IntValue<'c>) -> R<IntValue<'c>> {
        self.b.build_int_nsw_mul(a, b, "m").map_err(e)
    }

    pub fn add(&self, a: IntValue<'c>, b: IntValue<'c>) -> R<IntValue<'c>> {
        self.b.build_int_nsw_add(a, b, "a").map_err(e)
    }

    // ------------------------------------------------------------ casts

    pub fn cast(&self, v: V<'c>, from: DType, to: DType) -> R<V<'c>> {
        if from == to {
            return Ok(v);
        }
        let b = &self.b;
        let tt = self.regty(to);
        Ok(match (from.is_float(), to.is_float()) {
            (true, true) => b.build_float_cast(v.into_float_value(), tt.into_float_type(), "fc").map_err(e)?.into(),
            (false, true) => {
                if from.is_signed_int() {
                    b.build_signed_int_to_float(v.into_int_value(), tt.into_float_type(), "if").map_err(e)?.into()
                } else {
                    b.build_unsigned_int_to_float(v.into_int_value(), tt.into_float_type(), "uf").map_err(e)?.into()
                }
            }
            (true, false) => {
                if to == DType::Bool {
                    let z = v.into_float_value().get_type().const_zero();
                    b.build_float_compare(FloatPredicate::UNE, v.into_float_value(), z, "fb").map_err(e)?.into()
                } else if to.is_signed_int() {
                    b.build_float_to_signed_int(v.into_float_value(), tt.into_int_type(), "fi").map_err(e)?.into()
                } else {
                    b.build_float_to_unsigned_int(v.into_float_value(), tt.into_int_type(), "fu").map_err(e)?.into()
                }
            }
            (false, false) => {
                let iv = v.into_int_value();
                if to == DType::Bool {
                    let z = iv.get_type().const_zero();
                    b.build_int_compare(IntPredicate::NE, iv, z, "ib").map_err(e)?.into()
                } else if to.bits() > from.bits() {
                    if from.is_signed_int() {
                        b.build_int_s_extend(iv, tt.into_int_type(), "se").map_err(e)?.into()
                    } else {
                        b.build_int_z_extend(iv, tt.into_int_type(), "ze").map_err(e)?.into()
                    }
                } else if to.bits() < from.bits() {
                    b.build_int_truncate(iv, tt.into_int_type(), "tr").map_err(e)?.into()
                } else {
                    iv.into() // int8 <-> uint8: same bits
                }
            }
        })
    }

    // ------------------------------------------------------------ calls

    pub fn fintr(&self, intr: FIntr, dt: DType, args: &[V<'c>]) -> R<V<'c>> {
        let name = intr.llvm_name();
        let i = Intrinsic::find(name).ok_or_else(|| format!("no intrinsic {name}"))?;
        let f = i.get_declaration(&self.module, &[self.regty(dt)]).ok_or_else(|| format!("bad intrinsic decl {name}"))?;
        let args: Vec<BasicMetadataValueEnum> = args.iter().map(|a| (*a).into()).collect();
        let call = self.b.build_call(f, &args, "c").map_err(e)?;
        let v = call.try_as_basic_value().basic().ok_or("intrinsic returned void")?;
        Ok(self.fm(v))
    }

    /// Declare an external function (if needed).
    pub fn declare(&self, name: &str, ret: Option<BasicTypeEnum<'c>>, params: &[BasicMetadataTypeEnum<'c>]) -> FunctionValue<'c> {
        if let Some(f) = self.module.get_function(name) {
            return f;
        }
        let fty = match ret {
            Some(r) => r.fn_type(params, false),
            None => self.ctx.void_type().fn_type(params, false),
        };
        self.module.add_function(name, fty, Some(Linkage::External))
    }

    /// Call a pure external math function `name(args) -> ret`.
    fn pure_call(&self, name: &str, ret: BasicTypeEnum<'c>, args: &[V<'c>]) -> R<V<'c>> {
        let fresh = self.module.get_function(name).is_none();
        let ptys: Vec<BasicMetadataTypeEnum> = args.iter().map(|a| a.get_type().into()).collect();
        let f = self.declare(name, Some(ret), &ptys);
        if fresh {
            for a in ["nounwind", "willreturn", "nosync", "nofree"] {
                f.add_attribute(AttributeLoc::Function, self.attr(a));
            }
            // Pure: we never read errno (like -fno-math-errno). Without this the loop
            // vectorizer refuses to touch the call.
            let k = Attribute::get_named_enum_kind_id("memory");
            f.add_attribute(AttributeLoc::Function, self.ctx.create_enum_attribute(k, 0));
        }
        let args: Vec<BasicMetadataValueEnum> = args.iter().map(|a| (*a).into()).collect();
        let call = self.b.build_call(f, &args, "x").map_err(e)?;
        let v = call.try_as_basic_value().basic().ok_or("call returned void")?;
        Ok(self.fm(v))
    }

    /// libm function (`f` suffix for float32), advertising libmvec SIMD variants.
    pub fn libm(&self, func: Libm, dt: DType, args: &[V<'c>]) -> R<V<'c>> {
        let base = func.c_name();
        let n = if dt == DType::Float32 { format!("{base}f") } else { base.to_string() };
        let v = self.pure_call(&n, self.regty(dt), args)?;
        if !self.vfabi.is_empty() {
            let call = v.as_instruction_value().unwrap();
            let params = "v".repeat(args.len());
            let mut variants = Vec::new();
            for &(isa, lanes64) in &self.vfabi {
                let lanes = if dt == DType::Float32 { lanes64 * 2 } else { lanes64 };
                let vname = format!("_ZGV{isa}N{lanes}{params}_{n}");
                variants.push(format!("{vname}({vname})"));
                if self.module.get_function(&vname).is_none() {
                    let vt = self.regty(dt).into_float_type().vec_type(lanes);
                    let ptys: Vec<BasicMetadataTypeEnum> = (0..args.len()).map(|_| vt.into()).collect();
                    let f = self.module.add_function(&vname, vt.fn_type(&ptys, false), Some(Linkage::External));
                    self.used.borrow_mut().push(f);
                }
            }
            let attr = self.ctx.create_string_attribute("vector-function-abi-variant", &variants.join(","));
            // SAFETY: `call` is the call instruction just built.
            let cs = unsafe { inkwell::values::CallSiteValue::new(call.as_value_ref()) };
            cs.add_attribute(AttributeLoc::Function, attr);
        }
        Ok(v)
    }

    pub fn rust_math(&self, func: RustMath, dt: DType, args: &[V<'c>]) -> R<V<'c>> {
        self.pure_call(func.symbol(dt), self.regty(dt), args)
    }

    // ------------------------------------------------------------ arithmetic

    pub fn arith(&self, op: Arith, a: V<'c>, b: V<'c>, dt: DType) -> R<V<'c>> {
        let bl = &self.b;
        if dt.is_float() {
            let (a, b) = (a.into_float_value(), b.into_float_value());
            let v: V = match op {
                Arith::Add => bl.build_float_add(a, b, "fa").map_err(e)?.into(),
                Arith::Sub => bl.build_float_sub(a, b, "fs").map_err(e)?.into(),
                Arith::Mul => bl.build_float_mul(a, b, "fm").map_err(e)?.into(),
                Arith::Div => bl.build_float_div(a, b, "fd").map_err(e)?.into(),
            };
            Ok(self.fm(v))
        } else {
            let (a, b) = (a.into_int_value(), b.into_int_value());
            Ok(match op {
                Arith::Add => bl.build_int_add(a, b, "ia").map_err(e)?.into(),
                Arith::Sub => bl.build_int_sub(a, b, "is").map_err(e)?.into(),
                Arith::Mul => bl.build_int_mul(a, b, "im").map_err(e)?.into(),
                Arith::Div => return Err("integer true division".into()),
            })
        }
    }

    pub fn bitop(&self, op: BitOp, a: V<'c>, b: V<'c>) -> R<V<'c>> {
        let (a, b) = (a.into_int_value(), b.into_int_value());
        let bl = &self.b;
        Ok(match op {
            BitOp::And => bl.build_and(a, b, "an").map_err(e)?.into(),
            BitOp::Or => bl.build_or(a, b, "or").map_err(e)?.into(),
            BitOp::Xor => bl.build_xor(a, b, "xo").map_err(e)?.into(),
        })
    }

    pub fn select(&self, c: V<'c>, t: V<'c>, f: V<'c>) -> R<V<'c>> {
        self.b.build_select(c.into_int_value(), t, f, "sel").map_err(e)
    }

    pub fn cmp(&self, pred: Cmp, a: V<'c>, b: V<'c>, dt: DType) -> R<V<'c>> {
        if dt.is_float() {
            let p = match pred {
                Cmp::Lt => FloatPredicate::OLT,
                Cmp::Gt => FloatPredicate::OGT,
                Cmp::Le => FloatPredicate::OLE,
                Cmp::Ge => FloatPredicate::OGE,
                Cmp::Eq => FloatPredicate::OEQ,
                Cmp::Ne => FloatPredicate::UNE,
            };
            Ok(self.b.build_float_compare(p, a.into_float_value(), b.into_float_value(), "fc").map_err(e)?.into())
        } else {
            let s = dt.is_signed_int();
            let p = match (pred, s) {
                (Cmp::Lt, true) => IntPredicate::SLT,
                (Cmp::Lt, false) => IntPredicate::ULT,
                (Cmp::Gt, true) => IntPredicate::SGT,
                (Cmp::Gt, false) => IntPredicate::UGT,
                (Cmp::Le, true) => IntPredicate::SLE,
                (Cmp::Le, false) => IntPredicate::ULE,
                (Cmp::Ge, true) => IntPredicate::SGE,
                (Cmp::Ge, false) => IntPredicate::UGE,
                (Cmp::Eq, _) => IntPredicate::EQ,
                (Cmp::Ne, _) => IntPredicate::NE,
            };
            Ok(self.b.build_int_compare(p, a.into_int_value(), b.into_int_value(), "ic").map_err(e)?.into())
        }
    }

    /// NaN-propagating for floats, like numpy.maximum / numpy.minimum.
    pub fn maxmin(&self, is_max: bool, a: V<'c>, b: V<'c>, dt: DType) -> R<V<'c>> {
        if dt.is_float() {
            self.fintr(if is_max { FIntr::Maximum } else { FIntr::Minimum }, dt, &[a, b])
        } else {
            let c = self.cmp(if is_max { Cmp::Gt } else { Cmp::Lt }, a, b, dt)?;
            self.select(c, a, b)
        }
    }

    pub fn neg(&self, a: V<'c>, dt: DType) -> R<V<'c>> {
        if dt.is_float() {
            Ok(self.fm(self.b.build_float_neg(a.into_float_value(), "n").map_err(e)?.into()))
        } else {
            Ok(self.b.build_int_neg(a.into_int_value(), "n").map_err(e)?.into())
        }
    }

    // ------------------------------------------------------------ control flow

    /// for i in 0..n { body(i) }
    pub fn emit_loop(&self, n: IntValue<'c>, body: &mut dyn FnMut(IntValue<'c>) -> R<()>) -> R<()> {
        self.emit_acc_loop(n, &[], &mut |i, _| body(i).map(|_| vec![])).map(|_| ())
    }

    /// accs = inits; for i in 0..n { accs = body(i, accs) }; return accs
    pub fn emit_acc_loop(
        &self,
        n: IntValue<'c>,
        inits: &[V<'c>],
        body: &mut dyn FnMut(IntValue<'c>, &[V<'c>]) -> R<Vec<V<'c>>>,
    ) -> R<Vec<V<'c>>> {
        let f = self.cur_fn();
        let pre = self.b.get_insert_block().unwrap();
        let header = self.ctx.append_basic_block(f, "loop");
        let exit = self.ctx.append_basic_block(f, "loop.exit");
        let zero = self.i64c(0);
        let c = self.b.build_int_compare(IntPredicate::SGT, n, zero, "nz").map_err(e)?;
        self.b.build_conditional_branch(c, header, exit).map_err(e)?;
        self.b.position_at_end(header);
        let phi = self.b.build_phi(self.i64t(), "i").map_err(e)?;
        phi.add_incoming(&[(&zero, pre)]);
        let mut acc_phis = Vec::new();
        for init in inits {
            let p = self.b.build_phi(init.get_type(), "acc").map_err(e)?;
            p.add_incoming(&[(init, pre)]);
            acc_phis.push(p);
        }
        let i = phi.as_basic_value().into_int_value();
        let cur: Vec<V> = acc_phis.iter().map(|p| p.as_basic_value()).collect();
        let new = body(i, &cur)?;
        let latch = self.b.get_insert_block().unwrap();
        let next = self.b.build_int_nsw_add(i, self.i64c(1), "inext").map_err(e)?;
        let c = self.b.build_int_compare(IntPredicate::SLT, next, n, "c").map_err(e)?;
        self.b.build_conditional_branch(c, header, exit).map_err(e)?;
        phi.add_incoming(&[(&next, latch)]);
        for (p, v) in acc_phis.iter().zip(&new) {
            p.add_incoming(&[(v, latch)]);
        }
        self.b.position_at_end(exit);
        let mut res = Vec::new();
        for (init, v) in inits.iter().zip(&new) {
            let r = self.b.build_phi(init.get_type(), "racc").map_err(e)?;
            r.add_incoming(&[(init, pre), (v, latch)]);
            res.push(r.as_basic_value());
        }
        Ok(res)
    }

    /// if cond { then() }
    pub fn emit_if(&self, cond: IntValue<'c>, then: &mut dyn FnMut() -> R<()>) -> R<()> {
        let f = self.cur_fn();
        let tb = self.ctx.append_basic_block(f, "then");
        let done = self.ctx.append_basic_block(f, "endif");
        self.b.build_conditional_branch(cond, tb, done).map_err(e)?;
        self.b.position_at_end(tb);
        then()?;
        if self.b.get_insert_block().unwrap().get_terminator().is_none() {
            self.b.build_unconditional_branch(done).map_err(e)?;
        }
        self.b.position_at_end(done);
        Ok(())
    }

    /// A constant C string as a global.
    pub fn cstr(&self, s: &str) -> R<PointerValue<'c>> {
        let s = s.replace('\0', "?");
        Ok(self.b.build_global_string_ptr(&s, "msg").map_err(e)?.as_pointer_value())
    }

    pub fn const_int_value(v: IntValue<'c>) -> Option<i64> {
        v.get_sign_extended_constant()
    }
}
