//! LLVM IR generation for kernels (Elemwise/Composite and CAReduce).
//!
//! Kernels use the ABI in `abi.rs`. Generated kernels currently only have provided outputs
//! whose layout (C or F) is fixed in the spec; inputs with unknown layout get a runtime
//! contiguity check that selects a flat vectorized loop or a strided loop nest.
//!
//! The kernel body is emitted as a private `alwaysinline` function whose data pointers are
//! `noalias`; after inlining LLVM turns that into scoped alias metadata, which is what lets
//! the loop vectorizer work without runtime alias checks.

use crate::dtype::DType;
use crate::ir::{FusedOut, IndexedAxis, ReduceOp, SRef, ScalarGraph};
use crate::abi::PTARRAY_SIZE;
use crate::segment::{SegLoc, SegMember, SegmentSpec};
use crate::spec::{KernelSpec, Layout, Operand};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{
    AsValueRef,
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FastMathFlags, FunctionValue, IntValue,
    PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};

pub type R<T> = Result<T, String>;

/// Scalar ops understood by the code generator, with their (min, max) arity.
/// `usize::MAX` = variadic.
pub fn scalar_op_arity(op: &str) -> Option<(usize, usize)> {
    Some(match op {
        "add" | "mul" => (1, usize::MAX),
        "sub" | "truediv" | "intdiv" | "mod" | "pow" | "maximum" | "minimum" | "arctan2" | "lt" | "gt" | "le" | "ge"
        | "eq" | "neq" | "and" | "or" | "xor" | "second" => (2, 2),
        "clip" | "switch" => (3, 3),
        "neg" | "abs" | "sign" | "sqr" | "reciprocal" | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "sin"
        | "cos" | "floor" | "ceil" | "trunc" | "roundhalftoeven" | "roundhalfawayfromzero" | "expm1" | "log1p" | "tan"
        | "tanh" | "sinh" | "cosh" | "arcsin" | "arccos" | "arctan" | "arcsinh" | "arccosh" | "arctanh" | "erf"
        | "erfc" | "gammaln" | "gamma" | "psi" | "sigmoid" | "softplus" | "log1mexp" | "isnan" | "isinf" | "invert"
        | "identity" | "cast" => (1, 1),
        _ => return None,
    })
}

fn e<E: std::fmt::Debug>(err: E) -> String {
    format!("{err:?}")
}

pub struct Cg<'ctx> {
    pub ctx: &'ctx Context,
    pub module: Module<'ctx>,
    pub b: Builder<'ctx>,
    fm: Option<FastMathFlags>,
    /// glibc libmvec variants to advertise on libm calls: (ISA letter, f64 lanes).
    vfabi: Vec<(char, u32)>,
    used: std::cell::RefCell<Vec<FunctionValue<'ctx>>>,
    /// Generate kernels as private alwaysinline functions (segment members).
    internal: std::cell::Cell<bool>,
}

/// libm functions with SIMD variants in glibc's libmvec (>= 2.35) that LLVM 18's
/// built-in vector-library table does not know about.
const LIBMVEC: &[&str] = &[
    "expm1", "log1p", "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "asinh", "acosh",
    "atanh", "atan2", "exp2", "log2", "log10", "erf", "erfc",
];

type V<'ctx> = BasicValueEnum<'ctx>;

impl<'ctx> Cg<'ctx> {
    pub fn new(ctx: &'ctx Context, name: &str, fastmath: bool, vfabi: Vec<(char, u32)>) -> Self {
        let fm = if fastmath {
            Some(
                FastMathFlags::AllowReciprocal
                    | FastMathFlags::AllowContract
                    | FastMathFlags::ApproxFunc
                    | FastMathFlags::AllowReassoc
                    | FastMathFlags::NoSignedZeros,
            )
        } else {
            None
        };
        Cg {
            ctx,
            module: ctx.create_module(name),
            b: ctx.create_builder(),
            fm,
            vfabi,
            used: Default::default(),
            internal: std::cell::Cell::new(false),
        }
    }

    /// Keep vector-variant declarations alive until the loop vectorizer has seen them.
    fn finish_used(&self) {
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

    // ---------------------------------------------------------------- types & constants

    fn i64t(&self) -> inkwell::types::IntType<'ctx> {
        self.ctx.i64_type()
    }

    fn ptrt(&self) -> inkwell::types::PointerType<'ctx> {
        self.ctx.ptr_type(AddressSpace::default())
    }

    /// In-memory element type (bool is a byte).
    fn memty(&self, dt: DType) -> BasicTypeEnum<'ctx> {
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
    fn regty(&self, dt: DType) -> BasicTypeEnum<'ctx> {
        match dt {
            DType::Bool => self.ctx.bool_type().into(),
            _ => self.memty(dt),
        }
    }

    fn i64c(&self, v: i64) -> IntValue<'ctx> {
        self.i64t().const_int(v as u64, true)
    }

    fn constant(&self, dt: DType, f: f64, i: i64) -> V<'ctx> {
        match dt {
            DType::Float64 => self.ctx.f64_type().const_float(f).into(),
            DType::Float32 => self.ctx.f32_type().const_float(f).into(),
            DType::Bool => self.ctx.bool_type().const_int((i != 0) as u64, false).into(),
            _ => self.regty(dt).into_int_type().const_int(i as u64, true).into(),
        }
    }

    fn fconst(&self, dt: DType, f: f64) -> V<'ctx> {
        self.constant(dt, f, f as i64)
    }

    fn fm(&self, v: V<'ctx>) -> V<'ctx> {
        if let (Some(flags), Some(inst)) = (self.fm, v.as_instruction_value()) {
            let _ = inst.set_fast_math_flags(flags);
        }
        v
    }

    fn cur_fn(&self) -> FunctionValue<'ctx> {
        self.b.get_insert_block().unwrap().get_parent().unwrap()
    }

    // ---------------------------------------------------------------- memory

    fn load_elem(&self, ptr: PointerValue<'ctx>, dt: DType) -> R<V<'ctx>> {
        let v = self.b.build_load(self.memty(dt), ptr, "ld").map_err(e)?;
        if dt == DType::Bool {
            let z = self.ctx.i8_type().const_zero();
            Ok(self
                .b
                .build_int_compare(IntPredicate::NE, v.into_int_value(), z, "b")
                .map_err(e)?
                .into())
        } else {
            Ok(v)
        }
    }

    fn store_elem(&self, ptr: PointerValue<'ctx>, v: V<'ctx>, dt: DType) -> R<()> {
        let v = if dt == DType::Bool {
            self.b
                .build_int_z_extend(v.into_int_value(), self.ctx.i8_type(), "zb")
                .map_err(e)?
                .into()
        } else {
            v
        };
        self.b.build_store(ptr, v).map_err(e)?;
        Ok(())
    }

    fn gep_elem(&self, dt: DType, base: PointerValue<'ctx>, idx: IntValue<'ctx>) -> R<PointerValue<'ctx>> {
        unsafe { self.b.build_in_bounds_gep(self.memty(dt), base, &[idx], "p").map_err(e) }
    }

    fn gep_bytes(&self, base: PointerValue<'ctx>, off: IntValue<'ctx>) -> R<PointerValue<'ctx>> {
        unsafe {
            self.b
                .build_in_bounds_gep(self.ctx.i8_type(), base, &[off], "pb")
                .map_err(e)
        }
    }

    fn load_i64_at(&self, base: PointerValue<'ctx>, i: usize) -> R<IntValue<'ctx>> {
        let p = self.gep_elem(DType::Int64, base, self.i64c(i as i64))?;
        Ok(self.b.build_load(self.i64t(), p, "i").map_err(e)?.into_int_value())
    }

    // ---------------------------------------------------------------- casts

    pub fn cast(&self, v: V<'ctx>, from: DType, to: DType) -> R<V<'ctx>> {
        if from == to {
            return Ok(v);
        }
        let b = &self.b;
        let tt = self.regty(to);
        Ok(match (from.is_float(), to.is_float()) {
            (true, true) => b
                .build_float_cast(v.into_float_value(), tt.into_float_type(), "fc")
                .map_err(e)?
                .into(),
            (false, true) => {
                if from.is_signed_int() {
                    b.build_signed_int_to_float(v.into_int_value(), tt.into_float_type(), "if")
                        .map_err(e)?
                        .into()
                } else {
                    b.build_unsigned_int_to_float(v.into_int_value(), tt.into_float_type(), "uf")
                        .map_err(e)?
                        .into()
                }
            }
            (true, false) => {
                if to == DType::Bool {
                    let z = v.into_float_value().get_type().const_zero();
                    b.build_float_compare(FloatPredicate::UNE, v.into_float_value(), z, "fb")
                        .map_err(e)?
                        .into()
                } else if to.is_signed_int() {
                    b.build_float_to_signed_int(v.into_float_value(), tt.into_int_type(), "fi")
                        .map_err(e)?
                        .into()
                } else {
                    b.build_float_to_unsigned_int(v.into_float_value(), tt.into_int_type(), "fu")
                        .map_err(e)?
                        .into()
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

    // ---------------------------------------------------------------- calls

    fn intrinsic(&self, name: &str, tys: &[BasicTypeEnum<'ctx>], args: &[V<'ctx>]) -> R<V<'ctx>> {
        let intr = Intrinsic::find(name).ok_or_else(|| format!("no intrinsic {name}"))?;
        let f = intr
            .get_declaration(&self.module, tys)
            .ok_or_else(|| format!("bad intrinsic decl {name}"))?;
        let args: Vec<BasicMetadataValueEnum> = args.iter().map(|a| (*a).into()).collect();
        let call = self.b.build_call(f, &args, "c").map_err(e)?;
        let v = call
            .try_as_basic_value()
            .basic()
            .ok_or("intrinsic returned void")?;
        Ok(self.fm(v))
    }

    /// Call an external C function `name(args) -> ret`, declaring it if necessary.
    fn extcall(&self, name: &str, ret: BasicTypeEnum<'ctx>, args: &[V<'ctx>]) -> R<V<'ctx>> {
        let f = match self.module.get_function(name) {
            Some(f) => f,
            None => {
                let ptys: Vec<BasicMetadataTypeEnum> =
                    args.iter().map(|a| a.get_type().into()).collect();
                let fty = ret.fn_type(&ptys, false);
                let f = self.module.add_function(name, fty, Some(Linkage::External));
                for attr in ["nounwind", "willreturn", "nosync", "nofree"] {
                    let k = Attribute::get_named_enum_kind_id(attr);
                    f.add_attribute(AttributeLoc::Function, self.ctx.create_enum_attribute(k, 0));
                }
                // Pure: we never read errno (like -fno-math-errno). Without this the loop
                // vectorizer refuses to touch the call.
                let k = Attribute::get_named_enum_kind_id("memory");
                f.add_attribute(AttributeLoc::Function, self.ctx.create_enum_attribute(k, 0));
                f
            }
        };
        let args: Vec<BasicMetadataValueEnum> = args.iter().map(|a| (*a).into()).collect();
        let call = self.b.build_call(f, &args, "x").map_err(e)?;
        let v = call.try_as_basic_value().basic().ok_or("call returned void")?;
        Ok(self.fm(v))
    }

    /// libm function with `f` suffix for float32. If libmvec has SIMD versions, the call
    /// gets a `vector-function-abi-variant` attribute so the loop vectorizer can use them.
    fn libm(&self, name: &str, dt: DType, args: &[V<'ctx>]) -> R<V<'ctx>> {
        let n = if dt == DType::Float32 {
            format!("{name}f")
        } else {
            name.to_string()
        };
        let v = self.extcall(&n, self.regty(dt), args)?;
        if LIBMVEC.contains(&name) && !self.vfabi.is_empty() {
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
            let cs: inkwell::values::CallSiteValue = unsafe { inkwell::values::CallSiteValue::new(call.as_value_ref()) };
            cs.add_attribute(AttributeLoc::Function, attr);
        }
        Ok(v)
    }

    /// Precompiled Rust scalar function from `native.rs` (`ptl_<name>_f64|f32`).
    fn rustfn(&self, name: &str, dt: DType, args: &[V<'ctx>]) -> R<V<'ctx>> {
        let suffix = if dt == DType::Float32 { "f32" } else { "f64" };
        self.extcall(&format!("ptl_{name}_{suffix}"), self.regty(dt), args)
    }

    fn fintr(&self, name: &str, dt: DType, args: &[V<'ctx>]) -> R<V<'ctx>> {
        self.intrinsic(name, &[self.regty(dt)], args)
    }

    // ---------------------------------------------------------------- scalar ops

    fn fbin(&self, op: &str, a: V<'ctx>, b: V<'ctx>) -> R<V<'ctx>> {
        let (a, b) = (a.into_float_value(), b.into_float_value());
        let bl = &self.b;
        let v: V = match op {
            "add" => bl.build_float_add(a, b, "fa").map_err(e)?.into(),
            "sub" => bl.build_float_sub(a, b, "fs").map_err(e)?.into(),
            "mul" => bl.build_float_mul(a, b, "fm").map_err(e)?.into(),
            "div" => bl.build_float_div(a, b, "fd").map_err(e)?.into(),
            _ => return Err(format!("fbin {op}")),
        };
        Ok(self.fm(v))
    }

    fn ibin(&self, op: &str, a: V<'ctx>, b: V<'ctx>) -> R<V<'ctx>> {
        let (a, b) = (a.into_int_value(), b.into_int_value());
        let bl = &self.b;
        Ok(match op {
            "add" => bl.build_int_add(a, b, "ia").map_err(e)?.into(),
            "sub" => bl.build_int_sub(a, b, "is").map_err(e)?.into(),
            "mul" => bl.build_int_mul(a, b, "im").map_err(e)?.into(),
            "and" => bl.build_and(a, b, "an").map_err(e)?.into(),
            "or" => bl.build_or(a, b, "or").map_err(e)?.into(),
            "xor" => bl.build_xor(a, b, "xo").map_err(e)?.into(),
            _ => return Err(format!("ibin {op}")),
        })
    }

    fn arith(&self, op: &str, a: V<'ctx>, b: V<'ctx>, dt: DType) -> R<V<'ctx>> {
        if dt.is_float() {
            self.fbin(op, a, b)
        } else {
            self.ibin(op, a, b)
        }
    }

    fn select(&self, c: V<'ctx>, t: V<'ctx>, f: V<'ctx>) -> R<V<'ctx>> {
        self.b.build_select(c.into_int_value(), t, f, "sel").map_err(e)
    }

    fn cmp(&self, pred: &str, a: V<'ctx>, b: V<'ctx>, dt: DType) -> R<V<'ctx>> {
        if dt.is_float() {
            let p = match pred {
                "lt" => FloatPredicate::OLT,
                "gt" => FloatPredicate::OGT,
                "le" => FloatPredicate::OLE,
                "ge" => FloatPredicate::OGE,
                "eq" => FloatPredicate::OEQ,
                "neq" => FloatPredicate::UNE,
                _ => return Err(format!("cmp {pred}")),
            };
            Ok(self
                .b
                .build_float_compare(p, a.into_float_value(), b.into_float_value(), "fc")
                .map_err(e)?
                .into())
        } else {
            let s = dt.is_signed_int();
            let p = match (pred, s) {
                ("lt", true) => IntPredicate::SLT,
                ("lt", false) => IntPredicate::ULT,
                ("gt", true) => IntPredicate::SGT,
                ("gt", false) => IntPredicate::UGT,
                ("le", true) => IntPredicate::SLE,
                ("le", false) => IntPredicate::ULE,
                ("ge", true) => IntPredicate::SGE,
                ("ge", false) => IntPredicate::UGE,
                ("eq", _) => IntPredicate::EQ,
                ("neq", _) => IntPredicate::NE,
                _ => return Err(format!("cmp {pred}")),
            };
            Ok(self
                .b
                .build_int_compare(p, a.into_int_value(), b.into_int_value(), "ic")
                .map_err(e)?
                .into())
        }
    }

    fn maxmin(&self, is_max: bool, a: V<'ctx>, b: V<'ctx>, dt: DType) -> R<V<'ctx>> {
        if dt.is_float() {
            // NaN-propagating, like numpy.maximum / numpy.minimum
            self.fintr(if is_max { "llvm.maximum" } else { "llvm.minimum" }, dt, &[a, b])
        } else {
            let c = self.cmp(if is_max { "gt" } else { "lt" }, a, b, dt)?;
            self.select(c, a, b)
        }
    }

    fn neg(&self, a: V<'ctx>, dt: DType) -> R<V<'ctx>> {
        if dt.is_float() {
            Ok(self.fm(self.b.build_float_neg(a.into_float_value(), "n").map_err(e)?.into()))
        } else {
            Ok(self.b.build_int_neg(a.into_int_value(), "n").map_err(e)?.into())
        }
    }

    fn require_float(op: &str, dt: DType) -> R<()> {
        if dt.is_float() {
            Ok(())
        } else {
            Err(format!("scalar op {op} with non-float output {dt:?} not supported"))
        }
    }

    /// Emit one scalar op. `args` come with their own dtypes; `out` is the op's output dtype.
    fn scalar_op(&self, op: &str, args: &[(V<'ctx>, DType)], out: DType) -> R<V<'ctx>> {
        // Most ops compute in the output dtype.
        let conv = |i: usize| -> R<V<'ctx>> { self.cast(args[i].0, args[i].1, out) };
        let one = || self.fconst(out, 1.0);
        let zero = || self.fconst(out, 0.0);
        Ok(match op {
            "add" | "mul" => {
                let mut acc = conv(0)?;
                for i in 1..args.len() {
                    acc = self.arith(op, acc, conv(i)?, out)?;
                }
                acc
            }
            "sub" => self.arith("sub", conv(0)?, conv(1)?, out)?,
            "truediv" => {
                Self::require_float(op, out)?;
                self.fbin("div", conv(0)?, conv(1)?)?
            }
            "intdiv" => {
                if out.is_float() {
                    let q = self.fbin("div", conv(0)?, conv(1)?)?;
                    self.fintr("llvm.floor", out, &[q])?
                } else {
                    // Python floor division
                    let (a, b0) = (conv(0)?.into_int_value(), conv(1)?.into_int_value());
                    // x // 0 == 0 like numpy (division by zero is UB in LLVM)
                    let z0 = b0.get_type().const_zero();
                    let bz = self.b.build_int_compare(IntPredicate::EQ, b0, z0, "bz").map_err(e)?;
                    let b = self.b.build_select(bz, b0.get_type().const_int(1, false), b0, "b1").map_err(e)?.into_int_value();
                    let a = self.b.build_select(bz, z0, a, "a0").map_err(e)?.into_int_value();
                    if out.is_signed_int() {
                        let q = self.b.build_int_signed_div(a, b, "q").map_err(e)?;
                        let r = self.b.build_int_signed_rem(a, b, "r").map_err(e)?;
                        let z = r.get_type().const_zero();
                        let rnz = self.b.build_int_compare(IntPredicate::NE, r, z, "rnz").map_err(e)?;
                        let rneg = self.b.build_int_compare(IntPredicate::SLT, r, z, "rn").map_err(e)?;
                        let bneg = self.b.build_int_compare(IntPredicate::SLT, b, z, "bn").map_err(e)?;
                        let diff = self.b.build_xor(rneg, bneg, "d").map_err(e)?;
                        let adj = self.b.build_and(rnz, diff, "adj").map_err(e)?;
                        let qm1 = self.b.build_int_sub(q, q.get_type().const_int(1, false), "qm").map_err(e)?;
                        self.b.build_select(adj, qm1, q, "fq").map_err(e)?
                    } else {
                        self.b.build_int_unsigned_div(a, b, "q").map_err(e)?.into()
                    }
                }
            }
            "mod" if !out.is_float() => {
                // Python semantics (sign of the divisor); x % 0 == 0 like numpy
                let (a, b0) = (conv(0)?.into_int_value(), conv(1)?.into_int_value());
                let z = b0.get_type().const_zero();
                let bz = self.b.build_int_compare(IntPredicate::EQ, b0, z, "bz").map_err(e)?;
                let b = self.b.build_select(bz, b0.get_type().const_int(1, false), b0, "b1").map_err(e)?.into_int_value();
                let r = if out.is_signed_int() {
                    let r = self.b.build_int_signed_rem(a, b, "r").map_err(e)?;
                    let rnz = self.b.build_int_compare(IntPredicate::NE, r, z, "rnz").map_err(e)?;
                    let rneg = self.b.build_int_compare(IntPredicate::SLT, r, z, "rn").map_err(e)?;
                    let bneg = self.b.build_int_compare(IntPredicate::SLT, b, z, "bn").map_err(e)?;
                    let diff = self.b.build_xor(rneg, bneg, "d").map_err(e)?;
                    let adj = self.b.build_and(rnz, diff, "adj").map_err(e)?;
                    let rb = self.b.build_int_add(r, b, "rb").map_err(e)?;
                    self.b.build_select(adj, rb, r, "fr").map_err(e)?.into_int_value()
                } else {
                    self.b.build_int_unsigned_rem(a, b, "r").map_err(e)?
                };
                self.b.build_select(bz, z, r, "m").map_err(e)?
            }
            "mod" => {
                // Python semantics: a - floor(a/b)*b
                let (a, b) = (conv(0)?, conv(1)?);
                let q = self.fintr("llvm.floor", out, &[self.fbin("div", a, b)?])?;
                self.fbin("sub", a, self.fbin("mul", q, b)?)?
            }
            "pow" => {
                Self::require_float(op, out)?;
                self.fintr("llvm.pow", out, &[conv(0)?, conv(1)?])?
            }
            "maximum" => self.maxmin(true, conv(0)?, conv(1)?, out)?,
            "minimum" => self.maxmin(false, conv(0)?, conv(1)?, out)?,
            "clip" => {
                let x = self.maxmin(true, conv(0)?, conv(1)?, out)?;
                self.maxmin(false, x, conv(2)?, out)?
            }
            "neg" => self.neg(conv(0)?, out)?,
            "abs" => {
                if out.is_float() {
                    self.fintr("llvm.fabs", out, &[conv(0)?])?
                } else if out.is_signed_int() {
                    let x = conv(0)?;
                    let c = self.cmp("lt", x, zero(), out)?;
                    let n = self.neg(x, out)?;
                    self.select(c, n, x)?
                } else {
                    conv(0)?
                }
            }
            "sign" => {
                let x = conv(0)?;
                let pos = self.cmp("gt", x, zero(), out)?;
                let negc = self.cmp("lt", x, zero(), out)?;
                let m1 = self.neg(one(), out)?;
                let t = self.select(negc, m1, x)?; // keeps 0 and NaN
                self.select(pos, one(), t)?
            }
            "sqr" => {
                let x = conv(0)?;
                self.arith("mul", x, x, out)?
            }
            "reciprocal" => {
                Self::require_float(op, out)?;
                self.fbin("div", one(), conv(0)?)?
            }
            "exp2" | "log2" | "log10" => {
                Self::require_float(op, out)?;
                self.libm(op, out, &[conv(0)?])?
            }
            "exp" | "log" | "sqrt" | "sin" | "cos" | "floor"
            | "ceil" | "trunc" | "roundhalftoeven" | "roundhalfawayfromzero" => {
                Self::require_float(op, out)?;
                let name = match op {
                    "roundhalftoeven" => "llvm.roundeven".to_string(),
                    "roundhalfawayfromzero" => "llvm.round".to_string(),
                    _ => format!("llvm.{op}"),
                };
                self.fintr(&name, out, &[conv(0)?])?
            }
            "expm1" | "log1p" | "tan" | "tanh" | "sinh" | "cosh" | "arcsin" | "arccos"
            | "arctan" | "arcsinh" | "arccosh" | "arctanh" => {
                Self::require_float(op, out)?;
                let name = match op {
                    "arcsin" => "asin",
                    "arccos" => "acos",
                    "arctan" => "atan",
                    "arcsinh" => "asinh",
                    "arccosh" => "acosh",
                    "arctanh" => "atanh",
                    o => o,
                };
                self.libm(name, out, &[conv(0)?])?
            }
            "arctan2" => {
                Self::require_float(op, out)?;
                self.libm("atan2", out, &[conv(0)?, conv(1)?])?
            }
            "erf" | "erfc" | "gammaln" | "gamma" | "psi" => {
                Self::require_float(op, out)?;
                self.rustfn(op, out, &[conv(0)?])?
            }
            "sigmoid" => {
                Self::require_float(op, out)?;
                let x = conv(0)?;
                let ex = self.fintr("llvm.exp", out, &[self.neg(x, out)?])?;
                self.fbin("div", one(), self.fbin("add", one(), ex)?)?
            }
            "softplus" => {
                // Same branches as pytensor.scalar.math.Softplus
                Self::require_float(op, out)?;
                let x = conv(0)?;
                let ex = self.fintr("llvm.exp", out, &[x])?;
                let l1pex = self.libm("log1p", out, &[ex])?;
                let emx = self.fintr("llvm.exp", out, &[self.neg(x, out)?])?;
                let xpemx = self.fbin("add", x, emx)?;
                let c1 = self.cmp("lt", x, self.fconst(out, -37.0), out)?;
                let c2 = self.cmp("lt", x, self.fconst(out, 18.0), out)?;
                let c3 = self.cmp("lt", x, self.fconst(out, 33.3), out)?;
                let r3 = self.select(c3, xpemx, x)?;
                let r2 = self.select(c2, l1pex, r3)?;
                self.select(c1, ex, r2)?
            }
            "log1mexp" => {
                Self::require_float(op, out)?;
                let x = conv(0)?;
                let a = self.libm("log1p", out, &[self.neg(self.fintr("llvm.exp", out, &[x])?, out)?])?;
                let em1 = self.libm("expm1", out, &[x])?;
                let b = self.fintr("llvm.log", out, &[self.neg(em1, out)?])?;
                let c = self.cmp("lt", x, self.fconst(out, 0.5f64.ln()), out)?;
                self.select(c, a, b)?
            }
            "lt" | "gt" | "le" | "ge" | "eq" | "neq" => {
                let ct = DType::upcast(args[0].1, args[1].1);
                let a = self.cast(args[0].0, args[0].1, ct)?;
                let b = self.cast(args[1].0, args[1].1, ct)?;
                let c = self.cmp(op, a, b, ct)?;
                self.cast(c, DType::Bool, out)?
            }
            "isnan" | "isinf" => {
                let (x, dt) = args[0];
                if !dt.is_float() {
                    self.constant(out, 0.0, 0)
                } else if op == "isnan" {
                    let xf = x.into_float_value();
                    let c = self
                        .b
                        .build_float_compare(FloatPredicate::UNO, xf, xf, "nan")
                        .map_err(e)?;
                    self.cast(c.into(), DType::Bool, out)?
                } else {
                    let a = self.fintr("llvm.fabs", dt, &[x])?;
                    let c = self.cmp("eq", a, self.fconst(dt, f64::INFINITY), dt)?;
                    self.cast(c, DType::Bool, out)?
                }
            }
            "switch" => {
                let c = self.cast(args[0].0, args[0].1, DType::Bool)?;
                let t = self.cast(args[1].0, args[1].1, out)?;
                let f = self.cast(args[2].0, args[2].1, out)?;
                self.select(c, t, f)?
            }
            "and" | "or" | "xor" => {
                if out.is_float() {
                    return Err(format!("bitop {op} on float"));
                }
                self.ibin(op, conv(0)?, conv(1)?)?
            }
            "invert" => {
                if out.is_float() {
                    return Err("invert on float".into());
                }
                self.b.build_not(conv(0)?.into_int_value(), "inv").map_err(e)?.into()
            }
            "identity" | "cast" => conv(0)?,
            "second" => self.cast(args[1].0, args[1].1, out)?,
            _ => return Err(format!("unsupported scalar op '{op}'")),
        })
    }

    pub fn scalar_graph(&self, g: &ScalarGraph, ins: &[V<'ctx>]) -> R<Vec<(V<'ctx>, DType)>> {
        let mut vals: Vec<(V<'ctx>, DType)> = Vec::with_capacity(g.nodes.len());
        let get = |r: &SRef, vals: &Vec<(V<'ctx>, DType)>| -> (V<'ctx>, DType) {
            match r {
                SRef::In(i) => (ins[*i], g.inputs[*i]),
                SRef::Node(j) => vals[*j],
                SRef::Const { f, i, dtype } => (self.constant(*dtype, *f, *i), *dtype),
            }
        };
        for n in &g.nodes {
            let args: Vec<(V<'ctx>, DType)> = n.inputs.iter().map(|r| get(r, &vals)).collect();
            let v = self.scalar_op(&n.op, &args, n.dtype)?;
            vals.push((v, n.dtype));
        }
        Ok(g.outputs.iter().map(|r| get(r, &vals)).collect())
    }

    // ---------------------------------------------------------------- loops

    /// for i in 0..n { body(i) }
    fn emit_loop(
        &self,
        n: IntValue<'ctx>,
        body: &mut dyn FnMut(IntValue<'ctx>) -> R<()>,
    ) -> R<()> {
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
        let i = phi.as_basic_value().into_int_value();
        body(i)?;
        let latch = self.b.get_insert_block().unwrap();
        let next = self.b.build_int_nsw_add(i, self.i64c(1), "inext").map_err(e)?;
        let c = self.b.build_int_compare(IntPredicate::SLT, next, n, "c").map_err(e)?;
        self.b.build_conditional_branch(c, header, exit).map_err(e)?;
        phi.add_incoming(&[(&next, latch)]);
        self.b.position_at_end(exit);
        Ok(())
    }

    /// accs = inits; for i in 0..n { accs = body(i, accs) }; return accs
    fn emit_multi_acc_loop(
        &self,
        n: IntValue<'ctx>,
        inits: &[V<'ctx>],
        body: &mut dyn FnMut(IntValue<'ctx>, &[V<'ctx>]) -> R<Vec<V<'ctx>>>,
    ) -> R<Vec<V<'ctx>>> {
        let f = self.cur_fn();
        let pre = self.b.get_insert_block().unwrap();
        let header = self.ctx.append_basic_block(f, "mloop");
        let exit = self.ctx.append_basic_block(f, "mloop.exit");
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

    /// acc = init; for i in 0..n { acc = body(i, acc) }; return acc
    fn emit_acc_loop(
        &self,
        n: IntValue<'ctx>,
        init: V<'ctx>,
        body: &mut dyn FnMut(IntValue<'ctx>, V<'ctx>) -> R<V<'ctx>>,
    ) -> R<V<'ctx>> {
        let f = self.cur_fn();
        let pre = self.b.get_insert_block().unwrap();
        let header = self.ctx.append_basic_block(f, "rloop");
        let exit = self.ctx.append_basic_block(f, "rloop.exit");
        let zero = self.i64c(0);
        let c = self.b.build_int_compare(IntPredicate::SGT, n, zero, "nz").map_err(e)?;
        self.b.build_conditional_branch(c, header, exit).map_err(e)?;
        self.b.position_at_end(header);
        let phi = self.b.build_phi(self.i64t(), "i").map_err(e)?;
        let acc = self.b.build_phi(init.get_type(), "acc").map_err(e)?;
        phi.add_incoming(&[(&zero, pre)]);
        acc.add_incoming(&[(&init, pre)]);
        let i = phi.as_basic_value().into_int_value();
        let new_acc = body(i, acc.as_basic_value())?;
        let latch = self.b.get_insert_block().unwrap();
        let next = self.b.build_int_nsw_add(i, self.i64c(1), "inext").map_err(e)?;
        let c = self.b.build_int_compare(IntPredicate::SLT, next, n, "c").map_err(e)?;
        self.b.build_conditional_branch(c, header, exit).map_err(e)?;
        phi.add_incoming(&[(&next, latch)]);
        acc.add_incoming(&[(&new_acc, latch)]);
        self.b.position_at_end(exit);
        let res = self.b.build_phi(init.get_type(), "racc").map_err(e)?;
        res.add_incoming(&[(&init, pre), (&new_acc, latch)]);
        Ok(res.as_basic_value())
    }

    // ---------------------------------------------------------------- function scaffolding

    /// Emit `i32 name(ptr args, ptr cfg)` that unpacks the ArrayArgs and calls a private
    /// `alwaysinline` body taking `n_data` noalias data pointers followed by
    /// `n_meta` i64* pointers (shape / strides). `pick` maps body params to
    /// (arg index, field: 0=data 1=shape 2=strides).
    fn scaffold(
        &self,
        name: &str,
        params: &[(usize, usize)],
        n_data: usize,
        build_body: &mut dyn FnMut(&[PointerValue<'ctx>]) -> R<()>,
    ) -> R<()> {
        let ptr = self.ptrt();
        let i32t = self.ctx.i32_type();
        let linkage = if self.internal.get() { Linkage::Private } else { Linkage::External };
        let wrap = self.module.add_function(name, i32t.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false), Some(linkage));
        if self.internal.get() {
            let k = Attribute::get_named_enum_kind_id("alwaysinline");
            wrap.add_attribute(AttributeLoc::Function, self.ctx.create_enum_attribute(k, 0));
        }
        let ptys: Vec<BasicMetadataTypeEnum> = params.iter().map(|_| ptr.into()).collect();
        let body = self.module.add_function(
            &format!("{name}.body"),
            i32t.fn_type(&ptys, false), // 0 = ok; kernels may return an error code
            Some(Linkage::Private),
        );
        let attr = |n: &str| self.ctx.create_enum_attribute(Attribute::get_named_enum_kind_id(n), 0);
        body.add_attribute(AttributeLoc::Function, attr("alwaysinline"));
        body.add_attribute(AttributeLoc::Function, attr("nounwind"));
        for i in 0..params.len() {
            body.add_attribute(AttributeLoc::Param(i as u32), attr("nocapture"));
            if i < n_data {
                body.add_attribute(AttributeLoc::Param(i as u32), attr("noalias"));
            } else {
                body.add_attribute(AttributeLoc::Param(i as u32), attr("readonly"));
            }
        }
        wrap.add_attribute(AttributeLoc::Function, attr("nounwind"));

        // wrapper
        let bb = self.ctx.append_basic_block(wrap, "entry");
        self.b.position_at_end(bb);
        // (ctx, args, params): generated kernels only read `args` so far
        let args_ptr = wrap.get_nth_param(1).unwrap().into_pointer_value();
        let mut vals: Vec<BasicMetadataValueEnum> = Vec::new();
        for &(arg, field) in params {
            let off = self.i64c((arg * PTARRAY_SIZE + field * 8) as i64);
            let p = self.gep_bytes(args_ptr, off)?;
            vals.push(self.b.build_load(ptr, p, "a").map_err(e)?.into());
        }
        let rc = self.b.build_call(body, &vals, "rc").map_err(e)?.try_as_basic_value().basic().unwrap();
        self.b.build_return(Some(&rc)).map_err(e)?;

        // body
        let bb = self.ctx.append_basic_block(body, "entry");
        self.b.position_at_end(bb);
        let ps: Vec<PointerValue> = body
            .get_param_iter()
            .map(|p| p.into_pointer_value())
            .collect();
        build_body(&ps)?;
        self.b.build_return(Some(&i32t.const_zero())).map_err(e)?;
        Ok(())
    }

    // ---------------------------------------------------------------- elemwise

    pub fn gen_kernel(&self, name: &str, spec: &KernelSpec) -> R<()> {
        self.gen_kernel_inner(name, spec)?;
        self.finish_used();
        Ok(())
    }

    fn gen_kernel_inner(&self, name: &str, spec: &KernelSpec) -> R<()> {
        match spec {
            KernelSpec::Elemwise {
                ndim,
                inputs,
                outputs,
                inplace,
                scalar,
                ..
            } => self.gen_elemwise(name, *ndim, inputs, outputs, inplace, scalar),
            KernelSpec::Reduce {
                op,
                axes,
                input,
                output,
                ..
            } => self.gen_reduce(name, *op, axes, input, output.dtype),
            KernelSpec::Segment(seg) => self.gen_segment(name, seg),
            KernelSpec::Fused { ndim, inputs, n_elem, reads, outputs, outs, dim_src, scalar, .. } => {
                self.gen_fused(name, *ndim, inputs, *n_elem, reads, outputs, outs, dim_src, scalar)
            }
        }
    }

    fn gen_elemwise(
        &self,
        name: &str,
        ndim: usize,
        inputs: &[Operand],
        out_ops: &[Operand],
        inplace: &[Option<usize>],
        scalar: &ScalarGraph,
    ) -> R<()> {
        let outputs: Vec<DType> = out_ops.iter().map(|o| o.dtype).collect();
        let outputs = &outputs[..];
        // All outputs share one layout, chosen by the planner; it decides loop order.
        let layout = out_ops[0].layout;
        let order = layout.order(ndim); // outermost .. innermost
        let ni = inputs.len();
        // Operands that need their own data pointer: all inputs, plus non-inplace outputs.
        let own_out: Vec<usize> = (0..outputs.len()).filter(|k| inplace[*k].is_none()).collect();
        let n_ops = ni + own_out.len();
        // out_bcast[d]: every input is broadcastable in d, so the output dim is 1.
        let out_bcast: Vec<bool> = (0..ndim).map(|d| inputs.iter().all(|o| o.bcast[d])).collect();
        // effective broadcast: dims where an input is repeated along the output
        let eff = |o: &Operand, d: usize| o.bcast[d] && !out_bcast[d];
        #[derive(PartialEq, Clone, Copy)]
        enum Kind {
            Scalar,
            Full,
            Partial,
        }
        let kinds: Vec<Kind> = inputs
            .iter()
            .map(|o| {
                let nb = (0..ndim).filter(|&d| eff(o, d)).count();
                let nreal = (0..ndim).filter(|&d| !out_bcast[d]).count();
                if nb == 0 {
                    Kind::Full
                } else if nb == nreal {
                    Kind::Scalar
                } else {
                    Kind::Partial
                }
            })
            .collect();
        let any_partial = kinds.iter().any(|k| *k == Kind::Partial);
        let need_check: Vec<usize> = (0..ni)
            .filter(|&i| kinds[i] == Kind::Full && !inputs[i].layout.contiguous_as(layout, ndim))
            .collect();
        let emit_fast = !any_partial;
        let emit_generic = any_partial || !need_check.is_empty();

        // body params: data[0..n_ops], shape (of first output), strides[0..n_ops]
        let first_out_arg = ni;
        let mut params: Vec<(usize, usize)> = Vec::new();
        for i in 0..ni {
            params.push((i, 0));
        }
        for &k in &own_out {
            params.push((ni + k, 0));
        }
        params.push((first_out_arg, 1));
        for i in 0..ni {
            params.push((i, 2));
        }
        for &k in &own_out {
            params.push((ni + k, 2));
        }

        // operand index (into data/strides) where output k is written
        let target: Vec<usize> = (0..outputs.len())
            .map(|k| match inplace[k] {
                Some(i) => i,
                None => ni + own_out.iter().position(|x| *x == k).unwrap(),
            })
            .collect();

        self.scaffold(name, &params, n_ops, &mut |ps| {
            let data = &ps[0..n_ops];
            let shape_p = ps[n_ops];
            let strides = &ps[n_ops + 1..];
            let shape: Vec<IntValue> = (0..ndim)
                .map(|d| self.load_i64_at(shape_p, d))
                .collect::<R<_>>()?;
            // scalar operands are loaded once
            let mut scal: Vec<Option<V>> = vec![None; ni];
            for i in 0..ni {
                if kinds[i] == Kind::Scalar {
                    scal[i] = Some(self.load_elem(data[i], inputs[i].dtype)?);
                }
            }
            let f = self.cur_fn();
            let done = self.ctx.append_basic_block(f, "done");
            let fast_bb = self.ctx.append_basic_block(f, "fast");
            let gen_bb = self.ctx.append_basic_block(f, "strided");

            // dispatch
            if emit_fast && emit_generic {
                let mut ok = self.ctx.bool_type().const_int(1, false);
                for &i in &need_check {
                    let isz = inputs[i].dtype.itemsize() as i64;
                    let mut expected = self.i64c(isz);
                    for &d in order.iter().rev() {
                        let st = self.load_i64_at(strides[i], d)?;
                        let is1 = self.b.build_int_compare(IntPredicate::EQ, shape[d], self.i64c(1), "is1").map_err(e)?;
                        let eq = self.b.build_int_compare(IntPredicate::EQ, st, expected, "eq").map_err(e)?;
                        let good = self.b.build_or(is1, eq, "g").map_err(e)?;
                        ok = self.b.build_and(ok, good, "ok").map_err(e)?;
                        expected = self.b.build_int_nsw_mul(expected, shape[d], "ex").map_err(e)?;
                    }
                }
                self.b.build_conditional_branch(ok, fast_bb, gen_bb).map_err(e)?;
            } else if emit_fast {
                self.b.build_unconditional_branch(fast_bb).map_err(e)?;
            } else {
                self.b.build_unconditional_branch(gen_bb).map_err(e)?;
            }

            let compute = |ins: Vec<V<'ctx>>| -> R<Vec<V<'ctx>>> {
                let outs = self.scalar_graph(scalar, &ins)?;
                outs.iter()
                    .zip(outputs)
                    .map(|((v, dt), odt)| self.cast(*v, *dt, *odt))
                    .collect()
            };

            // fast path: flat loop over memory that is contiguous in the output's order
            self.b.position_at_end(fast_bb);
            if emit_fast {
                let mut total = self.i64c(1);
                for d in 0..ndim {
                    total = self.b.build_int_nsw_mul(total, shape[d], "tot").map_err(e)?;
                }
                self.emit_loop(total, &mut |i| {
                    let mut ins = Vec::with_capacity(ni);
                    for j in 0..ni {
                        ins.push(match scal[j] {
                            Some(v) => v,
                            None => self.load_elem(self.gep_elem(inputs[j].dtype, data[j], i)?, inputs[j].dtype)?,
                        });
                    }
                    let outs = compute(ins)?;
                    for (k, v) in outs.into_iter().enumerate() {
                        let p = self.gep_elem(outputs[k], data[target[k]], i)?;
                        self.store_elem(p, v, outputs[k])?;
                    }
                    Ok(())
                })?;
            }
            self.b.build_unconditional_branch(done).map_err(e)?;

            // generic strided path
            self.b.position_at_end(gen_bb);
            if emit_generic {
                // per-operand, per-dim byte strides (0 where broadcast)
                let mut st: Vec<Vec<Option<IntValue>>> = Vec::new();
                for j in 0..n_ops {
                    let mut v = Vec::new();
                    for d in 0..ndim {
                        let bc = j < ni && (eff(&inputs[j], d) || kinds[j] == Kind::Scalar);
                        v.push(if bc || out_bcast[d] { None } else { Some(self.load_i64_at(strides[j], d)?) });
                    }
                    st.push(v);
                }
                let zero = self.i64c(0);
                self.gen_nest(0, &order, &shape, &st, vec![zero; n_ops], &mut |offs| {
                    let mut ins = Vec::with_capacity(ni);
                    for j in 0..ni {
                        ins.push(match scal[j] {
                            Some(v) => v,
                            None => self.load_elem(self.gep_bytes(data[j], offs[j])?, inputs[j].dtype)?,
                        });
                    }
                    let outs = compute(ins)?;
                    for (k, v) in outs.into_iter().enumerate() {
                        let t = target[k];
                        self.store_elem(self.gep_bytes(data[t], offs[t])?, v, outputs[k])?;
                    }
                    Ok(())
                })?;
            }
            self.b.build_unconditional_branch(done).map_err(e)?;
            self.b.position_at_end(done);
            Ok(())
        })
    }

    /// Nested loops over `shape[d..]`, carrying per-operand byte offsets.
    fn gen_nest(
        &self,
        pos: usize,
        order: &[usize],
        shape: &[IntValue<'ctx>],
        st: &[Vec<Option<IntValue<'ctx>>>],
        offs: Vec<IntValue<'ctx>>,
        body: &mut dyn FnMut(&[IntValue<'ctx>]) -> R<()>,
    ) -> R<()> {
        if pos == order.len() {
            return body(&offs);
        }
        let d = order[pos];
        self.emit_loop(shape[d], &mut |i| {
            let mut new = offs.clone();
            for j in 0..offs.len() {
                if let Some(s) = st[j][d] {
                    let m = self.b.build_int_nsw_mul(i, s, "m").map_err(e)?;
                    new[j] = self.b.build_int_nsw_add(offs[j], m, "o").map_err(e)?;
                }
            }
            self.gen_nest(pos + 1, order, shape, st, new, body)
        })
    }

    // ---------------------------------------------------------------- reduce

    fn reduce_identity(&self, op: ReduceOp, dt: DType) -> V<'ctx> {
        let (f, i) = match op {
            ReduceOp::Add | ReduceOp::Or => (0.0, 0),
            ReduceOp::Mul | ReduceOp::And => (1.0, 1),
            ReduceOp::Max => (
                f64::NEG_INFINITY,
                match dt {
                    DType::Int64 => i64::MIN,
                    DType::Int32 => i32::MIN as i64,
                    DType::Int16 => i16::MIN as i64,
                    DType::Int8 => i8::MIN as i64,
                    _ => 0,
                },
            ),
            ReduceOp::Min => (
                f64::INFINITY,
                match dt {
                    DType::Int64 => i64::MAX,
                    DType::Int32 => i32::MAX as i64,
                    DType::Int16 => i16::MAX as i64,
                    DType::Int8 => i8::MAX as i64,
                    _ => -1, // all bits set for unsigned / bool
                },
            ),
        };
        if op == ReduceOp::And && dt != DType::Bool && !dt.is_float() {
            return self.constant(dt, 0.0, -1);
        }
        self.constant(dt, f, i)
    }

    fn reduce_combine(&self, op: ReduceOp, a: V<'ctx>, b: V<'ctx>, dt: DType) -> R<V<'ctx>> {
        match op {
            ReduceOp::Add => self.arith("add", a, b, dt),
            ReduceOp::Mul => self.arith("mul", a, b, dt),
            ReduceOp::Max => self.maxmin(true, a, b, dt),
            ReduceOp::Min => self.maxmin(false, a, b, dt),
            ReduceOp::And => self.ibin("and", a, b),
            ReduceOp::Or => self.ibin("or", a, b),
        }
    }

    fn gen_reduce(&self, name: &str, op: ReduceOp, axes: &[usize], input: &Operand, odt: DType) -> R<()> {
        let ndim = input.bcast.len();
        let idt = input.dtype;
        let kept: Vec<usize> = (0..ndim).filter(|d| !axes.contains(d)).collect();
        // params: din, dout, in_shape, in_strides, out_strides
        let params = vec![(0, 0), (1, 0), (0, 1), (0, 2), (1, 2)];
        self.scaffold(name, &params, 2, &mut |ps| {
            let (din, dout, shp, ist, ost) = (ps[0], ps[1], ps[2], ps[3], ps[4]);
            let shape: Vec<IntValue> = (0..ndim).map(|d| self.load_i64_at(shp, d)).collect::<R<_>>()?;
            // init output with the identity (output is provided C-contiguous)
            let mut nout = self.i64c(1);
            for &d in &kept {
                nout = self.b.build_int_nsw_mul(nout, shape[d], "no").map_err(e)?;
            }
            let ident = self.reduce_identity(op, odt);
            self.emit_loop(nout, &mut |i| self.store_elem(self.gep_elem(odt, dout, i)?, ident, odt))?;

            let mut st_in = Vec::new();
            let mut st_out = Vec::new();
            for d in 0..ndim {
                st_in.push(self.load_i64_at(ist, d)?);
                st_out.push(match kept.iter().position(|k| *k == d) {
                    Some(p) => Some(self.load_i64_at(ost, p)?),
                    None => None,
                });
            }
            let load_x = |off: IntValue<'ctx>| -> R<V<'ctx>> {
                let x = self.load_elem(self.gep_bytes(din, off)?, idt)?;
                self.cast(x, idt, odt)
            };
            if ndim == 0 {
                let x = load_x(self.i64c(0))?;
                let r = self.reduce_combine(op, ident, x, odt)?;
                return self.store_elem(dout, r, odt);
            }
            // loop order follows the input's memory order when it is known
            let order = input.layout.order(ndim);
            let inner = order[ndim - 1];
            let inner_reduced = st_out[inner].is_none();
            let sin = st_in[inner];
            // Version the whole nest on "innermost input stride == itemsize" so the inner
            // loop uses plain vector loads instead of gathers.
            let f = self.cur_fn();
            let contig_bb = self.ctx.append_basic_block(f, "inner.contig");
            let strided_bb = self.ctx.append_basic_block(f, "inner.strided");
            let done_bb = self.ctx.append_basic_block(f, "reduce.done");
            let isz = self.i64c(idt.itemsize() as i64);
            let is_c = if input.layout != Layout::Any {
                self.ctx.bool_type().const_int(1, false)
            } else {
                self.b.build_int_compare(IntPredicate::EQ, sin, isz, "ic").map_err(e)?
            };
            self.b.build_conditional_branch(is_c, contig_bb, strided_bb).map_err(e)?;
            for contig in [true, false] {
                self.b.position_at_end(if contig { contig_bb } else { strided_bb });
                let addr = |oi: IntValue<'ctx>, i: IntValue<'ctx>| -> R<IntValue<'ctx>> {
                    let s = if contig { isz } else { sin };
                    self.b.build_int_nsw_add(oi, self.b.build_int_nsw_mul(i, s, "m").map_err(e)?, "o").map_err(e)
                };
                self.reduce_nest(0, &order, &shape, &st_in, &st_out, self.i64c(0), self.i64c(0), &mut |oi, oo| {
                    let optr = self.gep_bytes(dout, oo)?;
                    if inner_reduced {
                        let acc0 = self.load_elem(optr, odt)?;
                        let acc = self.emit_acc_loop(shape[inner], acc0, &mut |i, acc| {
                            self.reduce_combine(op, acc, load_x(addr(oi, i)?)?, odt)
                        })?;
                        self.store_elem(optr, acc, odt)
                    } else {
                        let sout = st_out[inner].unwrap();
                        self.emit_loop(shape[inner], &mut |i| {
                            let offo = self.b.build_int_nsw_add(oo, self.b.build_int_nsw_mul(i, sout, "m").map_err(e)?, "o").map_err(e)?;
                            let p = self.gep_bytes(dout, offo)?;
                            let a = self.load_elem(p, odt)?;
                            let r = self.reduce_combine(op, a, load_x(addr(oi, i)?)?, odt)?;
                            self.store_elem(p, r, odt)
                        })
                    }
                })?;
                self.b.build_unconditional_branch(done_bb).map_err(e)?;
            }
            self.b.position_at_end(done_bb);
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_nest(
        &self,
        pos: usize,
        order: &[usize],
        shape: &[IntValue<'ctx>],
        st_in: &[IntValue<'ctx>],
        st_out: &[Option<IntValue<'ctx>>],
        oi: IntValue<'ctx>,
        oo: IntValue<'ctx>,
        body: &mut dyn FnMut(IntValue<'ctx>, IntValue<'ctx>) -> R<()>,
    ) -> R<()> {
        if pos + 1 == order.len() {
            return body(oi, oo);
        }
        let d = order[pos];
        self.emit_loop(shape[d], &mut |i| {
            let ni = self.b.build_int_nsw_add(oi, self.b.build_int_nsw_mul(i, st_in[d], "m").map_err(e)?, "o").map_err(e)?;
            let no = match st_out[d] {
                Some(s) => self.b.build_int_nsw_add(oo, self.b.build_int_nsw_mul(i, s, "m").map_err(e)?, "o").map_err(e)?,
                None => oo,
            };
            self.reduce_nest(pos + 1, order, shape, st_in, st_out, ni, no, body)
        })
    }

    // ---------------------------------------------------------------- segments

    /// One function for a whole segment. Member kernels are private `alwaysinline`
    /// functions; every segment value gets a PtArray struct on the stack, so after
    /// inlining SROA turns intermediates into registers. Returns member index + 1 on
    /// failure.
    fn gen_segment(&self, name: &str, seg: &SegmentSpec) -> R<()> {
        // member kernels first
        self.internal.set(true);
        let mut member_fns: Vec<Option<FunctionValue<'ctx>>> = Vec::new();
        for (j, m) in seg.members.iter().enumerate() {
            if let SegMember::Kernel { spec, .. } = m {
                let mname = format!("{name}.m{j}");
                self.gen_kernel_inner(&mname, spec)?;
                member_fns.push(self.module.get_function(&mname));
            } else {
                member_fns.push(None);
            }
        }
        self.internal.set(false);

        let ptr = self.ptrt();
        let i32t = self.ctx.i32_type();
        let i64t = self.i64t();
        let f = self.module.add_function(name, i32t.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false), Some(Linkage::External));
        let entry = self.ctx.append_basic_block(f, "entry");
        self.b.position_at_end(entry);
        let ctx_p = f.get_nth_param(0).unwrap().into_pointer_value();
        let args_p = f.get_nth_param(1).unwrap().into_pointer_value();
        let arr_t = self.ctx.struct_type(&[ptr.into(), ptr.into(), ptr.into(), ptr.into(), i64t.into()], false);

        let field = |s: PointerValue<'ctx>, k: usize| -> R<PointerValue<'ctx>> {
            let p = self.gep_bytes(s, self.i64c(8 * k as i64))?;
            Ok(self.b.build_load(ptr, p, "f").map_err(e)?.into_pointer_value())
        };
        let i64_array = |vals: &[IntValue<'ctx>]| -> R<PointerValue<'ctx>> {
            let a = self.b.build_array_alloca(i64t, self.i64c(vals.len().max(1) as i64), "dims").map_err(e)?;
            for (k, v) in vals.iter().enumerate() {
                self.b.build_store(self.gep_elem(DType::Int64, a, self.i64c(k as i64))?, *v).map_err(e)?;
            }
            Ok(a)
        };
        let make_struct = |data: PointerValue<'ctx>, shape: PointerValue<'ctx>, strides: PointerValue<'ctx>, ndim: usize| -> R<PointerValue<'ctx>> {
            let s = self.b.build_alloca(arr_t, "arr").map_err(e)?;
            let vals: [BasicValueEnum; 5] = [data.into(), shape.into(), strides.into(), ptr.const_null().into(), self.i64c(ndim as i64).into()];
            for (k, v) in vals.iter().enumerate() {
                self.b.build_store(self.gep_bytes(s, self.i64c(8 * k as i64))?, *v).map_err(e)?;
            }
            Ok(s)
        };

        // values (all allocas live in the entry block)
        let mut vp: Vec<PointerValue<'ctx>> = Vec::with_capacity(seg.values.len());
        for v in &seg.values {
            let p = match &v.loc {
                SegLoc::ExtIn(i) => self.gep_bytes(args_p, self.i64c((i * PTARRAY_SIZE) as i64))?,
                SegLoc::ExtOut(k) => self.gep_bytes(args_p, self.i64c(((seg.n_in + k) * PTARRAY_SIZE) as i64))?,
                SegLoc::Temp => {
                    let shape = v.shape.as_ref().ok_or("temporary without static shape")?;
                    let n: i64 = shape.iter().product();
                    let isz = v.dtype.itemsize() as i64;
                    let bytes = (n * isz).max(8) as u32;
                    let data = self.b.build_alloca(self.ctx.i8_type().array_type(bytes), "tmp").map_err(e)?;
                    if let Some(inst) = data.as_instruction() {
                        let _ = inst.set_alignment(64);
                    }
                    let mut strides = vec![0i64; shape.len()];
                    Layout::C.strides(shape, isz, &mut strides);
                    let sh: Vec<IntValue> = shape.iter().map(|&x| self.i64c(x)).collect();
                    let st: Vec<IntValue> = strides.iter().map(|&x| self.i64c(x)).collect();
                    make_struct(data, i64_array(&sh)?, i64_array(&st)?, v.ndim)?
                }
                SegLoc::View { base, order } => {
                    let b = vp[*base];
                    let (bd, bsh, bst) = (field(b, 0)?, field(b, 1)?, field(b, 2)?);
                    let mut sh = Vec::new();
                    let mut st = Vec::new();
                    for o in order {
                        match o {
                            Some(d) => {
                                sh.push(self.load_i64_at(bsh, *d)?);
                                st.push(self.load_i64_at(bst, *d)?);
                            }
                            None => {
                                sh.push(self.i64c(1));
                                st.push(self.i64c(0));
                            }
                        }
                    }
                    make_struct(bd, i64_array(&sh)?, i64_array(&st)?, v.ndim)?
                }
            };
            vp.push(p);
        }
        // member argument arrays
        let mut member_args: Vec<Option<PointerValue<'ctx>>> = Vec::new();
        for m in &seg.members {
            if let SegMember::Kernel { args, .. } = m {
                let arr = self.b.build_array_alloca(arr_t, self.i64c(args.len() as i64), "margs").map_err(e)?;
                for (k, &a) in args.iter().enumerate() {
                    let val = self.b.build_load(arr_t, vp[a], "a").map_err(e)?;
                    let dst = unsafe { self.b.build_in_bounds_gep(arr_t, arr, &[self.i64c(k as i64)], "d").map_err(e)? };
                    self.b.build_store(dst, val).map_err(e)?;
                }
                member_args.push(Some(arr));
            } else {
                member_args.push(None);
            }
        }

        let fail_with = |code: usize, cond: IntValue<'ctx>| -> R<()> {
            let fail = self.ctx.append_basic_block(f, "fail");
            let cont = self.ctx.append_basic_block(f, "cont");
            self.b.build_conditional_branch(cond, fail, cont).map_err(e)?;
            self.b.position_at_end(fail);
            self.b.build_return(Some(&i32t.const_int(code as u64, false))).map_err(e)?;
            self.b.position_at_end(cont);
            Ok(())
        };
        for (j, m) in seg.members.iter().enumerate() {
            match m {
                SegMember::Kernel { .. } => {
                    let call = self
                        .b
                        .build_call(member_fns[j].unwrap(), &[ctx_p.into(), member_args[j].unwrap().into(), ptr.const_null().into()], "rc")
                        .map_err(e)?;
                    let rc = call.try_as_basic_value().basic().unwrap().into_int_value();
                    let bad = self.b.build_int_compare(IntPredicate::NE, rc, i32t.const_zero(), "bad").map_err(e)?;
                    fail_with(j + 1, bad)?;
                }
                SegMember::ShapeI { i, input, output } => {
                    let v = self.load_i64_at(field(vp[*input], 1)?, *i)?;
                    self.b.build_store(field(vp[*output], 0)?, v).map_err(e)?;
                }
                SegMember::MakeVector { inputs, output } => {
                    let out = &seg.values[*output];
                    let od = field(vp[*output], 0)?;
                    let os = self.load_i64_at(field(vp[*output], 2)?, 0)?;
                    for (k, &x) in inputs.iter().enumerate() {
                        let val = self.load_elem(field(vp[x], 0)?, seg.values[x].dtype)?;
                        let off = self.b.build_int_nsw_mul(self.i64c(k as i64), os, "o").map_err(e)?;
                        self.store_elem(self.gep_bytes(od, off)?, val, out.dtype)?;
                    }
                }
                SegMember::Assert { conds } => {
                    for &c in conds {
                        let b = self.b.build_load(self.ctx.i8_type(), field(vp[c], 0)?, "c").map_err(e)?.into_int_value();
                        let z = self.b.build_int_compare(IntPredicate::EQ, b, self.ctx.i8_type().const_zero(), "z").map_err(e)?;
                        fail_with(j + 1, z)?;
                    }
                }
            }
        }
        self.b.build_return(Some(&i32t.const_zero())).map_err(e)?;
        Ok(())
    }

    // ---------------------------------------------------------------- fused indexed elemwise

    /// Loop nest over `order`, where `step(dim, i, offs)` returns the operand offsets
    /// one level deeper.
    #[allow(clippy::type_complexity)]
    fn nest_custom(
        &self,
        pos: usize,
        order: &[usize],
        dims: &[IntValue<'ctx>],
        offs: Vec<IntValue<'ctx>>,
        step: &mut dyn FnMut(usize, IntValue<'ctx>, &[IntValue<'ctx>]) -> R<Vec<IntValue<'ctx>>>,
        body: &mut dyn FnMut(&[IntValue<'ctx>]) -> R<()>,
    ) -> R<()> {
        if pos == order.len() {
            return body(&offs);
        }
        let d = order[pos];
        self.emit_loop(dims[d], &mut |i| {
            let new = step(d, i, &offs)?;
            self.nest_custom(pos + 1, order, dims, new, step, body)
        })
    }

    /// One Elemwise loop with indexed reads (`source[.., idx[i], ..]`), indexed writes
    /// (`out = target; out[.., idx[i], ..] (+)= value`) and reductions into the outputs.
    /// Returns 1 if an index is out of bounds.
    #[allow(clippy::too_many_arguments)]
    fn gen_fused(
        &self,
        name: &str,
        ndim: usize,
        inputs: &[Operand],
        n_elem: usize,
        reads: &[Option<IndexedAxis>],
        outputs: &[Operand],
        outs: &[FusedOut],
        dim_src: &[Option<(usize, usize)>],
        scalar: &ScalarGraph,
    ) -> R<()> {
        let ni = inputs.len();
        let na = ni + outputs.len();
        let mut params: Vec<(usize, usize)> = Vec::new();
        for field in 0..3 {
            for a in 0..na {
                params.push((a, field));
            }
        }
        let operand = |a: usize| if a < ni { &inputs[a] } else { &outputs[a - ni] };
        self.scaffold(name, &params, na, &mut |ps| {
            let (data, shp, strd) = (&ps[0..na], &ps[na..2 * na], &ps[2 * na..3 * na]);
            let f = self.cur_fn();
            let fail = self.ctx.append_basic_block(f, "oob");
            let dims: Vec<IntValue> = dim_src
                .iter()
                .map(|s| match s {
                    Some((a, d)) => self.load_i64_at(shp[*a], *d),
                    None => Ok(self.i64c(1)),
                })
                .collect::<R<_>>()?;
            // 0-d (scalar) Elemwise inputs are loaded once
            let mut scal: Vec<Option<V>> = vec![None; n_elem];
            for j in 0..n_elem {
                if reads[j].is_none() && inputs[j].bcast.len() == 0 && ndim > 0 {
                    scal[j] = Some(self.load_elem(data[j], inputs[j].dtype)?);
                }
            }
            // initialize outputs: reductions get the identity, writes a copy of the target
            for (k, o) in outs.iter().enumerate() {
                let a = ni + k;
                let odt = outputs[k].dtype;
                match o {
                    FusedOut::Reduce { op, axes } => {
                        let mut n = self.i64c(1);
                        for d in 0..(ndim - axes.len()) {
                            n = self.b.build_int_nsw_mul(n, self.load_i64_at(shp[a], d)?, "n").map_err(e)?;
                        }
                        let ident = self.reduce_identity(*op, odt);
                        self.emit_loop(n, &mut |i| self.store_elem(self.gep_elem(odt, data[a], i)?, ident, odt))?;
                    }
                    FusedOut::Write { target, .. } => {
                        let t = *target;
                        let same = self.b.build_ptr_diff(self.ctx.i8_type(), data[t], data[a], "d").map_err(e)?;
                        let is_same = self.b.build_int_compare(IntPredicate::EQ, same, self.i64c(0), "same").map_err(e)?;
                        let copy_bb = self.ctx.append_basic_block(f, "copy");
                        let after = self.ctx.append_basic_block(f, "copied");
                        self.b.build_conditional_branch(is_same, after, copy_bb).map_err(e)?;
                        self.b.position_at_end(copy_bb);
                        let tdims: Vec<IntValue> = (0..ndim).map(|d| self.load_i64_at(shp[t], d)).collect::<R<_>>()?;
                        let st: Vec<Vec<Option<IntValue>>> = [t, a]
                            .iter()
                            .map(|&x| (0..ndim).map(|d| self.load_i64_at(strd[x], d).map(Some)).collect::<R<Vec<_>>>())
                            .collect::<R<_>>()?;
                        let order: Vec<usize> = (0..ndim).collect();
                        let z = self.i64c(0);
                        self.gen_nest(0, &order, &tdims, &st, vec![z, z], &mut |offs| {
                            let v = self.load_elem(self.gep_bytes(data[t], offs[0])?, odt)?;
                            self.store_elem(self.gep_bytes(data[a], offs[1])?, v, odt)
                        })?;
                        self.b.build_unconditional_branch(after).map_err(e)?;
                        self.b.position_at_end(after);
                    }
                    FusedOut::Plain => {}
                }
            }

            // how each operand's offset advances per loop dim
            #[derive(Clone, Copy)]
            enum Adv<'c> {
                None,
                Stride(IntValue<'c>),
                /// through index array `idx`, bounded by `len`
                Indexed { idx: usize, stride: IntValue<'c>, len: IntValue<'c> },
            }
            let mut ops: Vec<usize> = Vec::new(); // argument per operand
            let mut adv: Vec<Vec<Adv>> = Vec::new();
            for j in 0..n_elem {
                if scal[j].is_some() {
                    continue;
                }
                let o = operand(j);
                let mut v = Vec::new();
                for d in 0..ndim {
                    v.push(match &reads[j] {
                        Some(r) if r.axis == d => Adv::Indexed {
                            idx: r.idx,
                            stride: self.load_i64_at(strd[j], d)?,
                            len: self.load_i64_at(shp[j], d)?,
                        },
                        _ if o.bcast.get(d).copied().unwrap_or(true) => Adv::None,
                        _ => Adv::Stride(self.load_i64_at(strd[j], d)?),
                    });
                }
                ops.push(j);
                adv.push(v);
            }
            let first_out = ops.len();
            for (k, o) in outs.iter().enumerate() {
                let a = ni + k;
                let mut v = Vec::new();
                for d in 0..ndim {
                    v.push(match o {
                        FusedOut::Plain => Adv::Stride(self.load_i64_at(strd[a], d)?),
                        FusedOut::Reduce { axes, .. } => {
                            if axes.contains(&d) {
                                Adv::None
                            } else {
                                let pos = (0..d).filter(|x| !axes.contains(x)).count();
                                Adv::Stride(self.load_i64_at(strd[a], pos)?)
                            }
                        }
                        FusedOut::Write { at, .. } if at.axis == d => Adv::Indexed {
                            idx: at.idx,
                            stride: self.load_i64_at(strd[a], d)?,
                            len: self.load_i64_at(shp[a], d)?,
                        },
                        FusedOut::Write { .. } => Adv::Stride(self.load_i64_at(strd[a], d)?),
                    });
                }
                ops.push(a);
                adv.push(v);
            }
            // index arrays: load idx[i] once per loop level
            let mut idx_stride: Vec<Option<IntValue>> = vec![None; ni];
            for a in adv.iter().flatten() {
                if let Adv::Indexed { idx, .. } = a {
                    if idx_stride[*idx].is_none() {
                        idx_stride[*idx] = Some(self.load_i64_at(strd[*idx], 0)?);
                    }
                }
            }
            let order: Vec<usize> = (0..ndim).collect();
            let zero = self.i64c(0);
            let mut step = |d: usize, i: IntValue<'ctx>, offs: &[IntValue<'ctx>]| -> R<Vec<IntValue<'ctx>>> {
                let mut loaded: Vec<(usize, IntValue)> = Vec::new();
                let mut new = offs.to_vec();
                for (q, a) in adv.iter().enumerate() {
                    let inc = match a[d] {
                        Adv::None => continue,
                        Adv::Stride(s) => self.b.build_int_nsw_mul(i, s, "m").map_err(e)?,
                        Adv::Indexed { idx, stride, len } => {
                            let raw = match loaded.iter().find(|(x, _)| *x == idx) {
                                Some((_, v)) => *v,
                                None => {
                                    let off = self.b.build_int_nsw_mul(i, idx_stride[idx].unwrap(), "io").map_err(e)?;
                                    let v = self.load_elem(self.gep_bytes(data[idx], off)?, inputs[idx].dtype)?;
                                    let v = self.cast(v, inputs[idx].dtype, DType::Int64)?.into_int_value();
                                    loaded.push((idx, v));
                                    v
                                }
                            };
                            // negative indices wrap; anything outside [0, len) fails
                            let neg = self.b.build_int_compare(IntPredicate::SLT, raw, zero, "neg").map_err(e)?;
                            let wrapped = self.b.build_int_add(raw, len, "w").map_err(e)?;
                            let v = self.b.build_select(neg, wrapped, raw, "v").map_err(e)?.into_int_value();
                            let ok = self.b.build_int_compare(IntPredicate::ULT, v, len, "ok").map_err(e)?;
                            let cont = self.ctx.append_basic_block(f, "inb");
                            self.b.build_conditional_branch(ok, cont, fail).map_err(e)?;
                            self.b.position_at_end(cont);
                            self.b.build_int_nsw_mul(v, stride, "m").map_err(e)?
                        }
                    };
                    new[q] = self.b.build_int_nsw_add(offs[q], inc, "o").map_err(e)?;
                }
                Ok(new)
            };
            // dtype of every operand's argument
            let op_dt: Vec<DType> = ops.iter().map(|&a| operand(a).dtype).collect();
            // Evaluate one element: `ptrs[q]` is operand q's element. Reductions with a
            // register accumulator (`reg[k] = Some(r)`) update `accs[r]` instead of memory.
            let compute = |ptrs: &[PointerValue<'ctx>], reg: &[Option<usize>], accs: &[V<'ctx>]| -> R<Vec<V<'ctx>>> {
                let mut ins = Vec::with_capacity(n_elem);
                let mut q = 0;
                for j in 0..n_elem {
                    ins.push(match scal[j] {
                        Some(v) => v,
                        None => {
                            let v = self.load_elem(ptrs[q], inputs[j].dtype)?;
                            q += 1;
                            v
                        }
                    });
                }
                let vals = self.scalar_graph(scalar, &ins)?;
                let mut new_accs = accs.to_vec();
                for (k, o) in outs.iter().enumerate() {
                    let odt = outputs[k].dtype;
                    let v = self.cast(vals[k].0, vals[k].1, odt)?;
                    let p = ptrs[first_out + k];
                    match o {
                        FusedOut::Plain | FusedOut::Write { set: true, .. } => self.store_elem(p, v, odt)?,
                        FusedOut::Reduce { op, .. } => match reg[k] {
                            Some(r) => new_accs[r] = self.reduce_combine(*op, accs[r], v, odt)?,
                            None => {
                                let acc = self.load_elem(p, odt)?;
                                self.store_elem(p, self.reduce_combine(*op, acc, v, odt)?, odt)?;
                            }
                        },
                        FusedOut::Write { set: false, .. } => {
                            let acc = self.load_elem(p, odt)?;
                            let r = if odt == DType::Bool { self.ibin("or", acc, v)? } else { self.arith("add", acc, v, odt)? };
                            self.store_elem(p, r, odt)?;
                        }
                    }
                }
                Ok(new_accs)
            };
            // offset of an indexed operand: load idx[i] (cached per iteration), wrap, check
            let indexed_off = |loaded: &mut Vec<(usize, IntValue<'ctx>)>, i: IntValue<'ctx>, idx: usize, stride: IntValue<'ctx>, len: IntValue<'ctx>| -> R<IntValue<'ctx>> {
                let raw = match loaded.iter().find(|(x, _)| *x == idx) {
                    Some((_, v)) => *v,
                    None => {
                        let off = self.b.build_int_nsw_mul(i, idx_stride[idx].unwrap(), "io").map_err(e)?;
                        let v = self.load_elem(self.gep_bytes(data[idx], off)?, inputs[idx].dtype)?;
                        let v = self.cast(v, inputs[idx].dtype, DType::Int64)?.into_int_value();
                        loaded.push((idx, v));
                        v
                    }
                };
                let neg = self.b.build_int_compare(IntPredicate::SLT, raw, zero, "neg").map_err(e)?;
                let wrapped = self.b.build_int_add(raw, len, "w").map_err(e)?;
                let v = self.b.build_select(neg, wrapped, raw, "v").map_err(e)?.into_int_value();
                let ok = self.b.build_int_compare(IntPredicate::ULT, v, len, "ok").map_err(e)?;
                let cont = self.ctx.append_basic_block(f, "inb");
                self.b.build_conditional_branch(ok, cont, fail).map_err(e)?;
                self.b.position_at_end(cont);
                self.b.build_int_nsw_mul(v, stride, "m").map_err(e)
            };

            if ndim == 0 {
                let ptrs: Vec<PointerValue> = ops.iter().map(|&a| self.gep_bytes(data[a], zero)).collect::<R<_>>()?;
                compute(&ptrs, &vec![None; outs.len()], &[])?;
            } else {
                let inner_dim = order[ndim - 1];
                // Innermost loop: register accumulators for reductions along it, and two
                // versions -- unit element strides (typed GEPs, vectorizes) or byte strides.
                let mut inner = |offs: &[IntValue<'ctx>]| -> R<()> {
                    let d = inner_dim;
                    let n = dims[d];
                    let bases: Vec<PointerValue> = ops.iter().enumerate().map(|(q, &a)| self.gep_bytes(data[a], offs[q])).collect::<R<_>>()?;
                    let mut reg = vec![None; outs.len()];
                    let mut acc_ptrs = Vec::new();
                    let mut acc0 = Vec::new();
                    for (k, o) in outs.iter().enumerate() {
                        if matches!(o, FusedOut::Reduce { .. }) && matches!(adv[first_out + k][d], Adv::None) {
                            reg[k] = Some(acc_ptrs.len());
                            let p = bases[first_out + k];
                            acc0.push(self.load_elem(p, outputs[k].dtype)?);
                            acc_ptrs.push((p, outputs[k].dtype));
                        }
                    }
                    let strided: Vec<(usize, IntValue)> = adv
                        .iter()
                        .enumerate()
                        .filter_map(|(q, a)| if let Adv::Stride(s) = a[d] { Some((q, s)) } else { None })
                        .collect();
                    let version = |contig: bool| -> R<(Vec<V<'ctx>>, inkwell::basic_block::BasicBlock<'ctx>)> {
                        let accs = self.emit_multi_acc_loop(n, &acc0, &mut |i, accs| {
                            let mut loaded = Vec::new();
                            let mut ptrs = Vec::with_capacity(ops.len());
                            for (q, a) in adv.iter().enumerate() {
                                ptrs.push(match a[d] {
                                    Adv::None => bases[q],
                                    Adv::Stride(s) => {
                                        if contig {
                                            self.gep_elem(op_dt[q], bases[q], i)?
                                        } else {
                                            self.gep_bytes(bases[q], self.b.build_int_nsw_mul(i, s, "m").map_err(e)?)?
                                        }
                                    }
                                    Adv::Indexed { idx, stride, len } => {
                                        let off = indexed_off(&mut loaded, i, idx, stride, len)?;
                                        self.gep_bytes(bases[q], off)?
                                    }
                                });
                            }
                            compute(&ptrs, &reg, accs)
                        })?;
                        Ok((accs, self.b.get_insert_block().unwrap()))
                    };
                    let finals: Vec<V> = if strided.is_empty() {
                        version(false)?.0
                    } else {
                        // all unit-strided operands move by exactly one element?
                        let mut ok = self.ctx.bool_type().const_int(1, false);
                        for &(q, s) in &strided {
                            let isz = self.i64c(op_dt[q].itemsize() as i64);
                            let c = self.b.build_int_compare(IntPredicate::EQ, s, isz, "unit").map_err(e)?;
                            ok = self.b.build_and(ok, c, "ok").map_err(e)?;
                        }
                        let fast = self.ctx.append_basic_block(f, "inner.contig");
                        let slow = self.ctx.append_basic_block(f, "inner.strided");
                        let join = self.ctx.append_basic_block(f, "inner.join");
                        self.b.build_conditional_branch(ok, fast, slow).map_err(e)?;
                        self.b.position_at_end(fast);
                        let (fa, fb) = version(true)?;
                        self.b.build_unconditional_branch(join).map_err(e)?;
                        self.b.position_at_end(slow);
                        let (sa, sb) = version(false)?;
                        self.b.build_unconditional_branch(join).map_err(e)?;
                        self.b.position_at_end(join);
                        let mut out = Vec::new();
                        for (x, y) in fa.iter().zip(&sa) {
                            let p = self.b.build_phi(x.get_type(), "acc").map_err(e)?;
                            p.add_incoming(&[(x, fb), (y, sb)]);
                            out.push(p.as_basic_value());
                        }
                        out
                    };
                    for ((p, dt), v) in acc_ptrs.iter().zip(finals) {
                        self.store_elem(*p, v, *dt)?;
                    }
                    Ok(())
                };
                self.nest_custom(0, &order[..ndim - 1], &dims, vec![zero; ops.len()], &mut step, &mut inner)?;
            }
            // the scaffold returns 0 from the current block; the failure block returns 1
            let cur = self.b.get_insert_block().unwrap();
            self.b.position_at_end(fail);
            self.b.build_return(Some(&self.ctx.i32_type().const_int(1, false))).map_err(e)?;
            self.b.position_at_end(cur);
            Ok(())
        })
    }
}
