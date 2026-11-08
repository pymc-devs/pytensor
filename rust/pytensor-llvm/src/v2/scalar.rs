//! Scalar ops: functions from scalars to scalars, always generated and inlined into the
//! loop that calls them. A Composite is a scalar graph and is itself a scalar op.
//!
//! Each op is its own type. Ops get their argument dtypes and the output dtypes that
//! PyTensor computed; most compute in the output dtype.

use super::emit::{Arith, BitOp, Cmp, Emitter, FIntr, Libm, RustMath, V};
use super::index::{Idx, IndexVec};
use super::op::{Key, R};
use crate::define_index;
use crate::dtype::DType;
use inkwell::values::BasicValue;
use inkwell::FloatPredicate;
use std::sync::Arc;

pub trait ScalarOp: Send + Sync + std::fmt::Debug {
    /// Allowed number of inputs (inclusive range).
    fn arity(&self) -> (usize, usize);

    fn n_outputs(&self) -> usize {
        1
    }

    /// Emit the op. `args` carry their own dtypes; `outs` are the output dtypes.
    fn emit<'c>(&self, e: &Emitter<'c>, args: &[(V<'c>, DType)], outs: &[DType]) -> R<Vec<V<'c>>> {
        Ok(vec![self.emit1(e, args, outs[0])?])
    }

    /// Single-output ops implement this instead of `emit`.
    fn emit1<'c>(&self, _e: &Emitter<'c>, _args: &[(V<'c>, DType)], _out: DType) -> R<V<'c>> {
        Err(format!("{self:?} has several outputs"))
    }

    fn key(&self, k: &mut Key);
}

/// Argument `i` converted to dtype `to`.
fn conv<'c>(e: &Emitter<'c>, args: &[(V<'c>, DType)], i: usize, to: DType) -> R<V<'c>> {
    e.cast(args[i].0, args[i].1, to)
}

fn require_float(op: &dyn std::fmt::Debug, dt: DType) -> R<()> {
    if dt.is_float() {
        Ok(())
    } else {
        Err(format!("{op:?} with non-float output {dt:?}"))
    }
}

/// Define a parameterless single-output scalar op.
macro_rules! scalar_op {
    ($(#[$m:meta])* $name:ident, ($lo:expr, $hi:expr), |$e:ident, $args:ident, $out:ident| $body:expr) => {
        $(#[$m])*
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $name;

        impl ScalarOp for $name {
            fn arity(&self) -> (usize, usize) {
                ($lo, $hi)
            }

            fn emit1<'c>(&self, $e: &Emitter<'c>, $args: &[(V<'c>, DType)], $out: DType) -> R<V<'c>> {
                $body
            }

            fn key(&self, k: &mut Key) {
                k.tag::<Self>();
            }
        }
    };
}

/// A float function of one argument via an LLVM intrinsic.
macro_rules! float_intrinsic {
    ($name:ident, $intr:expr) => {
        scalar_op!($name, (1, 1), |e, a, out| {
            require_float(&$name, out)?;
            e.fintr($intr, out, &[conv(e, a, 0, out)?])
        });
    };
}

/// A float function via libm.
macro_rules! float_libm {
    ($name:ident, $f:expr, $n:literal) => {
        scalar_op!($name, ($n, $n), |e, a, out| {
            require_float(&$name, out)?;
            let args: Vec<V> = (0..$n).map(|i| conv(e, a, i, out)).collect::<R<_>>()?;
            e.libm($f, out, &args)
        });
    };
}

/// A float function precompiled in Rust.
macro_rules! float_rust {
    ($name:ident, $f:expr) => {
        scalar_op!($name, (1, 1), |e, a, out| {
            require_float(&$name, out)?;
            e.rust_math($f, out, &[conv(e, a, 0, out)?])
        });
    };
}

/// A comparison: computed in the common dtype of the arguments.
macro_rules! comparison {
    ($name:ident, $c:expr) => {
        scalar_op!($name, (2, 2), |e, a, out| {
            let ct = DType::upcast(a[0].1, a[1].1);
            let c = e.cmp($c, conv(e, a, 0, ct)?, conv(e, a, 1, ct)?, ct)?;
            e.cast(c, DType::Bool, out)
        });
    };
}

macro_rules! bitwise {
    ($name:ident, $op:expr) => {
        scalar_op!($name, (2, 2), |e, a, out| {
            if out.is_float() {
                return Err(format!("{:?} on {out:?}", $name));
            }
            e.bitop($op, conv(e, a, 0, out)?, conv(e, a, 1, out)?)
        });
    };
}

// ---------------------------------------------------------------- arithmetic

scalar_op!(Add, (1, usize::MAX), |e, a, out| {
    let mut acc = conv(e, a, 0, out)?;
    for i in 1..a.len() {
        acc = e.arith(Arith::Add, acc, conv(e, a, i, out)?, out)?;
    }
    Ok(acc)
});

scalar_op!(Mul, (1, usize::MAX), |e, a, out| {
    let mut acc = conv(e, a, 0, out)?;
    for i in 1..a.len() {
        acc = e.arith(Arith::Mul, acc, conv(e, a, i, out)?, out)?;
    }
    Ok(acc)
});

scalar_op!(Sub, (2, 2), |e, a, out| e.arith(Arith::Sub, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out));

scalar_op!(TrueDiv, (2, 2), |e, a, out| {
    require_float(&TrueDiv, out)?;
    e.arith(Arith::Div, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out)
});

scalar_op!(
    /// Python floor division; `x // 0 == 0` for integers, like numpy.
    IntDiv,
    (2, 2),
    |e, a, out| {
        if out.is_float() {
            let q = e.arith(Arith::Div, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out)?;
            return e.fintr(FIntr::Floor, out, &[q]);
        }
        let b = &e.b;
        let (x, y0) = (conv(e, a, 0, out)?.into_int_value(), conv(e, a, 1, out)?.into_int_value());
        let z = y0.get_type().const_zero();
        let yz = b.build_int_compare(inkwell::IntPredicate::EQ, y0, z, "bz").map_err(super::emit::e)?;
        let y = b.build_select(yz, y0.get_type().const_int(1, false), y0, "b1").map_err(super::emit::e)?.into_int_value();
        let x = b.build_select(yz, z, x, "a0").map_err(super::emit::e)?.into_int_value();
        if !out.is_signed_int() {
            return Ok(b.build_int_unsigned_div(x, y, "q").map_err(super::emit::e)?.into());
        }
        let q = b.build_int_signed_div(x, y, "q").map_err(super::emit::e)?;
        let r = b.build_int_signed_rem(x, y, "r").map_err(super::emit::e)?;
        let adj = floor_adjust(e, r, y)?;
        let qm1 = b.build_int_sub(q, q.get_type().const_int(1, false), "qm").map_err(super::emit::e)?;
        b.build_select(adj, qm1, q, "fq").map_err(super::emit::e)
    }
);

scalar_op!(
    /// Python modulo (sign of the divisor); `x % 0 == 0` for integers, like numpy.
    Mod,
    (2, 2),
    |e, a, out| {
        let b = &e.b;
        if out.is_float() {
            let (x, y) = (conv(e, a, 0, out)?, conv(e, a, 1, out)?);
            let q = e.fintr(FIntr::Floor, out, &[e.arith(Arith::Div, x, y, out)?])?;
            return e.arith(Arith::Sub, x, e.arith(Arith::Mul, q, y, out)?, out);
        }
        let (x, y0) = (conv(e, a, 0, out)?.into_int_value(), conv(e, a, 1, out)?.into_int_value());
        let z = y0.get_type().const_zero();
        let yz = b.build_int_compare(inkwell::IntPredicate::EQ, y0, z, "bz").map_err(super::emit::e)?;
        let y = b.build_select(yz, y0.get_type().const_int(1, false), y0, "b1").map_err(super::emit::e)?.into_int_value();
        let r = if out.is_signed_int() {
            let r = b.build_int_signed_rem(x, y, "r").map_err(super::emit::e)?;
            let adj = floor_adjust(e, r, y)?;
            let ry = b.build_int_add(r, y, "rb").map_err(super::emit::e)?;
            b.build_select(adj, ry, r, "fr").map_err(super::emit::e)?.into_int_value()
        } else {
            b.build_int_unsigned_rem(x, y, "r").map_err(super::emit::e)?
        };
        b.build_select(yz, z, r, "m").map_err(super::emit::e)
    }
);

/// For C-style remainder `r` of a division by `y`: is floor rounding different?
fn floor_adjust<'c>(e: &Emitter<'c>, r: inkwell::values::IntValue<'c>, y: inkwell::values::IntValue<'c>) -> R<inkwell::values::IntValue<'c>> {
    use inkwell::IntPredicate::{NE, SLT};
    let b = &e.b;
    let z = r.get_type().const_zero();
    let rnz = b.build_int_compare(NE, r, z, "rnz").map_err(super::emit::e)?;
    let rneg = b.build_int_compare(SLT, r, z, "rn").map_err(super::emit::e)?;
    let yneg = b.build_int_compare(SLT, y, z, "bn").map_err(super::emit::e)?;
    let diff = b.build_xor(rneg, yneg, "d").map_err(super::emit::e)?;
    b.build_and(rnz, diff, "adj").map_err(super::emit::e)
}

scalar_op!(Pow, (2, 2), |e, a, out| {
    require_float(&Pow, out)?;
    e.fintr(FIntr::Pow, out, &[conv(e, a, 0, out)?, conv(e, a, 1, out)?])
});

scalar_op!(Maximum, (2, 2), |e, a, out| e.maxmin(true, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out));
scalar_op!(Minimum, (2, 2), |e, a, out| e.maxmin(false, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out));

scalar_op!(Clip, (3, 3), |e, a, out| {
    let x = e.maxmin(true, conv(e, a, 0, out)?, conv(e, a, 1, out)?, out)?;
    e.maxmin(false, x, conv(e, a, 2, out)?, out)
});

scalar_op!(Neg, (1, 1), |e, a, out| e.neg(conv(e, a, 0, out)?, out));

scalar_op!(Abs, (1, 1), |e, a, out| {
    let x = conv(e, a, 0, out)?;
    if out.is_float() {
        e.fintr(FIntr::Fabs, out, &[x])
    } else if out.is_signed_int() {
        let c = e.cmp(Cmp::Lt, x, e.fconst(out, 0.0), out)?;
        e.select(c, e.neg(x, out)?, x)
    } else {
        Ok(x)
    }
});

scalar_op!(Sign, (1, 1), |e, a, out| {
    let x = conv(e, a, 0, out)?;
    let (zero, one) = (e.fconst(out, 0.0), e.fconst(out, 1.0));
    let pos = e.cmp(Cmp::Gt, x, zero, out)?;
    let neg = e.cmp(Cmp::Lt, x, zero, out)?;
    let t = e.select(neg, e.neg(one, out)?, x)?; // keeps 0 and NaN
    e.select(pos, one, t)
});

scalar_op!(Sqr, (1, 1), |e, a, out| {
    let x = conv(e, a, 0, out)?;
    e.arith(Arith::Mul, x, x, out)
});

scalar_op!(Reciprocal, (1, 1), |e, a, out| {
    require_float(&Reciprocal, out)?;
    e.arith(Arith::Div, e.fconst(out, 1.0), conv(e, a, 0, out)?, out)
});

// ---------------------------------------------------------------- math functions

float_intrinsic!(Exp, FIntr::Exp);
float_intrinsic!(Log, FIntr::Log);
float_intrinsic!(Sqrt, FIntr::Sqrt);
float_intrinsic!(Sin, FIntr::Sin);
float_intrinsic!(Cos, FIntr::Cos);
float_intrinsic!(Floor, FIntr::Floor);
float_intrinsic!(Ceil, FIntr::Ceil);
float_intrinsic!(Trunc, FIntr::Trunc);
float_intrinsic!(RoundHalfToEven, FIntr::RoundEven);
float_intrinsic!(RoundHalfAwayFromZero, FIntr::Round);

float_libm!(Exp2, Libm::Exp2, 1);
float_libm!(Log2, Libm::Log2, 1);
float_libm!(Log10, Libm::Log10, 1);
float_libm!(Expm1, Libm::Expm1, 1);
float_libm!(Log1p, Libm::Log1p, 1);
float_libm!(Tan, Libm::Tan, 1);
float_libm!(Tanh, Libm::Tanh, 1);
float_libm!(Sinh, Libm::Sinh, 1);
float_libm!(Cosh, Libm::Cosh, 1);
float_libm!(ArcSin, Libm::Asin, 1);
float_libm!(ArcCos, Libm::Acos, 1);
float_libm!(ArcTan, Libm::Atan, 1);
float_libm!(ArcSinh, Libm::Asinh, 1);
float_libm!(ArcCosh, Libm::Acosh, 1);
float_libm!(ArcTanh, Libm::Atanh, 1);
float_libm!(ArcTan2, Libm::Atan2, 2);

float_rust!(Erf, RustMath::Erf);
float_rust!(Erfc, RustMath::Erfc);
float_rust!(GammaLn, RustMath::GammaLn);
float_rust!(Gamma, RustMath::Gamma);
float_rust!(Psi, RustMath::Psi);

scalar_op!(Sigmoid, (1, 1), |e, a, out| {
    require_float(&Sigmoid, out)?;
    let one = e.fconst(out, 1.0);
    let ex = e.fintr(FIntr::Exp, out, &[e.neg(conv(e, a, 0, out)?, out)?])?;
    e.arith(Arith::Div, one, e.arith(Arith::Add, one, ex, out)?, out)
});

scalar_op!(
    /// The same branches as pytensor.scalar.math.Softplus.
    Softplus,
    (1, 1),
    |e, a, out| {
        require_float(&Softplus, out)?;
        let x = conv(e, a, 0, out)?;
        let ex = e.fintr(FIntr::Exp, out, &[x])?;
        let l1pex = e.libm(Libm::Log1p, out, &[ex])?;
        let emx = e.fintr(FIntr::Exp, out, &[e.neg(x, out)?])?;
        let xpemx = e.arith(Arith::Add, x, emx, out)?;
        let c1 = e.cmp(Cmp::Lt, x, e.fconst(out, -37.0), out)?;
        let c2 = e.cmp(Cmp::Lt, x, e.fconst(out, 18.0), out)?;
        let c3 = e.cmp(Cmp::Lt, x, e.fconst(out, 33.3), out)?;
        let r3 = e.select(c3, xpemx, x)?;
        let r2 = e.select(c2, l1pex, r3)?;
        e.select(c1, ex, r2)
    }
);

scalar_op!(Log1mexp, (1, 1), |e, a, out| {
    require_float(&Log1mexp, out)?;
    let x = conv(e, a, 0, out)?;
    let small = e.libm(Libm::Log1p, out, &[e.neg(e.fintr(FIntr::Exp, out, &[x])?, out)?])?;
    let em1 = e.libm(Libm::Expm1, out, &[x])?;
    let large = e.fintr(FIntr::Log, out, &[e.neg(em1, out)?])?;
    let c = e.cmp(Cmp::Lt, x, e.fconst(out, 0.5f64.ln()), out)?;
    e.select(c, small, large)
});

// ---------------------------------------------------------------- comparisons & logic

comparison!(LT, Cmp::Lt);
comparison!(GT, Cmp::Gt);
comparison!(LE, Cmp::Le);
comparison!(GE, Cmp::Ge);
comparison!(EQ, Cmp::Eq);
comparison!(NEQ, Cmp::Ne);

scalar_op!(IsNan, (1, 1), |e, a, out| {
    let (x, dt) = a[0];
    if !dt.is_float() {
        return Ok(e.constant(out, 0.0, 0));
    }
    let xf = x.into_float_value();
    let c = e.b.build_float_compare(FloatPredicate::UNO, xf, xf, "nan").map_err(super::emit::e)?;
    e.cast(c.as_basic_value_enum(), DType::Bool, out)
});

scalar_op!(IsInf, (1, 1), |e, a, out| {
    let (x, dt) = a[0];
    if !dt.is_float() {
        return Ok(e.constant(out, 0.0, 0));
    }
    let ax = e.fintr(FIntr::Fabs, dt, &[x])?;
    let c = e.cmp(Cmp::Eq, ax, e.fconst(dt, f64::INFINITY), dt)?;
    e.cast(c, DType::Bool, out)
});

scalar_op!(Switch, (3, 3), |e, a, out| {
    let c = conv(e, a, 0, DType::Bool)?;
    e.select(c, conv(e, a, 1, out)?, conv(e, a, 2, out)?)
});

bitwise!(AND, BitOp::And);
bitwise!(OR, BitOp::Or);
bitwise!(XOR, BitOp::Xor);

scalar_op!(Invert, (1, 1), |e, a, out| {
    if out.is_float() {
        return Err(format!("Invert on {out:?}"));
    }
    Ok(e.b.build_not(conv(e, a, 0, out)?.into_int_value(), "inv").map_err(super::emit::e)?.into())
});

scalar_op!(Identity, (1, 1), |e, a, out| conv(e, a, 0, out));
scalar_op!(Cast, (1, 1), |e, a, out| conv(e, a, 0, out));
scalar_op!(Second, (2, 2), |e, a, out| conv(e, a, 1, out));

// ---------------------------------------------------------------- composites

define_index!(
    /// An input of a scalar graph.
    ScalarIn
);
define_index!(
    /// A node of a scalar graph.
    ScalarNodeId
);

#[derive(Clone, Copy, Debug)]
pub enum SRef {
    Input(ScalarIn),
    /// output `k` of a node
    Node(ScalarNodeId, u32),
    Const { dtype: DType, f: f64, i: i64 },
}

#[derive(Debug)]
pub struct ScalarNode {
    pub op: Arc<dyn ScalarOp>,
    pub inputs: Vec<SRef>,
    pub outputs: Vec<DType>,
}

/// PyTensor's Composite: a scalar graph, itself a scalar op.
#[derive(Default)]
pub struct Composite {
    pub inputs: IndexVec<ScalarIn, DType>,
    pub nodes: IndexVec<ScalarNodeId, ScalarNode>,
    pub outputs: Vec<SRef>,
}

impl Composite {
    pub fn dtype_of(&self, r: SRef) -> DType {
        match r {
            SRef::Input(i) => self.inputs[i],
            SRef::Node(n, k) => self.nodes[n].outputs[k as usize],
            SRef::Const { dtype, .. } => dtype,
        }
    }

    pub fn output_dtypes(&self) -> Vec<DType> {
        self.outputs.iter().map(|&r| self.dtype_of(r)).collect()
    }

    pub fn add(&mut self, op: Arc<dyn ScalarOp>, inputs: Vec<SRef>, outputs: Vec<DType>) -> R<ScalarNodeId> {
        let (lo, hi) = op.arity();
        if inputs.len() < lo || inputs.len() > hi {
            return Err(format!("{op:?} takes {lo}..={hi} inputs, got {}", inputs.len()));
        }
        if outputs.len() != op.n_outputs() {
            return Err(format!("{op:?} has {} outputs, got {} dtypes", op.n_outputs(), outputs.len()));
        }
        for r in &inputs {
            match *r {
                SRef::Input(i) if i.index() >= self.inputs.len() => return Err(format!("{op:?}: no scalar input {}", i.0)),
                SRef::Node(n, k) if n.index() >= self.nodes.len() || k as usize >= self.nodes[n].outputs.len() => {
                    return Err(format!("{op:?}: input refers to a later or unknown node"));
                }
                _ => {}
            }
        }
        Ok(self.nodes.push(ScalarNode { op, inputs, outputs }))
    }

    /// Emit the graph for the given input values.
    pub fn emit_graph<'c>(&self, e: &Emitter<'c>, ins: &[V<'c>]) -> R<Vec<(V<'c>, DType)>> {
        let mut vals: IndexVec<ScalarNodeId, Vec<V<'c>>> = IndexVec::new();
        let get = |r: SRef, vals: &IndexVec<ScalarNodeId, Vec<V<'c>>>| -> (V<'c>, DType) {
            match r {
                SRef::Input(i) => (ins[i.index()], self.inputs[i]),
                SRef::Node(n, k) => (vals[n][k as usize], self.nodes[n].outputs[k as usize]),
                SRef::Const { dtype, f, i } => (e.constant(dtype, f, i), dtype),
            }
        };
        for n in &self.nodes {
            let args: Vec<(V<'c>, DType)> = n.inputs.iter().map(|&r| get(r, &vals)).collect();
            let out = n.op.emit(e, &args, &n.outputs)?;
            vals.push(out);
        }
        Ok(self.outputs.iter().map(|&r| get(r, &vals)).collect())
    }
}

impl std::fmt::Debug for Composite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ops: Vec<String> = self.nodes.iter().map(|n| format!("{:?}", n.op)).collect();
        write!(f, "Composite{{{}}}", ops.join(", "))
    }
}

impl ScalarOp for Composite {
    fn arity(&self) -> (usize, usize) {
        (self.inputs.len(), self.inputs.len())
    }

    fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn emit<'c>(&self, e: &Emitter<'c>, args: &[(V<'c>, DType)], outs: &[DType]) -> R<Vec<V<'c>>> {
        let ins: Vec<V<'c>> = args
            .iter()
            .zip(self.inputs.iter())
            .map(|(&(v, dt), &want)| e.cast(v, dt, want))
            .collect::<R<_>>()?;
        let res = self.emit_graph(e, &ins)?;
        res.iter().zip(outs).map(|(&(v, dt), &want)| e.cast(v, dt, want)).collect()
    }

    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
        k.u64(self.inputs.len() as u64);
        for dt in &self.inputs {
            k.debug(dt);
        }
        let sref = |k: &mut Key, r: &SRef| match *r {
            SRef::Input(i) => {
                k.u64(0);
                k.u64(i.0 as u64);
            }
            SRef::Node(n, o) => {
                k.u64(1);
                k.u64(n.0 as u64);
                k.u64(o as u64);
            }
            SRef::Const { dtype, f, i } => {
                k.u64(2);
                k.debug(&dtype);
                k.f64(f);
                k.i64(i);
            }
        };
        k.u64(self.nodes.len() as u64);
        for n in &self.nodes {
            n.op.key(k);
            k.u64(n.inputs.len() as u64);
            for r in &n.inputs {
                sref(k, r);
            }
            for dt in &n.outputs {
                k.debug(dt);
            }
        }
        k.u64(self.outputs.len() as u64);
        for r in &self.outputs {
            sref(k, r);
        }
    }
}
