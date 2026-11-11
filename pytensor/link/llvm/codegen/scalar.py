"""Scalar ops: one emitter per PyTensor scalar op class.

An emitter gets the builder, the arguments as (value, dtype) pairs and the output dtypes
PyTensor computed, and returns the output values. Most ops compute in the output dtype.
"""

from __future__ import annotations

import math
from functools import singledispatch

import pytensor.scalar.basic as ps
import pytensor.scalar.math as psm
from pytensor.graph.basic import Constant
from pytensor.scalar.loop import ScalarLoop
from pytensor.link.llvm.codegen.emit import cast, constant
from pytensor.link.llvm.codegen.types import check_dtype, ir, is_float, is_signed, value_type


@singledispatch
def llvm_scalar(op, b, args: list, out_dtypes: list) -> list:
    raise NotImplementedError(f"LLVM backend: scalar op {op} is not supported")


def emit_scalar(op, b, args, out_dtypes):
    for d in out_dtypes:
        check_dtype(d)
    return llvm_scalar(op, b, args, out_dtypes)


def _single(fn):
    """Register helper: fn(b, args, out_dtype) -> value."""

    def wrapper(op, b, args, out_dtypes):
        return [fn(b, args, out_dtypes[0])]

    return wrapper


def register(*classes):
    def deco(fn):
        w = _single(fn)
        for c in classes:
            llvm_scalar.register(c)(w)
        return fn

    return deco


def conv(b, args, k, dtype):
    v, d = args[k]
    return cast(b, v, d, dtype)


def require_float(dtype, what):
    if not is_float(dtype):
        raise NotImplementedError(f"LLVM backend: {what} with output dtype {dtype}")


def _upcast(*dtypes):
    import numpy as np

    return str(np.result_type(*dtypes))


# ------------------------------------------------------------------ arithmetic


@register(ps.Add)
def _(b, args, out):
    acc = conv(b, args, 0, out)
    for k in range(1, len(args)):
        x = conv(b, args, k, out)
        acc = b.fadd(acc, x) if is_float(out) else (b.or_(acc, x) if out == "bool" else b.add(acc, x))
    return acc


@register(ps.Mul)
def _(b, args, out):
    acc = conv(b, args, 0, out)
    for k in range(1, len(args)):
        x = conv(b, args, k, out)
        acc = b.fmul(acc, x) if is_float(out) else (b.and_(acc, x) if out == "bool" else b.mul(acc, x))
    return acc


@register(ps.Sub)
def _(b, args, out):
    x, y = conv(b, args, 0, out), conv(b, args, 1, out)
    return b.fsub(x, y) if is_float(out) else b.sub(x, y)


@register(ps.TrueDiv)
def _(b, args, out):
    require_float(out, "true_div")
    return b.fdiv(conv(b, args, 0, out), conv(b, args, 1, out))


def _floor_adjust(b, r, y):
    """For a C-style remainder r of a division by y: does floor rounding differ?"""
    z = b.const_int(r.type, 0)
    rnz = b.ne(r, z)
    differ = b.xor(b.slt(r, z), b.slt(y, z))
    return b.and_(rnz, differ)


def _safe_divisor(b, y):
    """(y with 0 replaced by 1, is-zero flag): integer division by zero is undefined."""
    z = b.const_int(y.type, 0)
    yz = b.eq(y, z)
    return b.select(yz, b.const_int(y.type, 1), y), yz


@register(ps.IntDiv)
def _(b, args, out):
    """Python floor division; x // 0 == 0 for integers, like numpy."""
    x, y = conv(b, args, 0, out), conv(b, args, 1, out)
    if is_float(out):
        return b.intrinsic(ir.Intrinsic.Floor, [b.fdiv(x, y)])
    y1, yz = _safe_divisor(b, y)
    zero = b.const_int(x.type, 0)
    if not is_signed(out):
        return b.select(yz, zero, b.udiv(x, y1))
    # x // -1 is -x; computing it as a division would overflow for MIN (numpy wraps)
    minus1 = b.eq(y1, b.const_int(x.type, -1))
    y1 = b.select(minus1, b.const_int(x.type, 1), y1)
    q = b.sdiv(x, y1)
    r = b.srem(x, y1)
    q = b.select(_floor_adjust(b, r, y1), b.sub(q, b.const_int(x.type, 1)), q)
    q = b.select(minus1, b.neg(x), q)
    return b.select(yz, zero, q)


@register(ps.Mod)
def _(b, args, out):
    """Python modulo (sign of the divisor); x % 0 == 0 for integers, like numpy."""
    x, y = conv(b, args, 0, out), conv(b, args, 1, out)
    if is_float(out):
        q = b.intrinsic(ir.Intrinsic.Floor, [b.fdiv(x, y)])
        return b.fsub(x, b.fmul(q, y))
    y1, yz = _safe_divisor(b, y)
    zero = b.const_int(x.type, 0)
    if not is_signed(out):
        return b.select(yz, zero, b.urem(x, y1))
    # x % -1 == 0 (and MIN % -1 would overflow)
    minus1 = b.eq(y1, b.const_int(x.type, -1))
    y1 = b.select(minus1, b.const_int(x.type, 1), y1)
    r = b.srem(x, y1)
    r = b.select(_floor_adjust(b, r, y1), b.add(r, y1), r)
    return b.select(yz, zero, r)


@register(ps.Pow)
def _(b, args, out):
    if is_float(out):
        return ir_intrinsic(b, "Pow", conv(b, args, 0, out), conv(b, args, 1, out))
    if out == "bool":
        raise NotImplementedError("LLVM backend: pow with output dtype bool")
    return int_pow(b, conv(b, args, 0, out), conv(b, args, 1, out), out)


def int_pow(b, x, e, dtype):
    """x**e by squaring (wrapping like numpy). A negative e gives 1 for x == 1,
    +-1 for x == -1 and 0 otherwise, as an integer truncation of the real power."""
    one, zero = constant(b, dtype, 1), constant(b, dtype, 0)
    bits = int(dtype.lstrip("uint") or 8) if dtype not in ("int8", "uint8") else 8

    def step(_i, acc, base, k):
        odd = b.ne(b.and_(k, one), zero)
        k2 = b.lshr(k, one)
        return [b.select(odd, b.mul(acc, base), acc), b.mul(base, base), k2, b.eq(k2, zero)]

    acc, _, _, _ = b.fold(constant(b, "int64", bits), [one, x, e], step, until=True)
    if not is_signed(dtype):
        return acc
    neg = b.slt(e, zero)
    minus_one = constant(b, dtype, -1)
    odd_e = b.ne(b.and_(e, one), zero)
    at_neg = b.select(
        b.eq(x, one),
        one,
        b.select(b.eq(x, minus_one), b.select(odd_e, minus_one, one), zero),
    )
    return b.select(neg, at_neg, acc)


def ir_intrinsic(b, name, *xs):
    return b.intrinsic(getattr(ir.Intrinsic, name), list(xs))


def maxmin(b, is_max, x, y, dtype):
    """NaN-propagating for floats, like numpy.maximum / numpy.minimum."""
    if is_float(dtype):
        return ir_intrinsic(b, "Maximum" if is_max else "Minimum", x, y)
    if dtype == "bool":
        return b.or_(x, y) if is_max else b.and_(x, y)
    pred = (ir.ICmp.SGT if is_max else ir.ICmp.SLT) if is_signed(dtype) else (ir.ICmp.UGT if is_max else ir.ICmp.ULT)
    return b.select(b.icmp(pred, x, y), x, y)


@register(ps.Maximum, ps.ScalarMaximum)
def _(b, args, out):
    return maxmin(b, True, conv(b, args, 0, out), conv(b, args, 1, out), out)


@register(ps.Minimum, ps.ScalarMinimum)
def _(b, args, out):
    return maxmin(b, False, conv(b, args, 0, out), conv(b, args, 1, out), out)


@register(ps.Clip)
def _(b, args, out):
    x = maxmin(b, True, conv(b, args, 0, out), conv(b, args, 1, out), out)
    return maxmin(b, False, x, conv(b, args, 2, out), out)


@register(ps.Neg)
def _(b, args, out):
    x = conv(b, args, 0, out)
    return b.fneg(x) if is_float(out) else b.neg(x)


@register(ps.Abs)
def _(b, args, out):
    x = conv(b, args, 0, out)
    if is_float(out):
        return ir_intrinsic(b, "Fabs", x)
    if is_signed(out):
        return b.select(b.slt(x, b.const_int(x.type, 0)), b.neg(x), x)
    return x


@register(ps.Sign)
def _(b, args, out):
    x = conv(b, args, 0, out)
    zero, one = constant(b, out, 0), constant(b, out, 1)
    if is_float(out):
        pos, neg = b.fcmp(ir.FCmp.OGT, x, zero), b.fcmp(ir.FCmp.OLT, x, zero)
        minus = b.fneg(one)
    else:
        pos = b.icmp(ir.ICmp.SGT if is_signed(out) else ir.ICmp.UGT, x, zero)
        neg = b.slt(x, zero) if is_signed(out) else b.const_bool(False)
        minus = b.neg(one)
    return b.select(pos, one, b.select(neg, minus, x))  # keeps 0 and NaN


@register(ps.Sqr)
def _(b, args, out):
    x = conv(b, args, 0, out)
    return b.fmul(x, x) if is_float(out) else b.mul(x, x)


@register(ps.Reciprocal)
def _(b, args, out):
    require_float(out, "reciprocal")
    return b.fdiv(constant(b, out, 1.0), conv(b, args, 0, out))


# ------------------------------------------------------------------ math functions


def _intrinsic_op(name):
    def fn(b, args, out):
        require_float(out, name)
        return ir_intrinsic(b, name, conv(b, args, 0, out))

    return fn


for _cls, _name in [
    (ps.Exp, "Exp"),
    (ps.Exp2, "Exp2"),
    (ps.Log, "Log"),
    (ps.Log2, "Log2"),
    (ps.Log10, "Log10"),
    (ps.Sqrt, "Sqrt"),
    (ps.Sin, "Sin"),
    (ps.Cos, "Cos"),
    (ps.Floor, "Floor"),
    (ps.Ceil, "Ceil"),
    (ps.Trunc, "Trunc"),
    (ps.RoundHalfToEven, "RoundEven"),
    (ps.RoundHalfAwayFromZero, "Round"),
]:
    register(_cls)(_intrinsic_op(_name))


def _libm_op(name, arity=1):
    def fn(b, args, out):
        require_float(out, name)
        return b.libm(getattr(ir.Libm, name), [conv(b, args, k, out) for k in range(arity)])

    return fn


for _cls, _name in [
    (ps.Expm1, "Expm1"),
    (ps.Log1p, "Log1p"),
    (ps.Tan, "Tan"),
    (ps.Tanh, "Tanh"),
    (ps.Sinh, "Sinh"),
    (ps.Cosh, "Cosh"),
    (ps.ArcSin, "Asin"),
    (ps.ArcCos, "Acos"),
    (ps.ArcTan, "Atan"),
    (ps.ArcSinh, "Asinh"),
    (ps.ArcCosh, "Acosh"),
    (ps.ArcTanh, "Atanh"),
    (psm.Erf, "Erf"),
    (psm.Erfc, "Erfc"),
]:
    register(_cls)(_libm_op(_name))
register(ps.ArcTan2)(_libm_op("Atan2", 2))


def _rust_op(name):
    def fn(b, args, out):
        require_float(out, name)
        return b.rust_math(getattr(ir.RustMath, name), conv(b, args, 0, out))

    return fn


register(psm.GammaLn)(_rust_op("GammaLn"))
register(psm.Gamma)(_rust_op("Gamma"))
register(psm.Psi)(_rust_op("Psi"))


@register(psm.Sigmoid)
def _(b, args, out):
    require_float(out, "sigmoid")
    one = constant(b, out, 1.0)
    e = ir_intrinsic(b, "Exp", b.fneg(conv(b, args, 0, out)))
    return b.fdiv(one, b.fadd(one, e))


@register(psm.Softplus)
def _(b, args, out):
    """The same branches as pytensor.scalar.math.Softplus."""
    require_float(out, "softplus")
    x = conv(b, args, 0, out)
    ex = ir_intrinsic(b, "Exp", x)
    log1pex = b.libm(ir.Libm.Log1p, [ex])
    x_plus = b.fadd(x, ir_intrinsic(b, "Exp", b.fneg(x)))

    def lt(c):
        return b.fcmp(ir.FCmp.OLT, x, constant(b, out, c))

    return b.select(lt(-37.0), ex, b.select(lt(18.0), log1pex, b.select(lt(33.3), x_plus, x)))


@register(psm.Log1mexp)
def _(b, args, out):
    require_float(out, "log1mexp")
    x = conv(b, args, 0, out)
    small = b.libm(ir.Libm.Log1p, [b.fneg(ir_intrinsic(b, "Exp", x))])
    large = ir_intrinsic(b, "Log", b.fneg(b.libm(ir.Libm.Expm1, [x])))
    return b.select(b.fcmp(ir.FCmp.OLT, x, constant(b, out, math.log(0.5))), small, large)


# ------------------------------------------------------------------ comparisons & logic


def _comparison(fpred, spred, upred):
    def fn(b, args, out):
        ct = _upcast(args[0][1], args[1][1])
        x, y = conv(b, args, 0, ct), conv(b, args, 1, ct)
        if is_float(ct):
            c = b.fcmp(getattr(ir.FCmp, fpred), x, y)
        else:
            c = b.icmp(getattr(ir.ICmp, spred if is_signed(ct) else upred), x, y)
        return cast(b, c, "bool", out)

    return fn


register(ps.LT)(_comparison("OLT", "SLT", "ULT"))
register(ps.GT)(_comparison("OGT", "SGT", "UGT"))
register(ps.LE)(_comparison("OLE", "SLE", "ULE"))
register(ps.GE)(_comparison("OGE", "SGE", "UGE"))
register(ps.EQ)(_comparison("OEQ", "EQ", "EQ"))
register(ps.NEQ)(_comparison("UNE", "NE", "NE"))


@register(ps.IsNan)
def _(b, args, out):
    x, d = args[0]
    if not is_float(d):
        return constant(b, out, 0)
    return cast(b, b.fcmp(ir.FCmp.UNO, x, x), "bool", out)


@register(ps.IsInf)
def _(b, args, out):
    x, d = args[0]
    if not is_float(d):
        return constant(b, out, 0)
    c = b.fcmp(ir.FCmp.OEQ, ir_intrinsic(b, "Fabs", x), constant(b, d, math.inf))
    return cast(b, c, "bool", out)


@register(ps.Switch)
def _(b, args, out):
    return b.select(conv(b, args, 0, "bool"), conv(b, args, 1, out), conv(b, args, 2, out))


def _bitwise(method):
    def fn(b, args, out):
        if is_float(out):
            raise NotImplementedError(f"LLVM backend: bitwise op on {out}")
        return getattr(b, method)(conv(b, args, 0, out), conv(b, args, 1, out))

    return fn


register(ps.AND)(_bitwise("and_"))
register(ps.OR)(_bitwise("or_"))
register(ps.XOR)(_bitwise("xor"))


@register(ps.Invert)
def _(b, args, out):
    if is_float(out):
        raise NotImplementedError(f"LLVM backend: invert on {out}")
    return b.not_(conv(b, args, 0, out))


@register(ps.Identity, ps.Cast)
def _(b, args, out):
    return conv(b, args, 0, out)


@register(ps.Second)
def _(b, args, out):
    return conv(b, args, 1, out)


# ------------------------------------------------------------------ composites


def emit_inner(b, fg, args):
    """Emit the scalar graph `fg` on `args` ((value, dtype) per input); return the
    (value, dtype) of each output."""
    vals = {}
    for i, (v, d) in zip(fg.inputs, args, strict=True):
        vals[i] = (cast(b, v, d, i.type.dtype), i.type.dtype)

    def get(var):
        if var in vals:
            return vals[var]
        if isinstance(var, Constant):
            d = check_dtype(var.type.dtype)
            return constant(b, d, var.data.item() if hasattr(var.data, "item") else var.data), d
        raise NotImplementedError(f"LLVM backend: unresolved scalar variable {var}")

    for node in fg.toposort():
        outs = emit_scalar(node.op, b, [get(i) for i in node.inputs], [o.type.dtype for o in node.outputs])
        for o, v in zip(node.outputs, outs, strict=True):
            vals[o] = (v, o.type.dtype)
    return [get(o) for o in fg.outputs]


@llvm_scalar.register(ps.Composite)
def _(op, b, args, out_dtypes):
    outs = emit_inner(b, op.fgraph, args)
    return [cast(b, v, d, od) for (v, d), od in zip(outs, out_dtypes, strict=True)]


@llvm_scalar.register(ScalarLoop)
def _(op, b, args, out_dtypes):
    """The loop is a `fold` whose body is the inner graph, emitted inline; a while
    loop stops after the first step whose until flag is set (and returns the flag)."""
    (n, nd), *rest = args
    fg = op.fgraph
    n_carry = len(fg.outputs) - (1 if op.is_while else 0)
    carry_dtypes = [i.type.dtype for i in fg.inputs[:n_carry]]
    init = [cast(b, v, d, cd) for (v, d), cd in zip(rest[:n_carry], carry_dtypes, strict=True)]
    fixed = rest[n_carry:]

    def body(_i, *carry):
        outs = emit_inner(b, fg, [*zip(carry, carry_dtypes), *fixed])
        new = [cast(b, v, d, cd) for (v, d), cd in zip(outs[:n_carry], carry_dtypes, strict=True)]
        if op.is_while:
            new.append(cast(b, *outs[n_carry], "bool"))
        return new

    res = b.fold(cast(b, n, nd, "int64"), init, body, until=op.is_while)
    dtypes = [*carry_dtypes, "bool"] if op.is_while else carry_dtypes
    return [cast(b, v, d, od) for v, d, od in zip(res, dtypes, out_dtypes, strict=True)]


__all__ = ["emit_scalar", "llvm_scalar", "value_type"]
