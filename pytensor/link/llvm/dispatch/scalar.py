"""Scalar ops: one emitter per PyTensor scalar op class.

An emitter gets the builder, the arguments as (value, ScalarType) pairs and the output
types PyTensor computed, and returns the output values. Most ops compute in the output
type: `nums(b, args, out)` gives the arguments as `Num`s of that type, whose operators pick
the instructions.
"""

from __future__ import annotations

import math
import operator
from functools import reduce, singledispatch

import pytensor.scalar.basic as ps
import pytensor.scalar.math as psm
from pytensor.graph.basic import Constant
from pytensor.link.llvm.codegen.emit import cast, constant
from pytensor.link.llvm.codegen.num import Num
from pytensor.link.llvm.codegen.types import (
    element_type,
    ir,
    value_type,
)
from pytensor.link.llvm.dispatch import gamma
from pytensor.scalar.loop import ScalarLoop


@singledispatch
def llvm_scalar(op, b, args: list, out_types: list) -> list:
    raise NotImplementedError(f"LLVM backend: scalar op {op} is not supported")


def emit_scalar(op, b, args, out_types):
    """Emit scalar op `op` on `args` ((value, ScalarType) pairs) for outputs of
    `out_types`; return the output values."""
    out_types = [element_type(t) for t in out_types]
    return llvm_scalar(op, b, args, out_types)


def _single(fn):
    """Register helper: fn(b, args, out_type) -> value (or a Num of the output type)."""

    def wrapper(op, b, args, out_types):
        (out,) = out_types
        result = fn(b, args, out)
        if isinstance(result, Num):
            if result.scalar_type != out:
                raise AssertionError(
                    f"scalar op {op}: emitted {result.scalar_type} for an output of {out}"
                )
            result = result.value
        return [result]

    return wrapper


def register(*classes):
    def deco(fn):
        w = _single(fn)
        for c in classes:
            llvm_scalar.register(c)(w)
        return fn

    return deco


def conv(b, args, k, scalar_type):
    v, d = args[k]
    return cast(b, v, d, scalar_type)


def nums(b, args, scalar_type) -> list[Num]:
    """The arguments converted to `scalar_type`."""
    return [Num(b, v, d).astype(scalar_type) for v, d in args]


def require_float(scalar_type, what):
    if scalar_type not in ps.float_types:
        raise NotImplementedError(
            f"LLVM backend: {what} with output dtype {scalar_type}"
        )


def _upcast(*types):
    import numpy as np

    return element_type(str(np.result_type(*(t.dtype for t in types))))


# ------------------------------------------------------------------ arithmetic


@register(ps.Add)
def _(b, args, out):
    return reduce(operator.add, nums(b, args, out))


@register(ps.Mul)
def _(b, args, out):
    return reduce(operator.mul, nums(b, args, out))


@register(ps.Sub)
def _(b, args, out):
    x, y = nums(b, args, out)
    return x - y


@register(ps.TrueDiv)
def _(b, args, out):
    require_float(out, "true_div")
    x, y = nums(b, args, out)
    return x / y


@register(ps.IntDiv)
def _(b, args, out):
    """Python floor division; x // 0 == 0 for integers, like numpy."""
    x, y = nums(b, args, out)
    return x // y


@register(ps.Mod)
def _(b, args, out):
    """Python modulo (sign of the divisor); x % 0 == 0 for integers, like numpy."""
    x, y = nums(b, args, out)
    return x % y


@register(ps.Pow)
def _(b, args, out):
    if out in ps.float_types:
        return b.libm(ir.Libm.Pow, [conv(b, args, 0, out), conv(b, args, 1, out)])
    if out == ps.bool:
        raise NotImplementedError("LLVM backend: pow with output dtype bool")
    return int_pow(b, conv(b, args, 0, out), conv(b, args, 1, out), out)


def int_pow(b, x, e, scalar_type):
    """x**e by squaring (wrapping like numpy). A negative e gives 1 for x == 1,
    +-1 for x == -1 and 0 otherwise, as an integer truncation of the real power."""
    one, zero = constant(b, scalar_type, 1), constant(b, scalar_type, 0)
    bits = value_type(scalar_type).bits

    def step(_i, acc, base, k):
        odd = b.ne(b.and_(k, one), zero)
        k2 = b.lshr(k, one)
        return [
            b.select(odd, b.mul(acc, base), acc),
            b.mul(base, base),
            k2,
            b.eq(k2, zero),
        ]

    acc, _, _, _ = b.fold(constant(b, ps.int64, bits), [one, x, e], step, until=True)
    if scalar_type not in ps.int_types:
        return acc
    neg = b.slt(e, zero)
    minus_one = constant(b, scalar_type, -1)
    odd_e = b.ne(b.and_(e, one), zero)
    at_neg = b.select(
        b.eq(x, one),
        one,
        b.select(b.eq(x, minus_one), b.select(odd_e, minus_one, one), zero),
    )
    return b.select(neg, at_neg, acc)


def ir_intrinsic(b, name, *xs):
    return b.intrinsic(getattr(ir.Intrinsic, name), list(xs))


@register(ps.Maximum, ps.ScalarMaximum)
def _(b, args, out):
    x, y = nums(b, args, out)
    return x.maximum(y)


@register(ps.Minimum, ps.ScalarMinimum)
def _(b, args, out):
    x, y = nums(b, args, out)
    return x.minimum(y)


@register(ps.Clip)
def _(b, args, out):
    x, lo, hi = nums(b, args, out)
    return x.maximum(lo).minimum(hi)


@register(ps.Neg)
def _(b, args, out):
    (x,) = nums(b, args, out)
    return -x


@register(ps.Abs)
def _(b, args, out):
    (x,) = nums(b, args, out)
    return abs(x)


@register(ps.Sign)
def _(b, args, out):
    (x,) = nums(b, args, out)
    return (x > 0).select(1, (x < 0).select(-1, x))  # keeps 0 and NaN


@register(ps.Sqr)
def _(b, args, out):
    (x,) = nums(b, args, out)
    return x * x


@register(ps.Reciprocal)
def _(b, args, out):
    require_float(out, "reciprocal")
    (x,) = nums(b, args, out)
    return 1.0 / x


# ------------------------------------------------------------------ math functions


def _intrinsic_op(name):
    def fn(b, args, out):
        require_float(out, name)
        return ir_intrinsic(b, name, conv(b, args, 0, out))

    return fn


for _cls, _name in [
    (ps.Sqrt, "Sqrt"),
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
        return b.libm(
            getattr(ir.Libm, name), [conv(b, args, k, out) for k in range(arity)]
        )

    return fn


# libm calls, not intrinsics: LLVM only vectorizes them through libmvec when the call
# carries a vector-function-abi-variant attribute (see Builder.libm)
for _cls, _name in [
    (ps.Exp, "Exp"),
    (ps.Exp2, "Exp2"),
    (ps.Log, "Log"),
    (ps.Log2, "Log2"),
    (ps.Log10, "Log10"),
    (ps.Sin, "Sin"),
    (ps.Cos, "Cos"),
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


register(psm.Gamma)(_rust_op("Gamma"))
register(psm.Erfcx)(_rust_op("Erfcx"))


@register(psm.GammaLn)
def _(b, args, out):
    require_float(out, "gammaln")
    return gamma.gammaln(b, conv(b, args, 0, out), out)


@register(psm.Psi)
def _(b, args, out):
    require_float(out, "psi")
    return gamma.psi(b, conv(b, args, 0, out), out)


@register(psm.Sigmoid)
def _(b, args, out):
    require_float(out, "sigmoid")
    one = constant(b, out, 1.0)
    e = b.libm(ir.Libm.Exp, [b.fneg(conv(b, args, 0, out))])
    return b.fdiv(one, b.fadd(one, e))


@register(psm.Softplus)
def _(b, args, out):
    """max(x, 0) + log1p(exp(-|x|)), branch-free like the C and Numba backends."""
    require_float(out, "softplus")
    (x,) = nums(b, args, out)
    e = b.libm(ir.Libm.Exp, [(-abs(x)).value])
    return x.maximum(0.0) + Num(b, b.libm(ir.Libm.Log1p, [e]), out)


@register(psm.Log1mexp)
def _(b, args, out):
    require_float(out, "log1mexp")
    x = conv(b, args, 0, out)
    # a branch, not a select: where the loop isn't vectorized, only one pair of libm
    # calls runs (the vectorizer if-converts it)
    (result,) = b.if_else(
        b.fcmp(ir.FCmp.OLT, x, constant(b, out, math.log(0.5))),
        lambda: b.libm(ir.Libm.Log1p, [b.fneg(b.libm(ir.Libm.Exp, [x]))]),
        lambda: b.libm(ir.Libm.Log, [b.fneg(b.libm(ir.Libm.Expm1, [x]))]),
    )
    return result


# ------------------------------------------------------------------ comparisons & logic


def _comparison(compare):
    def fn(b, args, out):
        x, y = nums(b, args, _upcast(args[0][1], args[1][1]))
        return compare(x, y).astype(out)

    return fn


register(ps.LT)(_comparison(operator.lt))
register(ps.GT)(_comparison(operator.gt))
register(ps.LE)(_comparison(operator.le))
register(ps.GE)(_comparison(operator.ge))
register(ps.EQ)(_comparison(Num.eq))
register(ps.NEQ)(_comparison(Num.ne))


@register(ps.IsNan)
def _(b, args, out):
    x, d = args[0]
    if d not in ps.float_types:
        return constant(b, out, 0)
    return cast(b, b.fcmp(ir.FCmp.UNO, x, x), ps.bool, out)


@register(ps.IsInf)
def _(b, args, out):
    x, d = args[0]
    if d not in ps.float_types:
        return constant(b, out, 0)
    c = b.fcmp(ir.FCmp.OEQ, ir_intrinsic(b, "Fabs", x), constant(b, d, math.inf))
    return cast(b, c, ps.bool, out)


@register(ps.Switch)
def _(b, args, out):
    x, y = nums(b, args[1:], out)
    return Num(b, *args[0]).astype(ps.bool).select(x, y)


def _bitwise(op):
    def fn(b, args, out):
        if out in ps.float_types:
            raise NotImplementedError(f"LLVM backend: bitwise op on {out}")
        return op(*nums(b, args, out))

    return fn


register(ps.AND)(_bitwise(operator.and_))
register(ps.OR)(_bitwise(operator.or_))
register(ps.XOR)(_bitwise(operator.xor))
register(ps.Invert)(_bitwise(operator.invert))


@register(ps.Identity, ps.Cast)
def _(b, args, out):
    return conv(b, args, 0, out)


@register(ps.Second)
def _(b, args, out):
    return conv(b, args, 1, out)


# ------------------------------------------------------------------ composites


def emit_inner(b, fg, args):
    """Emit the scalar graph `fg` on `args` ((value, ScalarType) per input); return the
    (value, ScalarType) of each output."""
    values = {}
    for i, (v, d) in zip(fg.inputs, args, strict=True):
        values[i] = (cast(b, v, d, element_type(i.type)), element_type(i.type))

    def get(var):
        if var in values:
            return values[var]
        if isinstance(var, Constant):
            d = element_type(var.type)
            return constant(
                b, d, var.data.item() if hasattr(var.data, "item") else var.data
            ), d
        raise NotImplementedError(f"LLVM backend: unresolved scalar variable {var}")

    for node in fg.toposort():
        outs = emit_scalar(
            node.op,
            b,
            [get(i) for i in node.inputs],
            [element_type(var.type) for var in node.outputs],
        )
        for var, v in zip(node.outputs, outs, strict=True):
            values[var] = (v, element_type(var.type))
    return [get(var) for var in fg.outputs]


@llvm_scalar.register(ps.Composite)
def _(op, b, args, out_types):
    outs = emit_inner(b, op.fgraph, args)
    return [cast(b, v, d, od) for (v, d), od in zip(outs, out_types, strict=True)]


@llvm_scalar.register(ScalarLoop)
def _(op, b, args, out_types):
    """The loop is a `fold` whose body is the inner graph, emitted inline; a while
    loop stops after the first step whose until flag is set (and returns the flag)."""
    (n, nd), *rest = args
    fg = op.fgraph
    n_carry = len(fg.outputs) - (1 if op.is_while else 0)
    carry_types = [element_type(i.type) for i in fg.inputs[:n_carry]]
    init = [
        cast(b, v, d, cd)
        for (v, d), cd in zip(rest[:n_carry], carry_types, strict=True)
    ]
    fixed = rest[n_carry:]

    def body(_i, *carry):
        outs = emit_inner(b, fg, [*zip(carry, carry_types), *fixed])
        new = [
            cast(b, v, d, cd)
            for (v, d), cd in zip(outs[:n_carry], carry_types, strict=True)
        ]
        if op.is_while:
            new.append(cast(b, *outs[n_carry], ps.bool))
        return new

    res = b.fold(cast(b, n, nd, ps.int64), init, body, until=op.is_while)
    types = [*carry_types, ps.bool] if op.is_while else carry_types
    return [cast(b, v, d, od) for v, d, od in zip(res, types, out_types, strict=True)]


__all__ = ["emit_scalar", "llvm_scalar", "value_type"]
