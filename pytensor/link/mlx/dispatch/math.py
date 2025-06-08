from functools import singledispatch

import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx
from pytensor.scalar.basic import (
    AND,
    EQ,
    GE,
    GT,
    LE,
    LT,
    NEQ,
    OR,
    Abs,
    Add,
    Cast,
    Cos,
    Exp,
    IntDiv,
    Invert,
    IsNan,
    Log,
    Log1p,
    Mul,
    Neg,
    Pow,
    ScalarMaximum,
    ScalarMinimum,
    Sign,
    Sin,
    Sqr,
    Sqrt,
    Sub,
    Switch,
    TrueDiv,
)
from pytensor.scalar.math import Erfc, Erfcx, Sigmoid, Softplus
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Dot


@mlx_funcify.register(Dot)
def mlx_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return mx.matmul(x, y)

    return dot


# Second-level dispatch for scalar operations in Elemwise
@singledispatch
def mlx_funcify_Elemwise_scalar_op(scalar_op):
    """Simplified implementation for MLX scalar operations."""

    # Try using the operation name directly (most common case)
    op_name = getattr(scalar_op, "name", None)
    if op_name is not None:
        try:
            mlx_func = getattr(mx, op_name)
            # Handle variadic functions like Add
            if hasattr(scalar_op, "inputs") and len(scalar_op.inputs) > 2:

                def variadic_func(*args):
                    result = args[0]
                    for arg in args[1:]:
                        result = mlx_func(result, arg)
                    return result

                return variadic_func
            else:
                return mlx_func
        except AttributeError:
            pass

    raise NotImplementedError(f"MLX does not support Elemwise scalar op {scalar_op}")


@mlx_funcify_Elemwise_scalar_op.register(Add)
def _(scalar_op):
    def add(*args):
        result = args[0]
        for arg in args[1:]:
            result = mx.add(result, arg)
        return result

    return add


@mlx_funcify_Elemwise_scalar_op.register(Sub)
def _(scalar_op):
    def sub(x, y):
        return mx.subtract(x, y)

    return sub


@mlx_funcify_Elemwise_scalar_op.register(Mul)
def _(scalar_op):
    def mul(*args):
        result = args[0]
        for arg in args[1:]:
            result = mx.multiply(result, arg)
        return result

    return mul


@mlx_funcify_Elemwise_scalar_op.register(TrueDiv)
def _(scalar_op):
    def true_div(x, y):
        return mx.divide(x, y)

    return true_div


@mlx_funcify_Elemwise_scalar_op.register(IntDiv)
def _(scalar_op):
    def int_div(x, y):
        return mx.floor_divide(x, y)

    return int_div


@mlx_funcify_Elemwise_scalar_op.register(Pow)
def _(scalar_op):
    def pow(x, y):
        return mx.power(x, y)

    return pow


@mlx_funcify_Elemwise_scalar_op.register(Exp)
def _(scalar_op):
    def exp(x):
        return mx.exp(x)

    return exp


@mlx_funcify_Elemwise_scalar_op.register(Log)
def _(scalar_op):
    def log(x):
        return mx.log(x)

    return log


@mlx_funcify_Elemwise_scalar_op.register(Log1p)
def _(scalar_op):
    def log1p(x):
        return mx.log1p(x)

    return log1p


@mlx_funcify_Elemwise_scalar_op.register(Sin)
def _(scalar_op):
    def sin(x):
        return mx.sin(x)

    return sin


@mlx_funcify_Elemwise_scalar_op.register(Cos)
def _(scalar_op):
    def cos(x):
        return mx.cos(x)

    return cos


@mlx_funcify_Elemwise_scalar_op.register(Sqrt)
def _(scalar_op):
    def sqrt(x):
        return mx.sqrt(x)

    return sqrt


@mlx_funcify_Elemwise_scalar_op.register(Sqr)
def _(scalar_op):
    def sqr(x):
        return mx.square(x)

    return sqr


@mlx_funcify_Elemwise_scalar_op.register(Abs)
def _(scalar_op):
    def abs(x):
        return mx.abs(x)

    return abs


@mlx_funcify_Elemwise_scalar_op.register(Neg)
def _(scalar_op):
    def neg(x):
        return mx.negative(x)

    return neg


@mlx_funcify_Elemwise_scalar_op.register(Sign)
def _(scalar_op):
    def sign(x):
        return mx.sign(x)

    return sign


@mlx_funcify_Elemwise_scalar_op.register(LE)
def _(scalar_op):
    def le(x, y):
        return mx.less_equal(x, y)

    return le


@mlx_funcify_Elemwise_scalar_op.register(LT)
def _(scalar_op):
    def lt(x, y):
        return mx.less(x, y)

    return lt


@mlx_funcify_Elemwise_scalar_op.register(GE)
def _(scalar_op):
    def ge(x, y):
        return mx.greater_equal(x, y)

    return ge


@mlx_funcify_Elemwise_scalar_op.register(GT)
def _(scalar_op):
    def gt(x, y):
        return mx.greater(x, y)

    return gt


@mlx_funcify_Elemwise_scalar_op.register(EQ)
def _(scalar_op):
    def eq(x, y):
        return mx.equal(x, y)

    return eq


@mlx_funcify_Elemwise_scalar_op.register(NEQ)
def _(scalar_op):
    def neq(x, y):
        return mx.not_equal(x, y)

    return neq


@mlx_funcify_Elemwise_scalar_op.register(Switch)
def _(scalar_op):
    def switch(cond, x, y):
        return mx.where(cond, x, y)

    return switch


@mlx_funcify_Elemwise_scalar_op.register(AND)
def _(scalar_op):
    def bitwise_and(x, y):
        return mx.bitwise_and(x, y)

    return bitwise_and


@mlx_funcify_Elemwise_scalar_op.register(OR)
def _(scalar_op):
    def bitwise_or(x, y):
        return mx.bitwise_or(x, y)

    return bitwise_or


@mlx_funcify_Elemwise_scalar_op.register(ScalarMaximum)
def _(scalar_op):
    def maximum(x, y):
        return mx.maximum(x, y)

    return maximum


@mlx_funcify_Elemwise_scalar_op.register(ScalarMinimum)
def _(scalar_op):
    def minimum(x, y):
        return mx.minimum(x, y)

    return minimum


@mlx_funcify_Elemwise_scalar_op.register(Cast)
def _(scalar_op):
    def cast(x):
        dtype = convert_dtype_to_mlx(scalar_op.o_type.dtype)
        return x.astype(dtype)

    return cast


@mlx_funcify_Elemwise_scalar_op.register(Sigmoid)
def _(scalar_op):
    def sigmoid(x):
        return mx.sigmoid(x)

    return sigmoid


@mlx_funcify_Elemwise_scalar_op.register(Invert)
def _(scalar_op):
    def invert(x):
        return mx.bitwise_invert(x)

    return invert


@mlx_funcify_Elemwise_scalar_op.register(IsNan)
def _(scalar_op):
    def isnan(x):
        return mx.isnan(x)

    return isnan


@mlx_funcify_Elemwise_scalar_op.register(Erfc)
def _(scalar_op):
    def erfc(x):
        return 1.0 - mx.erf(x)

    return erfc


@mlx_funcify_Elemwise_scalar_op.register(Erfcx)
def _(scalar_op):
    def erfcx(x):
        return mx.exp(x * x) * (1.0 - mx.erf(x))

    return erfcx


@mlx_funcify_Elemwise_scalar_op.register(Softplus)
def _(scalar_op):
    def softplus(x):
        # Numerically stable implementation of log(1 + exp(x))
        # Following the same logic as the original PyTensor implementation
        return mx.where(
            x < -37.0,
            mx.exp(x),
            mx.where(
                x < 18.0, mx.log1p(mx.exp(x)), mx.where(x < 33.3, x + mx.exp(-x), x)
            ),
        )

    return softplus


@mlx_funcify.register(Elemwise)
def mlx_funcify_Elemwise(op, node, **kwargs):
    # Dispatch to the appropriate scalar op handler
    scalar_func = mlx_funcify_Elemwise_scalar_op(op.scalar_op)

    def elemwise(*inputs):
        return scalar_func(*inputs)

    return elemwise
