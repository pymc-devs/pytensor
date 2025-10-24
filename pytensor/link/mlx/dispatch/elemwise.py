from functools import singledispatch

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
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
    IsInf,
    IsNan,
    Log,
    Log1p,
    Maximum,
    Minimum,
    Mul,
    Neg,
    Pow,
    Sign,
    Sin,
    Sqr,
    Sqrt,
    Sub,
    Switch,
    TrueDiv,
)
from pytensor.scalar.math import Erfc, Erfcx, Sigmoid, Softplus
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@mlx_funcify.register(DimShuffle)
def mlx_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        # Convert scalar to array if needed
        if isinstance(x, int | float) or (
            isinstance(x, np.number) and not isinstance(x, np.ndarray)
        ):
            x = mx.array(x)
        res = mx.transpose(x, op.transposition)
        shape = list(res.shape[: len(op.shuffle)])
        for augm in op.augment:
            shape.insert(augm, 1)
        return mx.reshape(res, shape)

    return dimshuffle


# Second-level dispatch for scalar operations in CAReduce
@singledispatch
def mlx_funcify_CAReduce_scalar_op(scalar_op, axis):
    raise NotImplementedError(
        f"MLX does not support CAReduce with scalar op {scalar_op}"
    )


@mlx_funcify.register(CAReduce)
def mlx_funcify_CAReduce(op, **kwargs):
    return mlx_funcify_CAReduce_scalar_op(op.scalar_op, op.axis)


@mlx_funcify_CAReduce_scalar_op.register(Add)
def mlx_funcify_CAReduce_scalar_Add(scalar_op, axis):
    def sum_reduce(x):
        return mx.sum(x, axis=axis)

    return sum_reduce


@mlx_funcify_CAReduce_scalar_op.register(Mul)
def mlx_funcify_CAReduce_scalar_Mul(scalar_op, axis):
    def prod_reduce(x):
        return mx.prod(x, axis=axis)

    return prod_reduce


@mlx_funcify_CAReduce_scalar_op.register(AND)
def mlx_funcify_CAReduce_scalar_AND(scalar_op, axis):
    def all_reduce(x):
        return x.all(axis=axis)

    return all_reduce


@mlx_funcify_CAReduce_scalar_op.register(OR)
def mlx_funcify_CARreduce_OR(scalar_op, axis):
    def any_reduce(x):
        return mx.any(x, axis=axis)

    return any_reduce


@mlx_funcify_CAReduce_scalar_op.register(Maximum)
def mlx_funcify_CARreduce_Maximum(scalar_op, axis):
    def max_reduce(x):
        return mx.max(x, axis=axis)

    return max_reduce


@mlx_funcify_CAReduce_scalar_op.register(Minimum)
def mlx_funcify_CARreduce_Minimum(scalar_op, axis):
    def min_reduce(x):
        return mx.min(x, axis=axis)

    return min_reduce


@mlx_funcify.register(Softmax)
def mlx_funcify_Softmax(op, **kwargs):
    axis = op.axis

    def softmax(x):
        return mx.softmax(x, axis=axis)

    return softmax


@mlx_funcify.register(SoftmaxGrad)
def mlx_funcify_SoftmaxGrad(op, **kwargs):
    axis = op.axis

    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        return dy_times_sm - mx.sum(dy_times_sm, axis=axis, keepdims=True) * sm

    return softmax_grad


@mlx_funcify.register(LogSoftmax)
def mlx_funcify_LogSoftmax(op, **kwargs):
    axis = op.axis

    def log_softmax(x):
        return mlx_nn.log_softmax(x, axis=axis)

    return log_softmax


@mlx_funcify.register(Softplus)
def mlx_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return mx.where(
            x < -37.0,
            mx.exp(x),
            mx.where(
                x < 18.0,
                mx.log1p(mx.exp(x)),
                mx.where(
                    x < 33.3,
                    x + mx.exp(-x),
                    x,
                ),
            ),
        )

    return softplus


@mlx_funcify.register(Cast)
def mlx_funcify_Cast(op, **kwargs):
    def cast(x):
        dtype = convert_dtype_to_mlx(op.scalar_op.o_type.dtype)
        try:
            return x.astype(dtype)
        except ValueError as e:
            if "is not supported on the GPU" in str(e):
                # MLX GPU limitation - try auto-casting with warning
                import warnings

                warnings.warn(
                    f"MLX GPU limitation: {e}. Attempting automatic fallback casting.",
                    UserWarning,
                    stacklevel=2,
                )
                # Get the auto-cast version
                fallback_dtype = convert_dtype_to_mlx(
                    op.scalar_op.o_type.dtype, auto_cast_unsupported=True
                )
                return x.astype(fallback_dtype)
            else:
                # Re-raise other ValueError exceptions
                raise

    return cast


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
def mlx_funcify_Elemwise_scalar_Add(scalar_op):
    def add(*args):
        result = args[0]
        for arg in args[1:]:
            result = mx.add(result, arg)
        return result

    return add


@mlx_funcify_Elemwise_scalar_op.register(Sub)
def mlx_funcify_Elemwise_scalar_Sub(scalar_op):
    return mx.subtract


@mlx_funcify_Elemwise_scalar_op.register(Mul)
def mlx_funcify_Elemwise_scalar_Mul(scalar_op):
    def mul(*args):
        result = args[0]
        for arg in args[1:]:
            result = mx.multiply(result, arg)
        return result

    return mul


@mlx_funcify_Elemwise_scalar_op.register(TrueDiv)
def mlx_funcify_Elemwise_scalar_TrueDiv(scalar_op):
    return mx.divide


@mlx_funcify_Elemwise_scalar_op.register(IntDiv)
def mlx_funcify_Elemwise_scalar_IntDiv(scalar_op):
    return mx.floor_divide


@mlx_funcify_Elemwise_scalar_op.register(Pow)
def mlx_funcify_Elemwise_scalar_Pow(scalar_op):
    return mx.power


@mlx_funcify_Elemwise_scalar_op.register(Exp)
def mlx_funcify_Elemwise_scalar_Exp(scalar_op):
    return mx.exp


@mlx_funcify_Elemwise_scalar_op.register(Log)
def mlx_funcify_Elemwise_scalar_Log(scalar_op):
    return mx.log


@mlx_funcify_Elemwise_scalar_op.register(Log1p)
def mlx_funcify_Elemwise_scalar_Log1p(scalar_op):
    return mx.log1p


@mlx_funcify_Elemwise_scalar_op.register(Sin)
def mlx_funcify_Elemwise_scalar_Sin(scalar_op):
    return mx.sin


@mlx_funcify_Elemwise_scalar_op.register(Cos)
def mlx_funcify_Elemwise_scalar_Cos(scalar_op):
    return mx.cos


@mlx_funcify_Elemwise_scalar_op.register(Sqrt)
def mlx_funcify_Elemwise_scalar_Sqrt(scalar_op):
    return mx.sqrt


@mlx_funcify_Elemwise_scalar_op.register(Sqr)
def mlx_funcify_Elemwise_scalar_Sqr(scalar_op):
    return mx.square


@mlx_funcify_Elemwise_scalar_op.register(Abs)
def mlx_funcify_Elemwise_scalar_Abs(scalar_op):
    return mx.abs


@mlx_funcify_Elemwise_scalar_op.register(Neg)
def mlx_funcify_Elemwise_scalar_Neg(scalar_op):
    return mx.negative


@mlx_funcify_Elemwise_scalar_op.register(Sign)
def mlx_funcify_Elemwise_scalar_Sign(scalar_op):
    return mx.sign


@mlx_funcify_Elemwise_scalar_op.register(LE)
def mlx_funcify_Elemwise_scalar_LE(scalar_op):
    return mx.less_equal


@mlx_funcify_Elemwise_scalar_op.register(LT)
def mlx_funcify_Elemwise_scalar_LT(scalar_op):
    return mx.less


@mlx_funcify_Elemwise_scalar_op.register(GE)
def mlx_funcify_Elemwise_scalar_GE(scalar_op):
    return mx.greater_equal


@mlx_funcify_Elemwise_scalar_op.register(GT)
def mlx_funcify_Elemwise_scalar_GT(scalar_op):
    return mx.greater


@mlx_funcify_Elemwise_scalar_op.register(EQ)
def mlx_funcify_Elemwise_scalar_EQ(scalar_op):
    return mx.equal


@mlx_funcify_Elemwise_scalar_op.register(NEQ)
def mlx_funcify_Elemwise_scalar_NEQ(scalar_op):
    return mx.not_equal


@mlx_funcify_Elemwise_scalar_op.register(Switch)
def mlx_funcify_Elemwise_scalar_Switch(scalar_op):
    return mx.where


@mlx_funcify_Elemwise_scalar_op.register(AND)
def mlx_funcify_Elemwise_scalar_AND(scalar_op):
    return mx.bitwise_and


@mlx_funcify_Elemwise_scalar_op.register(OR)
def mlx_funcify_Elemwise_scalar_OR(scalar_op):
    return mx.bitwise_or


@mlx_funcify_Elemwise_scalar_op.register(Maximum)
def mlx_funcify_Elemwise_scalar_Maximum(scalar_op):
    return mx.maximum


@mlx_funcify_Elemwise_scalar_op.register(Minimum)
def mlx_funcify_Elemwise_scalar_Minimum(scalar_op):
    return mx.minimum


@mlx_funcify_Elemwise_scalar_op.register(Cast)
def mlx_funcify_Elemwise_scalar_Cast(scalar_op):
    def cast(x):
        dtype = convert_dtype_to_mlx(scalar_op.o_type.dtype)
        try:
            return x.astype(dtype)
        except ValueError as e:
            if "is not supported on the GPU" in str(e):
                import warnings

                warnings.warn(
                    f"MLX GPU limitation: {e}. Attempting automatic fallback casting.",
                    UserWarning,
                    stacklevel=2,
                )
                fallback_dtype = convert_dtype_to_mlx(
                    scalar_op.o_type.dtype, auto_cast_unsupported=True
                )
                return x.astype(fallback_dtype)
            else:
                raise e

    return cast


@mlx_funcify_Elemwise_scalar_op.register(Sigmoid)
def mlx_funcify_Elemwise_scalar_Sigmoid(scalar_op):
    return mx.sigmoid


@mlx_funcify_Elemwise_scalar_op.register(Invert)
def mlx_funcify_Elemwise_scalar_Invert(scalar_op):
    return mx.bitwise_invert


@mlx_funcify_Elemwise_scalar_op.register(IsNan)
def mlx_funcify_Elemwise_scalar_IsNan(scalar_op):
    return mx.isnan


@mlx_funcify_Elemwise_scalar_op.register(IsInf)
def mlx_funcify_Elemwise_scalar_IsInf(scalar_op):
    return mx.isinf


@mlx_funcify_Elemwise_scalar_op.register(Erfc)
def mlx_funcify_Elemwise_scalar_Erfc(scalar_op):
    def erfc(x):
        return 1.0 - mx.erf(x)

    return erfc


@mlx_funcify_Elemwise_scalar_op.register(Erfcx)
def mlx_funcify_Elemwise_scalar_Erfcx(scalar_op):
    def erfcx(x):
        return mx.exp(x * x) * (1.0 - mx.erf(x))

    return erfcx


@mlx_funcify_Elemwise_scalar_op.register(Softplus)
def mlx_funcify_Elemwise_scalar_softplus(scalar_op):
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
    return mlx_funcify_Elemwise_scalar_op(op.scalar_op)
