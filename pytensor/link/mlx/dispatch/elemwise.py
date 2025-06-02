from functools import singledispatch

import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx
from pytensor.scalar import Softplus
from pytensor.scalar.basic import (
    AND,
    OR,
    Add,
    Cast,
    Mul,
    ScalarMaximum,
    ScalarMinimum,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle
from pytensor.tensor.special import Softmax, SoftmaxGrad


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
def mlx_funcify_CAReduce_scalar_op(scalar_op):
    raise NotImplementedError(
        f"MLX does not support CAReduce with scalar op {scalar_op}"
    )


@mlx_funcify_CAReduce_scalar_op.register(Add)
def _(scalar_op):
    def sum_reduce(x, axis):
        return mx.sum(x, axis=axis)

    return sum_reduce


@mlx_funcify_CAReduce_scalar_op.register(Mul)
def _(scalar_op):
    def prod_reduce(x, axis):
        return mx.prod(x, axis=axis)

    return prod_reduce


@mlx_funcify_CAReduce_scalar_op.register(AND)
def _(scalar_op):
    def all_reduce(x, axis):
        return x.all(axis=axis)

    return all_reduce


@mlx_funcify_CAReduce_scalar_op.register(OR)
def _(scalar_op):
    def any_reduce(x, axis):
        return mx.any(x, axis=axis)

    return any_reduce


@mlx_funcify_CAReduce_scalar_op.register(ScalarMaximum)
def _(scalar_op):
    def max_reduce(x, axis):
        return mx.max(x, axis=axis)

    return max_reduce


@mlx_funcify_CAReduce_scalar_op.register(ScalarMinimum)
def _(scalar_op):
    def min_reduce(x, axis):
        return mx.min(x, axis=axis)

    return min_reduce


@mlx_funcify.register(CAReduce)
def mlx_funcify_CAReduce(op, **kwargs):
    # Dispatch to the appropriate scalar op handler
    scalar_reduce_fn = mlx_funcify_CAReduce_scalar_op(op.scalar_op)
    axis = op.axis

    def reduce(x):
        return scalar_reduce_fn(x, axis)

    return reduce


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
        return x.astype(dtype)

    return cast
