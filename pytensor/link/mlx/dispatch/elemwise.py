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


@mlx_funcify.register(CAReduce)
def mlx_funcify_CAReduce(op, **kwargs):
    if isinstance(op.scalar_op, Add):

        def sum(x):
            return mx.sum(x, axis=op.axis)

        return sum
    elif isinstance(op.scalar_op, Mul):

        def prod(x):
            return mx.prod(x, axis=op.axis)

        return prod
    elif isinstance(op.scalar_op, AND):

        def all(x):
            return x.all(axis=op.axis)

        return all
    elif isinstance(op.scalar_op, OR):

        def any(x):
            return mx.any(x, axis=op.axis)

        return any
    else:
        raise NotImplementedError(f"MLX does not support Elemwise {op.scalar_op}")


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
