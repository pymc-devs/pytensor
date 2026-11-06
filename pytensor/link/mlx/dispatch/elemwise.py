from functools import singledispatch

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.scalar.basic import (
    AND,
    OR,
    Add,
    Maximum,
    Minimum,
    Mul,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import MulWithoutZeros
from pytensor.tensor.special import LogSoftmax, Softmax


@mlx_funcify.register(DimShuffle)
def mlx_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        # Convert scalar to array if needed
        if isinstance(x, int | float) or (
            isinstance(x, np.number) and not isinstance(x, np.ndarray)
        ):
            x = mx.array(x)
        res = mx.transpose(x, op._transposition)
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


@mlx_funcify_CAReduce_scalar_op.register(MulWithoutZeros)
def mlx_funcify_CAReduce_scalar_MulWithoutZeros(scalar_op, axis):
    def prod_without_zeros_reduce(x):
        nonzero = x != 0
        prod = mx.prod(mx.where(nonzero, x, 1), axis=axis)
        # All-zero and empty slices reduce to 0, not 1
        return mx.where(mx.any(nonzero, axis=axis), prod, 0)

    return prod_without_zeros_reduce


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


@mlx_funcify.register(LogSoftmax)
def mlx_funcify_LogSoftmax(op, **kwargs):
    axis = op.axis

    def log_softmax(x):
        return mlx_nn.log_softmax(x, axis=axis)

    return log_softmax


@mlx_funcify.register(Elemwise)
def mlx_funcify_Elemwise(op, node, **kwargs):
    base_fn = mlx_funcify(op.scalar_op, node=node, **kwargs)
    if len(node.inputs) < 2:
        return base_fn

    def elemwise(*inputs):
        # Other backends reject a runtime size-1 dim that is not statically
        # broadcastable; match them rather than silently broadcasting here. Bool
        # constants arrive as Python scalars, which have no shape to check
        Elemwise._check_runtime_broadcast(
            node,
            tuple(x if isinstance(x, mx.array) else np.asarray(x) for x in inputs),
        )
        return base_fn(*inputs)

    return elemwise
