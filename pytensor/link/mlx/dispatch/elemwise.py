import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.scalar.basic import AND, OR, Add, Mul, ScalarMaximum, ScalarMinimum, Switch
from pytensor.tensor.elemwise import CAReduce, DimShuffle
from pytensor.tensor.special import Softmax, SoftmaxGrad


@mlx_funcify.register(DimShuffle)
def mlx_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
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
            return mx.all(x, axis=op.axis)

        return all
    elif isinstance(op.scalar_op, OR):

        def any(x):
            return mx.any(x, axis=op.axis)

        return any
    elif isinstance(op.scalar_op, ScalarMaximum):

        def max(x):
            return mx.max(x, axis=op.axis)

        return max
    elif isinstance(op.scalar_op, ScalarMinimum):

        def min(x):
            return mx.min(x, axis=op.axis)

        return min

    else:
        raise NotImplementedError(f"MLX does not support {op.scalar_op}")


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
