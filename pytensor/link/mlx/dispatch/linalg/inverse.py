import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor._linalg.inverse import MatrixInverse, MatrixPinv


@mlx_funcify.register(MatrixInverse)
def mlx_funcify_MatrixInverse(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def inv(x):
        return mx.linalg.inv(x.astype(dtype=X_dtype, stream=mx.cpu), stream=mx.cpu)

    return inv


@mlx_funcify.register(MatrixPinv)
def mlx_funcify_MatrixPinv(op, node, **kwargs):
    x_dtype = getattr(mx, node.inputs[0].dtype)

    def pinv(x):
        return mx.linalg.pinv(x.astype(dtype=x_dtype, stream=mx.cpu), stream=mx.cpu)

    return pinv
