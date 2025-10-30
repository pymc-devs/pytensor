import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.nlinalg import SVD, KroneckerProduct, MatrixInverse, MatrixPinv


@mlx_funcify.register(SVD)
def mlx_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    X_dtype = getattr(mx, node.inputs[0].dtype)

    if not full_matrices:
        raise TypeError("full_matrices=False is not supported in the mlx backend.")

    def svd_S_only(x):
        return mx.linalg.svd(
            x.astype(dtype=X_dtype, stream=mx.cpu), compute_uv=False, stream=mx.cpu
        )

    def svd_full(x):
        outputs = mx.linalg.svd(
            x.astype(dtype=X_dtype, stream=mx.cpu), compute_uv=True, stream=mx.cpu
        )
        return outputs

    if compute_uv:
        return svd_full
    else:
        return svd_S_only


@mlx_funcify.register(KroneckerProduct)
def mlx_funcify_KroneckerProduct(op, node, **kwargs):
    otype = node.outputs[0].dtype
    stream = mx.cpu if otype == "float64" else mx.gpu

    A_dtype = getattr(mx, node.inputs[0].dtype)
    B_dtype = getattr(mx, node.inputs[1].dtype)

    def kron(a, b):
        return mx.kron(
            a.astype(dtype=A_dtype, stream=stream),
            b.astype(dtype=B_dtype, stream=stream),
            stream=stream,
        )

    return kron


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
