import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.nlinalg import SVD, KroneckerProduct, MatrixInverse, MatrixPinv


@mlx_funcify.register(SVD)
def mlx_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv
    otype = (
        getattr(mx, node.outputs[0].dtype)
        if not compute_uv
        else [getattr(mx, output.dtype) for output in node.outputs]
    )

    if not full_matrices:
        raise TypeError("full_matrices=False is not supported in the mlx backend.")

    def svd_S_only(x):
        return mx.linalg.svd(x, compute_uv=False, stream=mx.cpu).astype(
            otype, stream=mx.cpu
        )

    def svd_full(x):
        outputs = mx.linalg.svd(x, compute_uv=True, stream=mx.cpu)
        return tuple(
            output.astype(typ, stream=mx.cpu)
            for output, typ in zip(outputs, otype, strict=True)
        )

    if compute_uv:
        return svd_full
    else:
        return svd_S_only


@mlx_funcify.register(KroneckerProduct)
def mlx_funcify_KroneckerProduct(op, node, **kwargs):
    otype = node.outputs[0].dtype
    mx_otype = getattr(mx, otype)
    stream = mx.cpu if otype == "float64" else mx.gpu

    def kron(a, b):
        return mx.kron(a, b, stream=stream).astype(mx_otype, stream=stream)

    return kron


@mlx_funcify.register(MatrixInverse)
def mlx_funcify_MatrixInverse(op, node, **kwargs):
    otype = getattr(mx, node.outputs[0].dtype)

    def inv(x):
        return mx.linalg.inv(x, stream=mx.cpu).astype(otype, stream=mx.cpu)

    return inv


@mlx_funcify.register(MatrixPinv)
def mlx_funcify_MatrixPinv(op, node, **kwargs):
    otype = getattr(mx, node.outputs[0].dtype)

    def pinv(x):
        return mx.linalg.pinv(x, stream=mx.cpu).astype(otype, stream=mx.cpu)

    return pinv
