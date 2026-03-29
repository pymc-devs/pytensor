import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
)


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


def _lu_det_parts(x):
    """Shared helper: compute sign and log|det| via LU factorization."""
    lu, pivots = mx.linalg.lu_factor(x, stream=mx.cpu)
    diag_u = mx.diagonal(lu, stream=mx.cpu)
    n_swaps = mx.sum(
        pivots != mx.arange(pivots.shape[0], dtype=pivots.dtype, stream=mx.cpu),
        stream=mx.cpu,
    )
    pivot_sign = 1 - 2 * (n_swaps % 2)
    sign = mx.multiply(
        pivot_sign,
        mx.prod(mx.sign(diag_u, stream=mx.cpu), stream=mx.cpu),
        stream=mx.cpu,
    )
    logabsdet = mx.sum(
        mx.log(mx.abs(diag_u, stream=mx.cpu), stream=mx.cpu),
        stream=mx.cpu,
    )
    return sign, logabsdet


@mlx_funcify.register(Det)
def mlx_funcify_Det(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def det(x):
        sign, logabsdet = _lu_det_parts(x.astype(dtype=X_dtype, stream=mx.cpu))
        return mx.multiply(sign, mx.exp(logabsdet, stream=mx.cpu), stream=mx.cpu)

    return det


@mlx_funcify.register(SLogDet)
def mlx_funcify_SLogDet(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def slogdet(x):
        return _lu_det_parts(x.astype(dtype=X_dtype, stream=mx.cpu))

    return slogdet


@mlx_funcify.register(Eig)
def mlx_funcify_Eig(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def eig(x):
        return mx.linalg.eig(x.astype(dtype=X_dtype, stream=mx.cpu), stream=mx.cpu)

    return eig


@mlx_funcify.register(Eigh)
def mlx_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def eigh(x):
        return mx.linalg.eigh(
            x.astype(dtype=X_dtype, stream=mx.cpu), UPLO=uplo, stream=mx.cpu
        )

    return eigh


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
