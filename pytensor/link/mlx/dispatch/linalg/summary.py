import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.summary import Det, SLogDet


def _lu_det_parts(x):
    """Shared helper: compute sign and logdet via LU factorization."""
    lu, pivots = mx.linalg.lu_factor(x)
    diag_u = mx.diagonal(lu)
    n_swaps = mx.sum(
        pivots != mx.arange(pivots.shape[0], dtype=pivots.dtype),
    )
    pivot_sign = 1 - 2 * (n_swaps % 2)
    sign = mx.multiply(
        pivot_sign,
        mx.prod(mx.sign(diag_u)),
    )
    logabsdet = mx.sum(
        mx.log(mx.abs(diag_u)),
    )
    return sign, logabsdet


@mlx_funcify.register(Det)
def mlx_funcify_Det(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def det(x):
        sign, logabsdet = _lu_det_parts(x.astype(dtype=X_dtype))
        return mx.multiply(sign, mx.exp(logabsdet))

    return det


@mlx_funcify.register(SLogDet)
def mlx_funcify_SLogDet(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def slogdet(x):
        return _lu_det_parts(x.astype(dtype=X_dtype))

    return slogdet
