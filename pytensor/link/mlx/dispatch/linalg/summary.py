import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.summary import Det, SLogDet


def _lu_det_parts(x):
    """Compute sign and logdet via LU factorization. Call within a CPU stream context."""
    lu, pivots = mx.linalg.lu_factor(x)
    diag_u = mx.diagonal(lu, axis1=-2, axis2=-1)
    n_swaps = mx.sum(pivots != mx.arange(pivots.shape[-1], dtype=pivots.dtype), axis=-1)
    pivot_sign = 1 - 2 * (n_swaps % 2)
    sign = pivot_sign * mx.prod(mx.sign(diag_u), axis=-1)
    logabsdet = mx.sum(mx.log(mx.abs(diag_u)), axis=-1)
    return sign, logabsdet


@mlx_funcify.register(Det)
def mlx_funcify_Det(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def det(x):
        with mx.stream(mx.cpu):
            sign, logabsdet = _lu_det_parts(x.astype(dtype=X_dtype))
            return sign * mx.exp(logabsdet)

    det.natively_batched = True
    return det


@mlx_funcify.register(SLogDet)
def mlx_funcify_SLogDet(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def slogdet(x):
        with mx.stream(mx.cpu):
            return _lu_det_parts(x.astype(dtype=X_dtype))

    slogdet.natively_batched = True
    return slogdet
