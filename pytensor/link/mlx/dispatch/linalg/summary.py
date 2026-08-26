import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.blockwise import mlx_funcify_batched
from pytensor.tensor.linalg.summary import Det, SLogDet


def _lu_det_parts(x):
    """Compute sign and logdet via LU factorization. Call within a CPU stream context.

    Reduces over the trailing axes only, so a stack of matrices is handled
    directly -- `mx.linalg.lu_factor` is natively batched (#2385).
    """
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

    return det


@mlx_funcify.register(SLogDet)
def mlx_funcify_SLogDet(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def slogdet(x):
        with mx.stream(mx.cpu):
            return _lu_det_parts(x.astype(dtype=X_dtype))

    return slogdet


@mlx_funcify_batched.register(Det)
def mlx_funcify_batched_Det(op, node, **kwargs):
    """`Det` goes through `mx.linalg.lu_factor`, which `mx.vmap` cannot batch (#2385)."""
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def det(x):
        with mx.stream(mx.cpu):
            sign, logabsdet = _lu_det_parts(x.astype(dtype=X_dtype))
            return sign * mx.exp(logabsdet)

    return det


@mlx_funcify_batched.register(SLogDet)
def mlx_funcify_batched_SLogDet(op, node, **kwargs):
    """As for `Det`."""
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def slogdet(x):
        with mx.stream(mx.cpu):
            return _lu_det_parts(x.astype(dtype=X_dtype))

    return slogdet
