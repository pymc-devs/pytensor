import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eig
from pytensor.tensor.linalg.decomposition.lu import LU
from pytensor.tensor.linalg.decomposition.svd import SVD


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


@mlx_funcify.register(Cholesky)
def mlx_funcify_Cholesky(op, node, **kwargs):
    lower = op.lower
    a_dtype = getattr(mx, node.inputs[0].dtype)

    def cholesky(a):
        return mx.linalg.cholesky(
            a.astype(dtype=a_dtype, stream=mx.cpu), upper=not lower, stream=mx.cpu
        )

    return cholesky


@mlx_funcify.register(LU)
def mlx_funcify_LU(op, node, **kwargs):
    permute_l = op.permute_l
    A_dtype = getattr(mx, node.inputs[0].dtype)
    p_indices = op.p_indices

    if permute_l:
        raise ValueError("permute_l=True is not supported in the mlx backend.")
    if not p_indices:
        raise ValueError("p_indices=False is not supported in the mlx backend.")

    def lu(a):
        p_idx, L, U = mx.linalg.lu(
            a.astype(dtype=A_dtype, stream=mx.cpu), stream=mx.cpu
        )

        return (
            p_idx.astype(mx.int32, stream=mx.cpu),
            L,
            U,
        )

    return lu


@mlx_funcify.register(Eig)
def mlx_funcify_Eig(op, node, **kwargs):
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def eig(x):
        return mx.linalg.eig(x.astype(dtype=X_dtype, stream=mx.cpu), stream=mx.cpu)

    return eig
