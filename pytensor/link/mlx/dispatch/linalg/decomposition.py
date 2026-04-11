import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eig, Eigh, Eigvalsh
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor, PivotToPermutations
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


@mlx_funcify.register(Eigh)
def mlx_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def eigh(x):
        return mx.linalg.eigh(
            x.astype(dtype=X_dtype, stream=mx.cpu), UPLO=uplo, stream=mx.cpu
        )

    return eigh


@mlx_funcify.register(Eigvalsh)
def mlx_funcify_Eigvalsh(op, node, **kwargs):
    UPLO = "L" if op.lower else "U"
    X_dtype = getattr(mx, node.inputs[0].dtype)

    if len(node.inputs) == 2:
        raise NotImplementedError(
            "mlx.core.linalg.eigvalsh does not support generalized "
            "eigenvector problems (b != None)"
        )

    def eigvalsh(a):
        return mx.linalg.eigvalsh(
            a.astype(dtype=X_dtype, stream=mx.cpu), UPLO=UPLO, stream=mx.cpu
        )

    return eigvalsh


@mlx_funcify.register(LUFactor)
def mlx_funcify_LUFactor(op, node, **kwargs):
    A_dtype = getattr(mx, node.inputs[0].dtype)

    def lu_factor(a):
        lu, pivots = mx.linalg.lu_factor(
            a.astype(dtype=A_dtype, stream=mx.cpu), stream=mx.cpu
        )
        return lu, pivots.astype(mx.int32, stream=mx.cpu)

    return lu_factor


@mlx_funcify.register(PivotToPermutations)
def mlx_funcify_PivotToPermutations(op, **kwargs):
    inverse = op.inverse

    def pivot_to_permutations(pivots):
        pivots = mx.array(pivots)
        n = pivots.shape[0]
        p_inv = mx.arange(n, dtype=mx.int32)
        for i in range(n):
            p_inv[i], p_inv[pivots[i]] = p_inv[pivots[i]], p_inv[i]
        if inverse:
            return p_inv
        return mx.argsort(p_inv)

    return pivot_to_permutations
