import jax

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eig, Eigh, Eigvalsh
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor, PivotToPermutations
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import Schur
from pytensor.tensor.linalg.decomposition.svd import SVD


@jax_funcify.register(SVD)
def jax_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x, full_matrices=full_matrices, compute_uv=compute_uv):
        return jax.numpy.linalg.svd(
            x, full_matrices=full_matrices, compute_uv=compute_uv
        )

    return svd


@jax_funcify.register(Eig)
def jax_funcify_Eig(op, **kwargs):
    def eig(x):
        return jax.numpy.linalg.eig(x)

    return eig


@jax_funcify.register(Eigh)
def jax_funcify_Eigh(op, **kwargs):
    uplo = op.UPLO

    def eigh(x, uplo=uplo):
        return jax.numpy.linalg.eigh(x, UPLO=uplo)

    return eigh


@jax_funcify.register(Eigvalsh)
def jax_funcify_Eigvalsh(op, node, **kwargs):
    if op.lower:
        UPLO = "L"
    else:
        UPLO = "U"

    if len(node.inputs) == 2:
        raise NotImplementedError(
            "jax.numpy.linalg.eigvalsh does not support generalized eigenvector problems (b != None)"
        )

    def eigvalsh(a):
        return jax.numpy.linalg.eigvalsh(a, UPLO=UPLO)

    return eigvalsh


@jax_funcify.register(Cholesky)
def jax_funcify_Cholesky(op, **kwargs):
    lower = op.lower

    def cholesky(a, lower=lower):
        return jax.scipy.linalg.cholesky(a, lower=lower).astype(a.dtype)

    return cholesky


@jax_funcify.register(PivotToPermutations)
def jax_funcify_PivotToPermutation(op, **kwargs):
    inverse = op.inverse

    def pivot_to_permutations(pivots):
        p_inv = jax.lax.linalg.lu_pivots_to_permutation(pivots, pivots.shape[0])
        if inverse:
            return p_inv
        return jax.numpy.argsort(p_inv)

    return pivot_to_permutations


@jax_funcify.register(LU)
def jax_funcify_LU(op, **kwargs):
    permute_l = op.permute_l
    p_indices = op.p_indices

    if p_indices:
        raise ValueError("JAX does not support the p_indices argument")

    def lu(*inputs):
        return jax.scipy.linalg.lu(*inputs, permute_l=permute_l, check_finite=False)

    return lu


@jax_funcify.register(LUFactor)
def jax_funcify_LUFactor(op, **kwargs):
    overwrite_a = op.overwrite_a

    def lu_factor(a):
        return jax.scipy.linalg.lu_factor(
            a, check_finite=False, overwrite_a=overwrite_a
        )

    return lu_factor


@jax_funcify.register(QR)
def jax_funcify_QR(op, **kwargs):
    mode = op.mode

    def qr(x, mode=mode):
        res = jax.scipy.linalg.qr(x, mode=mode)
        return res[0] if len(res) == 1 else res

    return qr


@jax_funcify.register(Schur)
def jax_funcify_Schur(op, **kwargs):
    import warnings

    output = op.output

    if op.sort is not None:
        warnings.warn(
            "jax.scipy.linalg.schur only supports sort=None. The sort argument is ignored."
        )

    def schur(a):
        T, Z = jax.scipy.linalg.schur(a, output=output)
        return T, Z

    return schur
