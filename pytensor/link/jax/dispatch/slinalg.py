import warnings

import jax

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.slinalg import (
    LU,
    BlockDiagonal,
    Cholesky,
    Eigvalsh,
    LUFactor,
    PivotToPermutations,
    Solve,
    SolveTriangular,
)


@jax_funcify.register(Eigvalsh)
def jax_funcify_Eigvalsh(op, **kwargs):
    if op.lower:
        UPLO = "L"
    else:
        UPLO = "U"

    def eigvalsh(a, b):
        if b is not None:
            raise NotImplementedError(
                "jax.numpy.linalg.eigvalsh does not support generalized eigenvector problems (b != None)"
            )
        return jax.numpy.linalg.eigvalsh(a, UPLO=UPLO)

    return eigvalsh


@jax_funcify.register(Cholesky)
def jax_funcify_Cholesky(op, **kwargs):
    lower = op.lower

    def cholesky(a, lower=lower):
        return jax.scipy.linalg.cholesky(a, lower=lower).astype(a.dtype)

    return cholesky


@jax_funcify.register(Solve)
def jax_funcify_Solve(op, **kwargs):
    assume_a = op.assume_a
    lower = op.lower

    if assume_a == "tridiagonal":
        # jax.scipy.solve does not yet support tridiagonal matrices
        # But there's a jax.lax.linalg.tridiaonal_solve we can use instead.
        def solve(a, b):
            dl = jax.numpy.diagonal(a, offset=-1, axis1=-2, axis2=-1)
            d = jax.numpy.diagonal(a, offset=0, axis1=-2, axis2=-1)
            du = jax.numpy.diagonal(a, offset=1, axis1=-2, axis2=-1)
            return jax.lax.linalg.tridiagonal_solve(dl, d, du, b, lower=lower)

    else:
        if assume_a not in ("gen", "sym", "her", "pos"):
            warnings.warn(
                f"JAX solve does not support assume_a={op.assume_a}. Defaulting to assume_a='gen'.\n"
                f"If appropriate, you may want to set assume_a to one of 'sym', 'pos', 'her' or 'tridiagonal' to improve performance.",
                UserWarning,
            )
            assume_a = "gen"

        def solve(a, b):
            return jax.scipy.linalg.solve(a, b, lower=lower, assume_a=assume_a)

    return solve


@jax_funcify.register(SolveTriangular)
def jax_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite

    def solve_triangular(A, b):
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            lower=lower,
            trans=0,  # this is handled by explicitly transposing A, so it will always be 0 when we get to here.
            unit_diagonal=unit_diagonal,
            check_finite=check_finite,
        )

    return solve_triangular


@jax_funcify.register(BlockDiagonal)
def jax_funcify_BlockDiagonalMatrix(op, **kwargs):
    def block_diag(*inputs):
        return jax.scipy.linalg.block_diag(*inputs)

    return block_diag


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
    check_finite = op.check_finite

    if p_indices:
        raise ValueError("JAX does not support the p_indices argument")

    def lu(*inputs):
        return jax.scipy.linalg.lu(
            *inputs, permute_l=permute_l, check_finite=check_finite
        )

    return lu


@jax_funcify.register(LUFactor)
def jax_funcify_LUFactor(op, **kwargs):
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a

    def lu_factor(a):
        return jax.scipy.linalg.lu_factor(
            a, check_finite=check_finite, overwrite_a=overwrite_a
        )

    return lu_factor
