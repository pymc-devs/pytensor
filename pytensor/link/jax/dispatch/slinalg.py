import jax

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.slinalg import (
    BlockDiagonal,
    Cholesky,
    Eigvalsh,
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
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b
    transposed = op.transposed

    def solve(a, b):
        if transposed:
            a = a.T

        return jax.scipy.linalg.solve(
            a,
            b,
            assume_a=assume_a,
            lower=lower,
            check_finite=check_finite,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
        )

    return solve


@jax_funcify.register(SolveTriangular)
def jax_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    trans = op.trans
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite

    def solve_triangular(A, b):
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            lower=lower,
            trans=trans,
            unit_diagonal=unit_diagonal,
            check_finite=check_finite,
        )

    return solve_triangular


@jax_funcify.register(BlockDiagonal)
def jax_funcify_BlockDiagonalMatrix(op, **kwargs):
    def block_diag(*inputs):
        return jax.scipy.linalg.block_diag(*inputs)

    return block_diag
