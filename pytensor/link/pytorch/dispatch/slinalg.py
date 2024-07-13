import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.slinalg import (
    BlockDiagonal,
    Cholesky,
    Eigvalsh,
    Solve,
    SolveTriangular,
)


@pytorch_funcify.register(Eigvalsh)
def pytorch_funcify_Eigvalsh(op, **kwargs):
    if op.lower:
        UPLO = "L"
    else:
        UPLO = "U"

    def eigvalsh(a, b):
        if b is not None:
            raise NotImplementedError(
                "torch.linalg.eigvalsh does not support generalized eigenvector problems (b != None)"
            )
        return torch.linalg.eigvalsh(a, UPLO=UPLO)

    return eigvalsh


@pytorch_funcify.register(Cholesky)
def pytorch_funcify_Cholesky(op, **kwargs):
    upper = not op.lower

    def cholesky(a):
        return torch.linalg.cholesky(a, upper=upper)

    return cholesky


@pytorch_funcify.register(Solve)
def pytorch_funcify_Solve(op, **kwargs):
    lower = False
    if op.assume_a != "gen" and op.lower:
        lower = True

    def solve(a, b):
        if lower:
            return torch.linalg.solve(torch.tril(a), b)

        return torch.linalg.solve(a, b)

    return solve


@pytorch_funcify.register(SolveTriangular)
def pytorch_funcify_SolveTriangular(op, **kwargs):
    if op.check_finite:
        raise NotImplementedError(
            "Option check_finite is not implemented in torch.linalg.solve_triangular"
        )

    upper = not op.lower
    unit_diagonal = op.unit_diagonal
    trans = op.trans

    def solve_triangular(A, b):
        A_p = A
        if trans == 1 or trans == "T":
            A_p = A.T

        if trans == 2 or trans == "C":
            A_p = A.H

        b_p = b
        if b.ndim == 1:
            b_p = b[:, None]

        res = torch.linalg.solve_triangular(
            A_p, b_p, upper=upper, unitriangular=unit_diagonal
        )

        if b.ndim == 1 and res.shape[1] == 1:
            return res.flatten()

        return res

    return solve_triangular


@pytorch_funcify.register(BlockDiagonal)
def pytorch_funcify_BlockDiagonalMatrix(op, **kwargs):
    def block_diag(*inputs):
        return torch.block_diag(*inputs)

    return block_diag
