import torch.linalg

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.slinalg import Cholesky, SolveTriangular


@pytorch_funcify.register(Cholesky)
def pytorch_funcify_Cholesky(op, **kwargs):
    lower = op.lower

    def cholesky(a, lower=lower):
        return torch.linalg.cholesky(a, upper=not lower)

    return cholesky


@pytorch_funcify.register(SolveTriangular)
def pytorch_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    trans = op.trans
    unit_diagonal = op.unit_diagonal

    def solve_triangular(A, b):
        return torch.linalg.solve_triangular(
            A, b, upper=not lower, unit_triangle=unit_diagonal, left=trans == "T"
        )

    return solve_triangular
