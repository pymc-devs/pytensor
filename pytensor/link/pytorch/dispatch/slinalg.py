import torch

from pytensor.link.pytorch.dispatch.basic import torch_funcify
from pytensor.tensor.slinalg import Cholesky, Solve, SolveTriangular


@torch_funcify.register(Cholesky)
def torch_funcify_Cholesky(op, **kwargs):
    lower = op.lower

    def cholesky(a, lower=lower):
        return torch.cholesky(a, upper=not lower).to(a.dtype)

    return cholesky


@torch_funcify.register(Solve)
def torch_funcify_Solve(op, **kwargs):
    if op.assume_a != "gen" and op.lower:
        lower = True
    else:
        lower = False

    def solve(a, b, lower=lower):
        if lower:
            return torch.triangular_solve(b, a, upper=False)[0]
        else:
            return torch.solve(b, a)[0]

    return solve


@torch_funcify.register(SolveTriangular)
def torch_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    trans = op.trans
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite

    def solve_triangular(A, b):
        return torch.triangular_solve(b, A, upper=not lower, transpose=trans)[0]

    return solve_triangular