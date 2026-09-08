import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.psd import CholeskySolve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


@mlx_funcify.register(Solve)
def mlx_funcify_Solve(op, node, **kwargs):
    assume_a = op.assume_a
    a_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    if assume_a != "gen":
        warnings.warn(
            f"MLX solve does not support assume_a={op.assume_a}. Defaulting to assume_a='gen'.",
            UserWarning,
        )

    def solve(a, b):
        return mx.linalg.solve(
            a.astype(stream=mx.cpu, dtype=a_dtype),
            b.astype(stream=mx.cpu, dtype=b_dtype),
            stream=mx.cpu,
        )

    return solve


@mlx_funcify.register(SolveTriangular)
def mlx_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    def solve_triangular(A, b):
        A = A.astype(stream=mx.cpu, dtype=A_dtype)

        if unit_diagonal:
            # MLX's `solve_triangular` has no `unit_diagonal`. LAPACK's `trtrs`
            # never reads the diagonal in that mode, so overwriting it with ones
            # gives the same answer.
            diagonal_mask = mx.eye(A.shape[-1], dtype=mx.bool_, stream=mx.cpu)
            A = mx.where(diagonal_mask, mx.array(1, dtype=A_dtype), A, stream=mx.cpu)

        return mx.linalg.solve_triangular(
            A,
            b.astype(stream=mx.cpu, dtype=b_dtype),
            upper=not lower,
            stream=mx.cpu,
        )

    return solve_triangular


@mlx_funcify.register(CholeskySolve)
def mlx_funcify_CholeskySolve(op, node, **kwargs):
    lower = op.lower
    c_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    # MLX has no cho_solve, so with A = L L.T we solve L y = b then L.T x = y.
    def cho_solve(c, b):
        c = c.astype(stream=mx.cpu, dtype=c_dtype)
        b = b.astype(stream=mx.cpu, dtype=b_dtype)
        c_T = mx.swapaxes(c, -1, -2, stream=mx.cpu)
        L, L_T = (c, c_T) if lower else (c_T, c)

        y = mx.linalg.solve_triangular(L, b, upper=False, stream=mx.cpu)
        return mx.linalg.solve_triangular(L_T, y, upper=True, stream=mx.cpu)

    return cho_solve
