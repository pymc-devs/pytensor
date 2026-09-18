import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.psd import CholeskySolve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


def _as_column(b, b_ndim):
    # MLX treats a 2-d rhs as a matrix, so a batched vector rhs needs an explicit column.
    return mx.expand_dims(b, -1, stream=mx.cpu) if b_ndim == 1 else b


def _from_column(out, b_ndim):
    return mx.squeeze(out, -1, stream=mx.cpu) if b_ndim == 1 else out


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

    b_ndim = op.b_ndim

    def solve(a, b):
        out = mx.linalg.solve(
            a.astype(stream=mx.cpu, dtype=a_dtype),
            _as_column(b.astype(stream=mx.cpu, dtype=b_dtype), b_ndim),
            stream=mx.cpu,
        )
        return _from_column(out, b_ndim)

    solve.natively_batched = True
    return solve


@mlx_funcify.register(SolveTriangular)
def mlx_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)
    b_ndim = op.b_ndim

    def solve_triangular(A, b):
        A = A.astype(stream=mx.cpu, dtype=A_dtype)

        if unit_diagonal:
            # MLX's `solve_triangular` has no `unit_diagonal`. LAPACK's `trtrs`
            # never reads the diagonal in that mode, so overwriting it with ones
            # gives the same answer.
            diagonal_mask = mx.eye(A.shape[-1], dtype=mx.bool_, stream=mx.cpu)
            A = mx.where(diagonal_mask, mx.array(1, dtype=A_dtype), A, stream=mx.cpu)

        out = mx.linalg.solve_triangular(
            A,
            _as_column(b.astype(stream=mx.cpu, dtype=b_dtype), b_ndim),
            upper=not lower,
            stream=mx.cpu,
        )
        return _from_column(out, b_ndim)

    solve_triangular.natively_batched = True
    return solve_triangular


@mlx_funcify.register(CholeskySolve)
def mlx_funcify_CholeskySolve(op, node, **kwargs):
    lower = op.lower
    c_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)
    b_ndim = op.b_ndim

    # MLX has no cho_solve, so with A = L L.T we solve L y = b then L.T x = y.
    def cho_solve(c, b):
        c = c.astype(stream=mx.cpu, dtype=c_dtype)
        b = _as_column(b.astype(stream=mx.cpu, dtype=b_dtype), b_ndim)
        c_T = mx.swapaxes(c, -1, -2, stream=mx.cpu)
        L, L_T = (c, c_T) if lower else (c_T, c)

        y = mx.linalg.solve_triangular(L, b, upper=False, stream=mx.cpu)
        out = mx.linalg.solve_triangular(L_T, y, upper=True, stream=mx.cpu)
        return _from_column(out, b_ndim)

    cho_solve.natively_batched = True
    return cho_solve
