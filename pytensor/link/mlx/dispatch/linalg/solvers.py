import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.blockwise import mlx_funcify_batched
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


def _unit_diagonal(A):
    """Replace the stored diagonal of ``A`` with ones.

    Call within a CPU stream context. Broadcasts over any leading batch dims.
    """
    eye = mx.eye(A.shape[-1], dtype=A.dtype)
    return A * (1 - eye) + eye


@mlx_funcify.register(SolveTriangular)
def mlx_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    def solve_triangular(A, b):
        with mx.stream(mx.cpu):
            A = A.astype(dtype=A_dtype)
            b = b.astype(dtype=b_dtype)
            if unit_diagonal:
                # `mx.linalg.solve_triangular` has no `unit_diagonal` argument,
                # so the stored diagonal would be used instead of ones and the
                # wrong answer returned silently (#2384).
                A = _unit_diagonal(A)
            return mx.linalg.solve_triangular(A, b, upper=not lower)

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


def _as_column(b, is_vector):
    """Give a vector right-hand side an explicit trailing column axis.

    MLX only special-cases a 1-d `b`; with a batch it rejects `A (*batch, m, m)`
    against `b (*batch, m)`, so the column has to be materialised.
    """
    return mx.expand_dims(b, -1) if is_vector else b


@mlx_funcify_batched.register(Solve)
def mlx_funcify_batched_Solve(op, node, **kwargs):
    """`mx.linalg.solve` is already batched, so `Blockwise` must not `vmap` it.

    `mx.vmap` has no rule for the `LUF` primitive that `solve` builds, so every
    batched solve raised "[Primitive::vmap] Not implemented for LUF" (#2385).
    """
    a_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)
    b_is_vector = node.inputs[1].type.ndim == 1

    def solve(a, b):
        with mx.stream(mx.cpu):
            out = mx.linalg.solve(
                a.astype(dtype=a_dtype),
                _as_column(b.astype(dtype=b_dtype), b_is_vector),
            )
            return out.squeeze(-1) if b_is_vector else out

    return solve


@mlx_funcify_batched.register(SolveTriangular)
def mlx_funcify_batched_SolveTriangular(op, node, **kwargs):
    """Bypass `mx.vmap`, whose `solve_triangular` rule drops the `upper` flag.

    A batched `solve_triangular` did not raise -- it silently solved the *full*
    system instead of the triangular one, so any input that was not already
    exactly triangular returned wrong numbers. That is the normal case for
    `cho_solve` and `lu_solve`, which pack `L` and `U` into one array. The
    eager `mx.linalg.solve_triangular` batches correctly.
    """
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)
    b_is_vector = node.inputs[1].type.ndim == 1

    def solve_triangular(A, b):
        with mx.stream(mx.cpu):
            A = A.astype(dtype=A_dtype)
            if unit_diagonal:
                A = _unit_diagonal(A)
            out = mx.linalg.solve_triangular(
                A, _as_column(b.astype(dtype=b_dtype), b_is_vector), upper=not lower
            )
            return out.squeeze(-1) if b_is_vector else out

    return solve_triangular


@mlx_funcify_batched.register(CholeskySolve)
def mlx_funcify_batched_CholeskySolve(op, node, **kwargs):
    """As for `SolveTriangular`: `cho_solve` is two triangular solves."""
    lower = op.lower
    c_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)
    b_is_vector = node.inputs[1].type.ndim == 1

    def cho_solve(c, b):
        with mx.stream(mx.cpu):
            c = c.astype(dtype=c_dtype)
            c_T = mx.swapaxes(c, -1, -2)
            L, L_T = (c, c_T) if lower else (c_T, c)

            b = _as_column(b.astype(dtype=b_dtype), b_is_vector)
            y = mx.linalg.solve_triangular(L, b, upper=False)
            out = mx.linalg.solve_triangular(L_T, y, upper=True)
            return out.squeeze(-1) if b_is_vector else out

    return cho_solve
