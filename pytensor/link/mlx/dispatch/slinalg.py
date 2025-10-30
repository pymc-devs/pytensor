import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.slinalg import LU, Cholesky, Solve, SolveTriangular


@mlx_funcify.register(Cholesky)
def mlx_funcify_Cholesky(op, node, **kwargs):
    lower = op.lower
    a_dtype = getattr(mx, node.inputs[0].dtype)

    def cholesky(a):
        return mx.linalg.cholesky(
            a.astype(dtype=a_dtype, stream=mx.cpu), upper=not lower, stream=mx.cpu
        )

    return cholesky


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
        # MLX only supports solve on CPU
        return mx.linalg.solve(
            a.astype(stream=mx.cpu, dtype=a_dtype),
            b.astype(stream=mx.cpu, dtype=b_dtype),
            stream=mx.cpu,
        )

    return solve


@mlx_funcify.register(SolveTriangular)
def mlx_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    def solve_triangular(A, b):
        return mx.linalg.solve_triangular(
            A.astype(stream=mx.cpu, dtype=A_dtype),
            b.astype(stream=mx.cpu, dtype=b_dtype),
            upper=not lower,
            stream=mx.cpu,  # MLX only supports solve_triangular on CPU
        )

    return solve_triangular


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
