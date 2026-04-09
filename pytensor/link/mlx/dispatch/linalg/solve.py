import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor._linalg.solve.general import Solve
from pytensor.tensor._linalg.solve.triangular import SolveTriangular


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
    A_dtype = getattr(mx, node.inputs[0].dtype)
    b_dtype = getattr(mx, node.inputs[1].dtype)

    def solve_triangular(A, b):
        return mx.linalg.solve_triangular(
            A.astype(stream=mx.cpu, dtype=A_dtype),
            b.astype(stream=mx.cpu, dtype=b_dtype),
            upper=not lower,
            stream=mx.cpu,
        )

    return solve_triangular
