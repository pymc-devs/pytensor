import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.slinalg import (
    LU,
    QR,
    Cholesky,
    Eigvalsh,
    LUFactor,
    PivotToPermutations,
    Solve,
    SolveTriangular,
)


@mlx_funcify.register(Eigvalsh)
def mlx_funcify_Eigvalsh(op, node, **kwargs):
    UPLO = "L" if op.lower else "U"
    X_dtype = getattr(mx, node.inputs[0].dtype)

    def eigvalsh(a, b=None):
        if b is not None:
            raise NotImplementedError(
                "mlx.core.linalg.eigvalsh does not support generalized "
                "eigenvector problems (b != None)"
            )
        return mx.linalg.eigvalsh(
            a.astype(dtype=X_dtype, stream=mx.cpu), UPLO=UPLO, stream=mx.cpu
        )

    return eigvalsh


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


@mlx_funcify.register(QR)
def mlx_funcify_QR(op, node, **kwargs):
    mode = op.mode
    A_dtype = getattr(mx, node.inputs[0].dtype)

    if mode not in ("economic", "r"):
        raise NotImplementedError(
            f"mode='{mode}' is not supported in the MLX backend. "
            "Only 'economic' and 'r' modes are available."
        )

    def qr(a):
        Q, R = mx.linalg.qr(a.astype(dtype=A_dtype, stream=mx.cpu), stream=mx.cpu)
        if mode == "r":
            M = a.shape[-2]
            K = R.shape[-2]
            if M > K:
                # Pytensor follows scipy convention for mode = 'r', which returns R with the same
                # leading shape as the input.
                pad_width = [(0, 0)] * (R.ndim - 2) + [(0, M - K), (0, 0)]
                return mx.pad(R, pad_width, stream=mx.cpu)
            return R
        return Q, R

    return qr


@mlx_funcify.register(LUFactor)
def mlx_funcify_LUFactor(op, node, **kwargs):
    A_dtype = getattr(mx, node.inputs[0].dtype)

    def lu_factor(a):
        lu, pivots = mx.linalg.lu_factor(
            a.astype(dtype=A_dtype, stream=mx.cpu), stream=mx.cpu
        )
        return lu, pivots.astype(mx.int32, stream=mx.cpu)

    return lu_factor


@mlx_funcify.register(PivotToPermutations)
def mlx_funcify_PivotToPermutations(op, **kwargs):
    inverse = op.inverse

    def pivot_to_permutations(pivots):
        pivots = mx.array(pivots)
        n = pivots.shape[0]
        p_inv = mx.arange(n, dtype=mx.int32)
        for i in range(n):
            p_inv[i], p_inv[pivots[i]] = p_inv[pivots[i]], p_inv[i]
        if inverse:
            return p_inv
        return mx.argsort(p_inv)

    return pivot_to_permutations
