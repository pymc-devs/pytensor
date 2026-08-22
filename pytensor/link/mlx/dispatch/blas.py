import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.utils import get_static_scalar
from pytensor.tensor.blas import BatchedDot, Gemm, Gemv, Ger


@mlx_funcify.register(BatchedDot)
def mlx_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match along the first dimension of BatchedDot")
        return mx.matmul(a, b)

    return batched_dot


@mlx_funcify.register(Gemm)
def mlx_funcify_Gemm(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)
    static_beta = get_static_scalar(node, 4)

    if static_alpha is not None and static_beta is not None:

        def gemm(z, alpha, x, y, beta):
            return mx.addmm(z, x, y, alpha=static_alpha, beta=static_beta)

    else:

        def gemm(z, alpha, x, y, beta):
            return beta * z + alpha * mx.matmul(x, y)

    return gemm


@mlx_funcify.register(Gemv)
def mlx_funcify_Gemv(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)
    static_beta = get_static_scalar(node, 4)

    if static_alpha is not None and static_beta is not None:

        def gemv(y, alpha, A, x, beta):
            return mx.addmm(y, A, x, alpha=static_alpha, beta=static_beta)

    else:

        def gemv(y, alpha, A, x, beta):
            return beta * y + alpha * mx.matmul(A, x)

    return gemv


@mlx_funcify.register(Ger)
def mlx_funcify_Ger(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)

    if static_alpha is not None:

        def ger(A, alpha, x, y):
            # GER is the rank-1 update A + alpha * outer(x, y). Expressed as a
            # matmul of (m, 1) @ (1, n), this maps directly onto mx.addmm with beta=1.
            return mx.addmm(
                A, x.reshape(-1, 1), y.reshape(1, -1), alpha=static_alpha, beta=1.0
            )

    else:

        def ger(A, alpha, x, y):
            return A + alpha * mx.outer(x, y)

    return ger
