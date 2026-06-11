import mlx.core as mx

from pytensor.graph.basic import Constant
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blas import BatchedDot, Gemv, Ger


@mlx_funcify.register(BatchedDot)
def mlx_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match along the first dimension of BatchedDot")
        return mx.matmul(a, b)

    return batched_dot


@mlx_funcify.register(Gemv)
def mlx_funcify_Gemv(op, node=None, **kwargs):
    static_alpha = _as_float_constant(node.inputs[1]) if node is not None else None
    static_beta = _as_float_constant(node.inputs[4]) if node is not None else None

    if static_alpha is not None and static_beta is not None:

        def gemv(y, alpha, A, x, beta):
            return mx.addmm(y, A, x, alpha=static_alpha, beta=static_beta)

    else:

        def gemv(y, alpha, A, x, beta):
            return beta * y + alpha * mx.matmul(A, x)

    return gemv


@mlx_funcify.register(Ger)
def mlx_funcify_Ger(op, node=None, **kwargs):
    static_alpha = _as_float_constant(node.inputs[1]) if node is not None else None

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


def _as_float_constant(var):
    if not isinstance(var, Constant):
        return None
    try:
        return float(var.data)
    except (TypeError, ValueError):
        return None
