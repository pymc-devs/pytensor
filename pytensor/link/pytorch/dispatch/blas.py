import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.link.utils import get_static_scalar
from pytensor.tensor.blas import BatchedDot, Gemm, Gemv, Ger


@pytorch_funcify.register(BatchedDot)
def pytorch_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        return torch.bmm(a, b)

    return batched_dot


@pytorch_funcify.register(Gemm)
def pytorch_funcify_Gemm(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)
    static_beta = get_static_scalar(node, 4)

    if static_alpha is not None and static_beta is not None:

        def gemm(z, alpha, x, y, beta):
            return torch.addmm(z, x, y, beta=static_beta, alpha=static_alpha)

    else:

        def gemm(z, alpha, x, y, beta):
            return beta * z + alpha * torch.matmul(x, y)

    return gemm


@pytorch_funcify.register(Gemv)
def pytorch_funcify_Gemv(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)
    static_beta = get_static_scalar(node, 4)

    if static_alpha is not None and static_beta is not None:

        def gemv(y, alpha, A, x, beta):
            return torch.addmv(y, A, x, beta=static_beta, alpha=static_alpha)

    else:

        def gemv(y, alpha, A, x, beta):
            return beta * y + alpha * torch.matmul(A, x)

    return gemv


@pytorch_funcify.register(Ger)
def pytorch_funcify_Ger(op, node=None, **kwargs):
    static_alpha = get_static_scalar(node, 1)

    if static_alpha is not None:

        def ger(A, alpha, x, y):
            return torch.addr(A, x, y, beta=1.0, alpha=static_alpha)

    else:

        def ger(A, alpha, x, y):
            return A + alpha * torch.outer(x, y)

    return ger
