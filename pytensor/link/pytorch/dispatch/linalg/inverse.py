import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor._linalg.inverse import MatrixInverse, MatrixPinv


@pytorch_funcify.register(MatrixInverse)
def pytorch_funcify_MatrixInverse(op, **kwargs):
    def matrix_inverse(x):
        return torch.linalg.inv(x)

    return matrix_inverse


@pytorch_funcify.register(MatrixPinv)
def pytorch_funcify_Pinv(op, **kwargs):
    hermitian = op.hermitian

    def pinv(x):
        return torch.linalg.pinv(x, hermitian=hermitian)

    return pinv
