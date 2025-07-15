import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
)


@pytorch_funcify.register(SVD)
def pytorch_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x):
        U, S, V = torch.linalg.svd(x, full_matrices=full_matrices)
        if compute_uv:
            return U, S, V
        return S

    return svd


@pytorch_funcify.register(Det)
def pytorch_funcify_Det(op, **kwargs):
    def det(x):
        return torch.linalg.det(x)

    return det


@pytorch_funcify.register(SLogDet)
def pytorch_funcify_SLogDet(op, **kwargs):
    def slogdet(x):
        return torch.linalg.slogdet(x)

    return slogdet


@pytorch_funcify.register(Eig)
def pytorch_funcify_Eig(op, **kwargs):
    def eig(x):
        return torch.linalg.eig(x)

    return eig


@pytorch_funcify.register(Eigh)
def pytorch_funcify_Eigh(op, **kwargs):
    uplo = op.UPLO

    def eigh(x, uplo=uplo):
        return torch.linalg.eigh(x, UPLO=uplo)

    return eigh


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


@pytorch_funcify.register(KroneckerProduct)
def pytorch_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return torch.kron(x, y)

    return _kron
