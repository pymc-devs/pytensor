import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.math import Argmax, Dot, Max
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    QRFull,
    SLogDet,
)


@pytorch_funcify.register(SVD)
def pytorch_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x):
        U, S, V = torch.linalg.svd(x, full_matrices=full_matrices)
        return U, S, V if compute_uv else S

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


@pytorch_funcify.register(QRFull)
def pytorch_funcify_QRFull(op, **kwargs):
    mode = op.mode
    if mode == "raw":
        raise NotImplementedError("raw mode not implemented in PyTorch")

    def qr_full(x):
        Q, R = torch.linalg.qr(x, mode=mode)
        if mode == "r":
            return R
        return Q, R

    return qr_full


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return torch.dot(x, y)

    return dot


@pytorch_funcify.register(MatrixPinv)
def pytorch_funcify_Pinv(op, **kwargs):
    hermitian = op.hermitian

    def pinv(x):
        return torch.linalg.pinv(x, hermitian=hermitian)

    return pinv


@pytorch_funcify.register(BatchedDot)
def pytorch_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        return torch.matmul(a, b)

    return batched_dot


@pytorch_funcify.register(KroneckerProduct)
def pytorch_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return torch.kron(x, y)

    return _kron


@pytorch_funcify.register(Max)
def pytorch_funcify_Max(op, **kwargs):
    axis = op.axis

    def max(x):
        if axis is None:
            max_res = torch.max(x.flatten())
            return max_res

        # PyTorch doesn't support multiple axes for max;
        # this is a work-around
        axes = [int(ax) for ax in axis]

        new_dim = torch.prod(torch.tensor([x.size(ax) for ax in axes])).item()
        keep_axes = [i for i in range(x.ndim) if i not in axes]
        permute_order = keep_axes + axes
        permuted_x = x.permute(*permute_order)
        kept_shape = permuted_x.shape[: len(keep_axes)]

        new_shape = (*kept_shape, new_dim)
        reshaped_x = permuted_x.reshape(new_shape)
        max_res, _ = torch.max(reshaped_x, dim=-1)
        return max_res

    return max


@pytorch_funcify.register(Argmax)
def pytorch_funcify_Argmax(op, **kwargs):
    axis = op.axis

    def argmax(x):
        if axis is None:
            return torch.argmax(x.view(-1))

        # PyTorch doesn't support multiple axes for argmax;
        # this is a work-around
        axes = [int(ax) for ax in axis]

        new_dim = torch.prod(torch.tensor([x.size(ax) for ax in axes])).item()
        keep_axes = [i for i in range(x.ndim) if i not in axes]
        permute_order = keep_axes + axes
        permuted_x = x.permute(*permute_order)
        kept_shape = permuted_x.shape[: len(keep_axes)]

        new_shape = (*kept_shape, new_dim)
        reshaped_x = permuted_x.reshape(new_shape)
        return torch.argmax(reshaped_x, dim=-1)

    return argmax
