import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.math import Dot, MaxAndArgmax
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    MatrixInverse,
    MatrixPinv,
    QRFull,
    SLogDet,
)


@pytorch_funcify.register(SVD)
def pytorch_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x, full_matrices=full_matrices, compute_uv=compute_uv):
        return torch.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    return svd


@pytorch_funcify.register(Det)
def pytorch_funcify_Det(op, **kwargs):
    def det(x):
        return torch.det(x)

    return det


@pytorch_funcify.register(SLogDet)
def pytorch_funcify_SLogDet(op, **kwargs):
    def slogdet(x):
        return torch.slogdet(x)

    return slogdet


@pytorch_funcify.register(Eig)
def pytorch_funcify_Eig(op, **kwargs):
    def eig(x):
        return torch.eig(x)

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
        return torch.inverse(x)

    return matrix_inverse


@pytorch_funcify.register(QRFull)
def pytorch_funcify_QRFull(op, **kwargs):
    mode = op.mode

    def qr_full(x, mode=mode):
        return torch.qr(x, mode=mode)

    return qr_full


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return torch.dot(x, y)

    return dot


@pytorch_funcify.register(MatrixPinv)
def pytorch_funcify_Pinv(op, **kwargs):
    def pinv(x):
        return torch.pinverse(x)

    return pinv


@pytorch_funcify.register(BatchedDot)
def pytorch_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        if a.ndim == 2 or b.ndim == 2:
            return torch.einsum("n...j,nj...->n...", a, b)
        return torch.einsum("nij,njk->nik", a, b)

    return batched_dot


@pytorch_funcify.register(MaxAndArgmax)
def pytorch_funcify_MaxAndArgmax(op, **kwargs):
    axis = op.axis

    def maxandargmax(x, axis=axis):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        max_res = torch.max(x, axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = torch.tensor(
            [i for i in range(x.ndim) if i not in axes], dtype=torch.int64
        )
        # Not-reduced axes in front
        transposed_x = torch.transpose(
            x, torch.cat((keep_axes, torch.tensor(axes, dtype=torch.int64)))
        )
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = kept_shape + (
            torch.prod(torch.tensor(reduced_shape, dtype=torch.int64), dtype=torch.int64),
        )
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx_res = torch.argmax(reshaped_x, axis=-1).type(torch.int64)

        return max_res, max_idx_res

    return maxandargmax