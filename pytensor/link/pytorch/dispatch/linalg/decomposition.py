import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor._linalg.decomposition.eigen import Eig, Eigh
from pytensor.tensor._linalg.decomposition.qr import QR
from pytensor.tensor._linalg.decomposition.svd import SVD


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


@pytorch_funcify.register(QR)
def pytorch_funcify_QR(op, **kwargs):
    mode = op.mode
    if mode == "raw":
        raise NotImplementedError("raw mode not implemented in PyTorch")
    elif mode == "full":
        mode = "complete"
    elif mode == "economic":
        mode = "reduced"

    def qr(x):
        Q, R = torch.linalg.qr(x, mode=mode)
        if mode == "r":
            return R
        return Q, R

    return qr
