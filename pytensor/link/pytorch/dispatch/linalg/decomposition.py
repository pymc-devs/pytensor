import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.linalg.decomposition.eigen import Eig, Eigh, Eigvalsh
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.svd import SVD


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
def pytorch_funcify_Eigh(op, node, **kwargs):
    UPLO = "L" if op.lower else "U"

    if len(node.inputs) == 2:
        raise NotImplementedError(
            "torch.linalg.eigh does not support generalized eigenvalue problems (b != None)"
        )

    def eigh(a):
        return torch.linalg.eigh(a, UPLO=UPLO)

    return eigh


@pytorch_funcify.register(Eigvalsh)
def pytorch_funcify_Eigvalsh(op, node, **kwargs):
    UPLO = "L" if op.lower else "U"

    if len(node.inputs) == 2:
        raise NotImplementedError(
            "torch.linalg.eigvalsh does not support generalized eigenvalue problems (b != None)"
        )

    def eigvalsh(a):
        return torch.linalg.eigvalsh(a, UPLO=UPLO)

    return eigvalsh


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
