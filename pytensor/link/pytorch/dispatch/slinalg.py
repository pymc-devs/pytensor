import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.slinalg import QR


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
