import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor._linalg.summary import Det, SLogDet


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
