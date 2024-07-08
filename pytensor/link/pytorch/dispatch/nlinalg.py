import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.math import Dot


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return torch.matmul(x, y)

    return dot
