import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.math import Dot


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, node, **kwargs):
    dtype = getattr(torch, node.outputs[0].dtype)

    def dot(x, y):
        return torch.matmul(x.to(dtype=dtype), y.to(dtype=dtype))

    return dot
