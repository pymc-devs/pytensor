import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.math import Argmax


@pytorch_funcify.register(Argmax)
def pytorch_funcify_Argmax(op, **kwargs):
    dim = op.axis
    keepdim = op.keepdims

    def argmax(x):
        return torch.argmax(x, dim=dim, keepdim=keepdim)

    return argmax
