import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.math import Dot


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, node, **kwargs):
    # torch.matmul needs matching dtypes; Dot declares the promoted one.
    out_dtype = getattr(torch, node.outputs[0].dtype)

    def dot(x, y):
        return torch.matmul(x.to(out_dtype), y.to(out_dtype))

    return dot
