import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.math import Dot


@pytorch_funcify.register(Dot)
def pytorch_funcify_Dot(op, **kwargs):
    def dot(x, y):
        # Case 1: Vector Product/Matrix Multiplication/1-D Broadcastable Vector
        if x.shape < 3 and y.shape < 3:
            return torch.matmul(x, y)
        else:
            # Case 2: Stackable batch dimension
            return torch.tensordot(x, y, dims=([-1], [-2]))

    return dot
