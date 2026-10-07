import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.blas import BatchedDot


@pytorch_funcify.register(BatchedDot)
def pytorch_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        return torch.bmm(a, b)

    return batched_dot
