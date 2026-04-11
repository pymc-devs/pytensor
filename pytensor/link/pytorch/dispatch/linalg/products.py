import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.linalg.products import KroneckerProduct


@pytorch_funcify.register(KroneckerProduct)
def pytorch_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return torch.kron(x, y)

    return _kron
