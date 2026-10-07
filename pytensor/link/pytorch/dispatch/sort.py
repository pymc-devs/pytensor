import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.sort import ArgSortOp, SortOp


@pytorch_funcify.register(SortOp)
def pytorch_funcify_Sort(op, **kwargs):
    stable = op.kind == "stable"
    axis = op.axis

    def sort(arr):
        sorted, _ = torch.sort(arr, dim=axis, stable=stable)
        return sorted

    return sort


@pytorch_funcify.register(ArgSortOp)
def pytorch_funcify_ArgSort(op, **kwargs):
    stable = op.kind == "stable"
    axis = op.axis

    def argsort(arr):
        return torch.argsort(arr, dim=axis, stable=stable)

    return argsort
