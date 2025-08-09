import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.extra_ops import CumOp, Repeat, Unique


@pytorch_funcify.register(CumOp)
def pytorch_funcify_Cumop(op, **kwargs):
    axis = op.axis
    mode = op.mode

    def cumop(x):
        if mode == "add":
            return torch.cumsum(x, dim=axis)
        else:
            return torch.cumprod(x, dim=axis)

    return cumop


@pytorch_funcify.register(Repeat)
def pytorch_funcify_Repeat(op, **kwargs):
    axis = op.axis

    def repeat(x, repeats):
        return x.repeat_interleave(repeats, dim=axis)

    return repeat


@pytorch_funcify.register(Unique)
def pytorch_funcify_Unique(op, **kwargs):
    return_index = op.return_index

    if return_index:
        # TODO: evaluate whether is worth implementing this param
        # (see https://github.com/pytorch/pytorch/issues/36748)
        raise NotImplementedError("return_index is not implemented for pytorch")

    axis = op.axis
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    def unique(x):
        return torch.unique(
            x,
            sorted=True,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=axis,
        )

    return unique
