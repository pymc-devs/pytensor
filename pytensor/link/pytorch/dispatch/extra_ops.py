import warnings
import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.extra_ops import (
    Bartlett,
    CumOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    Repeat,
    Unique,
    UnravelIndex,
)


@pytorch_funcify.register(Bartlett)
def pytorch_funcify_Bartlett(op, **kwargs):
    def bartlett(x):
        return torch.bartlett_window(x)

    return bartlett


@pytorch_funcify.register(CumOp)
def pytorch_funcify_CumOp(op, **kwargs):
    axis = op.axis
    mode = op.mode

    def cumop(x, axis=axis, mode=mode):
        if mode == "add":
            return torch.cumsum(x, axis=axis)
        else:
            return torch.cumprod(x, axis=axis)

    return cumop


@pytorch_funcify.register(Repeat)
def pytorch_funcify_Repeat(op, **kwargs):
    axis = op.axis

    def repeatop(x, repeats, axis=axis):
        return x.repeat(repeats, axis=axis)

    return repeatop


@pytorch_funcify.register(Unique)
def pytorch_funcify_Unique(op, **kwargs):
    axis = op.axis

    if axis is not None:
        raise NotImplementedError(
            "torch.unique is not implemented for the axis argument"
        )

    return_index = op.return_index
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    def unique(
        x,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    ):
        ret = torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    return unique


@pytorch_funcify.register(UnravelIndex)
def pytorch_funcify_UnravelIndex(op, **kwargs):
    order = op.order

    warnings.warn("PyTorch ignores the `order` parameter in `unravel_index`.")

    def unravelindex(indices, dims, order=order):
        return torch.unravel_index(indices, dims)

    return unravelindex


@pytorch_funcify.register(RavelMultiIndex)
def pytorch_funcify_RavelMultiIndex(op, **kwargs):
    mode = op.mode
    order = op.order

    def ravelmultiindex(*inp, mode=mode, order=order):
        multi_index, dims = inp[:-1], inp[-1]
        return torch.ravel_multi_index(multi_index, dims, mode=mode, order=order)

    return ravelmultiindex


@pytorch_funcify.register(FillDiagonal)
def pytorch_funcify_FillDiagonal(op, **kwargs):
    def filldiagonal(value, diagonal):
        value.fill_diagonal_(diagonal)
        return value

    return filldiagonal


@pytorch_funcify.register(FillDiagonalOffset)
def pytorch_funcify_FillDiagonalOffset(op, **kwargs):
    def filldiagonaloffset(a, val, offset):
        height, width = a.shape

        if offset >= 0:
            start = offset
            num_of_step = min(min(width, height), width - offset)
        else:
            start = -offset * a.shape[1]
            num_of_step = min(min(width, height), height + offset)

        step = a.shape[1] + 1
        end = start + step * num_of_step
        a.view(-1)[start:end:step] = val

        return a

    return filldiagonaloffset
