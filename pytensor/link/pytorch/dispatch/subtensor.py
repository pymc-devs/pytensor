import torch
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type_other import MakeSlice

BOOLEAN_MASK_ERROR = """PyTorch does not support resizing arrays with boolean
masks. In some cases, however, it is possible to re-express your model
in a form that PyTorch can compile:

>>> import pytensor.tensor as at
>>> x_at = at.vector('x')
>>> y_at = x_at[x_at > 0].sum()

can be re-expressed as:

>>> import pytensor.tensor as at
>>> x_at = at.vector('x')
>>> y_at = at.where(x_at > 0, x_at, 0).sum()
"""

DYNAMIC_SLICE_LENGTH_ERROR = """PyTorch does not support slicing arrays with a dynamic
slice length.
"""


def subtensor_assert_indices_pytorch_compatible(node, idx_list):
    from pytensor.graph.basic import Constant
    from pytensor.tensor.variable import TensorVariable

    ilist = indices_from_subtensor(node.inputs[1:], idx_list)
    for idx in ilist:
        if isinstance(idx, TensorVariable):
            if idx.type.dtype == "bool":
                raise NotImplementedError(BOOLEAN_MASK_ERROR)
        elif isinstance(idx, slice):
            for slice_arg in (idx.start, idx.stop, idx.step):
                if slice_arg is not None and not isinstance(slice_arg, Constant):
                    raise NotImplementedError(DYNAMIC_SLICE_LENGTH_ERROR)


@pytorch_funcify.register(Subtensor)
@pytorch_funcify.register(AdvancedSubtensor)
@pytorch_funcify.register(AdvancedSubtensor1)
def pytorch_funcify_Subtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)
    subtensor_assert_indices_pytorch_compatible(node, idx_list)

    def subtensor_constant(x, *ilists):
        indices = indices_from_subtensor(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor_constant


@pytorch_funcify.register(IncSubtensor)
@pytorch_funcify.register(AdvancedIncSubtensor1)
def pytorch_funcify_IncSubtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):

        def pytorch_fn(x, indices, y):
            x[indices] = y
            return x

    else:

        def pytorch_fn(x, indices, y):
            x[indices] += y
            return x

    def incsubtensor(x, y, *ilist, pytorch_fn=pytorch_fn, idx_list=idx_list):
        indices = indices_from_subtensor(ilist, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return pytorch_fn(x, indices, y)

    return incsubtensor


@pytorch_funcify.register(AdvancedIncSubtensor)
def pytorch_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    if getattr(op, "set_instead_of_inc", False):

        def pytorch_fn(x, indices, y):
            x[indices] = y
            return x

    else:

        def pytorch_fn(x, indices, y):
            x[indices] += y
            return x

    def advancedincsubtensor(x, y, *ilist, pytorch_fn=pytorch_fn):
        return pytorch_fn(x, ilist, y)

    return advancedincsubtensor


@pytorch_funcify.register(MakeSlice)
def pytorch_funcify_MakeSlice(op, **kwargs):
    def makeslice(*x):
        return slice(*x)

    return makeslice