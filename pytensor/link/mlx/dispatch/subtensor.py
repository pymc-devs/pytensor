from copy import deepcopy

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)


@mlx_funcify.register(Subtensor)
def mlx_funcify_Subtensor(op, node, **kwargs):
    def subtensor(x, *ilists):
        indices = indices_from_subtensor(
            [int(element) for element in ilists], op.idx_list
        )
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor


@mlx_funcify.register(AdvancedSubtensor)
@mlx_funcify.register(AdvancedSubtensor1)
def mlx_funcify_AdvancedSubtensor(op, node, **kwargs):
    def advanced_subtensor(x, *ilists):
        indices = indices_from_subtensor(ilists, op.idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return advanced_subtensor


@mlx_funcify.register(IncSubtensor)
@mlx_funcify.register(AdvancedIncSubtensor1)
def mlx_funcify_IncSubtensor(op, node, **kwargs):
    if getattr(op, "set_instead_of_inc", False):

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] = y
            return x

    else:

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] += y
            return x

    def incsubtensor(x, y, *ilist, mlx_fn=mlx_fn, idx_list=op.idx_list):
        indices = indices_from_subtensor(ilist, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return mlx_fn(x, indices, y)

    return incsubtensor


@mlx_funcify.register(AdvancedIncSubtensor)
def mlx_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    if getattr(op, "set_instead_of_inc", False):

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] = y
            return x

    else:

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] += y
            return x

    def advancedincsubtensor(x, y, *ilist, mlx_fn=mlx_fn):
        return mlx_fn(x, ilist, y)

    return advancedincsubtensor
