from copy import deepcopy

import mlx.core as mx
import numpy as np

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
from pytensor.tensor.type_other import MakeSlice


def normalize_indices_for_mlx(ilist, idx_list):
    """Convert numpy integers to Python integers for MLX indexing.

    MLX requires index values to be Python int, not np.int64 or other NumPy types.
    """

    def normalize_element(element):
        if element is None:
            return None
        elif isinstance(element, slice):
            return slice(
                normalize_element(element.start),
                normalize_element(element.stop),
                normalize_element(element.step),
            )
        elif isinstance(element, mx.array) and element.ndim == 0:
            return int(element.item())
        elif isinstance(element, np.integer):
            return int(element)
        else:
            return element

    indices = indices_from_subtensor(ilist, idx_list)
    return tuple(normalize_element(idx) for idx in indices)


@mlx_funcify.register(Subtensor)
def mlx_funcify_Subtensor(op, node, **kwargs):
    """MLX implementation of Subtensor."""
    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor


@mlx_funcify.register(AdvancedSubtensor)
@mlx_funcify.register(AdvancedSubtensor1)
def mlx_funcify_AdvancedSubtensor(op, node, **kwargs):
    """MLX implementation of AdvancedSubtensor."""
    idx_list = getattr(op, "idx_list", None)

    def advanced_subtensor(x, *ilists):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return advanced_subtensor


@mlx_funcify.register(IncSubtensor)
@mlx_funcify.register(AdvancedIncSubtensor1)
def mlx_funcify_IncSubtensor(op, node, **kwargs):
    """MLX implementation of IncSubtensor."""
    idx_list = getattr(op, "idx_list", None)

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

    def incsubtensor(x, y, *ilist, mlx_fn=mlx_fn, idx_list=idx_list):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilist, idx_list)

        if len(indices) == 1:
            indices = indices[0]

        return mlx_fn(x, indices, y)

    return incsubtensor


@mlx_funcify.register(AdvancedIncSubtensor)
def mlx_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    """MLX implementation of AdvancedIncSubtensor."""
    idx_list = getattr(op, "idx_list", None)

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

    def advancedincsubtensor(x, y, *ilist, mlx_fn=mlx_fn, idx_list=idx_list):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilist, idx_list)

        # For advanced indexing, if we have a single tuple of indices, unwrap it
        if len(indices) == 1:
            indices = indices[0]

        return mlx_fn(x, indices, y)

    return advancedincsubtensor


@mlx_funcify.register(MakeSlice)
def mlx_funcify_MakeSlice(op, **kwargs):
    def makeslice(*x):
        return slice(*x)

    return makeslice
