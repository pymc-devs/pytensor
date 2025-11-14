import warnings

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_default_op_cache_key,
)
from pytensor.tensor.sort import ArgSortOp, SortOp


@register_funcify_default_op_cache_key(SortOp)
def numba_funcify_SortOp(op, node, **kwargs):
    if op.kind != "quicksort":
        warnings.warn(
            (
                f'Numba function sort doesn\'t support kind="{op.kind}"'
                " switching to `quicksort`."
            ),
            UserWarning,
        )

    @numba_basic.numba_njit
    def sort_f(a, axis):
        axis = axis.item()

        a_swapped = np.swapaxes(a, axis, -1)
        a_sorted = np.sort(a_swapped)
        a_sorted_swapped = np.swapaxes(a_sorted, -1, axis)

        return a_sorted_swapped

    return sort_f


@register_funcify_default_op_cache_key(ArgSortOp)
def numba_funcify_ArgSortOp(op, node, **kwargs):
    kind = op.kind

    if kind not in ["quicksort", "mergesort"]:
        kind = "quicksort"
        warnings.warn(
            (
                f'Numba function argsort doesn\'t support kind="{op.kind}"'
                " switching to `quicksort`."
            ),
            UserWarning,
        )

    @numba_basic.numba_njit
    def argort_f(X, axis):
        axis = axis.item()

        Y = np.swapaxes(X, axis, 0)
        result = np.empty_like(Y, dtype="int64")

        indices = list(np.ndindex(Y.shape[1:]))

        for idx in indices:
            result[(slice(None), *idx)] = np.argsort(Y[(slice(None), *idx)], kind=kind)

        result = np.swapaxes(result, 0, axis)
        return result

    return argort_f
