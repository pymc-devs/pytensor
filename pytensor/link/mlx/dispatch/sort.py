import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.sort import ArgSortOp, SortOp


@mlx_funcify.register(SortOp)
def mlx_funcify_Sort(op, **kwargs):
    kind = op.kind
    if kind != "quicksort":
        warnings.warn(
            message=f"MLX sort does not support the kind argument (got kind={kind}). The argument will be "
            f"ignored.",
            category=UserWarning,
        )

    def sort(x, axis):
        return mx.sort(x, axis=axis)

    return sort


@mlx_funcify.register(ArgSortOp)
def mlx_funcify_ArgSort(op, **kwargs):
    kind = op.kind
    if kind != "quicksort":
        warnings.warn(
            message=f"MLX argsort does not support the kind argument (got kind={kind}). The argument will be "
            f"ignored.",
            category=UserWarning,
        )

    def argsort(x, axis):
        return mx.argsort(x, axis=axis)

    return argsort
