import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.sort import ArgSortOp, SortOp


@mlx_funcify.register(SortOp)
def mlx_funcify_Sort(op, **kwargs):
    kind = op.kind

    def sort(x, axis, kind=kind):
        if kind != "quicksort":
            raise NotImplementedError(
                f"MLX sort does not support kind={kind}, only 'quicksort'."
            )
        return mx.sort(x, axis=axis)

    return sort


@mlx_funcify.register(ArgSortOp)
def mlx_funcify_ArgSort(op, **kwargs):
    kind = op.kind

    def argsort(x, axis, kind=kind):
        if kind != "quicksort":
            raise NotImplementedError(
                f"MLX argsort does not support kind={kind}, only 'quicksort'."
            )
        return mx.argsort(x, axis=axis)

    return argsort
