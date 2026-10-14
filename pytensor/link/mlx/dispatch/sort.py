import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import convert_dtype_to_mlx, mlx_funcify
from pytensor.tensor.sort import ArgSortOp, SortOp


def _warn_unsupported_kind(op, name):
    if op.kind != "quicksort":
        warnings.warn(
            message=f"MLX {name} does not support the kind argument (got kind={op.kind}). "
            "The argument will be ignored.",
            category=UserWarning,
        )


@mlx_funcify.register(SortOp)
def mlx_funcify_Sort(op, node, **kwargs):
    _warn_unsupported_kind(op, "sort")
    axis = op.axis

    def sort(x):
        return mx.sort(x, axis=axis)

    return sort


@mlx_funcify.register(ArgSortOp)
def mlx_funcify_ArgSort(op, node, **kwargs):
    _warn_unsupported_kind(op, "argsort")
    axis = op.axis
    out_dtype = convert_dtype_to_mlx(node.outputs[0].dtype)

    def argsort(x):
        return mx.argsort(x, axis=axis).astype(out_dtype)

    return argsort
