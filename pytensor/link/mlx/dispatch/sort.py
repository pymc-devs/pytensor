import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.tensor_basic import coerce_to_int
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.sort import ArgSortOp, SortOp


def _warn_unsupported_kind(op, name):
    if op.kind != "quicksort":
        warnings.warn(
            message=f"MLX {name} does not support the kind argument (got kind={op.kind}). "
            "The argument will be ignored.",
            category=UserWarning,
        )


def _static_axis(node):
    try:
        return int(get_scalar_constant_value(node.inputs[1]))
    except NotScalarConstantError:
        return None


def _resolve_axis(static_axis, axis):
    return coerce_to_int(axis) if static_axis is None else static_axis


@mlx_funcify.register(SortOp)
def mlx_funcify_Sort(op, node, **kwargs):
    _warn_unsupported_kind(op, "sort")
    static_axis = _static_axis(node)

    def sort(x, axis):
        return mx.sort(x, axis=_resolve_axis(static_axis, axis))

    return sort


@mlx_funcify.register(ArgSortOp)
def mlx_funcify_ArgSort(op, node, **kwargs):
    _warn_unsupported_kind(op, "argsort")
    static_axis = _static_axis(node)

    def argsort(x, axis):
        return mx.argsort(x, axis=_resolve_axis(static_axis, axis)).astype(mx.int64)

    return argsort
