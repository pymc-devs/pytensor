from collections.abc import Sequence
from typing import cast as type_cast

import numpy as np
from numpy.lib._array_utils_impl import normalize_axis_tuple

from pytensor import Variable
from pytensor.graph import Apply
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.tensor import TensorLike, as_tensor_variable
from pytensor.tensor.basic import infer_static_shape
from pytensor.tensor.math import prod
from pytensor.tensor.shape import ShapeValueType
from pytensor.tensor.type import tensor
from pytensor.tensor.variable import TensorVariable


class JoinDims(Op):
    __props__ = (
        "start_axis",
        "n_axes",
    )
    view_map = {0: [0]}

    def __init__(self, start_axis: int, n_axes: int):
        if start_axis < 0:
            raise ValueError("JoinDims start_axis must be non-negative")
        if n_axes < 0:
            raise ValueError("JoinDims n_axes must be non-negative")

        self.start_axis = start_axis
        self.n_axes = n_axes

    @property
    def axis_range(self):
        return range(self.start_axis, self.start_axis + self.n_axes)

    def output_shapes(self, input_shapes, joined_shape):
        return (
            *input_shapes[: self.start_axis],
            joined_shape,
            *input_shapes[self.start_axis + self.n_axes :],
        )

    def make_node(self, x: Variable) -> Apply:  # type: ignore[override]
        x = as_tensor_variable(x)

        static_shapes = x.type.shape
        axis_range = self.axis_range

        joined_shape = (
            int(np.prod([static_shapes[i] for i in axis_range]))
            if all(static_shapes[i] is not None for i in axis_range)
            else None
        )

        output_shapes = self.output_shapes(static_shapes, joined_shape)
        output_type = tensor(shape=output_shapes, dtype=x.type.dtype)

        return Apply(self, [x], [output_type])

    def infer_shape(self, fgraph, node, shapes):
        [input_shape] = shapes
        axis_range = self.axis_range

        joined_shape = prod([input_shape[i] for i in axis_range])
        return [self.output_shapes(input_shape, joined_shape)]

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs

        output_shape = (
            *x.shape[: self.start_axis],
            -1,
            *x.shape[self.start_axis + self.n_axes :],
        )

        out[0] = x.reshape(output_shape)


@_vectorize_node.register(JoinDims)
def _vectorize_joindims(op, node, x):
    [old_x] = node.inputs

    batched_ndims = x.type.ndim - old_x.type.ndim
    start_axis = op.start_axis
    n_axes = op.n_axes

    return JoinDims(start_axis + batched_ndims, n_axes).make_node(x)


def join_dims(x: TensorLike, axis: Sequence[int] | int | None = None) -> TensorVariable:
    """Join consecutive dimensions of a tensor into a single dimension.

    Parameters
    ----------
    x : Variable
        The input tensor.
    axis : int or sequence of int, optional
        The dimensions to join. If None, all dimensions are joined.

    Returns
    -------
    joined_x : Variable
        The reshaped tensor with joined dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor("x", shape=(2, 3, 4, 5))
    >>> y = pt.join_dims(x, axis=(1, 2))
    >>> y.type.shape
    (2, 12, 5)
    """
    x = as_tensor_variable(x)

    if axis is None:
        axis = list(range(x.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    elif not isinstance(axis, list | tuple):
        raise TypeError("axis must be an int, a list/tuple of ints, or None")

    axis = normalize_axis_tuple(axis, x.ndim)

    if len(axis) <= 1:
        return x  # type: ignore[unreachable]

    if np.diff(axis).max() > 1:
        raise ValueError(
            f"join_dims axis must be consecutive, got normalized axis: {axis}"
        )

    start_axis = min(axis)
    n_axes = len(axis)

    return type_cast(
        TensorVariable,
        JoinDims(start_axis=start_axis, n_axes=n_axes)(x),
    )


class SplitDims(Op):
    __props__ = ("axis",)
    view_map = {0: [0]}

    def __init__(self, axis: int):
        if axis < 0:
            raise ValueError("SplitDims axis must be non-negative")
        self.axis = axis

    def make_node(self, x: Variable, shape: Variable) -> Apply:  # type: ignore[override]
        if shape.type.numpy_dtype.kind not in "iu":
            raise TypeError("shape must be an integer tensor")

        x = as_tensor_variable(x)
        shape = as_tensor_variable(shape, dtype=int, ndim=1)

        axis = self.axis
        _, constant_shape = infer_static_shape(shape)

        output_shapes = [
            *x.type.shape[:axis],
            *constant_shape,
            *x.type.shape[axis + 1 :],
        ]

        output = tensor(
            shape=tuple(x if isinstance(x, int) else None for x in output_shapes),
            dtype=x.type.dtype,
        )
        return Apply(self, [x, shape], [output])

    def infer_shape(self, fgraph, node, shapes):
        [input_shape, _] = shapes
        _, shape = node.inputs
        output_shapes = list(input_shape)
        axis = self.axis

        inferred_shape = [*output_shapes[:axis], *shape, *output_shapes[axis + 1 :]]
        return [inferred_shape]

    def perform(self, node, inputs, outputs):
        (x, shape) = inputs
        (out,) = outputs

        output_shape = (*x.shape[: self.axis], *shape, *x.shape[self.axis + 1 :])

        out[0] = x.reshape(output_shape)


@_vectorize_node.register(SplitDims)
def _vectorize_splitdims(op, node, x, shape):
    from pytensor.tensor.blockwise import vectorize_node_fallback

    old_x, _ = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim

    if as_tensor_variable(shape).type.ndim != 1:
        return vectorize_node_fallback(op, node, x, shape)

    axis = op.axis
    return SplitDims(axis=axis + batched_ndims).make_node(x, shape)


def split_dims(
    x: TensorLike,
    shape: ShapeValueType | Sequence[ShapeValueType],
    axis: int | None = None,
) -> TensorVariable:
    """Split a dimension of a tensor into multiple dimensions.

    Parameters
    ----------
    x : TensorLike
        The input tensor.
    shape : int or sequence of int
        The new shape to split the specified dimension into.
    axis : int, optional
        The dimension to split. If None, the input is assumed to be 1D and axis 0 is used.

    Returns
    -------
    split_x : Variable
        The reshaped tensor with split dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor("x", shape=(6, 4, 6))
    >>> y = pt.split_dims(x, shape=(2, 3), axis=0)
    >>> y.type.shape
    (2, 3, 4, 6)
    """
    x = as_tensor_variable(x)

    if axis is None:
        if x.ndim != 1:
            raise ValueError(
                "split_dims can only be called with axis=None for 1d inputs"
            )
        axis = 0

    if isinstance(shape, int):
        shape = [shape]
    else:
        shape = list(shape)  # type: ignore[arg-type]

    if not shape:
        # If we get an empty shape, there is potentially a dummy dimension at the requested axis. This happens for
        # example when splitting a packed tensor that had its dims expanded before packing (e.g. when packing shapes
        # (3, ) and (3, 3) to (3, 4)
        return type_cast(TensorVariable, x.squeeze(axis=axis))

    [axis] = normalize_axis_tuple(axis, x.ndim)  # type: ignore[misc]
    shape = as_tensor_variable(shape, dtype="int64", ndim=1)  # type: ignore[arg-type]

    split_op = SplitDims(axis=axis)
    return type_cast(TensorVariable, split_op(x, shape))


__all__ = ["join_dims", "split_dims"]
