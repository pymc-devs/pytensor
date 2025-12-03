from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
from numpy.lib._array_utils_impl import normalize_axis_tuple

from pytensor import Variable
from pytensor.graph import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike, as_tensor_variable
from pytensor.tensor.basic import join, split
from pytensor.tensor.math import prod
from pytensor.tensor.shape import ShapeValueType
from pytensor.tensor.type import tensor
from pytensor.tensor.variable import TensorConstant


class JoinDims(Op):
    def __init__(self, axis: Sequence[int] | int | None = None):
        self.axis = axis

    def make_node(self, x: Variable) -> Apply:
        static_shapes = x.type.shape
        joined_shape = (
            int(np.prod([static_shapes[i] for i in self.axis]))
            if all(static_shapes[i] is not None for i in self.axis)
            else None
        )

        output_shapes = (
            *static_shapes[: min(self.axis)],
            joined_shape,
            *static_shapes[max(self.axis) + 1 :],
        )

        output_type = tensor(shape=output_shapes, dtype=x.type.dtype)
        return Apply(self, [x], [output_type])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs

        output_shape = [
            *x.shape[: min(self.axis)],
            -1,
            *x.shape[max(self.axis) + 1 :],
        ]

        out[0] = x.reshape(tuple(output_shape))


def join_dims(x: Variable, axis: Sequence[int] | int | None = None) -> Variable:
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
    if axis is None:
        axis = range(x.ndim).tolist()
    if not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        # The user passed an empty list/tuple, so we return the input as is
        return x

    axis = normalize_axis_tuple(axis, x.ndim)

    if len(axis) > 1 and np.diff(axis).max() > 1:
        raise ValueError(
            f"join_dims axis must be consecutive, got normalized axis: {axis}"
        )

    return JoinDims(axis)(x)


class SplitDims(Op):
    def __init__(self, axis: int | None = None):
        self.axis = axis

    def _make_output_shape(self, input_shape, shape):
        axis = self.axis

        output_shapes = list(input_shape)
        shape = list(shape)

        output_shapes[axis] = shape.pop(-1)
        for s in shape[::-1]:
            output_shapes.insert(axis, s)

        return tuple(output_shapes)

    def make_node(self, x: Variable, shape: Variable) -> Apply:
        output_shapes = self._make_output_shape(x.type.shape, shape)

        output = tensor(
            shape=tuple([x if isinstance(x, int) else None for x in output_shapes]),
            dtype=x.type.dtype,
        )
        return Apply(self, [x, as_tensor_variable(shape)], [output])

    def perform(self, node, inputs, outputs):
        (x, shape) = inputs
        (out,) = outputs

        out[0] = x.reshape(self._make_output_shape(x.shape, shape))


def split_dims(
    x: TensorLike, shape: ShapeValueType, axis: int | None = None
) -> Variable:
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
    elif isinstance(shape, TensorConstant):
        shape = shape.data.tolist()
    else:
        shape = list(shape)

    if not shape:
        # If we get an empty shape, there is potentially a dummy dimension at the requested axis. This happens for example
        # when splitting a packed tensor that had its dims expanded before packing (e.g. when packing shapes (3, ) and
        # (3, 3) to (3, 4)
        return x.squeeze(axis=axis)

    return SplitDims(axis)(x, shape)


def _analyze_axes_list(axes) -> tuple[int, int, int]:
    """
    Analyze the provided axes list to determine how many axes are before and after the interval to be raveled, as
    well as the minimum and maximum number of axes that the inputs can have.

    The rules are:
    - Axes must be strictly increasing in both the positive and negative parts of the list.
    - Negative axes must come after positive axes.
    - There can be at most one "hole" in the axes list, which can be either an implicit hole on an endpoint
      (e.g. [0, 1]) or an explicit hole in the middle (e.g. [0, 2] or [1, -1]).

    Returns
    -------
    n_axes_before: int
        The number of axes before the interval to be raveled.
    n_axes_after: int
        The number of axes after the interval to be raveled.
    min_axes: int
        The minimum number of axes that the inputs must have.
    """
    if axes is None:
        return 0, 0, 0

    if isinstance(axes, int):
        axes = [axes]
    elif not isinstance(axes, Iterable):
        raise TypeError("axes must be an int, an iterable of ints, or None")

    axes = list(axes)

    if len(axes) == 0:
        raise ValueError("axes=[] is ambiguous; use None to ravel all")

    if len(set(axes)) != len(axes):
        raise ValueError("axes must have no duplicates")

    first_negative_idx = next((i for i, a in enumerate(axes) if a < 0), len(axes))
    positive_axes = list(axes[:first_negative_idx])
    negative_axes = list(axes[first_negative_idx:])

    if not all(a < 0 for a in negative_axes):
        raise ValueError("Negative axes must come after positive")

    def strictly_increasing(s):
        return all(b > a for a, b in pairwise(s))

    if positive_axes and not strictly_increasing(positive_axes):
        raise ValueError("Axes must be strictly increasing in the positive part")
    if negative_axes and not strictly_increasing(negative_axes):
        raise ValueError("Axes must be strictly increasing in the negative part")

    def find_gaps(s):
        """Return positions where b - a > 1."""
        return [i for i, (a, b) in enumerate(pairwise(s)) if b - a > 1]

    pos_gaps = find_gaps(positive_axes)
    neg_gaps = find_gaps(negative_axes)

    if pos_gaps:
        raise ValueError("Positive axes must be contiguous")
    if neg_gaps:
        raise ValueError("Negative axes must be contiguous")

    if positive_axes and positive_axes[0] != 0:
        raise ValueError(
            "If positive axes are provided, the first positive axis must be 0 to avoid ambiguity. To ravel indices "
            "starting from the front, use negative axes only."
        )

    if negative_axes and negative_axes[-1] != -1:
        raise ValueError(
            "If negative axes are provided, the last negative axis must be -1 to avoid ambiguity. To ravel indices "
            "up to the end, use positive axes only."
        )

    positive_only = positive_axes and not negative_axes
    negative_only = negative_axes and not positive_axes

    if positive_only:
        n_before = len(positive_axes)
        n_after = 0
        min_axes = n_before

        return n_before, n_after, min_axes

    elif negative_only:
        n_before = 0
        n_after = len(negative_axes)
        min_axes = n_after

        return n_before, n_after, min_axes

    else:
        n_before = len(positive_axes)
        n_after = len(negative_axes)
        min_axes = n_before + n_after

        return n_before, n_after, min_axes


def pack(*tensors: TensorLike, axes: Sequence[int] | int | None = None):
    n_before, n_after, min_axes = _analyze_axes_list(axes)

    if all([n_before == 0, n_after == 0, min_axes == 0]):
        # Special case -- we're raveling everything
        packed_shapes = [tensor.shape for tensor in tensors]
        reshaped_tensors = [tensor.ravel() for tensor in tensors]

        return join(0, *reshaped_tensors), packed_shapes

    reshaped_tensors: list[TensorLike] = []
    packed_shapes: list[ShapeValueType] = []

    for i, input_tensor in enumerate(tensors):
        n_dim = input_tensor.ndim

        if n_dim < min_axes:
            raise ValueError(
                f"Input {i} (zero indexed) to pack has {n_dim} dimensions, "
                f"but axes={axes} assumes at least {min_axes} dimension{'s' if min_axes != 1 else ''}."
            )

        axis_after_packed_axes = n_dim - n_after
        packed_shapes.append(input_tensor.shape[n_before:axis_after_packed_axes])

        new_shape = (
            *input_tensor.shape[:n_before],
            -1,
            *input_tensor.shape[axis_after_packed_axes:],
        )
        reshaped_tensors.append(input_tensor.reshape(new_shape))

        # Using join_dims could look like this, but it does not insert extra shapes when needed. For example, it fails
        # on pack(pt.tensor("x", shape=(3, )), pt.tensor("y", shape=(3, 3)), axes=0), because the first tensor needs to
        # have its single dimension expanded before the join.

        # join_axes = {n_before, axis_after_packed_axes - 1}
        # reshaped_tensors.append(join_dims(input_tensor, tuple(join_axes)))

    return join(n_before, *reshaped_tensors), packed_shapes


def unpack(packed_input, axes, packed_shapes):
    if axes is None:
        if packed_input.ndim != 1:
            raise ValueError(
                "unpack can only be called with keep_axis=None for 1d inputs"
            )
        split_axis = 0
    else:
        axes = normalize_axis_tuple(axes, ndim=packed_input.ndim)
        try:
            [split_axis] = (i for i in range(packed_input.ndim) if i not in axes)
        except ValueError as err:
            raise ValueError(
                "Unpack must have exactly one more dimension that implied by axes"
            ) from err

    split_inputs = split(
        packed_input,
        splits_size=[prod(shape).astype(int) for shape in packed_shapes],
        n_splits=len(packed_shapes),
        axis=split_axis,
    )

    return [
        split_dims(inp, shape, split_axis)
        for inp, shape in zip(split_inputs, packed_shapes, strict=True)
    ]


__all__ = ["join_dims", "pack", "split_dims", "unpack"]
