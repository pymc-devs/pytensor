from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
from numpy.lib._array_utils_impl import normalize_axis_tuple

from pytensor import Variable
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.tensor import TensorLike, as_tensor_variable
from pytensor.tensor.basic import expand_dims, infer_static_shape, join, split
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.math import prod
from pytensor.tensor.shape import ShapeValueType, shape
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

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        output_grads: Sequence[Variable],
    ) -> list[Variable]:
        (x,) = inputs
        (g_out,) = output_grads

        x_shape = shape(x)
        packed_shape = [x_shape[i] for i in self.axis_range]
        return [split_dims(g_out, shape=packed_shape, axis=self.start_axis)]


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
    x : TensorLike
        The input tensor.
    axis : int or sequence of int, optional
        The dimensions to join. If None, all dimensions are joined.

    Returns
    -------
    joined_x : TensorVariable
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

    return JoinDims(start_axis=start_axis, n_axes=n_axes)(x)  # type: ignore[return-value]


class SplitDims(Op):
    __props__ = ("axis",)
    view_map = {0: [0]}

    def __init__(self, axis: int):
        if axis < 0:
            raise ValueError("SplitDims axis must be non-negative")
        self.axis = axis

    def make_node(self, x: Variable, shape: Variable) -> Apply:  # type: ignore[override]
        x = as_tensor_variable(x)
        shape = as_tensor_variable(shape, dtype=int, ndim=1)

        if shape.type.numpy_dtype.kind not in "iu":
            raise TypeError("shape must be an integer tensor")

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

    def connection_pattern(self, node):
        return [[True], [False]]

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        output_grads: Sequence[Variable],
    ) -> list[Variable]:
        (x, _) = inputs
        (g_out,) = output_grads

        n_axes = g_out.ndim - x.ndim + 1  # type: ignore[attr-defined]
        axis_range = list(range(self.axis, self.axis + n_axes))

        return [join_dims(g_out, axis=axis_range), DisconnectedType()()]


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
    split_x : TensorVariable
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
        return squeeze(x, axis=axis)  # type: ignore[no-any-return]

    [axis] = normalize_axis_tuple(axis, x.ndim)  # type: ignore[misc]
    shape = as_tensor_variable(shape, dtype="int64", ndim=1)  # type: ignore[arg-type]

    return SplitDims(axis=axis)(x, shape)  # type: ignore[return-value]


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
        axes = (axes,)
    elif not isinstance(axes, Iterable):
        raise TypeError("axes must be an int, an iterable of ints, or None")

    axes = tuple(axes)

    if len(axes) == 0:
        raise ValueError("axes=[] is ambiguous; use None to ravel all")

    if len(set(axes)) != len(axes):
        raise ValueError("axes must have no duplicates")

    first_negative_idx = next((i for i, a in enumerate(axes) if a < 0), len(axes))
    positive_axes = list(axes[:first_negative_idx])
    negative_axes = list(axes[first_negative_idx:])

    if not all(a < 0 for a in negative_axes):
        raise ValueError("Negative axes must come after positive")

    def not_strictly_increasing(s):
        if len(s) < 1:
            return False
        return any(b <= a for a, b in pairwise(s))

    if not_strictly_increasing(positive_axes):
        raise ValueError("Axes must be strictly increasing in the positive part")
    if not_strictly_increasing(negative_axes):
        raise ValueError("Axes must be strictly increasing in the negative part")

    def find_gaps(s):
        """Find if there are gaps in a strictly increasing sequence."""
        return any(b - a > 1 for a, b in pairwise(s))

    if find_gaps(positive_axes):
        raise ValueError("Positive axes must be contiguous")
    if find_gaps(negative_axes):
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

    n_before = len(positive_axes)
    n_after = len(negative_axes)
    min_axes = n_before + n_after

    return n_before, n_after, min_axes


def pack(
    *tensors: TensorLike, axes: Sequence[int] | int | None = None
) -> tuple[TensorVariable, list[ShapeValueType]]:
    """
    Combine multiple tensors by preserving the specified axes and raveling the rest into a single axis.

    Parameters
    ----------
    *tensors : TensorLike
        Input tensors to be packed.
    axes : int, sequence of int, or None, optional
        Axes to preserve during packing. If None, all axes are raveled. See the Notes section for the rules.

    Returns
    -------
    packed_tensor : TensorVariable
        The packed tensor with specified axes preserved and others raveled.
    packed_shapes : list of ShapeValueType
        A list containing the shapes of the raveled dimensions for each input tensor.

    Notes
    -----
    The `axes` parameter determines which axes are preserved during packing. Axes can be specified using positive or
    negative indices, but must follow these rules:
        - If axes is None, all axes are raveled.
        - If a single integer is provided, it can be positive or negative, and can take any value up to the smallest
            number of dimensions among the input tensors.
        - If a list is provided, it can be all positive, all negative, or a combination of positive and negative.
        - Positive axes must be contiguous and start from 0.
        - Negative axes must be contiguous and end at -1.
        - If positive and negative axes are combined, positive axes must come before negative axes, and both 0 and -1
            must be included.

    Examples
    --------
    The easiest way to understand pack is through examples. The simplest case is using axes=None, which is equivalent
    to ``join(0, *[t.ravel() for t in tensors])``:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 3))
        y = pt.tensor("y", shape=(4, 5, 6))

        packed_tensor, packed_shapes = pt.pack(x, y, axes=None)
        # packed_tensor has shape (6 + 120,) == (126,)
        # packed_shapes is [(2, 3), (4, 5, 6)]

    If we want to preserve a single axis, we can use either positive or negative indexing. Notice that all tensors
    must have the same size along the preserved axis. For example, using axes=0:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 3))
        y = pt.tensor("y", shape=(2, 5, 6))
        packed_tensor, packed_shapes = pt.pack(x, y, axes=0)
        # packed_tensor has shape (2, 3 + 30) == (2, 33)
        # packed_shapes is [(3,), (5, 6)]


    Using negative indexing we can preserve the last two axes:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(4, 2, 3))
        y = pt.tensor("y", shape=(5, 2, 3))
        packed_tensor, packed_shapes = pt.pack(x, y, axes=(-2, -1))
        # packed_tensor has shape (4 + 5, 2, 3) == (9, 2, 3)
        # packed_shapes is [(4,), (5,

    Or using a mix of positive and negative axes, we can preserve the first and last axes:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 4, 3))
        y = pt.tensor("y", shape=(2, 5, 3))
        packed_tensor, packed_shapes = pt.pack(x, y, axes=(0, -1))
        # packed_tensor has shape (2, 4 + 5, 3) == (2, 9, 3)
        # packed_shapes is [(4,), (5,)]
    """
    tensor_list = [as_tensor_variable(t) for t in tensors]

    n_before, n_after, min_axes = _analyze_axes_list(axes)

    reshaped_tensors: list[Variable] = []
    packed_shapes: list[ShapeValueType] = []

    for i, input_tensor in enumerate(tensor_list):
        n_dim = input_tensor.ndim

        if n_dim < min_axes:
            raise ValueError(
                f"Input {i} (zero indexed) to pack has {n_dim} dimensions, "
                f"but axes={axes} assumes at least {min_axes} dimension{'s' if min_axes != 1 else ''}."
            )
        n_after_packed = n_dim - n_after
        packed_shapes.append(input_tensor.shape[n_before:n_after_packed])

        if n_dim == min_axes:
            # If an input has the minimum number of axes, pack implicitly inserts a new axis based on the pattern
            # implied by the axes.
            input_tensor = expand_dims(input_tensor, axis=n_before)
            reshaped_tensors.append(input_tensor)
            continue

        # The reshape we want is (shape[:before], -1, shape[n_after_packed:]). join_dims does (shape[:min(axes)], -1,
        # shape[max(axes)+1:]). So this will work if we choose axes=(n_before, n_after_packed - 1). Because of the
        # rules on the axes input, we will always have n_before <= n_after_packed - 1. A set is used here to cover the
        # corner case when n_before == n_after_packed - 1 (i.e., when there is only one axis to ravel --> do nothing).
        join_axes = range(n_before, n_after_packed)
        joined = join_dims(input_tensor, tuple(join_axes))
        reshaped_tensors.append(joined)

    return join(n_before, *reshaped_tensors), packed_shapes


def unpack(
    packed_input: TensorLike,
    axes: int | Sequence[int] | None,
    packed_shapes: list[ShapeValueType],
) -> list[TensorVariable]:
    """
    Unpack a packed tensor into multiple tensors by splitting along the specified axes and reshaping.

    The unpacking process reverses the packing operation, restoring the original shapes of the input tensors. `axes`
    corresponds to the axes that were preserved during packing, and `packed_shapes` contains the shapes of the raveled
    dimensions for each output tensor (that is, the shapes that were destroyed during packing).

    The signature of unpack is such that the same `axes` should be passed to both `pack` and `unpack` to create a
    "round-trip" operation. For details on the rules for `axes`, see the documentation for `pack`.

    Parameters
    ----------
    packed_input : TensorLike
        The packed tensor to be unpacked.
    axes : int, sequence of int, or None
        Axes that were preserved during packing. If None, the input is assumed to be 1D and axis 0 is used.
    packed_shapes : list of ShapeValueType
        A list containing the shapes of the raveled dimensions for each output tensor.

    Returns
    -------
    unpacked_tensors : list of TensorVariable
        A list of unpacked tensors with their original shapes restored.
    """
    packed_input = as_tensor_variable(packed_input)

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
        splits_size=[prod(shape, dtype=int) for shape in packed_shapes],
        n_splits=len(packed_shapes),
        axis=split_axis,
    )

    return [
        split_dims(inp, shape, split_axis)
        for inp, shape in zip(split_inputs, packed_shapes, strict=True)
    ]


__all__ = ["join_dims", "pack", "split_dims", "unpack"]
