from collections.abc import Callable
from typing import Literal

from pytensor.scan import scan
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import (
    TensorVariable,
    arange,
    as_tensor,
    moveaxis,
    switch,
    zeros,
)
from pytensor.tensor.extra_ops import broadcast_to, linspace
from pytensor.tensor.math import divmod as pt_divmod
from pytensor.tensor.math import eq, mean, minimum
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import min as pt_min
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.subtensor import set_subtensor


PadMode = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "minimum",
    "mean",
    "median",
    "wrap",
    "symmetric",
    "reflect",
]
stat_funcs = {"maximum": pt_max, "minimum": pt_min, "mean": mean}


def _slice_at_axis(sl: slice, axis: int) -> tuple[slice, ...]:
    """
    Construct tuple of slices to slice an array in the given dimension.

    Copied from numpy.lib.arraypad._slice_at_axis
    https://github.com/numpy/numpy/blob/300096d384046eee479b0c7a70f79e308da52bff/numpy/lib/_arraypad_impl.py#L33

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> _slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)  # type: ignore


def _get_edges(
    padded: TensorVariable, axis: int, width_pair: tuple[TensorVariable, TensorVariable]
) -> tuple[TensorVariable, TensorVariable]:
    """
    Retrieve edge values from empty-padded array in given dimension.

    Copied from numpy.lib.arraypad._get_edges
    https://github.com/numpy/numpy/blob/300096d384046eee479b0c7a70f79e308da52bff/numpy/lib/_arraypad_impl.py#L154

    Parameters
    ----------
    padded : TensorVariable
        Empty-padded array.
    axis : int
        Dimension in which the edges are considered.
    width_pair : (TensorVariable, TensorVariable)
        Pair of widths that mark the pad area on both sides in the given
        dimension.

    Returns
    -------
    left_edge, right_edge : TensorVariable
        Edge values of the valid area in `padded` in the given dimension. Its
        shape will always match `padded` except for the dimension given by
        `axis` which will have a length of 1.
    """
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]

    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]

    return left_edge, right_edge


def _symbolic_pad(
    x: TensorVariable, pad_width: TensorVariable
) -> tuple[TensorVariable, tuple[slice, ...], TensorVariable]:
    pad_width = broadcast_to(pad_width, as_tensor((x.ndim, 2)))
    new_shape = as_tensor(
        [pad_width[i][0] + size + pad_width[i][1] for i, size in enumerate(x.shape)]
    )
    original_area_slice = tuple(
        slice(pad_width[i][0], pad_width[i][0] + size) for i, size in enumerate(x.shape)
    )
    padded: TensorVariable = set_subtensor(zeros(new_shape)[original_area_slice], x)
    return padded, original_area_slice, pad_width


def _get_padding_slices(
    dim_shape: TensorVariable,
    width_pair: tuple[TensorVariable, TensorVariable],
    axis: int,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    right_slice = _slice_at_axis(slice(dim_shape - width_pair[1], None), axis)

    return left_slice, right_slice


def _constant_pad(
    x: TensorVariable, pad_width: TensorVariable, constant_values: TensorVariable
) -> TensorVariable:
    padded, area_slice, pad_width = _symbolic_pad(x, pad_width)
    values = broadcast_to(constant_values, as_tensor((padded.ndim, 2)))

    for axis in range(padded.ndim):
        width_pair = pad_width[axis]
        value_pair = values[axis]
        dim_shape = padded.shape[axis]

        left_slice, right_slice = _get_padding_slices(dim_shape, width_pair, axis)
        padded = set_subtensor(padded[left_slice], value_pair[0])
        padded = set_subtensor(padded[right_slice], value_pair[1])

    return padded


def _edge_pad(x: TensorVariable, pad_width: TensorVariable) -> TensorVariable:
    padded, area_slice, pad_width = _symbolic_pad(x, pad_width)
    for axis in range(padded.ndim):
        width_pair = pad_width[axis]
        dim_shape = padded.shape[axis]

        left_edge, right_edge = _get_edges(padded, axis, width_pair)
        left_slice, right_slice = _get_padding_slices(dim_shape, width_pair, axis)

        padded = set_subtensor(padded[left_slice], left_edge)
        padded = set_subtensor(padded[right_slice], right_edge)

    return padded


def _get_stats(
    padded: TensorVariable,
    axis: int,
    width_pair: TensorVariable,
    length_pair: tuple[TensorVariable, TensorVariable] | tuple[None, None],
    stat_func: Callable,
):
    """
    Calculate statistic for the empty-padded array in given dimension.

    Copied from numpy.lib.arraypad._get_stats
    https://github.com/numpy/numpy/blob/300096d384046eee479b0c7a70f79e308da52bff/numpy/lib/_arraypad_impl.py#L230

    Parameters
    ----------
    padded : TensorVariable
        Empty-padded array.
    axis : int
        Dimension in which the statistic is calculated.
    width_pair : (TensorVariable, TensorVariable)
        Pair of widths that mark the pad area on both sides in the given dimension.
    length_pair : 2-element sequence of None or TensorVariable
        Gives the number of values in valid area from each side that is taken into account when calculating the
        statistic. If None the entire valid area in `padded` is considered.
    stat_func : function
        Function to compute statistic. The expected signature is
        ``stat_func(x: TensorVariable, axis: int, keepdims: bool) -> TensorVariable``.

    Returns
    -------
    left_stat, right_stat : TensorVariable
        Calculated statistic for both sides of `padded`.
    """
    # Calculate indices of the edges of the area with original values
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    # as well as its length
    max_length = right_index - left_index

    # Limit stat_lengths to max_length
    left_length, right_length = length_pair

    # Calculate statistic for the left side
    left_length = (
        minimum(left_length, max_length) if left_length is not None else max_length
    )
    left_slice = _slice_at_axis(slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    if left_length is None and right_length is None:
        # We could also return early in the more general case of left_length == right_length, but we don't necessarily
        # know these shapes.
        # TODO: Add rewrite to simplify in this case
        return left_stat, left_stat

    # Calculate statistic for the right side
    right_length = (
        minimum(right_length, max_length) if right_length is not None else max_length
    )
    right_slice = _slice_at_axis(slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)

    return left_stat, right_stat


def _stat_pad(
    x: TensorVariable, pad_width: TensorVariable, stat_func, stat_length=None
):
    padded, area_slice, pad_width = _symbolic_pad(x, pad_width)
    if stat_length is None:
        stat_length = [[None, None]] * padded.ndim
    else:
        stat_length = broadcast_to(stat_length, as_tensor((padded.ndim, 2)))

    for axis in range(padded.ndim):
        width_pair = pad_width[axis]
        length_pair = stat_length[axis]
        dim_shape = padded.shape[axis]

        left_stat, right_stat = _get_stats(
            padded, axis, width_pair, length_pair, stat_func
        )
        left_slice, right_slice = _get_padding_slices(dim_shape, width_pair, axis)
        padded = set_subtensor(padded[left_slice], left_stat)
        padded = set_subtensor(padded[right_slice], right_stat)

    return padded


def _linear_ramp_pad(
    x: TensorVariable, pad_width: TensorVariable, end_values: TensorVariable | int = 0
) -> TensorVariable:
    padded, area_slice, pad_width = _symbolic_pad(x, pad_width)
    end_values = as_tensor(end_values)
    end_values = broadcast_to(end_values, as_tensor((padded.ndim, 2)))

    for axis in range(padded.ndim):
        width_pair = pad_width[axis]
        end_value_pair = end_values[axis]
        edge_pair = _get_edges(padded, axis, width_pair)
        dim_shape = padded.shape[axis]
        left_slice, right_slice = _get_padding_slices(dim_shape, width_pair, axis)

        left_ramp, right_ramp = (
            linspace(
                start=end_value,
                stop=specify_broadcastable(edge, axis).squeeze(axis),
                num=width,
                endpoint=False,
                dtype=padded.dtype,
                axis=axis,
            )
            for end_value, edge, width in zip(end_value_pair, edge_pair, width_pair)
        )

        # Reverse the direction of the ramp for the "right" side
        right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]  # type: ignore

        padded = set_subtensor(padded[left_slice], left_ramp)
        padded = set_subtensor(padded[right_slice], right_ramp)

    return padded


def flip(x, axis=None):
    if axis is None:
        index = ((slice(None, None, -1)),) * x.ndim
    else:
        if isinstance(axis, int):
            axis = [axis]
        index = [
            slice(None, None, -1) if i in axis else slice(None, None, None)
            for i in range(x.ndim)
        ]
    return x[index]


def _looping_pad(
    x: TensorVariable, pad_width: TensorVariable, kind: str
) -> TensorVariable:
    pad_width = broadcast_to(pad_width, as_tensor((x.ndim, 2)))

    for axis in range(x.ndim):
        if kind == "wrap":

            def inner_func(i, x):
                return x

        elif kind == "symmetric":
            # Delay creation of this function to here because we want to use the axis global inside the scan
            def inner_func(i, x):
                return switch(eq(i % 2, 0), flip(x, axis=axis), x)

        size = x.shape[axis]
        repeats, (left_remainder, right_remainder) = pt_divmod(pad_width[axis], size)

        left_trim = size - left_remainder
        right_trim = size - right_remainder
        total_repeats = repeats.sum() + 3  # left, right, center

        parts, _ = scan(inner_func, non_sequences=[x], sequences=arange(total_repeats))

        parts = moveaxis(parts, 0, axis)
        new_shape = [-1 if i == axis else x.shape[i] for i in range(x.ndim)]
        x = parts.reshape(new_shape)
        trim_slice = _slice_at_axis(slice(left_trim, -right_trim), axis)
        x = x[trim_slice]

    return x


def pad(x: TensorLike, pad_width: TensorLike, mode: PadMode = "constant", **kwargs):
    allowed_kwargs = {
        "edge": [],
        "wrap": [],
        "constant": ["constant_values"],
        "linear_ramp": ["end_values"],
        "maximum": ["stat_length"],
        "mean": ["stat_length"],
        "median": ["stat_length"],
        "minimum": ["stat_length"],
        "reflect": ["reflect_type"],
        "symmetric": ["reflect_type"],
    }

    if any(value not in allowed_kwargs[mode] for value in kwargs.keys()):
        raise ValueError(
            f"Invalid keyword arguments for mode '{mode}': {kwargs.keys()}"
        )
    x = as_tensor(x)
    pad_width = as_tensor(pad_width)

    if mode == "constant":
        constant_values = as_tensor(kwargs.pop("constant_values", 0))
        return _constant_pad(x, pad_width, constant_values)
    elif mode == "edge":
        return _edge_pad(x, pad_width)
    elif mode in ["maximum", "minimum", "mean", "median"]:
        if mode == "median":
            # TODO: pt.quantile? pt.median?
            raise NotImplementedError("Median padding not implemented")
        stat_func = stat_funcs[mode]
        return _stat_pad(x, pad_width, stat_func, **kwargs)
    elif mode == "linear_ramp":
        end_values = kwargs.pop("end_values", 0)
        return _linear_ramp_pad(x, pad_width, end_values)
    elif mode == "wrap":
        return _looping_pad(x, pad_width, kind="wrap")
    elif mode == "symmetric":
        reflect_type = kwargs.pop("reflect_type", "even")
        if reflect_type == "odd":
            raise NotImplementedError("Odd reflection not implemented")
        return _looping_pad(x, pad_width, kind="symmetric")
    elif mode == "reflect":
        reflect_type = kwargs.pop("reflect_type", "even")
        if reflect_type == "odd":
            raise NotImplementedError("Odd reflection not implemented")
        raise NotImplementedError("Reflect padding not implemented")
    else:
        raise ValueError(f"Invalid mode: {mode}")


__all__ = ["pad"]
