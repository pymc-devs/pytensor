from collections.abc import Callable, Sequence
from typing import Literal, cast

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import (
    TensorVariable,
    arange,
    as_tensor,
    switch,
    zeros,
)
from pytensor.tensor.extra_ops import broadcast_to, linspace
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import maximum, mean, minimum
from pytensor.tensor.math import min as pt_min
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.subtensor import flip, set_subtensor, slice_at_axis, take
from pytensor.tensor.symbolic import TensorSymbolicOp


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

mode_options = {
    "edge": set(),
    "wrap": set(),
    "constant": {"constant_values"},
    "linear_ramp": {"end_values"},
    "maximum": {"stat_length"},
    "mean": {"stat_length"},
    "median": {"stat_length"},
    "minimum": {"stat_length"},
    "reflect": {"reflect_type"},
    "symmetric": {"reflect_type"},
}


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
    left_slice = slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]

    right_index = padded.shape[axis] - width_pair[1]
    right_slice = slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]

    # The slices are symbolic, so `axis` comes back with an unknown length. Callers
    # broadcast these edges across the pad area, and the gradient of that
    # broadcast is only summed when the length is known to be 1.
    return (
        specify_broadcastable(left_edge, axis),
        specify_broadcastable(right_edge, axis),
    )


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
    left_slice = slice_at_axis(slice(None, width_pair[0]), axis)
    right_slice = slice_at_axis(slice(dim_shape - width_pair[1], None), axis)

    return left_slice, right_slice


def _constant_pad(
    x: TensorVariable, pad_width: TensorVariable, constant_values: TensorVariable
) -> TensorVariable:
    padded, _area_slice, pad_width = _symbolic_pad(x, pad_width)
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
    padded, _area_slice, pad_width = _symbolic_pad(x, pad_width)
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
    left_slice = slice_at_axis(slice(left_index, left_index + left_length), axis)
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
    right_slice = slice_at_axis(slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)

    return left_stat, right_stat


def _stat_pad(
    x: TensorVariable,
    pad_width: TensorVariable,
    stat_func: Callable,
    stat_length: TensorVariable | None,
):
    padded, _area_slice, pad_width = _symbolic_pad(x, pad_width)
    if stat_length is None:
        stat_length = [[None, None]] * padded.ndim  # type: ignore
    else:
        stat_length = broadcast_to(stat_length, as_tensor((padded.ndim, 2)))

    for axis in range(padded.ndim):
        width_pair = pad_width[axis]
        length_pair = stat_length[axis]  # type: ignore
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
    padded, _area_slice, pad_width = _symbolic_pad(x, pad_width)
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
            for end_value, edge, width in zip(
                end_value_pair, edge_pair, width_pair, strict=True
            )
        )

        # Reverse the direction of the ramp for the "right" side
        right_ramp = right_ramp[slice_at_axis(slice(None, None, -1), axis)]  # type: ignore

        padded = set_subtensor(padded[left_slice], left_ramp)
        padded = set_subtensor(padded[right_slice], right_ramp)

    return padded


def _static_pad_width(
    pad_width: TensorVariable, ndim: int
) -> tuple[tuple[int, int], ...] | None:
    """``pad_width`` broadcast to ``ndim`` (before, after) pairs, or None if not constant."""
    if not isinstance(pad_width, Constant):
        return None
    widths = np.asarray(pad_width.data)
    if not np.issubdtype(widths.dtype, np.integer):
        raise TypeError("`pad_width` must be of integral type.")
    return tuple(
        (int(before), int(after))
        for before, after in np.broadcast_to(widths, (ndim, 2))
    )


def _gather_indices(
    mode: Literal["wrap", "symmetric", "reflect"],
    size: int | TensorVariable,
    before: int | TensorVariable,
    after: int | TensorVariable,
) -> TensorVariable:
    """Map output positions along one axis back to the input positions they copy.

    ``wrap``, ``symmetric`` and ``reflect`` never compute new values: every
    padded element is some element of the input, selected by a periodic index
    map. ``wrap`` has period ``size``, ``symmetric`` period ``2 * size`` (each
    edge value repeated once at the turning point), and ``reflect`` period
    ``2 * size - 2`` (turning points not repeated).

    When ``size``, ``before`` and ``after`` are all Python ints the map is built
    with NumPy, which keeps the padded output shape statically known.
    """
    if all(isinstance(value, int) for value in (size, before, after)):
        offsets = np.arange(before + size + after) - before
        where, clamp = np.where, max
    else:
        offsets = arange(before + size + after) - before
        where, clamp = switch, maximum

    if mode == "wrap":
        indices = offsets % size
    elif mode == "symmetric":
        period = 2 * size
        folded = offsets % period
        indices = where(folded < size, folded, period - 1 - folded)
    elif mode == "reflect":
        # A size of 1 degenerates the period to 0. Clamping it to 1 maps every
        # position onto the single element, which is what reflecting a length-1
        # axis means.
        period = clamp(2 * size - 2, 1)
        folded = offsets % period
        indices = where(folded < size, folded, period - folded)
    else:
        raise ValueError(f"Invalid gather pad mode: {mode}")

    return as_tensor(indices)


def _gather_pad(
    x: TensorVariable,
    pad_width: TensorVariable,
    mode: Literal["wrap", "symmetric", "reflect"],
) -> TensorVariable:
    """Pad by gathering input elements, one axis at a time."""
    width_pairs: Sequence[tuple[int | TensorVariable, int | TensorVariable]]
    static_widths = _static_pad_width(pad_width, x.ndim)
    if static_widths is None:
        widths = broadcast_to(pad_width, as_tensor((x.ndim, 2)))
        width_pairs = [(widths[axis, 0], widths[axis, 1]) for axis in range(x.ndim)]
    else:
        width_pairs = static_widths

    for axis, (before, after) in enumerate(width_pairs):
        size = x.type.shape[axis]
        if size is None:
            size = x.shape[axis]

        x = take(x, _gather_indices(mode, size, before, after), axis=axis)

    return x


class Pad(TensorSymbolicOp):
    """Pad an array, with the padding graph built from the input types."""

    __props__ = ("pad_mode", "reflect_type", "has_stat_length", "static_pad_width")

    def __init__(
        self,
        *,
        pad_mode: PadMode,
        reflect_type: str | None = None,
        has_stat_length: bool = False,
        static_pad_width: tuple[tuple[int, int], ...] | None = None,
        **kwargs,
    ):
        self.pad_mode = pad_mode
        self.reflect_type = reflect_type
        self.has_stat_length = has_stat_length
        # `pad_width` reaches build_inner_graph as a dummy with no value, so a
        # constant one has to travel alongside it to stay visible inside the graph.
        self.static_pad_width = static_pad_width

        super().__init__(**kwargs)

    def build_inner_graph(self, x, pad_width, *extra_inputs):
        if self.static_pad_width is not None:
            pad_width = as_tensor(np.asarray(self.static_pad_width, dtype="int64"))

        mode = self.pad_mode
        if mode == "constant":
            (constant_values,) = extra_inputs
            return [_constant_pad(x, pad_width, constant_values)]
        if mode == "edge":
            return [_edge_pad(x, pad_width)]
        if mode == "linear_ramp":
            (end_values,) = extra_inputs
            return [_linear_ramp_pad(x, pad_width, end_values)]
        if mode in ("maximum", "minimum", "mean"):
            stat_length = extra_inputs[0] if self.has_stat_length else None
            return [_stat_pad(x, pad_width, stat_funcs[mode], stat_length)]
        return [_gather_pad(x, pad_width, mode)]


def pad(
    x: TensorLike,
    pad_width: TensorLike,
    mode: PadMode = "constant",
    *,
    constant_values: TensorLike | None = None,
    end_values: TensorLike | None = None,
    stat_length: TensorLike | None = None,
    reflect_type: Literal["even", "odd"] | None = None,
) -> TensorVariable:
    """
    Pad an array.

    Parameters
    ----------
    x : array_like of rank N
        The array to pad.

    pad_width : sequence, array_like, or int
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.

    mode : str, optional
        One of the following string values. Default is 'constant'.

        'constant'
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    constant_values : sequence or scalar, optional
        Only valid for mode 'constant'. The values to set the padded values to
        for each axis.

        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.

    end_values : sequence or scalar, optional
        Only valid for mode 'linear_ramp'. The values used for the ending value
        of the linear ramp, which form the edge of the padded array.

        ``((before_1, after_1), ... (before_N, after_N))`` unique end values
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after end values for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.

    stat_length : sequence or int, optional
        Only valid for modes 'maximum', 'mean', and 'minimum'. Number of values
        at the edge of each axis used to compute the statistic.

        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic
        lengths for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.

        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default is ``None``, to use the entire axis.

    reflect_type : str, optional
        Only valid for modes 'reflect' and 'symmetric'. Only 'even' is currently
        accepted, which reflects around the edge value without altering it.
        Default is 'even'.

    Returns
    -------
    padded : TensorVariable
        Padded array of rank equal to ``x`` with shape increased according to
        ``pad_width``.

    Raises
    ------
    ValueError
        If ``mode`` is not a recognized padding mode, or a keyword argument is
        given that ``mode`` does not accept.
    TypeError
        If ``pad_width`` is a constant of non-integral dtype.
    NotImplementedError
        If ``mode`` is 'median', or ``reflect_type`` is 'odd'.

    Examples
    --------

    .. testcode::

        import pytensor.tensor as pt
        a = [1, 2, 3, 4, 5]
        print(pt.pad(a, (2, 3), 'constant', constant_values=(4, 6)).eval())

    .. testoutput::

        [4. 4. 1. 2. 3. 4. 5. 6. 6. 6.]

    .. testcode::

        print(pt.pad(a, (2, 3), 'edge').eval())

    .. testoutput::

         [1. 1. 1. 2. 3. 4. 5. 5. 5. 5.]

    .. testcode::

        print(pt.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4)).eval())

    .. testoutput::

        [ 5.  3.  1.  2.  3.  4.  5.  2. -1. -4.]

    .. testcode::

        print(pt.pad(a, (2,), 'maximum').eval())

    .. testoutput::

        [5. 5. 1. 2. 3. 4. 5. 5. 5.]

    .. testcode::

        print(pt.pad(a, (2,), 'mean').eval())

    .. testoutput::

        [3. 3. 1. 2. 3. 4. 5. 3. 3.]

    .. testcode::

        a = [[1, 2], [3, 4]]
        print(pt.pad(a, ((3, 2), (2, 3)), 'minimum').eval())

    .. testoutput::

        [[1. 1. 1. 2. 1. 1. 1.]
         [1. 1. 1. 2. 1. 1. 1.]
         [1. 1. 1. 2. 1. 1. 1.]
         [1. 1. 1. 2. 1. 1. 1.]
         [3. 3. 3. 4. 3. 3. 3.]
         [1. 1. 1. 2. 1. 1. 1.]
         [1. 1. 1. 2. 1. 1. 1.]]

    .. testcode::

        a = [1, 2, 3, 4, 5]
        print(pt.pad(a, (2, 3), 'reflect').eval())

    .. testoutput::

        [3 2 1 2 3 4 5 4 3 2]

    .. testcode::

        print(pt.pad(a, (2, 3), 'symmetric').eval())

    .. testoutput::

        [2 1 1 2 3 4 5 5 4 3]

    .. testcode::

        print(pt.pad(a, (2, 3), 'wrap').eval())

    .. testoutput::

        [4 5 1 2 3 4 5 1 2 3]

    """
    if mode not in mode_options:
        raise ValueError(f"Invalid mode: {mode}")

    supplied = {
        name
        for name, value in (
            ("constant_values", constant_values),
            ("end_values", end_values),
            ("stat_length", stat_length),
            ("reflect_type", reflect_type),
        )
        if value is not None
    }
    unsupported = supplied - mode_options[mode]
    if unsupported:
        raise ValueError(
            f"Invalid keyword arguments for mode '{mode}': {sorted(unsupported)}. "
            f"Mode '{mode}' accepts {sorted(mode_options[mode])}"
        )

    x = as_tensor(x, name="x")
    pad_width = as_tensor(pad_width, name="pad_width")
    inputs = [x, pad_width]
    has_stat_length = False

    if mode == "constant":
        if constant_values is None:
            constant_values = 0
        inputs += [as_tensor(constant_values, name="constant_values")]

    elif mode == "linear_ramp":
        if end_values is None:
            end_values = 0
        inputs += [as_tensor(end_values, name="end_values")]

    elif mode in ("maximum", "minimum", "mean", "median"):
        if mode == "median":
            # TODO: Revisit this after we implement a quantile function.
            #  See https://github.com/pymc-devs/pytensor/issues/53
            raise NotImplementedError("Median padding not implemented")
        if stat_length is not None:
            has_stat_length = True
            inputs += [as_tensor(stat_length, name="stat_length")]

    elif mode in ("symmetric", "reflect"):
        if reflect_type is None:
            reflect_type = "even"
        if reflect_type == "odd":
            raise NotImplementedError(
                "Odd reflection not implemented. If you need this feature, please open an "
                "issue at https://github.com/pymc-devs/pytensor/issues"
            )

    # Every other mode is rejected above if it was given a `reflect_type`, so this stays
    # None outside the reflecting modes.
    op = Pad(
        pad_mode=mode,
        reflect_type=reflect_type,
        has_stat_length=has_stat_length,
        static_pad_width=_static_pad_width(pad_width, x.ndim),
    )
    return cast(TensorVariable, op(*inputs))


__all__ = ["flip", "pad"]
