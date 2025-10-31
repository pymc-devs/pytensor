from collections.abc import Callable
from functools import partial
from typing import Literal, cast

from pytensor.compile.builders import OpFromGraph
from pytensor.ifelse import ifelse
from pytensor.scan import scan
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import (
    TensorVariable,
    as_tensor,
    concatenate,
    expand_dims,
    moveaxis,
    switch,
    zeros,
)
from pytensor.tensor.extra_ops import broadcast_to, linspace
from pytensor.tensor.math import divmod as pt_divmod
from pytensor.tensor.math import eq, gt, mean, minimum
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import min as pt_min
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.subtensor import flip, set_subtensor, slice_at_axis


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


def _wrap_pad(x: TensorVariable, pad_width: TensorVariable) -> TensorVariable:
    pad_width = broadcast_to(pad_width, as_tensor((x.ndim, 2)))

    for axis in range(x.ndim):
        size = x.shape[axis]

        # Compute how many complete copies of the input will be padded on this dimension, along with the amount of
        # overflow on the final copy
        repeats, (left_remainder, right_remainder) = pt_divmod(pad_width[axis], size)

        # In the next step we will generate extra copies of the input, and then trim them down to the correct size.
        left_trim = size - left_remainder
        right_trim = size - right_remainder

        # The total number of copies needed is always the sum of the number of complete copies to add, plus the original
        # input itself, plus the two edge copies that will be trimmed down.
        total_repeats = repeats.sum() + 3

        # Create a batch dimension and clone the input the required number of times
        parts = expand_dims(x, (0,)).repeat(total_repeats, axis=0)

        # Move the batch dimension to the active dimension
        parts = moveaxis(parts, 0, axis)

        # Ravel the active dimension while preserving the shapes of the inactive dimensions. This will expand the
        # active dimension to have the correctly padded shape, plus excess to be trimmed
        new_shape = [-1 if i == axis else x.shape[i] for i in range(x.ndim)]
        x = parts.reshape(new_shape)

        # Trim the excess on the active dimension
        trim_slice = slice_at_axis(slice(left_trim, -right_trim), axis)
        x = x[trim_slice]

    return x


def _build_padding_one_direction(array, array_flipped, repeats, *, inner_func, axis):
    [_, parts] = scan(
        inner_func,
        non_sequences=[array, array_flipped],
        outputs_info=[0, None],
        n_steps=repeats,
        return_updates=False,
    )

    parts = moveaxis(parts, 0, axis)
    new_shape = [-1 if i == axis else array.shape[i] for i in range(array.ndim)]
    padding = parts.reshape(new_shape)

    return padding


def _symmetric_pad(x, pad_width):
    def _symmetric_inner(i, x, x_flipped, padding_left):
        return i + 1, ifelse(eq(i % 2, int(padding_left)), x_flipped, x)

    pad_width = broadcast_to(pad_width, as_tensor((x.ndim, 2)))

    for axis in range(x.ndim):
        x_flipped = flip(x, axis=axis)
        original_size = x.shape[axis]

        repeats, remainders = pt_divmod(pad_width[axis], original_size)
        has_remainder = gt(remainders, 0)
        repeats = repeats + has_remainder

        left_padding = _build_padding_one_direction(
            x,
            x_flipped,
            repeats[0],
            axis=axis,
            inner_func=partial(_symmetric_inner, padding_left=True),
        )
        right_padding = _build_padding_one_direction(
            x,
            x_flipped,
            repeats[1],
            axis=axis,
            inner_func=partial(_symmetric_inner, padding_left=False),
        )

        x = concatenate([flip(left_padding, axis), x, right_padding], axis=axis)

        (left_trim, right_trim) = switch(
            has_remainder, original_size - remainders, remainders
        )
        right_trim = x.shape[axis] - right_trim

        trim_slice = slice_at_axis(slice(left_trim, right_trim), axis)
        x = x[trim_slice]

    return x


def _reflect_pad(x, pad_width):
    def _reflect_inner(i, x, x_flipped, padding_left):
        return i + 1, ifelse(eq(i % 2, int(padding_left)), x_flipped, x)

    pad_width = broadcast_to(pad_width, as_tensor((x.ndim, 2)))
    for axis in range(x.ndim):
        trimmed_size = x.shape[axis] - 1

        trim_slice = slice_at_axis(slice(None, -1), axis)
        x_trimmed = x[trim_slice]
        x_flipped = flip(x, axis=axis)[trim_slice]

        repeats, remainders = pt_divmod(pad_width[axis], trimmed_size)
        repeats = repeats + 1

        left_padding = _build_padding_one_direction(
            x_trimmed,
            x_flipped,
            repeats[0],
            axis=axis,
            inner_func=partial(_reflect_inner, padding_left=True),
        )
        right_padding = _build_padding_one_direction(
            x_trimmed,
            x_flipped,
            repeats[1],
            axis=axis,
            inner_func=partial(_reflect_inner, padding_left=False),
        )

        left_trim = slice_at_axis(slice(trimmed_size - remainders[0] - 1, -1), axis)
        right_trim = slice_at_axis(
            slice(1, right_padding.shape[axis] - trimmed_size + remainders[1] + 1), axis
        )

        x = concatenate(
            [flip(left_padding, axis)[left_trim], x, right_padding[right_trim]],
            axis=axis,
        )
    return x


class Pad(OpFromGraph):
    """
    Wrapper Op for Pad graphs
    """

    def __init__(
        self, inputs, outputs, pad_mode, reflect_type=None, has_stat_length=False
    ):
        self.pad_mode = pad_mode
        self.reflect_type = reflect_type
        self.has_stat_length = has_stat_length

        super().__init__(inputs=inputs, outputs=outputs)


def pad(
    x: TensorLike, pad_width: TensorLike, mode: PadMode = "constant", **kwargs
) -> TensorVariable:
    """
    Pad an array.

    Parameters
    ----------
    array : array_like of rank N
        The array to pad.

    pad_width : sequence, array_like, or int
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.

    mode : str or function, optional
        One of the following string values or a user supplied function.

        'constant' (default)
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

    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic
        lengths for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.

        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default is ``None``, to use the entire axis.

    constant_values : sequence or scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.

    end_values : sequence or scalar, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ``((before_1, after_1), ... (before_N, after_N))`` unique end values
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after end values for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.

    reflect_type : str, optional
        Only 'even' is currently accepted. Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

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
    if any(value not in allowed_kwargs[mode] for value in kwargs.keys()):
        raise ValueError(
            f"Invalid keyword arguments for mode '{mode}': {kwargs.keys()}"
        )
    x = as_tensor(x, name="x")
    pad_width = as_tensor(pad_width, name="pad_width")
    inputs = [x, pad_width]
    attrs = {}

    if mode == "constant":
        constant_values = as_tensor(
            kwargs.pop("constant_values", 0), name="constant_values"
        )
        inputs += [constant_values]
        outputs = _constant_pad(x, pad_width, constant_values)

    elif mode == "edge":
        outputs = _edge_pad(x, pad_width)

    elif mode in ["maximum", "minimum", "mean", "median"]:
        if mode == "median":
            # TODO: Revisit this after we implement a quantile function.
            #  See https://github.com/pymc-devs/pytensor/issues/53
            raise NotImplementedError("Median padding not implemented")
        stat_func = cast(Callable, stat_funcs[mode])
        stat_length = kwargs.get("stat_length")
        if stat_length is not None:
            attrs.update({"has_stat_length": True})
            stat_length = as_tensor(stat_length, name="stat_length")
            inputs += [stat_length]

        outputs = _stat_pad(x, pad_width, stat_func, stat_length)

    elif mode == "linear_ramp":
        end_values = kwargs.pop("end_values", 0)
        end_values = as_tensor(end_values)

        inputs += [end_values]
        outputs = _linear_ramp_pad(x, pad_width, end_values)

    elif mode == "wrap":
        outputs = _wrap_pad(x, pad_width)

    elif mode == "symmetric":
        reflect_type = kwargs.pop("reflect_type", "even")
        if reflect_type == "odd":
            raise NotImplementedError(
                "Odd reflection not implemented. If you need this feature, please open an "
                "issue at https://github.com/pymc-devs/pytensor/issues"
            )
        attrs.update({"reflect_type": reflect_type})
        outputs = _symmetric_pad(x, pad_width)

    elif mode == "reflect":
        reflect_type = kwargs.pop("reflect_type", "even")
        if reflect_type == "odd":
            raise NotImplementedError(
                "Odd reflection not implemented. If you need this feature, please open an "
                "issue at https://github.com/pymc-devs/pytensor/issues"
            )
        attrs.update({"reflect_type": reflect_type})
        outputs = _reflect_pad(x, pad_width)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    op = Pad(inputs=inputs, outputs=[outputs], pad_mode=mode, **attrs)(*inputs)
    return cast(TensorVariable, op)


__all__ = ["flip", "pad"]
