from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import TypeAlias, Union
from typing import cast as typing_cast

import numpy as np
from numpy.lib._array_utils_impl import normalize_axis_index, normalize_axis_tuple

from pytensor import Variable
from pytensor.gradient import disconnected_type
from pytensor.graph import Apply
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.link.c.op import COp
from pytensor.scalar import ScalarVariable
from pytensor.tensor import TensorLike, as_tensor_variable, get_vector_length
from pytensor.tensor import basic as ptb
from pytensor.tensor.basic import infer_static_shape, join, split
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import prod
from pytensor.tensor.shape import shape, specify_shape
from pytensor.tensor.type import int_dtypes, tensor
from pytensor.tensor.variable import TensorConstant, TensorVariable


ShapeValueType: TypeAlias = (
    int | np.integer | ScalarVariable | TensorVariable | np.ndarray
)


class JoinDims(Op):
    __props__ = ("start_axis", "n_axes")
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
        if (self.start_axis + self.n_axes) > x.type.ndim:
            raise ValueError(
                f"JoinDims was asked to join dimensions {self.start_axis} to {self.n_axes}, "
                f"but input {x} has only {x.type.ndim} dimensions."
            )

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
        joined_shape = prod([input_shape[i] for i in self.axis_range], dtype=int)
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

    def L_op(self, inputs, outputs, output_grads):
        (x,) = inputs
        (g_out,) = output_grads

        x_shape = x.shape
        packed_shape = [x_shape[i] for i in self.axis_range]
        return [split_dims(g_out, shape=packed_shape, axis=self.start_axis)]


@_vectorize_node.register(JoinDims)
def _vectorize_joindims(op, node, x):
    [old_x] = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim
    return JoinDims(op.start_axis + batched_ndims, op.n_axes).make_node(x)


def join_dims(
    x: TensorLike, start_axis: int = 0, n_axes: int | None = None
) -> TensorVariable:
    """Join consecutive dimensions of a tensor into a single dimension.

    Parameters
    ----------
    x : TensorLike
        The input tensor.
    start_axis : int, default 0
        The axis from which to start joining dimensions
    n_axes: int, optional.
        The number of axis to join after `axis`. If `None` joins all remaining axis.
        If 0, it inserts a new dimension of length 1.

    Returns
    -------
    joined_x : TensorVariable
        The reshaped tensor with joined dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor("x", shape=(2, 3, 4, 5))
    >>> y = pt.join_dims(x)
    >>> y.type.shape
    (120,)
    >>> y = pt.join_dims(x, start_axis=1)
    >>> y.type.shape
    (2, 60)
    >>> y = pt.join_dims(x, start_axis=1, n_axes=2)
    >>> y.type.shape
    (2, 12, 5)
    """
    x = as_tensor_variable(x)
    ndim = x.type.ndim

    if start_axis < 0:
        # We treat scalars as if they had a single axis
        start_axis += max(1, ndim)

    if not 0 <= start_axis <= ndim:
        raise IndexError(
            f"Axis {start_axis} is out of bounds for array of dimension {ndim}"
        )

    if n_axes is None:
        n_axes = ndim - start_axis

    return JoinDims(start_axis, n_axes)(x)  # type: ignore[return-value]


class SplitDims(Op):
    __props__ = ("axis",)
    view_map = {0: [0]}

    def __init__(self, axis: int):
        if axis < 0:
            raise ValueError("SplitDims axis must be non-negative")
        self.axis = axis

    def make_node(self, x, shape):
        x = as_tensor_variable(x)
        shape = as_tensor_variable(shape, dtype=int)

        if shape.type.numpy_dtype.kind not in "iu":
            raise TypeError("shape must be an integer tensor")

        if shape.type.ndim != 1:
            raise TypeError(
                f"shape must be a 1-D tensor, got {shape} with {shape.type.ndim} dimensions"
            )

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

    def L_op(self, inputs, outputs, output_grads):
        (x, _) = inputs
        (g_out,) = output_grads
        n_axes = g_out.ndim - x.ndim + 1
        return [
            join_dims(g_out, start_axis=self.axis, n_axes=n_axes),
            disconnected_type(),
        ]


@_vectorize_node.register(SplitDims)
def _vectorize_splitdims(op, node, x, shape):
    from pytensor.tensor.blockwise import vectorize_node_fallback

    old_x, _ = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim

    if as_tensor_variable(shape).type.ndim != 1:
        return vectorize_node_fallback(op, node, x, shape)

    return SplitDims(axis=op.axis + batched_ndims).make_node(x, shape)


def split_dims(
    x: TensorLike,
    shape: ShapeValueType | Sequence[ShapeValueType],
    axis: int = 0,
) -> TensorVariable:
    """Split a dimension of a tensor into multiple dimensions.

    Parameters
    ----------
    x : TensorLike
        The input tensor.
    shape : int or sequence of int
        The new shape to split the specified dimension into.
    axis : int, default 0
        The dimension to split.

    Returns
    -------
    split_x : TensorVariable
        The reshaped tensor with split dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor("x", shape=(6, 4, 6))
    >>> y = pt.split_dims(x, shape=(2, 3))
    >>> y.type.shape
    (2, 3, 4, 6)
    >>> y = pt.split_dims(x, shape=(2, 3), axis=-1)
    >>> y.type.shape
    (6, 4, 2, 3)
    """
    x = as_tensor_variable(x)
    axis = normalize_axis_index(axis, x.ndim)

    # Convert scalar shape to 1d tuple (shape,)
    # Which is basically a specify_shape
    if not isinstance(shape, Sequence):
        if isinstance(shape, TensorVariable | np.ndarray):
            if shape.ndim == 0:
                shape = (shape,)
        elif isinstance(shape, int | np.integer | ScalarVariable):
            shape = (shape,)

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
    *tensors: TensorLike, keep_axes: Sequence[int] | int | None = None
) -> tuple[TensorVariable, list[TensorVariable]]:
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
    The easiest way to understand pack is through examples.
    The simplest case is using the default keep_axes=None, which is equivalent to ``concatenate([t.ravel() for t in tensors])``:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 3))
        y = pt.tensor("y", shape=(4, 5, 6))

        packed_tensor, packed_shapes = pt.pack(x, y)
        # packed_tensor has shape (6 + 120,) == (126,)
        # packed_shapes is [(2, 3), (4, 5, 6)]

    If we want to preserve a single axis, we can use either positive or negative indexing.
    Notice that all tensors must have the same size along the preserved axis.
    For example, using keep_axes=0:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 3))
        y = pt.tensor("y", shape=(2, 5, 6))
        packed_tensor, packed_shapes = pt.pack(x, y, keep_axes=0)
        # packed_tensor has shape (2, 3 + 30) == (2, 33)
        # packed_shapes is [(3,), (5, 6)]


    Using negative indexing we can preserve the last two axes:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(4, 2, 3))
        y = pt.tensor("y", shape=(5, 2, 3))
        packed_tensor, packed_shapes = pt.pack(x, y, keep_axes=(-2, -1))
        # packed_tensor has shape (4 + 5, 2, 3) == (9, 2, 3)
        # packed_shapes is [(4,), (5,

    Or using a mix of positive and negative axes, we can preserve the first and last axes:

    .. code-block:: python
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(2, 4, 3))
        y = pt.tensor("y", shape=(2, 5, 3))
        packed_tensor, packed_shapes = pt.pack(x, y, keep_axes=(0, -1))
        # packed_tensor has shape (2, 4 + 5, 3) == (2, 9, 3)
        # packed_shapes is [(4,), (5,)]
    """
    tensor_list = [as_tensor_variable(t) for t in tensors]

    n_before, n_after, min_axes = _analyze_axes_list(keep_axes)

    reshaped_tensors: list[Variable] = []
    packed_shapes: list[TensorVariable] = []

    for i, input_tensor in enumerate(tensor_list):
        n_dim = input_tensor.ndim

        if n_dim < min_axes:
            raise ValueError(
                f"Input {i} (zero indexed) to pack has {n_dim} dimensions, "
                f"but {keep_axes=} assumes at least {min_axes} dimension{'s' if min_axes != 1 else ''}."
            )

        n_packed = n_dim - n_after - n_before
        packed_shapes.append(input_tensor.shape[n_before : n_before + n_packed])
        joined = join_dims(input_tensor, n_before, n_packed)
        reshaped_tensors.append(joined)

    return join(n_before, *reshaped_tensors), packed_shapes


def unpack(
    packed_input: TensorLike,
    packed_shapes: Sequence[ShapeValueType],
    keep_axes: int | Sequence[int] | None = None,
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
    packed_shapes : list of ShapeValueType
        A list containing the shapes of the raveled dimensions for each output tensor.
    keep_axes : int, sequence of int, optional
        Axes that were preserved during packing. Default is None

    Returns
    -------
    unpacked_tensors : list of TensorVariable
        A list of unpacked tensors with their original shapes restored.
    """
    packed_input = as_tensor_variable(packed_input)
    if keep_axes is None:
        if packed_input.ndim != 1:
            raise ValueError(
                "unpack can only be called with keep_axis=None for 1d inputs"
            )
        split_axis = 0
    else:
        keep_axes = normalize_axis_tuple(keep_axes, ndim=packed_input.ndim)
        try:
            [split_axis] = (i for i in range(packed_input.ndim) if i not in keep_axes)
        except ValueError as err:
            raise ValueError(
                f"unpack input must have exactly one more dimension that implied by keep_axes. "
                f"{packed_input} has {packed_input.type.ndim} dimensions, expected {len(keep_axes) + 1}"
            ) from err

    n_splits = len(packed_shapes)
    if n_splits == 1:
        # If there is only one tensor to unpack, no need to split
        split_inputs = [packed_input]
    else:
        split_inputs = split(
            packed_input,
            splits_size=[prod(shape, dtype=int) for shape in packed_shapes],
            axis=split_axis,
        )

    return [
        split_dims(inp, shape, split_axis)
        for inp, shape in zip(split_inputs, packed_shapes, strict=True)
    ]


class Reshape(COp):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time.
    """

    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True
    _output_type_depends_on_input_value = True

    check_input = False
    __props__ = ("ndim",)

    def __init__(self, ndim):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.ndim}}}"

    def make_node(self, x, shp):
        x = ptb.as_tensor_variable(x)
        shp_orig = shp
        shp = ptb.as_tensor_variable(shp, ndim=1)
        if shp.type.shape == (None,):
            shp = specify_shape(shp, self.ndim)
        if not (
            shp.dtype in int_dtypes
            or (isinstance(shp, TensorConstant) and shp.data.size == 0)
        ):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
            raise TypeError(f"Shape must be integers; got {shp.dtype}")

        assert shp.ndim == 1

        if isinstance(shp, TensorConstant):
            out_shape = [int(s) if s >= 0 else None for s in shp.data]
        else:
            out_shape = [None] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in range(self.ndim):
                y = shp_list[index]
                y = ptb.as_tensor_variable(y)
                try:
                    s_val = ptb.get_scalar_constant_value(y).item()
                    if s_val >= 0:
                        out_shape[index] = s_val
                except NotScalarConstantError:
                    pass

        # If we only don't know the size of one output dimension,
        # but we know all the input dimensions we can deduce it
        # This happens often when there is -1 as an input of Reshape
        if None not in x.type.shape and out_shape.count(None) == 1:
            full_size = np.prod(x.type.shape)
            known_size = np.prod([s for s in out_shape if s is not None])
            out_shape[out_shape.index(None)] = int(full_size // known_size)

        out_shape = tuple(out_shape)

        # Run some eager error checks
        if len(out_shape) != self.ndim:
            raise ValueError(
                "Shape argument to Reshape has incorrect length:"
                f" {len(out_shape)}, should be {self.ndim}"
            )

        if None not in x.type.shape and None not in out_shape:
            if np.prod(x.type.shape) != np.prod(out_shape):
                raise ValueError(
                    f"Reshape: Input shape {x.type.shape} is incompatible with new shape {out_shape}"
                )

        return Apply(self, [x, shp], [tensor(dtype=x.type.dtype, shape=out_shape)])

    def perform(self, node, inp, out_):
        x, shp = inp
        (out,) = out_
        if len(shp) != self.ndim:
            raise ValueError(
                "Shape argument to Reshape has incorrect"
                f" length: {len(shp)}, should be {self.ndim}"
            )
        out[0] = np.reshape(x, shp)

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, _shp = inp
        (g_out,) = grads
        return [reshape(g_out, shape(x), ndim=x.ndim), disconnected_type()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], return_list=True)

    def infer_shape(self, fgraph, node, ishapes):
        from pytensor.tensor.math import eq, maximum, mul

        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.
        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in range(self.ndim)]
        #    )]
        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.
        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError(
                    "shape argument to Reshape.perform"
                    " must have at most one entry equal to -1"
                )
            return [requ]
        else:
            requ = [requ[i] for i in range(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [
                tuple(
                    ptb.switch(eq(requ[i], -1), rest_size, requ[i])
                    for i in range(self.ndim)
                )
            ]

    def c_code_cache_version(self):
        return (10,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shp = inputs
        shp_dtype = node.inputs[1].type.dtype_specs()[1]
        (z,) = outputs
        fail = sub["fail"]
        ndim = self.ndim

        return f"""
        assert (PyArray_NDIM({shp}) == 1);

        // Unpack shape into new_dims
        npy_intp new_dims[{ndim}];
        for (int ii = 0; ii < {ndim}; ++ii)
        {{
            new_dims[ii] = (({shp_dtype}*)(PyArray_BYTES({shp}) + ii * PyArray_STRIDES({shp})[0]))[0];
        }}

        PyArray_Dims newshape;
        newshape.len = {ndim};
        newshape.ptr = new_dims;

        Py_XDECREF({z});
        {z} = (PyArrayObject *) PyArray_Newshape({x}, &newshape, NPY_CORDER);

        if (!{z}) {{
            //The error message should have been set by PyArray_Newshape
            {fail};
        }}
        """


@_vectorize_node.register(Reshape)
def _vectorize_reshape(op, node, x, shape):
    from pytensor.tensor.blockwise import vectorize_node_fallback

    old_x, old_shape = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim

    if as_tensor_variable(shape).type.ndim != 1:
        return vectorize_node_fallback(op, node, x, shape)

    if len(tuple(old_shape)) == len(tuple(shape)):
        new_shape = [*x.shape[:batched_ndims], *shape]
    elif len(tuple(old_shape)) == (len(tuple(shape)) - batched_ndims):
        new_shape = shape
    else:
        raise ValueError("Invalid shape length passed into vectorize node of Reshape")

    return reshape(x, new_shape, ndim=len(tuple(new_shape))).owner


def reshape(
    x: "TensorLike",
    newshape: Union["TensorLike", Sequence["TensorLike"]],
    *,
    ndim: int | None = None,
) -> TensorVariable:
    if ndim is None:
        newshape = ptb.as_tensor_variable(newshape)  # type: ignore
        if newshape.type.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                f" scalar. Got {newshape} after conversion to a vector."
            )
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                f"The length of the provided shape ({newshape}) cannot "
                "be automatically determined, so PyTensor is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem."
            )
    op = Reshape(ndim)
    rval = op(x, newshape)
    return typing_cast(TensorVariable, rval)


__all__ = ["Reshape", "join_dims", "pack", "reshape", "split_dims", "unpack"]
