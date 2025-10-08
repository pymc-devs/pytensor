import typing
import warnings
from collections.abc import Hashable, Sequence
from types import EllipsisType
from typing import Literal

import numpy as np

from pytensor.graph import Apply
from pytensor.scalar import discrete_dtypes, upcast
from pytensor.tensor import as_tensor, get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.type import integer_dtypes
from pytensor.tensor.utils import get_static_shape_from_size_variables
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.math import cast, second
from pytensor.xtensor.type import XTensorVariable, as_xtensor, xtensor
from pytensor.xtensor.vectorization import combine_dims_and_shape


class Stack(XOp):
    __props__ = ("new_dim_name", "stacked_dims")

    def __init__(self, new_dim_name: str, stacked_dims: tuple[str, ...]):
        super().__init__()
        if new_dim_name in stacked_dims:
            raise ValueError(
                f"Stacking dim {new_dim_name} must not be in {stacked_dims}"
            )
        if not stacked_dims:
            raise ValueError(f"Stacking dims must not be empty: got {stacked_dims}")
        self.new_dim_name = new_dim_name
        self.stacked_dims = stacked_dims

    def make_node(self, x):
        x = as_xtensor(x)
        if not (set(self.stacked_dims) <= set(x.type.dims)):
            raise ValueError(
                f"Stacking dims {self.stacked_dims} must be a subset of {x.type.dims}"
            )
        if self.new_dim_name in x.type.dims:
            raise ValueError(
                f"Stacking dim {self.new_dim_name} must not be in {x.type.dims}"
            )
        if len(self.stacked_dims) == x.type.ndim:
            batch_dims, batch_shape = (), ()
        else:
            batch_dims, batch_shape = zip(
                *(
                    (dim, shape)
                    for dim, shape in zip(x.type.dims, x.type.shape)
                    if dim not in self.stacked_dims
                )
            )
        stack_shape = 1
        for dim, shape in zip(x.type.dims, x.type.shape):
            if dim in self.stacked_dims:
                if shape is None:
                    stack_shape = None
                    break
                else:
                    stack_shape *= shape
        output = xtensor(
            dtype=x.type.dtype,
            shape=(*batch_shape, stack_shape),
            dims=(*batch_dims, self.new_dim_name),
        )
        return Apply(self, [x], [output])


def stack(x, dim: dict[str, Sequence[str]] | None = None, **dims: Sequence[str]):
    if dim is not None:
        if dims:
            raise ValueError("Cannot use both positional dim and keyword dims in stack")
        dims = dim

    y = x
    for new_dim_name, stacked_dims in dims.items():
        if isinstance(stacked_dims, str):
            raise TypeError(
                f"Stacking dims must be a sequence of strings, got a single string: {stacked_dims}"
            )
        y = Stack(new_dim_name, tuple(stacked_dims))(y)
    return y


class UnStack(XOp):
    __props__ = ("old_dim_name", "unstacked_dims")

    def __init__(
        self,
        old_dim_name: str,
        unstacked_dims: tuple[str, ...],
    ):
        super().__init__()
        if old_dim_name in unstacked_dims:
            raise ValueError(
                f"Dim to be unstacked {old_dim_name} can't be in {unstacked_dims}"
            )
        if not unstacked_dims:
            raise ValueError("Dims to unstack into can't be empty.")
        if len(unstacked_dims) == 1:
            raise ValueError("Only one dimension to unstack into, use rename instead")
        self.old_dim_name = old_dim_name
        self.unstacked_dims = unstacked_dims

    def make_node(self, x, *unstacked_length):
        x = as_xtensor(x)
        if self.old_dim_name not in x.type.dims:
            raise ValueError(
                f"Dim to unstack {self.old_dim_name} must be in {x.type.dims}"
            )
        if not set(self.unstacked_dims).isdisjoint(x.type.dims):
            raise ValueError(
                f"Dims to unstack into {self.unstacked_dims} must not be in {x.type.dims}"
            )

        if len(unstacked_length) != len(self.unstacked_dims):
            raise ValueError(
                f"Number of unstacked lengths {len(unstacked_length)} must match number of unstacked dims {len(self.unstacked_dims)}"
            )
        unstacked_lengths = [as_tensor(length, ndim=0) for length in unstacked_length]
        if not all(length.dtype in discrete_dtypes for length in unstacked_lengths):
            raise TypeError("Unstacked lengths must be discrete dtypes.")

        if x.type.ndim == 1:
            batch_dims, batch_shape = (), ()
        else:
            batch_dims, batch_shape = zip(
                *(
                    (dim, shape)
                    for dim, shape in zip(x.type.dims, x.type.shape)
                    if dim != self.old_dim_name
                )
            )

        static_unstacked_lengths = get_static_shape_from_size_variables(
            unstacked_lengths
        )

        output = xtensor(
            dtype=x.type.dtype,
            shape=(*batch_shape, *static_unstacked_lengths),
            dims=(*batch_dims, *self.unstacked_dims),
        )
        return Apply(self, [x, *unstacked_lengths], [output])


def unstack(x, dim: dict[str, dict[str, int]] | None = None, **dims: dict[str, int]):
    if dim is not None:
        if dims:
            raise ValueError(
                "Cannot use both positional dim and keyword dims in unstack"
            )
        dims = dim

    y = x
    for old_dim_name, unstacked_dict in dims.items():
        y = UnStack(old_dim_name, tuple(unstacked_dict.keys()))(
            y, *tuple(unstacked_dict.values())
        )
    return y


class Transpose(XOp):
    __props__ = ("dims",)

    def __init__(
        self,
        dims: Sequence[str],
    ):
        super().__init__()
        self.dims = tuple(dims)

    def make_node(self, x):
        x = as_xtensor(x)

        transpose_dims = self.dims
        x_shape = x.type.shape
        x_dims = x.type.dims
        if set(transpose_dims) != set(x_dims):
            raise ValueError(f"{transpose_dims} must be a permuted list of {x_dims}")

        output = xtensor(
            dtype=x.type.dtype,
            shape=tuple(x_shape[x_dims.index(d)] for d in transpose_dims),
            dims=transpose_dims,
        )
        return Apply(self, [x], [output])


def transpose(
    x,
    *dim: str | EllipsisType,
    missing_dims: Literal["raise", "warn", "ignore"] = "raise",
):
    """Transpose dimensions of the tensor.

    Parameters
    ----------
    x : XTensorVariable
        Input tensor to transpose.
    *dim : str
        Dimensions to transpose to. Can include ellipsis (...) to represent
        remaining dimensions in their original order.
    missing_dims : {"raise", "warn", "ignore"}, optional
        How to handle dimensions that don't exist in the input tensor:
        - "raise": Raise an error if any dimensions don't exist (default)
        - "warn": Warn if any dimensions don't exist
        - "ignore": Silently ignore any dimensions that don't exist

    Returns
    -------
    XTensorVariable
        Transposed tensor with reordered dimensions.

    Raises
    ------
    ValueError
        If any dimension in dims doesn't exist in the input tensor and missing_dims is "raise".
    """
    # Validate dimensions
    x = as_xtensor(x)
    x_dims = x.type.dims
    invalid_dims = set(dim) - {..., *x_dims}
    if invalid_dims:
        if missing_dims != "ignore":
            msg = f"Dimensions {invalid_dims} do not exist. Expected one or more of: {x_dims}"
            if missing_dims == "raise":
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        # Handle missing dimensions if not raising
        dim = tuple(d for d in dim if d in x_dims or d is ...)

    if dim == ():
        dim = tuple(reversed(x_dims))
    elif dim == (...,):
        dim = x_dims
    elif ... in dim:
        if dim.count(...) > 1:
            raise ValueError("Ellipsis (...) can only appear once in the dimensions")
        # Handle ellipsis expansion
        ellipsis_idx = dim.index(...)
        pre = dim[:ellipsis_idx]
        post = dim[ellipsis_idx + 1 :]
        middle = [d for d in x_dims if d not in pre + post]
        dim = (*pre, *middle, *post)

    if dim == x_dims:
        # No-op transpose
        return x

    return Transpose(dims=typing.cast(tuple[str], dim))(x)


class Concat(XOp):
    __props__ = ("dim",)

    def __init__(self, dim: str):
        self.dim = dim
        super().__init__()

    def make_node(self, *inputs):
        inputs = [as_xtensor(inp) for inp in inputs]
        concat_dim = self.dim

        dims_and_shape: dict[str, int | None] = {}
        for inp in inputs:
            for dim, dim_length in zip(inp.type.dims, inp.type.shape):
                if dim not in dims_and_shape:
                    dims_and_shape[dim] = dim_length
                else:
                    if dim == concat_dim:
                        if dim_length is None:
                            dims_and_shape[dim] = None
                        elif dims_and_shape[dim] is not None:
                            dims_and_shape[dim] += dim_length
                    elif dim_length is not None:
                        # Check for conflicting in non-concatenated shapes
                        if (dims_and_shape[dim] is not None) and (
                            dims_and_shape[dim] != dim_length
                        ):
                            raise ValueError(
                                f"Non-concatenated dimension {dim} has conflicting shapes"
                            )
                        # Keep the non-None shape
                        dims_and_shape[dim] = dim_length

        if concat_dim not in dims_and_shape:
            # It's a new dim, that should be located at the start
            dims_and_shape = {concat_dim: len(inputs)} | dims_and_shape
        elif dims_and_shape[concat_dim] is not None:
            # We need to add +1 for every input that doesn't have this dimension
            for inp in inputs:
                if concat_dim not in inp.type.dims:
                    dims_and_shape[concat_dim] += 1

        dims, shape = zip(*dims_and_shape.items())
        dtype = upcast(*[x.type.dtype for x in inputs])
        output = xtensor(dtype=dtype, dims=dims, shape=shape)
        return Apply(self, inputs, [output])


def concat(xtensors, dim: str):
    """Concatenate a sequence of XTensorVariables along a specified dimension.

    Parameters
    ----------
    xtensors : Sequence of XTensorVariable
        The tensors to concatenate.
    dim : str
        The dimension along which to concatenate the tensors.

    Returns
    -------
    XTensorVariable


    Example
    -------

    .. testcode::

        from pytensor.xtensor import as_xtensor, xtensor, concat

        x = xtensor("x", shape=(2, 3), dims=("a", "b"))
        zero = as_xtensor([0], dims=("a"))

        out = concat([zero, x, zero], dim="a")
        assert out.type.dims == ("a", "b")
        assert out.type.shape == (4, 3)

    """
    return Concat(dim=dim)(*xtensors)


class Squeeze(XOp):
    """Remove specified dimensions from an XTensorVariable.

    Only dimensions that are known statically to be size 1 will be removed.
    Symbolic dimensions must be explicitly specified, and are assumed safe.

    Parameters
    ----------
    dim : tuple of str
        The names of the dimensions to remove.
    """

    __props__ = ("dims",)

    def __init__(self, dims):
        self.dims = tuple(sorted(set(dims)))

    def make_node(self, x):
        x = as_xtensor(x)

        # Validate that dims exist and are size-1 if statically known
        dims_to_remove = []
        x_dims = x.type.dims
        x_shape = x.type.shape
        for d in self.dims:
            if d not in x_dims:
                raise ValueError(f"Dimension {d} not found in {x.type.dims}")
            idx = x_dims.index(d)
            dim_size = x_shape[idx]
            if dim_size is not None and dim_size != 1:
                raise ValueError(f"Dimension {d} has static size {dim_size}, not 1")
            dims_to_remove.append(idx)

        new_dims = tuple(
            d for i, d in enumerate(x.type.dims) if i not in dims_to_remove
        )
        new_shape = tuple(
            s for i, s in enumerate(x.type.shape) if i not in dims_to_remove
        )

        out = xtensor(
            dtype=x.type.dtype,
            shape=new_shape,
            dims=new_dims,
        )
        return Apply(self, [x], [out])


def squeeze(x, dim=None, drop=False, axis=None):
    """Remove dimensions of size 1 from an XTensorVariable."""
    x = as_xtensor(x)

    # drop parameter is ignored in pytensor.xtensor
    if drop is not None:
        warnings.warn("drop parameter has no effect in pytensor.xtensor", UserWarning)

    # dim and axis are mutually exclusive
    if dim is not None and axis is not None:
        raise ValueError("Cannot specify both `dim` and `axis`")

    # if axis is specified, it must be a sequence of ints
    if axis is not None:
        if not isinstance(axis, Sequence):
            axis = [axis]
        if not all(isinstance(a, int) for a in axis):
            raise ValueError("axis must be an integer or a sequence of integers")

        # convert axis to dims
        dims = tuple(x.type.dims[i] for i in axis)

    # if dim is specified, it must be a string or a sequence of strings
    if dim is None:
        dims = tuple(d for d, s in zip(x.type.dims, x.type.shape) if s == 1)
    elif isinstance(dim, str):
        dims = (dim,)
    else:
        dims = tuple(dim)

    if not dims:
        return x  # no-op if nothing to squeeze

    return Squeeze(dims=dims)(x)


class ExpandDims(XOp):
    """Add a new dimension to an XTensorVariable."""

    __props__ = ("dim",)

    def __init__(self, dim):
        if not isinstance(dim, str):
            raise TypeError(f"`dim` must be a string, got: {type(self.dim)}")

        self.dim = dim

    def make_node(self, x, size):
        x = as_xtensor(x)

        if self.dim in x.type.dims:
            raise ValueError(f"Dimension {self.dim} already exists in {x.type.dims}")

        size = as_xtensor(size, dims=())
        if not (size.dtype in integer_dtypes and size.ndim == 0):
            raise ValueError(f"size should be an integer scalar, got {size.type}")
        try:
            static_size = int(get_scalar_constant_value(size))
        except NotScalarConstantError:
            static_size = None

        # If size is a constant, validate it
        if static_size is not None and static_size < 0:
            raise ValueError(f"size must be 0 or positive, got: {static_size}")
        new_shape = (static_size, *x.type.shape)

        # Insert new dim at front
        new_dims = (self.dim, *x.type.dims)

        out = xtensor(
            dtype=x.type.dtype,
            shape=new_shape,
            dims=new_dims,
        )
        return Apply(self, [x, size], [out])


def expand_dims(x, dim=None, create_index_for_new_dim=None, axis=None, **dim_kwargs):
    """Add one or more new dimensions to an XTensorVariable."""
    x = as_xtensor(x)

    # Store original dimensions for axis handling
    original_dims = x.type.dims

    # Warn if create_index_for_new_dim is used (not supported)
    if create_index_for_new_dim is not None:
        warnings.warn(
            "create_index_for_new_dim=False has no effect in pytensor.xtensor",
            UserWarning,
            stacklevel=2,
        )

    if dim is None:
        dim = dim_kwargs
    elif dim_kwargs:
        raise ValueError("Cannot specify both `dim` and `**dim_kwargs`")

    # Check that dim is Hashable or a sequence of Hashable or dict
    if not isinstance(dim, Hashable):
        if not isinstance(dim, Sequence | dict):
            raise TypeError(f"unhashable type: {type(dim).__name__}")
        if not all(isinstance(d, Hashable) for d in dim):
            raise TypeError(f"unhashable type in {type(dim).__name__}")

    # Normalize to a dimension-size mapping
    if isinstance(dim, str):
        dims_dict = {dim: 1}
    elif isinstance(dim, Sequence) and not isinstance(dim, dict):
        dims_dict = dict.fromkeys(dim, 1)
    elif isinstance(dim, dict):
        dims_dict = {}
        for name, val in dim.items():
            if isinstance(val, str):
                raise TypeError(f"Dimension size cannot be a string: {val}")
            if isinstance(val, Sequence | np.ndarray):
                warnings.warn(
                    "When a sequence is provided as a dimension size, only its length is used. "
                    "The actual values (which would be coordinates in xarray) are ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                dims_dict[name] = len(val)
            else:
                # should be int or symbolic scalar
                dims_dict[name] = val
    else:
        raise TypeError(f"Invalid type for `dim`: {type(dim)}")

    # Insert each new dim at the front (reverse order preserves user intent)
    for name, size in reversed(dims_dict.items()):
        x = ExpandDims(dim=name)(x, size)

    # If axis is specified, transpose to put new dimensions in the right place
    if axis is not None:
        # Wrap non-sequence axis in a list
        if not isinstance(axis, Sequence):
            axis = [axis]

        # require len(axis) == len(dims_dict)
        if len(axis) != len(dims_dict):
            raise ValueError("lengths of dim and axis should be identical.")

        # Insert new dimensions at their specified positions
        target_dims = list(original_dims)
        for name, pos in zip(dims_dict, axis):
            # Convert negative axis to positive position relative to current dims
            if pos < 0:
                pos = len(target_dims) + pos + 1
            target_dims.insert(pos, name)
        x = Transpose(dims=tuple(target_dims))(x)

    return x


class Broadcast(XOp):
    """Broadcast multiple XTensorVariables against each other."""

    __props__ = ("exclude",)

    def __init__(self, exclude: Sequence[str] = ()):
        self.exclude = tuple(exclude)

    def make_node(self, *inputs):
        inputs = [as_xtensor(x) for x in inputs]

        exclude = self.exclude
        dims_and_shape = combine_dims_and_shape(inputs, exclude=exclude)

        broadcast_dims = tuple(dims_and_shape.keys())
        broadcast_shape = tuple(dims_and_shape.values())
        dtype = upcast(*[x.type.dtype for x in inputs])

        outputs = []
        for x in inputs:
            x_dims = x.type.dims
            x_shape = x.type.shape
            # The output has excluded dimensions in the order they appear in the op argument
            excluded_dims = tuple(d for d in exclude if d in x_dims)
            excluded_shape = tuple(x_shape[x_dims.index(d)] for d in excluded_dims)

            output = xtensor(
                dtype=dtype,
                shape=broadcast_shape + excluded_shape,
                dims=broadcast_dims + excluded_dims,
            )
            outputs.append(output)

        return Apply(self, inputs, outputs)


def broadcast(
    *args, exclude: str | Sequence[str] | None = None
) -> tuple[XTensorVariable, ...]:
    """Broadcast any number of XTensorVariables against each other.

    Parameters
    ----------
    *args : XTensorVariable
        The tensors to broadcast against each other.
    exclude : str or Sequence[str] or None, optional
    """
    if not args:
        return ()

    if exclude is None:
        exclude = ()
    elif isinstance(exclude, str):
        exclude = (exclude,)
    elif not isinstance(exclude, Sequence):
        raise TypeError(f"exclude must be None, str, or Sequence, got {type(exclude)}")
    # xarray broadcast always returns a tuple, even if there's only one tensor
    return tuple(Broadcast(exclude=exclude)(*args, return_list=True))  # type: ignore


def full_like(x, fill_value, dtype=None):
    """Create a new XTensorVariable with the same shape and dimensions, filled with a specified value.

    Parameters
    ----------
    x : XTensorVariable
        The tensor to fill.
    fill_value : scalar or XTensorVariable
        The value to fill the new tensor with.
    dtype : str or np.dtype, optional
        The data type of the new tensor. If None, uses the dtype of the input tensor.

    Returns
    -------
    XTensorVariable
        A new tensor with the same shape and dimensions as self, filled with fill_value.

    Examples
    --------
    >>> from pytensor.xtensor import xtensor, full_like
    >>> x = xtensor(dtype="float64", dims=("a", "b"), shape=(2, 3))
    >>> y = full_like(x, 5.0)
    >>> assert y.dims == ("a", "b")
    >>> assert y.type.shape == (2, 3)
    """
    x = as_xtensor(x)
    fill_value = as_xtensor(fill_value)

    # Check that fill_value is a scalar (ndim=0)
    if fill_value.type.ndim != 0:
        raise ValueError(
            f"fill_value must be a scalar, got ndim={fill_value.type.ndim}"
        )

    # Handle dtype conversion
    if dtype is not None:
        # If dtype is specified, cast the fill_value to that dtype
        fill_value = cast(fill_value, dtype)
    else:
        # If dtype is None, cast the fill_value to the input tensor's dtype
        # This matches xarray's behavior where it preserves the original dtype
        fill_value = cast(fill_value, x.type.dtype)

    # Use the xtensor second function
    return second(x, fill_value)


def ones_like(x, dtype=None):
    """Create a new XTensorVariable with the same shape and dimensions, filled with ones.

    Parameters
    ----------
    x : XTensorVariable
        The tensor to fill.
    dtype : str or np.dtype, optional
        The data type of the new tensor. If None, uses the dtype of the input tensor.

    Returns:
    XTensorVariable
        A new tensor with the same shape and dimensions as self, filled with ones.

    Examples
    --------
    >>> from pytensor.xtensor import xtensor, full_like
    >>> x = xtensor(dtype="float64", dims=("a", "b"), shape=(2, 3))
    >>> y = ones_like(x)
    >>> assert y.dims == ("a", "b")
    >>> assert y.type.shape == (2, 3)
    """
    return full_like(x, 1.0, dtype=dtype)


def zeros_like(x, dtype=None):
    """Create a new XTensorVariable with the same shape and dimensions, filled with zeros.

    Parameters
    ----------
    x : XTensorVariable
        The tensor to fill.
    dtype : str or np.dtype, optional
        The data type of the new tensor. If None, uses the dtype of the input tensor.

    Returns:
    XTensorVariable
        A new tensor with the same shape and dimensions as self, filled with zeros.

    Examples
    --------
    >>> from pytensor.xtensor import xtensor, full_like
    >>> x = xtensor(dtype="float64", dims=("a", "b"), shape=(2, 3))
    >>> y = zeros_like(x)
    >>> assert y.dims == ("a", "b")
    >>> assert y.type.shape == (2, 3)
    """
    return full_like(x, 0.0, dtype=dtype)
