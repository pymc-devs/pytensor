import warnings
from collections.abc import Sequence
from typing import Literal

from pytensor import Variable
from pytensor.graph import Apply
from pytensor.scalar import upcast
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.type import as_xtensor, xtensor


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


def expand_ellipsis(
    dims: tuple[str, ...],
    all_dims: tuple[str, ...],
    validate: bool = True,
    missing_dims: Literal["raise", "warn", "ignore"] = "raise",
) -> tuple[str, ...]:
    """Expand ellipsis in dimension permutation.

    Parameters
    ----------
    dims : tuple[str, ...]
        The dimension permutation, which may contain ellipsis
    all_dims : tuple[str, ...]
        All available dimensions
    validate : bool, default True
        Whether to check that all non-ellipsis elements in dims are valid dimension names.
    missing_dims : {"raise", "warn", "ignore"}, optional
        How to handle dimensions that don't exist in all_dims:
        - "raise": Raise an error if any dimensions don't exist (default)
        - "warn": Warn if any dimensions don't exist
        - "ignore": Silently ignore any dimensions that don't exist

    Returns
    -------
    tuple[str, ...]
        The expanded dimension permutation

    Raises
    ------
    ValueError
        If more than one ellipsis is present in dims.
        If any non-ellipsis element in dims is not a valid dimension name and validate is True.
        If missing_dims is "raise" and any dimension in dims doesn't exist in all_dims.
    """
    # Handle empty or full ellipsis case
    if dims == () or dims == (...,):
        return tuple(reversed(all_dims))

    # Check for multiple ellipses
    if dims.count(...) > 1:
        raise ValueError("an index can only have a single ellipsis ('...')")

    # Validate dimensions if requested
    if validate:
        invalid_dims = set(dims) - {..., *all_dims}
        if invalid_dims:
            if missing_dims == "raise":
                raise ValueError(
                    f"Invalid dimensions: {invalid_dims}. Available dimensions: {all_dims}"
                )
            elif missing_dims == "warn":
                warnings.warn(f"Dimensions {invalid_dims} do not exist in {all_dims}")

    # Handle missing dimensions if not raising
    if missing_dims in ("ignore", "warn"):
        dims = tuple(d for d in dims if d in all_dims or d is ...)

    # If no ellipsis, just return the dimensions
    if ... not in dims:
        return dims

    # Handle ellipsis expansion
    ellipsis_idx = dims.index(...)
    pre = list(dims[:ellipsis_idx])
    post = list(dims[ellipsis_idx + 1 :])
    middle = [d for d in all_dims if d not in pre + post]
    return tuple(pre + middle + post)


class Transpose(XOp):
    __props__ = ("dims", "missing_dims")

    def __init__(
        self,
        dims: tuple[str | Literal[...], ...],
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
    ):
        super().__init__()
        self.dims = dims
        self.missing_dims = missing_dims

    def make_node(self, x):
        x = as_xtensor(x)
        dims = expand_ellipsis(
            self.dims, x.type.dims, validate=True, missing_dims=self.missing_dims
        )

        output = xtensor(
            dtype=x.type.dtype,
            shape=tuple(x.type.shape[x.type.dims.index(d)] for d in dims),
            dims=dims,
        )
        return Apply(self, [x], [output])


def transpose(x, *dims, missing_dims: Literal["raise", "warn", "ignore"] = "raise"):
    """Transpose dimensions of the tensor.

    Parameters
    ----------
    x : XTensorVariable
        Input tensor to transpose.
    *dims : str
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
    return Transpose(dims, missing_dims=missing_dims)(x)


class Concat(XOp):
    __props__ = ("dim",)

    def __init__(self, dim: str):
        self.dim = dim
        super().__init__()

    def make_node(self, *inputs: Variable) -> Apply:
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
    return Concat(dim=dim)(*xtensors)


class ExpandDims(XOp):
    """Add a new dimension to an XTensorVariable.

    Parameters
    ----------
    dim : str or None
        The name of the new dimension. If None, the dimension will be unnamed.
    """

    def __init__(self, dim):
        self.dim = dim

    def make_node(self, x):
        x = as_xtensor(x)

        # Check if dimension already exists
        if self.dim is not None and self.dim in x.type.dims:
            raise ValueError(f"Dimension {self.dim} already exists")

        # Create new dimensions list with the new dimension
        new_dims = list(x.type.dims)
        new_dims.append(self.dim)

        # Create new shape with the new dimension
        new_shape = list(x.type.shape)
        new_shape.append(1)

        output = xtensor(
            dtype=x.type.dtype, shape=tuple(new_shape), dims=tuple(new_dims)
        )
        return Apply(self, [x], [output])


def expand_dims(x, dim: str):
    """Add a new dimension to an XTensorVariable.

    Parameters
    ----------
    x : XTensorVariable
        The input tensor
    dim : str
        The name of the new dimension

    Returns
    -------
    XTensorVariable
        A new tensor with the expanded dimension
    """
    return ExpandDims(dim=dim)(x)


class Squeeze(XOp):
    """Remove a dimension of size 1 from an XTensorVariable.

    Parameters
    ----------
    dim : str or None
        The name of the dimension to remove. If None, all dimensions of size 1 will be removed.
    """

    def __init__(self, dim=None):
        self.dim = dim

    def make_node(self, x):
        x = as_xtensor(x)

        # Get the index of the dimension to remove
        if self.dim is not None:
            if self.dim not in x.type.dims:
                raise ValueError(f"Dimension {self.dim} not found")
            dim_idx = x.type.dims.index(self.dim)
            if x.type.shape[dim_idx] != 1:
                raise ValueError(
                    f"Dimension {self.dim} has size {x.type.shape[dim_idx]}, not 1"
                )
        else:
            # Find all dimensions of size 1
            dim_idx = [i for i, s in enumerate(x.type.shape) if s == 1]
            if not dim_idx:
                raise ValueError("No dimensions of size 1 to remove")

        # Create new dimensions and shape lists
        new_dims = list(x.type.dims)
        new_shape = list(x.type.shape)
        if self.dim is not None:
            new_dims.pop(dim_idx)
            new_shape.pop(dim_idx)
        else:
            # Remove all dimensions of size 1
            new_dims = [d for i, d in enumerate(new_dims) if i not in dim_idx]
            new_shape = [s for i, s in enumerate(new_shape) if i not in dim_idx]

        output = xtensor(
            dtype=x.type.dtype, shape=tuple(new_shape), dims=tuple(new_dims)
        )
        return Apply(self, [x], [output])


def squeeze(x, dim=None):
    """Remove a dimension of size 1 from an XTensorVariable.

    Parameters
    ----------
    x : XTensorVariable
        The input tensor
    dim : str or None, optional
        The name of the dimension to remove. If None, all dimensions of size 1 will be removed.

    Returns
    -------
    XTensorVariable
        A new tensor with the specified dimension removed
    """
    return Squeeze(dim=dim)(x)
