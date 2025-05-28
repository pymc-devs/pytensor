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
    dims: tuple[str, ...], all_dims: tuple[str, ...], validate: bool = True
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

    Returns
    -------
    tuple[str, ...]
        The expanded dimension permutation

    Raises
    ------
    ValueError
        If more than one ellipsis is present in dims.
        If any non-ellipsis element in dims is not a valid dimension name and validate is True.
    """
    if dims == () or dims == (...,):
        return tuple(reversed(all_dims))

    if ... not in dims:
        if validate:
            invalid_dims = set(dims) - set(all_dims)
            if invalid_dims:
                raise ValueError(f"Invalid dimensions: {invalid_dims}. Available dimensions: {all_dims}")
        return dims

    if sum(d is ... for d in dims) > 1:
        raise ValueError("an index can only have a single ellipsis ('...')")

    pre = []
    post = []
    found = False
    for d in dims:
        if d is ...:
            found = True
        elif not found:
            pre.append(d)
        else:
            post.append(d)
    if validate:
        invalid_dims = set(pre + post) - set(all_dims)
        if invalid_dims:
            raise ValueError(f"Invalid dimensions: {invalid_dims}. Available dimensions: {all_dims}")
    middle = [d for d in all_dims if d not in pre + post]
    return tuple(pre + middle + post)


class Transpose(XOp):
    __props__ = ("dims", "missing_dims")

    def __init__(self, dims: tuple[str, ...], missing_dims: Literal["raise", "warn", "ignore"] = "raise"):
        super().__init__()
        self.dims = dims
        self.missing_dims = missing_dims

    def make_node(self, x):
        x = as_xtensor(x)
        dims = expand_ellipsis(self.dims, x.type.dims, validate=(self.missing_dims == "raise"))

        # Handle missing dimensions based on missing_dims setting
        if self.missing_dims == "ignore":
            # Filter out dimensions that don't exist in x.type.dims
            dims = tuple(d for d in dims if d in x.type.dims)
            # Add remaining dimensions in their original order
            remaining_dims = tuple(d for d in x.type.dims if d not in dims)
            dims = dims + remaining_dims
        elif self.missing_dims == "warn":
            missing = set(dims) - set(x.type.dims)
            if missing:
                warnings.warn(f"Dimensions {missing} do not exist in {x.type.dims}")
            # Filter out missing dimensions and add remaining ones
            dims = tuple(d for d in dims if d in x.type.dims)
            remaining_dims = tuple(d for d in x.type.dims if d not in dims)
            dims = dims + remaining_dims
        else:  # "raise"
            if set(dims) != set(x.type.dims):
                raise ValueError(f"Transpose dims {dims} must match {x.type.dims}")

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
