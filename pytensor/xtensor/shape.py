import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from pytensor import Variable
from pytensor.graph import Apply
from pytensor.scalar import discrete_dtypes, upcast
from pytensor.tensor import as_tensor, get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.type import XTensorVariable, as_xtensor, xtensor


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

        static_unstacked_lengths = [None] * len(unstacked_lengths)
        for i, length in enumerate(unstacked_lengths):
            try:
                static_length = get_scalar_constant_value(length)
            except NotScalarConstantError:
                pass
            else:
                static_unstacked_lengths[i] = int(static_length)

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
        dims: tuple[str | Literal[...], ...],
    ):
        super().__init__()
        if dims.count(...) > 1:
            raise ValueError("an index can only have a single ellipsis ('...')")
        self.dims = dims

    def make_node(self, x):
        x = as_xtensor(x)

        transpose_dims = self.dims
        x_dims = x.type.dims

        if transpose_dims == () or transpose_dims == (...,):
            out_dims = tuple(reversed(x_dims))
        elif ... in transpose_dims:
            # Handle ellipsis expansion
            ellipsis_idx = transpose_dims.index(...)
            pre = transpose_dims[:ellipsis_idx]
            post = transpose_dims[ellipsis_idx + 1 :]
            middle = [d for d in x_dims if d not in pre + post]
            out_dims = (*pre, *middle, *post)
            if set(out_dims) != set(x_dims):
                raise ValueError(f"{out_dims} must be a permuted list of {x_dims}")
        else:
            out_dims = transpose_dims
            if set(out_dims) != set(x_dims):
                raise ValueError(
                    f"{out_dims} must be a permuted list of {x_dims}, unless `...` is included"
                )

        output = xtensor(
            dtype=x.type.dtype,
            shape=tuple(x.type.shape[x.type.dims.index(d)] for d in out_dims),
            dims=out_dims,
        )
        return Apply(self, [x], [output])


def transpose(
    x,
    *dims: str | Literal[...],
    missing_dims: Literal["raise", "warn", "ignore"] = "raise",
) -> XTensorVariable:
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
    # Validate dimensions
    x = as_xtensor(x)
    all_dims = x.type.dims
    invalid_dims = set(dims) - {..., *all_dims}
    if invalid_dims:
        if missing_dims != "ignore":
            msg = f"Dimensions {invalid_dims} do not exist. Expected one or more of: {all_dims}"
            if missing_dims == "raise":
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        # Handle missing dimensions if not raising
        dims = tuple(d for d in dims if d in all_dims or d is ...)

    return Transpose(dims)(x)


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
    """Add a new dimension to an XTensorVariable."""

    __props__ = ("dims", "size")

    def __init__(self, dim, size=1):
        self.dims = dim
        self.size = size

    def make_node(self, x):
        x = as_xtensor(x)

        if self.dims is None:
            # No-op: return same variable
            return Apply(self, [x], [x])

        # Insert new dim at front
        new_dims = (self.dims, *x.type.dims)

        # Determine shape
        if isinstance(self.size, int | np.integer):
            new_shape = (self.size, *x.type.shape)
        else:
            new_shape = (None, *x.type.shape)  # symbolic size

        out = xtensor(
            dtype=x.type.dtype,
            shape=new_shape,
            dims=new_dims,
        )
        return Apply(self, [x], [out])

    def infer_shape(self, fgraph, node, input_shapes):
        (input_shape,) = input_shapes
        if self.dims is None:
            return [input_shape]
        return [(self.size, *list(input_shape))]


def expand_dims(x, dim: str | None, size=1):
    """Add a new dimension to an XTensorVariable.

    Parameters
    ----------
    x : XTensorVariable
        Input tensor
    dim : str or None
        Name of new dimension. If None, returns x unchanged.
    size : int or symbolic, optional
        Size of the new dimension (default 1)

    Returns
    -------
    XTensorVariable
        Tensor with the new dimension inserted
    """
    x = as_xtensor(x)

    if dim is None:
        return x  # No-op

    if not isinstance(dim, str):
        raise TypeError(f"`dim` must be a string or None, got: {type(dim)}")

    if dim in x.type.dims:
        raise ValueError(f"Dimension {dim} already exists in {x.type.dims}")

    if isinstance(size, int | np.integer):
        if size <= 0:
            raise ValueError(f"size must be positive, got: {size}")
    elif not (
        hasattr(size, "ndim") and getattr(size, "ndim", None) == 0  # symbolic scalar
    ):
        raise TypeError(f"size must be an int or scalar variable, got: {type(size)}")

    return ExpandDims(dim=dim, size=size)(x)


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

    def __init__(self, dim):
        self.dims = tuple(sorted(set(dim)))

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


def squeeze(x, dim=None):
    """Remove dimensions of size 1 from an XTensorVariable.

    Parameters
    ----------
    x : XTensorVariable
        The input tensor
    dim : str or None or iterable of str, optional
        The name(s) of the dimension(s) to remove. If None, all dimensions of size 1
        (known statically) will be removed. Dimensions with symbolic shape will be retained.

    Returns
    -------
    XTensorVariable
        A new tensor with the specified dimension(s) removed.
    """
    x = as_xtensor(x)

    if dim is None:
        dims = tuple(d for d, s in zip(x.type.dims, x.type.shape) if s == 1)
    elif isinstance(dim, str):
        dims = (dim,)
    else:
        dims = tuple(dim)

    if not dims:
        return x  # no-op if nothing to squeeze

    return Squeeze(dim=dims)(x)
