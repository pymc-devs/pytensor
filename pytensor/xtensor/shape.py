from collections.abc import Sequence

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


class Transpose(XOp):
    __props__ = ("dims",)

    def __init__(self, dims: tuple[str, ...]):
        super().__init__()
        self.dims = dims

    def make_node(self, x):
        x = as_xtensor(x)
        # Allow ellipsis for full transpose
        if self.dims == () or self.dims == (...,):
            dims = tuple(reversed(x.type.dims))
        else:
            # Expand ellipsis if present
            if ... in self.dims:
                pre = []
                post = []
                found = False
                for d in self.dims:
                    if d is ...:
                        found = True
                    elif not found:
                        pre.append(d)
                    else:
                        post.append(d)
                middle = [d for d in x.type.dims if d not in pre + post]
                dims = tuple(pre + middle + post)
            else:
                dims = self.dims
        if set(dims) != set(x.type.dims):
            raise ValueError(f"Transpose dims {dims} must match {x.type.dims}")
        output = xtensor(
            dtype=x.type.dtype,
            shape=tuple(x.type.shape[x.type.dims.index(d)] for d in dims),
            dims=dims,
        )
        return Apply(self, [x], [output])


def transpose(x, *dims):
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
