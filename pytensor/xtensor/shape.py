from collections.abc import Sequence

from pytensor.graph import Apply
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


class UnStack(XOp):
    __props__ = ("old_dim_name", "unstacked_dims", "unstacked_lengths")

    def __init__(
        self,
        old_dim_name: str,
        unstacked_dims: tuple[str, ...],
        unstacked_lengths: tuple[int, ...],
    ):
        super().__init__()
        if old_dim_name in unstacked_dims:
            raise ValueError(
                f"Dim to be unstacked {old_dim_name} can't be in {unstacked_dims}"
            )
        if len(unstacked_dims) != len(unstacked_lengths):
            raise ValueError(
                "Tuples with unstacked dim names and lengths must have the same length "
                f"but have {len(unstacked_dims)} and {len(unstacked_lengths)}"
            )
        if not unstacked_dims:
            raise ValueError("Dims to unstack into can't be empty.")
        if len(unstacked_dims) == 1:
            raise ValueError("Only one dimension to unstack into, use rename instead")
        self.old_dim_name = old_dim_name
        self.unstacked_dims = unstacked_dims
        self.unstacked_lengths = unstacked_lengths

    def make_node(self, x):
        x = as_xtensor(x)
        if self.old_dim_name not in x.type.dims:
            raise ValueError(
                f"Dim to unstack {self.old_dim_name} must be in {x.type.dims}"
            )
        if not set(self.unstacked_dims).isdisjoint(x.type.dims):
            raise ValueError(
                f"Dims to unstack into {self.unstacked_dims} must not be in {x.type.dims}"
            )
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

        output = xtensor(
            dtype=x.type.dtype,
            shape=(*batch_shape, *self.unstacked_lengths),
            dims=(*batch_dims, *self.unstacked_dims),
        )
        return Apply(self, [x], [output])


def unstack(x, dim: dict[str, dict[str, int]] | None = None, **dims: dict[str, int]):
    if dim is not None:
        if dims:
            raise ValueError(
                "Cannot use both positional dim and keyword dims in unstack"
            )
        dims = dim

    y = x
    for old_dim_name, unstacked_dict in dims.items():
        y = UnStack(
            old_dim_name, tuple(unstacked_dict.keys()), tuple(unstacked_dict.values())
        )(y)
    return y
