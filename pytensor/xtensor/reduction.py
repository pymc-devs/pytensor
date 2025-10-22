import typing
from collections.abc import Sequence
from functools import partial
from types import EllipsisType

import pytensor.scalar as ps
from pytensor.graph.basic import Apply
from pytensor.tensor.math import variadic_mul
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.math import neq, sqrt
from pytensor.xtensor.math import sqr as square
from pytensor.xtensor.type import as_xtensor, xtensor


REDUCE_DIM = str | Sequence[str] | EllipsisType | None


class XReduce(XOp):
    __slots__ = ("binary_op", "dims")

    def __init__(self, binary_op, dims: Sequence[str]):
        super().__init__()
        self.binary_op = binary_op
        # Order of reduce dims doesn't change the behavior of the Op
        self.dims = tuple(sorted(dims))

    def make_node(self, x):
        x = as_xtensor(x)
        x_dims = x.type.dims
        x_dims_set = set(x_dims)
        reduce_dims_set = set(self.dims)
        if x_dims_set == reduce_dims_set:
            out_dims, out_shape = [], []
        else:
            if not reduce_dims_set.issubset(x_dims_set):
                raise ValueError(
                    f"Reduced dims {self.dims} not found in array dimensions {x_dims}."
                )
            out_dims, out_shape = zip(
                *[
                    (d, s)
                    for d, s in zip(x_dims, x.type.shape)
                    if d not in reduce_dims_set
                ]
            )
        output = xtensor(dtype=x.type.dtype, shape=out_shape, dims=out_dims)
        return Apply(self, [x], [output])


def _process_user_dims(x, dim: REDUCE_DIM) -> Sequence[str]:
    if isinstance(dim, str):
        return (dim,)
    elif dim is None or dim is Ellipsis:
        x = as_xtensor(x)
        return typing.cast(tuple[str], x.type.dims)
    return dim


def reduce(x, dim: REDUCE_DIM = None, *, binary_op):
    dims = _process_user_dims(x, dim)
    return XReduce(binary_op=binary_op, dims=dims)(x)


sum = partial(reduce, binary_op=ps.add)
prod = partial(reduce, binary_op=ps.mul)
max = partial(reduce, binary_op=ps.maximum)
min = partial(reduce, binary_op=ps.minimum)


def bool_reduce(x, dim: REDUCE_DIM = None, *, binary_op):
    x = as_xtensor(x)
    if x.type.dtype != "bool":
        x = neq(x, 0)
    return reduce(x, dim=dim, binary_op=binary_op)


all = partial(bool_reduce, binary_op=ps.and_)
any = partial(bool_reduce, binary_op=ps.or_)


def _infer_reduced_size(original_var, reduced_var):
    reduced_dims = reduced_var.dims
    return variadic_mul(
        *[size for dim, size in original_var.sizes.items() if dim not in reduced_dims]
    )


def mean(x, dim: REDUCE_DIM):
    x = as_xtensor(x)
    sum_x = sum(x, dim)
    n = _infer_reduced_size(x, sum_x)
    return sum_x / n


def var(x, dim: REDUCE_DIM, *, ddof: int = 0):
    x = as_xtensor(x)
    x_mean = mean(x, dim)
    n = _infer_reduced_size(x, x_mean)
    return square(x - x_mean).sum(dim) / (n - ddof)


def std(x, dim: REDUCE_DIM, *, ddof: int = 0):
    return sqrt(var(x, dim, ddof=ddof))


class XCumReduce(XOp):
    __props__ = ("binary_op", "dims")

    def __init__(self, binary_op, dims: Sequence[str]):
        self.binary_op = binary_op
        self.dims = tuple(sorted(dims))  # Order doesn't matter

    def make_node(self, x):
        x = as_xtensor(x)
        out = x.type()
        return Apply(self, [x], [out])


def cumreduce(x, dim: REDUCE_DIM, *, binary_op):
    dims = _process_user_dims(x, dim)
    return XCumReduce(dims=dims, binary_op=binary_op)(x)


cumsum = partial(cumreduce, binary_op=ps.add)
cumprod = partial(cumreduce, binary_op=ps.mul)
