import sys
from collections.abc import Iterable, Sequence
from types import EllipsisType

import numpy as np

import pytensor.scalar as ps
from pytensor import config
from pytensor.graph.basic import Apply
from pytensor.scalar.basic import _cast_mapping, upcast
from pytensor.xtensor.basic import XOp, as_xtensor
from pytensor.xtensor.type import xtensor
from pytensor.xtensor.vectorization import XElemwise


this_module = sys.modules[__name__]


def _as_xelemwise(core_op):
    x_op = XElemwise(core_op)

    def decorator(func):
        def wrapper(*args, **kwargs):
            return x_op(*args, **kwargs)

        wrapper.__doc__ = f"Ufunc version of {core_op} for XTensorVariables"
        return wrapper

    return decorator


@_as_xelemwise(ps.abs)
def abs(): ...


@_as_xelemwise(ps.add)
def add(): ...


@_as_xelemwise(ps.and_)
def logical_and(): ...


@_as_xelemwise(ps.and_)
def bitwise_and(): ...


and_ = logical_and


@_as_xelemwise(ps.angle)
def angle(): ...


@_as_xelemwise(ps.arccos)
def arccos(): ...


@_as_xelemwise(ps.arccosh)
def arccosh(): ...


@_as_xelemwise(ps.arcsin)
def arcsin(): ...


@_as_xelemwise(ps.arcsinh)
def arcsinh(): ...


@_as_xelemwise(ps.arctan)
def arctan(): ...


@_as_xelemwise(ps.arctan2)
def arctan2(): ...


@_as_xelemwise(ps.arctanh)
def arctanh(): ...


@_as_xelemwise(ps.betainc)
def betainc(): ...


@_as_xelemwise(ps.betaincinv)
def betaincinv(): ...


@_as_xelemwise(ps.ceil)
def ceil(): ...


@_as_xelemwise(ps.clip)
def clip(): ...


@_as_xelemwise(ps.complex)
def complex(): ...


@_as_xelemwise(ps.conj)
def conjugate(): ...


conj = conjugate


@_as_xelemwise(ps.cos)
def cos(): ...


@_as_xelemwise(ps.cosh)
def cosh(): ...


@_as_xelemwise(ps.deg2rad)
def deg2rad(): ...


@_as_xelemwise(ps.eq)
def equal(): ...


eq = equal


@_as_xelemwise(ps.erf)
def erf(): ...


@_as_xelemwise(ps.erfc)
def erfc(): ...


@_as_xelemwise(ps.erfcinv)
def erfcinv(): ...


@_as_xelemwise(ps.erfcx)
def erfcx(): ...


@_as_xelemwise(ps.erfinv)
def erfinv(): ...


@_as_xelemwise(ps.exp)
def exp(): ...


@_as_xelemwise(ps.exp2)
def exp2(): ...


@_as_xelemwise(ps.expm1)
def expm1(): ...


@_as_xelemwise(ps.floor)
def floor(): ...


@_as_xelemwise(ps.int_div)
def floor_divide(): ...


floor_div = int_div = floor_divide


@_as_xelemwise(ps.gamma)
def gamma(): ...


@_as_xelemwise(ps.gammainc)
def gammainc(): ...


@_as_xelemwise(ps.gammaincc)
def gammaincc(): ...


@_as_xelemwise(ps.gammainccinv)
def gammainccinv(): ...


@_as_xelemwise(ps.gammaincinv)
def gammaincinv(): ...


@_as_xelemwise(ps.gammal)
def gammal(): ...


@_as_xelemwise(ps.gammaln)
def gammaln(): ...


@_as_xelemwise(ps.gammau)
def gammau(): ...


@_as_xelemwise(ps.ge)
def greater_equal(): ...


ge = greater_equal


@_as_xelemwise(ps.gt)
def greater(): ...


gt = greater


@_as_xelemwise(ps.hyp2f1)
def hyp2f1(): ...


@_as_xelemwise(ps.i0)
def i0(): ...


@_as_xelemwise(ps.i1)
def i1(): ...


@_as_xelemwise(ps.identity)
def identity(): ...


@_as_xelemwise(ps.imag)
def imag(): ...


@_as_xelemwise(ps.invert)
def logical_not(): ...


@_as_xelemwise(ps.invert)
def bitwise_not(): ...


@_as_xelemwise(ps.invert)
def bitwise_invert(): ...


@_as_xelemwise(ps.invert)
def invert(): ...


@_as_xelemwise(ps.isinf)
def isinf(): ...


@_as_xelemwise(ps.isnan)
def isnan(): ...


@_as_xelemwise(ps.iv)
def iv(): ...


@_as_xelemwise(ps.ive)
def ive(): ...


@_as_xelemwise(ps.j0)
def j0(): ...


@_as_xelemwise(ps.j1)
def j1(): ...


@_as_xelemwise(ps.jv)
def jv(): ...


@_as_xelemwise(ps.kve)
def kve(): ...


@_as_xelemwise(ps.le)
def less_equal(): ...


le = less_equal


@_as_xelemwise(ps.log)
def log(): ...


@_as_xelemwise(ps.log10)
def log10(): ...


@_as_xelemwise(ps.log1mexp)
def log1mexp(): ...


@_as_xelemwise(ps.log1p)
def log1p(): ...


@_as_xelemwise(ps.log2)
def log2(): ...


@_as_xelemwise(ps.lt)
def less(): ...


lt = less


@_as_xelemwise(ps.mod)
def mod(): ...


@_as_xelemwise(ps.mul)
def multiply(): ...


mul = multiply


@_as_xelemwise(ps.neg)
def negative(): ...


neg = negative


@_as_xelemwise(ps.neq)
def not_equal(): ...


neq = not_equal


@_as_xelemwise(ps.or_)
def logical_or(): ...


@_as_xelemwise(ps.or_)
def bitwise_or(): ...


or_ = logical_or


@_as_xelemwise(ps.owens_t)
def owens_t(): ...


@_as_xelemwise(ps.polygamma)
def polygamma(): ...


@_as_xelemwise(ps.pow)
def power(): ...


pow = power


@_as_xelemwise(ps.psi)
def psi(): ...


@_as_xelemwise(ps.rad2deg)
def rad2deg(): ...


@_as_xelemwise(ps.real)
def real(): ...


@_as_xelemwise(ps.reciprocal)
def reciprocal(): ...


@_as_xelemwise(ps.round_half_to_even)
def round(): ...


@_as_xelemwise(ps.maximum)
def maximum(): ...


@_as_xelemwise(ps.minimum)
def minimum(): ...


@_as_xelemwise(ps.second)
def second(): ...


@_as_xelemwise(ps.sigmoid)
def sigmoid(): ...


expit = sigmoid


@_as_xelemwise(ps.sign)
def sign(): ...


@_as_xelemwise(ps.sin)
def sin(): ...


@_as_xelemwise(ps.sinh)
def sinh(): ...


@_as_xelemwise(ps.softplus)
def softplus(): ...


@_as_xelemwise(ps.sqr)
def square(): ...


sqr = square


@_as_xelemwise(ps.sqrt)
def sqrt(): ...


@_as_xelemwise(ps.sub)
def subtract(): ...


sub = subtract


@_as_xelemwise(ps.switch)
def where(): ...


switch = where


@_as_xelemwise(ps.tan)
def tan(): ...


@_as_xelemwise(ps.tanh)
def tanh(): ...


@_as_xelemwise(ps.tri_gamma)
def tri_gamma(): ...


@_as_xelemwise(ps.true_div)
def true_divide(): ...


true_div = true_divide


@_as_xelemwise(ps.trunc)
def trunc(): ...


@_as_xelemwise(ps.xor)
def logical_xor(): ...


@_as_xelemwise(ps.xor)
def bitwise_xor(): ...


xor = logical_xor


_xelemwise_cast_op: dict[str, XElemwise] = {}


def cast(x, dtype):
    """Cast an XTensorVariable to a different dtype."""
    if dtype == "floatX":
        dtype = config.floatX
    else:
        dtype = np.dtype(dtype).name

    x = as_xtensor(x)
    if x.type.dtype == dtype:
        return x
    if x.type.dtype.startswith("complex") and not dtype.startswith("complex"):
        raise TypeError(
            "Casting from complex to real is ambiguous: consider"
            " real(), imag(), angle() or abs()"
        )

    if dtype not in _xelemwise_cast_op:
        _xelemwise_cast_op[dtype] = XElemwise(scalar_op=_cast_mapping[dtype])
    return _xelemwise_cast_op[dtype](x)


def softmax(x, dim=None):
    """Compute the softmax of an XTensorVariable along a specified dimension."""
    exp_x = exp(x)
    return exp_x / exp_x.sum(dim=dim)


def logsumexp(x, dim=None):
    """Compute the logsumexp of an XTensorVariable along a specified dimension."""
    return log(exp(x).sum(dim=dim))


class Dot(XOp):
    """Matrix multiplication between two XTensorVariables.

    This operation performs matrix multiplication between two tensors, automatically
    aligning and contracting dimensions. The behavior matches xarray's dot operation.

    Parameters
    ----------
    dims : tuple of str
        The dimensions to contract over. If None, will contract over all matching dimensions.
    """

    __props__ = ("dims",)

    def __init__(self, dims: Iterable[str]):
        self.dims = dims
        super().__init__()

    def make_node(self, x, y):
        x = as_xtensor(x)
        y = as_xtensor(y)

        x_shape_dict = dict(zip(x.type.dims, x.type.shape))
        y_shape_dict = dict(zip(y.type.dims, y.type.shape))

        # Check for dimension size mismatches (concrete only)
        for dim in self.dims:
            x_shape = x_shape_dict.get(dim, None)
            y_shape = y_shape_dict.get(dim, None)
            if (
                isinstance(x_shape, int)
                and isinstance(y_shape, int)
                and x_shape != y_shape
            ):
                raise ValueError(f"Size of dim '{dim}' does not match")

        # Determine output dimensions
        shape_dict = {**x_shape_dict, **y_shape_dict}
        out_dims = tuple(d for d in shape_dict if d not in self.dims)

        # Determine output shape
        out_shape = tuple(shape_dict[d] for d in out_dims)

        # Determine output dtype
        out_dtype = upcast(x.type.dtype, y.type.dtype)

        out = xtensor(dtype=out_dtype, shape=out_shape, dims=out_dims)
        return Apply(self, [x, y], [out])


def dot(x, y, dim: str | Sequence[str] | EllipsisType | None = None):
    """Generalized dot product for XTensorVariables.

    This operation performs multiplication followed by summation for shared dimensions
    or simply summation for non-shared dimensions.

    Parameters
    ----------
    x : XTensorVariable
        First input tensor
    y : XTensorVariable
        Second input tensor
    dim : str, Sequence[str], Ellipsis (...), or None, optional
        The dimensions to contract over. If None, will contract over all matching dimensions.
        If Ellipsis (...), will contract over all dimensions.

    Returns
    -------
    XTensorVariable


    Examples
    --------

    .. testcode::

        from pytensor.xtensor import xtensor, dot

        x = xtensor("x", dims=("a", "b"))
        y = xtensor("y", dims=("b", "c"))

        assert dot(x, y).dims == ("a", "c")  # Contract over shared `b` dimension
        assert dot(x, y, dim=("a", "b")).dims == ("c",)  # Contract over 'a' and 'b'
        assert dot(x, y, dim=...).dims == ()  # Contract over all dimensions

    """
    x = as_xtensor(x)
    y = as_xtensor(y)

    x_dims = set(x.type.dims)
    y_dims = set(y.type.dims)
    intersection = x_dims & y_dims
    union = x_dims | y_dims

    # Canonicalize dims
    if dim is None:
        dim_set = intersection
    elif dim is ...:
        dim_set = union
    elif isinstance(dim, str):
        dim_set = {dim}
    elif isinstance(dim, Iterable):
        dim_set = set(dim)

    # Validate provided dims
    # Check if any dimension is not found in either input
    for d in dim_set:
        if d not in union:
            raise ValueError(f"Dimension {d} not found in either input")

    result = Dot(dims=tuple(dim_set))(x, y)

    return result
