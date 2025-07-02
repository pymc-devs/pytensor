import sys
from collections.abc import Iterable
from types import EllipsisType

import numpy as np

import pytensor.scalar as ps
from pytensor import config
from pytensor.graph.basic import Apply
from pytensor.scalar import ScalarOp
from pytensor.scalar.basic import _cast_mapping, upcast
from pytensor.xtensor.basic import XOp, as_xtensor
from pytensor.xtensor.type import xtensor
from pytensor.xtensor.vectorization import XElemwise


this_module = sys.modules[__name__]


def _as_xelemwise(core_op: ScalarOp) -> XElemwise:
    out = XElemwise(core_op)
    out.__doc__ = f"Ufunc version of {core_op} for XTensorVariables"
    return out


abs = _as_xelemwise(ps.abs)
add = _as_xelemwise(ps.add)
logical_and = bitwise_and = and_ = _as_xelemwise(ps.and_)
angle = _as_xelemwise(ps.angle)
arccos = _as_xelemwise(ps.arccos)
arccosh = _as_xelemwise(ps.arccosh)
arcsin = _as_xelemwise(ps.arcsin)
arcsinh = _as_xelemwise(ps.arcsinh)
arctan = _as_xelemwise(ps.arctan)
arctan2 = _as_xelemwise(ps.arctan2)
arctanh = _as_xelemwise(ps.arctanh)
betainc = _as_xelemwise(ps.betainc)
betaincinv = _as_xelemwise(ps.betaincinv)
ceil = _as_xelemwise(ps.ceil)
clip = _as_xelemwise(ps.clip)
complex = _as_xelemwise(ps.complex)
conjugate = conj = _as_xelemwise(ps.conj)
cos = _as_xelemwise(ps.cos)
cosh = _as_xelemwise(ps.cosh)
deg2rad = _as_xelemwise(ps.deg2rad)
equal = eq = _as_xelemwise(ps.eq)
erf = _as_xelemwise(ps.erf)
erfc = _as_xelemwise(ps.erfc)
erfcinv = _as_xelemwise(ps.erfcinv)
erfcx = _as_xelemwise(ps.erfcx)
erfinv = _as_xelemwise(ps.erfinv)
exp = _as_xelemwise(ps.exp)
exp2 = _as_xelemwise(ps.exp2)
expm1 = _as_xelemwise(ps.expm1)
floor = _as_xelemwise(ps.floor)
floor_divide = floor_div = int_div = _as_xelemwise(ps.int_div)
gamma = _as_xelemwise(ps.gamma)
gammainc = _as_xelemwise(ps.gammainc)
gammaincc = _as_xelemwise(ps.gammaincc)
gammainccinv = _as_xelemwise(ps.gammainccinv)
gammaincinv = _as_xelemwise(ps.gammaincinv)
gammal = _as_xelemwise(ps.gammal)
gammaln = _as_xelemwise(ps.gammaln)
gammau = _as_xelemwise(ps.gammau)
greater_equal = ge = _as_xelemwise(ps.ge)
greater = gt = _as_xelemwise(ps.gt)
hyp2f1 = _as_xelemwise(ps.hyp2f1)
i0 = _as_xelemwise(ps.i0)
i1 = _as_xelemwise(ps.i1)
identity = _as_xelemwise(ps.identity)
imag = _as_xelemwise(ps.imag)
logical_not = bitwise_invert = bitwise_not = invert = _as_xelemwise(ps.invert)
isinf = _as_xelemwise(ps.isinf)
isnan = _as_xelemwise(ps.isnan)
iv = _as_xelemwise(ps.iv)
ive = _as_xelemwise(ps.ive)
j0 = _as_xelemwise(ps.j0)
j1 = _as_xelemwise(ps.j1)
jv = _as_xelemwise(ps.jv)
kve = _as_xelemwise(ps.kve)
less_equal = le = _as_xelemwise(ps.le)
log = _as_xelemwise(ps.log)
log10 = _as_xelemwise(ps.log10)
log1mexp = _as_xelemwise(ps.log1mexp)
log1p = _as_xelemwise(ps.log1p)
log2 = _as_xelemwise(ps.log2)
less = lt = _as_xelemwise(ps.lt)
mod = _as_xelemwise(ps.mod)
multiply = mul = _as_xelemwise(ps.mul)
negative = neg = _as_xelemwise(ps.neg)
not_equal = neq = _as_xelemwise(ps.neq)
logical_or = bitwise_or = or_ = _as_xelemwise(ps.or_)
owens_t = _as_xelemwise(ps.owens_t)
polygamma = _as_xelemwise(ps.polygamma)
power = pow = _as_xelemwise(ps.pow)
psi = _as_xelemwise(ps.psi)
rad2deg = _as_xelemwise(ps.rad2deg)
real = _as_xelemwise(ps.real)
reciprocal = _as_xelemwise(ps.reciprocal)
round = _as_xelemwise(ps.round_half_to_even)
maximum = _as_xelemwise(ps.scalar_maximum)
minimum = _as_xelemwise(ps.scalar_minimum)
second = _as_xelemwise(ps.second)
sigmoid = expit = _as_xelemwise(ps.sigmoid)
sign = _as_xelemwise(ps.sign)
sin = _as_xelemwise(ps.sin)
sinh = _as_xelemwise(ps.sinh)
softplus = _as_xelemwise(ps.softplus)
square = sqr = _as_xelemwise(ps.sqr)
sqrt = _as_xelemwise(ps.sqrt)
subtract = sub = _as_xelemwise(ps.sub)
where = switch = _as_xelemwise(ps.switch)
tan = _as_xelemwise(ps.tan)
tanh = _as_xelemwise(ps.tanh)
tri_gamma = _as_xelemwise(ps.tri_gamma)
true_divide = true_div = _as_xelemwise(ps.true_div)
trunc = _as_xelemwise(ps.trunc)
logical_xor = bitwise_xor = xor = _as_xelemwise(ps.xor)

_xelemwise_cast_op: dict[str, XElemwise] = {}


def cast(x, dtype):
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
    exp_x = exp(x)
    return exp_x / exp_x.sum(dim=dim)


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


def dot(x, y, dim: str | Iterable[str] | EllipsisType | None = None):
    """Matrix multiplication between two XTensorVariables.

    This operation performs matrix multiplication between two tensors, automatically
    aligning and contracting dimensions. The behavior matches xarray's dot operation.

    Parameters
    ----------
    x : XTensorVariable
        First input tensor
    y : XTensorVariable
        Second input tensor
    dim : str, Iterable[Hashable], EllipsisType, or None, optional
        The dimensions to contract over. If None, will contract over all matching dimensions.
        If Ellipsis (...), will contract over all dimensions.

    Returns
    -------
    XTensorVariable
        The result of the matrix multiplication.

    Examples
    --------
    >>> x = xtensor(dtype="float64", dims=("a", "b"), shape=(2, 3))
    >>> y = xtensor(dtype="float64", dims=("b", "c"), shape=(3, 4))
    >>> z = dot(x, y)  # Result has dimensions ("a", "c")
    >>> z = dot(x, y, dim=...)  # Contract over all dimensions
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
