import numpy as np
import xarray as xr

import pytensor.xtensor.basic as xbasic
from pytensor import Variable, _as_symbolic
from pytensor.graph import Apply, Constant
from pytensor.tensor import TensorVariable
from pytensor.tensor.utils import hash_from_ndarray
from pytensor.tensor.variable import TensorConstant, _tensor_py_operators
from pytensor.utils import hash_from_code
from pytensor.xtensor.spaces import OrderedSpace
from pytensor.xtensor.type import XTensorType


@_as_symbolic.register(xr.DataArray)
def as_symbolic_sparse(x, **kwargs):
    return as_xtensor_variable(x, **kwargs)


def as_xtensor_variable(x, name=None, ndim=None, **kwargs):
    """
    Wrapper around SparseVariable constructor to construct
    a Variable with a sparse matrix with the same dtype and
    format.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    object
        SparseVariable version of `x`.

    """

    # TODO
    # Verify that sp is sufficiently sparse, and raise a
    # warning if it is not

    if isinstance(x, Apply):
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a "
                "multi-output Op has to be fetched.",
                x,
            )
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if not isinstance(x.type, XTensorType):
            raise TypeError(f"Variable type field must be a XTensorType, got {x.type}")
        return x
    try:
        return constant(x, name=name)
    except TypeError as err:
        raise TypeError(f"Cannot convert {x} to XTensorType {type(x)}") from err


as_xtensor = as_xtensor_variable


def constant(x, name=None):
    if not isinstance(x, xr.DataArray):
        raise TypeError("xtensor.constant must be called on a Xarray DataArray")
    try:
        return XTensorConstant(
            XTensorType(dtype=x.dtype, dims=x.dims, shape=x.shape),
            x.values.copy(),
            name=name,
        )
    except TypeError:
        raise TypeError(f"Could not convert {x} to XTensorType")


# def sp_ones_like(x):
#     """
#     Construct a sparse matrix of ones with the same sparsity pattern.
#
#     Parameters
#     ----------
#     x
#         Sparse matrix to take the sparsity pattern.
#
#     Returns
#     -------
#     A sparse matrix
#         The same as `x` with data changed for ones.
#
#     """
#     # TODO: don't restrict to CSM formats
#     data, indices, indptr, _shape = csm_properties(x)
#     return CSM(format=x.format)(at.ones_like(data), indices, indptr, _shape)
#
#
# def sp_zeros_like(x):
#     """
#     Construct a sparse matrix of zeros.
#
#     Parameters
#     ----------
#     x
#         Sparse matrix to take the shape.
#
#     Returns
#     -------
#     A sparse matrix
#         The same as `x` with zero entries for all element.
#
#     """
#
#     # TODO: don't restrict to CSM formats
#     _, _, indptr, _shape = csm_properties(x)
#     return CSM(format=x.format)(
#         data=np.array([], dtype=x.type.dtype),
#         indices=np.array([], dtype="int32"),
#         indptr=at.zeros_like(indptr),
#         shape=_shape,
#     )
#
#
# def override_dense(*methods):
#     def decorate(cls):
#         def native(method):
#             original = getattr(cls.__base__, method)
#
#             def to_dense(self, *args, **kwargs):
#                 self = self.toarray()
#                 new_args = [
#                     arg.toarray()
#                     if hasattr(arg, "type") and isinstance(arg.type, SparseTensorType)
#                     else arg
#                     for arg in args
#                 ]
#                 warn(
#                     f"Method {method} is not implemented for sparse variables. The variable will be converted to dense."
#                 )
#                 return original(self, *new_args, **kwargs)
#
#             return to_dense
#
#         for method in methods:
#             setattr(cls, method, native(method))
#         return cls
#
#     return decorate


# @override_dense(
#     "__abs__",
#     "__ceil__",
#     "__floor__",
#     "__trunc__",
#     "transpose",
#     "any",
#     "all",
#     "flatten",
#     "ravel",
#     "arccos",
#     "arcsin",
#     "arctan",
#     "arccosh",
#     "arcsinh",
#     "arctanh",
#     "ceil",
#     "cos",
#     "cosh",
#     "deg2rad",
#     "exp",
#     "exp2",
#     "expm1",
#     "floor",
#     "log",
#     "log10",
#     "log1p",
#     "log2",
#     "rad2deg",
#     "sin",
#     "sinh",
#     "sqrt",
#     "tan",
#     "tanh",
#     "copy",
#     "prod",
#     "mean",
#     "var",
#     "std",
#     "min",
#     "max",
#     "argmin",
#     "argmax",
#     "round",
#     "trace",
#     "cumsum",
#     "cumprod",
#     "ptp",
#     "squeeze",
#     "diagonal",
#     "__and__",
#     "__or__",
#     "__xor__",
#     "__pow__",
#     "__mod__",
#     "__divmod__",
#     "__truediv__",
#     "__floordiv__",
#     "reshape",
#     "dimshuffle",
# )
class _xtensor_py_operators(_tensor_py_operators):
    T = property(
        lambda self: transpose(self), doc="Return aliased transpose of self (read-only)"
    )

    def astype(self, dtype):
        return cast(self, dtype)

    def __neg__(self):
        return neg(self)

    def __add__(left, right):
        return xbasic.add(left, right)

    def __radd__(right, left):
        return add(left, right)

    def __sub__(left, right):
        return sub(left, right)

    def __rsub__(right, left):
        return sub(left, right)

    def __mul__(left, right):
        return mul(left, right)

    def __rmul__(left, right):
        return mul(left, right)

    # comparison operators

    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    def __dot__(left, right):
        return structured_dot(left, right)

    def __rdot__(right, left):
        return structured_dot(left, right)

    def sum(self, axis=None, sparse_grad=False):
        return sp_sum(self, axis=axis, sparse_grad=sparse_grad)

    dot = __dot__

    def toarray(self):
        return dense_from_sparse(self)

    @property
    def shape(self):
        # TODO: The plan is that the ShapeFeature in at.opt will do shape
        # propagation and remove the dense_from_sparse from the graph.  This
        # will *NOT* actually expand your sparse matrix just to get the shape.
        return shape(dense_from_sparse(self))

    @property
    def ndim(self) -> int:
        return self.type.ndim

    @property
    def dtype(self):
        return self.type.dtype

    # Note that the `size` attribute of sparse matrices behaves differently
    # from dense matrices: it is the number of elements stored in the matrix
    # rather than the total number of elements that may be stored. Note also
    # that stored zeros *do* count in the size.
    size = property(lambda self: csm_data(self).size)

    def zeros_like(model):
        return sp_zeros_like(model)

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)

        if len(args) == 2:
            scalar_arg_1 = (
                np.isscalar(args[0]) or getattr(args[0], "type", None) == iscalar
            )
            scalar_arg_2 = (
                np.isscalar(args[1]) or getattr(args[1], "type", None) == iscalar
            )
            if scalar_arg_1 and scalar_arg_2:
                ret = get_item_scalar(self, args)
            elif isinstance(args[0], list):
                ret = get_item_2lists(self, args[0], args[1])
            else:
                ret = get_item_2d(self, args)
        elif isinstance(args[0], list):
            ret = get_item_list(self, args[0])
        else:
            ret = get_item_2d(self, args)
        return ret

    def conj(self):
        return conjugate(self)


class XTensorVariable(_xtensor_py_operators, TensorVariable):
    pass

    # def __str__(self):
    #     return f"{self.__class__.__name__}{{{self.format},{self.dtype}}}"

    # def __repr__(self):
    #     return str(self)


class XTensorConstantSignature(tuple):
    def __eq__(self, other):
        if type(self) != type(other):
            return False

        (t0, d0), (t1, d1) = self, other
        if t0 != t1 or d0.shape != d1.shape:
            return False

        return True

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        (a, b) = self
        return hash(type(self)) ^ hash(a) ^ hash(type(b))

    def pytensor_hash(self):
        t, d = self
        return "".join([hash_from_ndarray(d)] + [hash_from_code(dim) for dim in t.dims])


class XTensorConstant(TensorConstant, _xtensor_py_operators):
    def __init__(self, type: XTensorType, data, name=None):
        # TODO: Add checks that type and data are compatible
        # Check that the type carries ordered dims
        if not isinstance(type.dims, OrderedSpace):
            raise ValueError(f"XTensor constants require ordered dims, got {type.dims}")
        Constant.__init__(self, type, data, name)

    def signature(self):
        assert self.data is not None
        return XTensorConstantSignature((self.type, self.data))


XTensorType.variable_type = XTensorVariable
XTensorType.constant_type = XTensorConstant
