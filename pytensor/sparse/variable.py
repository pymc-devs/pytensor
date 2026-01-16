from warnings import warn

import numpy as np
import scipy.sparse as scipy_sparse

from pytensor.sparse.basic import (
    cast,
    csm_data,
    dense_from_sparse,
    get_item_2d,
    get_item_2lists,
    get_item_list,
    get_item_scalar,
    neg,
    sp_ones_like,
    sp_zeros_like,
    transpose,
)
from pytensor.sparse.math import (
    add,
    ge,
    gt,
    le,
    lt,
    multiply,
    sp_sum,
    structured_conjugate,
    structured_dot,
    subtract,
)
from pytensor.sparse.type import SparseTensorType
from pytensor.sparse.utils import hash_from_sparse
from pytensor.tensor.shape import shape
from pytensor.tensor.type import iscalar
from pytensor.tensor.variable import (
    TensorConstant,
    TensorVariable,
    _tensor_py_operators,
)


def constant(x, name=None):
    if not isinstance(x, scipy_sparse.spmatrix):
        raise TypeError("sparse.constant must be called on a scipy.sparse.spmatrix")
    try:
        return SparseConstant(
            SparseTensorType(format=x.format, dtype=x.dtype), x.copy(), name=name
        )
    except TypeError:
        raise TypeError(f"Could not convert {x} to SparseTensorType", type(x))


def override_dense(method):
    dense_method = getattr(_tensor_py_operators, method.__name__)

    def to_dense(self, *args, **kwargs):
        self = self.toarray()
        new_args = [
            arg.toarray()
            if hasattr(arg, "type") and isinstance(arg.type, SparseTensorType)
            else arg
            for arg in args
        ]
        warn(
            f"Method {method} is not implemented for sparse variables. The variable will be converted to dense."
        )
        return dense_method(self, *new_args, **kwargs)

    to_dense._is_dense_override = True
    return to_dense


class _sparse_py_operators:
    T = property(
        lambda self: transpose(self), doc="Return aliased transpose of self (read-only)"
    )

    def astype(self, dtype):
        return cast(self, dtype)

    def __neg__(self):
        return neg(self)

    def __add__(left, right):
        return add(left, right)

    def __radd__(right, left):
        return add(left, right)

    def __sub__(left, right):
        return subtract(left, right)

    def __rsub__(right, left):
        return subtract(left, right)

    def __mul__(left, right):
        return multiply(left, right)

    def __rmul__(left, right):
        return multiply(left, right)

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

    todense = toarray

    @property
    def shape(self):
        # TODO: The plan is that the ShapeFeature in ptb.opt will do shape
        # propagation and remove the dense_from_sparse from the graph.  This
        # will *NOT* actually expand your sparse matrix just to get the shape.
        return shape(dense_from_sparse(self))

    ndim = property(lambda self: self.type.ndim)
    dtype = property(lambda self: self.type.dtype)

    # Note that the `size` attribute of sparse matrices behaves differently
    # from dense matrices: it is the number of elements stored in the matrix
    # rather than the total number of elements that may be stored. Note also
    # that stored zeros *do* count in the size.
    size = property(lambda self: csm_data(self).size)

    def zeros_like(model):
        return sp_zeros_like(model)

    def ones_like(self):
        return sp_ones_like(self)

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
        return structured_conjugate(self)

    @override_dense
    def __abs__(self):
        raise NotImplementedError

    @override_dense
    def __ceil__(self):
        raise NotImplementedError

    @override_dense
    def __floor__(self):
        raise NotImplementedError

    @override_dense
    def __trunc__(self):
        raise NotImplementedError

    @override_dense
    def transpose(self):
        raise NotImplementedError

    @override_dense
    def any(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def all(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def nonzero(self):
        raise NotImplementedError

    @override_dense
    def nonzero_values(self):
        raise NotImplementedError

    @override_dense
    def flatten(self, ndim=1):
        raise NotImplementedError

    @override_dense
    def ravel(self):
        raise NotImplementedError

    @override_dense
    def arccos(self):
        raise NotImplementedError

    @override_dense
    def arcsin(self):
        raise NotImplementedError

    @override_dense
    def arctan(self):
        raise NotImplementedError

    @override_dense
    def arccosh(self):
        raise NotImplementedError

    @override_dense
    def arcsinh(self):
        raise NotImplementedError

    @override_dense
    def arctanh(self):
        raise NotImplementedError

    @override_dense
    def ceil(self):
        raise NotImplementedError

    @override_dense
    def cos(self):
        raise NotImplementedError

    @override_dense
    def cosh(self):
        raise NotImplementedError

    @override_dense
    def deg2rad(self):
        raise NotImplementedError

    @override_dense
    def exp(self):
        raise NotImplementedError

    @override_dense
    def exp2(self):
        raise NotImplementedError

    @override_dense
    def expm1(self):
        raise NotImplementedError

    @override_dense
    def floor(self):
        raise NotImplementedError

    @override_dense
    def log(self):
        raise NotImplementedError

    @override_dense
    def log10(self):
        raise NotImplementedError

    @override_dense
    def log1p(self):
        raise NotImplementedError

    @override_dense
    def log2(self):
        raise NotImplementedError

    @override_dense
    def rad2deg(self):
        raise NotImplementedError

    @override_dense
    def sin(self):
        raise NotImplementedError

    @override_dense
    def sinh(self):
        raise NotImplementedError

    @override_dense
    def sqrt(self):
        raise NotImplementedError

    @override_dense
    def tan(self):
        raise NotImplementedError

    @override_dense
    def tanh(self):
        raise NotImplementedError

    @override_dense
    def copy(self, name=None):
        raise NotImplementedError

    @override_dense
    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        raise NotImplementedError

    @override_dense
    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        raise NotImplementedError

    @override_dense
    def var(self, axis=None, ddof=0, keepdims=False, corrected=False):
        raise NotImplementedError

    @override_dense
    def std(self, axis=None, ddof=0, keepdims=False, corrected=False):
        raise NotImplementedError

    @override_dense
    def min(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def max(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def argmin(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def argmax(self, axis=None, keepdims=False):
        raise NotImplementedError

    @override_dense
    def argsort(self, axis=-1, kind="quicksort", order=None):
        raise NotImplementedError

    @override_dense
    def round(self, mode=None):
        raise NotImplementedError

    @override_dense
    def trace(self):
        raise NotImplementedError

    @override_dense
    def cumsum(self, axis=None):
        raise NotImplementedError

    @override_dense
    def cumprod(self, axis=None):
        raise NotImplementedError

    @override_dense
    def ptp(self, axis=None):
        raise NotImplementedError

    @override_dense
    def squeeze(self, axis=None):
        raise NotImplementedError

    @override_dense
    def diagonal(self, offset=0, axis1=0, axis2=1):
        raise NotImplementedError

    @override_dense
    def __and__(self, other):
        raise NotImplementedError

    @override_dense
    def __or__(self, other):
        raise NotImplementedError

    @override_dense
    def __xor__(self, other):
        raise NotImplementedError

    @override_dense
    def __pow__(self, other):
        raise NotImplementedError

    @override_dense
    def __mod__(self, other):
        raise NotImplementedError

    @override_dense
    def __divmod__(self, other):
        raise NotImplementedError

    @override_dense
    def __truediv__(self, other):
        raise NotImplementedError

    @override_dense
    def __floordiv__(self, other):
        raise NotImplementedError

    @override_dense
    def reshape(self, shape, *, ndim=None):
        raise NotImplementedError

    @override_dense
    def dimshuffle(self, *pattern):
        raise NotImplementedError


class SparseVariable(_sparse_py_operators, TensorVariable):  # type: ignore[misc]
    format = property(lambda self: self.type.format)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.format},{self.dtype}}}"

    def __repr__(self):
        return str(self)


class SparseConstantSignature(tuple):
    def __eq__(self, other):
        (a, b), (x, y) = self, other
        return (
            a == x
            and (b.dtype == y.dtype)
            and (type(b) is type(y))
            and (b.shape == y.shape)
            and (abs(b - y).sum() < 1e-6 * b.nnz)
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        (a, b) = self
        return hash(type(self)) ^ hash(a) ^ hash(type(b))

    def pytensor_hash(self):
        (_, d) = self
        return hash_from_sparse(d)


class SparseConstant(SparseVariable, TensorConstant):  # type: ignore[misc]
    format = property(lambda self: self.type.format)

    def signature(self):
        assert self.data is not None
        return SparseConstantSignature((self.type, self.data))

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.format},{self.dtype},shape={self.data.shape},nnz={self.data.nnz}}}"

    def __repr__(self):
        return str(self)

    @property
    def unique_value(self):
        return None


SparseTensorType.variable_type = SparseVariable
SparseTensorType.constant_type = SparseConstant
