import warnings

from pytensor.tensor import TensorType
from pytensor.tensor.math import variadic_mul


try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ModuleNotFoundError:
    XARRAY_AVAILABLE = False

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np

import pytensor.xtensor as px
from pytensor import _as_symbolic, config
from pytensor.graph import Apply, Constant
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.graph.type import HasDataType, HasShape, Type
from pytensor.tensor.basic import constant as tensor_constant
from pytensor.tensor.utils import hash_from_ndarray
from pytensor.tensor.variable import TensorVariable


class XTensorType(Type, HasDataType, HasShape):
    """A `Type` for Xtensors (Xarray-like tensors with dims)."""

    __props__ = ("dtype", "shape", "dims")

    def __init__(
        self,
        dtype: str | np.dtype,
        *,
        dims: Sequence[str],
        shape: Sequence[int | None] | None = None,
        name: str | None = None,
    ):
        if dtype == "floatX":
            self.dtype = config.floatX
        else:
            self.dtype = np.dtype(dtype).name

        self.dims = tuple(dims)
        if shape is None:
            self.shape = (None,) * len(self.dims)
        else:
            self.shape = tuple(shape)
        self.ndim = len(self.dims)
        self.name = name

    def clone(
        self,
        dtype=None,
        dims=None,
        shape=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = self.dtype
        if dims is None:
            dims = self.dims
        if shape is None:
            shape = self.shape
        return type(self)(dtype=dtype, shape=shape, dims=dims, **kwargs)

    def filter(self, value, strict=False, allow_downcast=None):
        # TODO implement this
        return value

    def convert_variable(self, var):
        # TODO: Implement this
        return var

    def __repr__(self):
        return f"XTensorType({self.dtype}, {self.dims}, {self.shape})"

    def __hash__(self):
        return hash((type(self), self.dtype, self.shape, self.dims))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.dims == other.dims
            and self.shape == other.shape
        )

    def is_super(self, otype):
        if type(self) is not type(otype):
            return False
        if self.dtype != otype.dtype:
            return False
        if self.dims != otype.dims:
            return False
        if any(
            s_dim_length is not None and s_dim_length != o_dim_length
            for s_dim_length, o_dim_length in zip(self.shape, otype.shape)
        ):
            return False
        return True


def xtensor(
    name: str | None = None,
    *,
    dims: Sequence[str],
    shape: Sequence[int | None] | None = None,
    dtype: str | np.dtype = "floatX",
):
    return XTensorType(dtype=dtype, dims=dims, shape=shape)(name=name)


_XTensorTypeType = TypeVar("_XTensorTypeType", bound=XTensorType)


class XTensorVariable(Variable[_XTensorTypeType, OptionalApplyType]):
    # These can't work because Python requires native output types
    def __bool__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python boolean. "
            "Call `.astype(bool)` for the symbolic equivalent."
        )

    def __index__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __int__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __float__(self):
        raise TypeError(
            "XTensorVariables cannot be converted to Python float. "
            "Call `.astype(float)` for the symbolic equivalent."
        )

    def __complex__(self):
        raise TypeError(
            "XTensorVariables cannot be converted to Python complex number. "
            "Call `.astype(complex)` for the symbolic equivalent."
        )

    # Python valid overloads
    def __abs__(self):
        return px.math.abs(self)

    def __neg__(self):
        return px.math.neg(self)

    def __lt__(self, other):
        return px.math.lt(self, other)

    def __le__(self, other):
        return px.math.le(self, other)

    def __gt__(self, other):
        return px.math.gt(self, other)

    def __ge__(self, other):
        return px.math.ge(self, other)

    def __invert__(self):
        return px.math.invert(self)

    def __and__(self, other):
        return px.math.and_(self, other)

    def __or__(self, other):
        return px.math.or_(self, other)

    def __xor__(self, other):
        return px.math.xor(self, other)

    def __rand__(self, other):
        return px.math.and_(other, self)

    def __ror__(self, other):
        return px.math.or_(other, self)

    def __rxor__(self, other):
        return px.math.xor(other, self)

    def __add__(self, other):
        return px.math.add(self, other)

    def __sub__(self, other):
        return px.math.sub(self, other)

    def __mul__(self, other):
        return px.math.mul(self, other)

    def __div__(self, other):
        return px.math.div(self, other)

    def __pow__(self, other):
        return px.math.pow(self, other)

    def __mod__(self, other):
        return px.math.mod(self, other)

    def __divmod__(self, other):
        return px.math.divmod(self, other)

    def __truediv__(self, other):
        return px.math.true_div(self, other)

    def __floordiv__(self, other):
        return px.math.floor_div(self, other)

    def __rtruediv__(self, other):
        return px.math.true_div(other, self)

    def __rfloordiv__(self, other):
        return px.math.floor_div(other, self)

    def __radd__(self, other):
        return px.math.add(other, self)

    def __rsub__(self, other):
        return px.math.sub(other, self)

    def __rmul__(self, other):
        return px.math.mul(other, self)

    def __rdiv__(self, other):
        return px.math.div_proxy(other, self)

    def __rmod__(self, other):
        return px.math.mod(other, self)

    def __rdivmod__(self, other):
        return px.math.divmod(other, self)

    def __rpow__(self, other):
        return px.math.pow(other, self)

    def __ceil__(self):
        return px.math.ceil(self)

    def __floor__(self):
        return px.math.floor(self)

    def __trunc__(self):
        return px.math.trunc(self)

    # DataArray-like attributes
    # https://docs.xarray.dev/en/latest/api.html#id1
    @property
    def values(self) -> TensorVariable:
        return px.basic.tensor_from_xtensor(self)

    # Can't provide property data because that's already taken by Constants!
    # data = values

    @property
    def coords(self):
        raise NotImplementedError("coords not implemented for XTensorVariable")

    @property
    def dims(self) -> tuple[str]:
        return self.type.dims

    @property
    def sizes(self) -> dict[str, TensorVariable]:
        return dict(zip(self.dims, self.shape))

    @property
    def as_numpy(self):
        # No-op, since the underlying data is always a numpy array
        return self

    # ndarray attributes
    # https://docs.xarray.dev/en/latest/api.html#ndarray-attributes
    @property
    def ndim(self) -> int:
        return self.type.ndim

    @property
    def shape(self) -> tuple[TensorVariable]:
        return tuple(px.basic.tensor_from_xtensor(self).shape)

    @property
    def size(self):
        return variadic_mul(*self.shape)

    @property
    def dtype(self):
        return self.type.dtype

    # DataArray contents
    # https://docs.xarray.dev/en/latest/api.html#dataarray-contents
    def rename(self, new_name_or_name_dict=None, **names):
        if isinstance(new_name_or_name_dict, str):
            new_name = new_name_or_name_dict
            name_dict = None
        else:
            new_name = None
            name_dict = new_name_or_name_dict
        new_out = px.basic.rename(self, name_dict, **names)
        new_out.name = new_name
        return new_out

    # def swap_dims(self, *args, **kwargs):
    #     ...
    #
    # def expand_dims(self, *args, **kwargs):
    #     ...
    #
    # def squeeze(self):
    #     ...

    def copy(self, name: str | None = None):
        out = px.math.identity(self)
        out.name = name
        return out

    def astype(self, dtype):
        return px.math.cast(self, dtype)

    def item(self):
        raise NotImplementedError("item not implemented for XTensorVariable")

    # Indexing
    # https://docs.xarray.dev/en/latest/api.html#id2
    def __setitem__(self, key, value):
        raise TypeError("XTensorVariable does not support item assignment.")

    @property
    def loc(self):
        raise NotImplementedError("loc not implemented for XTensorVariable")

    def sel(self, *args, **kwargs):
        raise NotImplementedError("sel not implemented for XTensorVariable")

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            return self.isel(idx)

        # Check for ellipsis not in the last position (last one is useless anyway)
        if any(idx_item is Ellipsis for idx_item in idx):
            if idx.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")
            # Convert intermediate Ellipsis to slice(None)
            ellipsis_loc = idx.index(Ellipsis)
            n_implied_none_slices = self.type.ndim - (len(idx) - 1)
            idx = (
                *idx[:ellipsis_loc],
                *((slice(None),) * n_implied_none_slices),
                *idx[ellipsis_loc + 1 :],
            )

        return px.indexing.index(self, *idx)

    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        drop: bool = False,  # Unused by PyTensor
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs,
    ):
        if indexers_kwargs:
            if indexers is not None:
                raise ValueError(
                    "Cannot pass both indexers and indexers_kwargs to isel"
                )
            indexers = indexers_kwargs

        if missing_dims not in {"raise", "warn", "ignore"}:
            raise ValueError(
                f"Unrecognized options {missing_dims} for missing_dims argument"
            )

        # Sort indices and pass them to index
        dims = self.type.dims
        indices = [slice(None)] * self.type.ndim
        for key, idx in indexers.items():
            if idx is Ellipsis:
                # Xarray raises a less informative error, suggesting indices must be integer
                # But slices are also fine
                raise TypeError("Ellipsis (...) is an invalid labeled index")
            try:
                indices[dims.index(key)] = idx
            except IndexError:
                if missing_dims == "raise":
                    raise ValueError(
                        f"Dimension {key} does not exist. Expected one of {dims}"
                    )
                elif missing_dims == "warn":
                    warnings.warn(
                        UserWarning,
                        f"Dimension {key} does not exist. Expected one of {dims}",
                    )

        return px.indexing.index(self, *indices)

    def _head_tail_or_thin(
        self,
        indexers: dict[str, Any] | int | None,
        indexers_kwargs: dict[str, Any],
        *,
        kind: Literal["head", "tail", "thin"],
    ):
        if indexers_kwargs:
            if indexers is not None:
                raise ValueError(
                    "Cannot pass both indexers and indexers_kwargs to head"
                )
            indexers = indexers_kwargs

        if indexers is None:
            if kind == "thin":
                raise TypeError(
                    "thin() indexers must be either dict-like or a single integer"
                )
            else:
                # Default to 5 for head and tail
                indexers = {dim: 5 for dim in self.type.dims}

        elif not isinstance(indexers, dict):
            indexers = {dim: indexers for dim in self.type.dims}

        if kind == "head":
            indices = {dim: slice(None, value) for dim, value in indexers.items()}
        elif kind == "tail":
            sizes = self.sizes
            # Can't use slice(-value, None), in case value is zero
            indices = {
                dim: slice(sizes[dim] - value, None) for dim, value in indexers.items()
            }
        elif kind == "thin":
            indices = {dim: slice(None, None, value) for dim, value in indexers.items()}
        return self.isel(indices)

    def head(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="head")

    def tail(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="tail")

    def thin(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="thin")

    # ndarray methods
    # https://docs.xarray.dev/en/latest/api.html#id7
    def clip(self, min, max):
        return px.math.clip(self, min, max)

    def conj(self):
        return px.math.conj(self)

    @property
    def imag(self):
        return px.math.imag(self)

    @property
    def real(self):
        return px.math.real(self)

    # Aggregation
    # https://docs.xarray.dev/en/latest/api.html#id6
    def all(self, dim):
        return px.reduction.all(self, dim)

    def any(self, dim):
        return px.reduction.any(self, dim)

    def max(self, dim):
        return px.reduction.max(self, dim)

    def min(self, dim):
        return px.reduction.min(self, dim)

    def mean(self, dim):
        return px.reduction.mean(self, dim)

    def prod(self, dim):
        return px.reduction.prod(self, dim)

    def sum(self, dim):
        return px.reduction.sum(self, dim)

    def std(self, dim):
        return px.reduction.std(self, dim)

    def var(self, dim):
        return px.reduction.var(self, dim)

    def cumsum(self, dim):
        return px.reduction.cumsum(self, dim)

    def cumprod(self, dim):
        return px.reduction.cumprod(self, dim)

    def diff(self, dim, n=1):
        """Compute the n-th discrete difference along the given dimension."""
        slice1 = {dim: slice(1, None)}
        slice2 = {dim: slice(None, -1)}
        x = self
        for _ in range(n):
            x = x[slice1] - x[slice2]
        return x


class XTensorConstantSignature(tuple):
    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        (ttype0, data0), (ttype1, data1) = self, other
        if ttype0 != ttype1 or data0.shape != data1.shape:
            return False

        # TODO: Cash sum and use it in hash like TensorConstant does
        return (data0 == data1).all()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        (ttype, data) = self
        return hash((type(self), ttype, data.shape))

    def pytensor_hash(self):
        _, data = self
        return hash_from_ndarray(data)


class XTensorConstant(XTensorVariable, Constant[_XTensorTypeType]):
    def __init__(self, type: _XTensorTypeType, data, name=None):
        # TODO: Add checks that type and data are compatible
        Constant.__init__(self, type, data, name)

    def signature(self):
        return XTensorConstantSignature((self.type, self.data))


XTensorType.variable_type = XTensorVariable
XTensorType.constant_type = XTensorConstant


def xtensor_constant(x, name=None, dims: None | Sequence[str] = None):
    if isinstance(x, xr.DataArray):
        x_dims = x.dims
        x_data = x.values

        if dims is not None and dims != x_dims:
            raise ValueError(
                f"xr.DataArray dims {x_dims} don't match requested specified {dims}. "
                "Use transpose or rename"
            )
    else:
        x_data = tensor_constant(x).data
        if dims is not None:
            x_dims = dims
        else:
            if x_data.ndim == 0:
                x_dims = ()
            else:
                "Cannot convert TensorLike constant to XTensorConstant without specifying dims."
    try:
        return XTensorConstant(
            XTensorType(dtype=x_data.dtype, dims=x_dims, shape=x_data.shape),
            x_data,
            name=name,
        )
    except TypeError:
        raise TypeError(f"Could not convert {x} to XTensorType")


if XARRAY_AVAILABLE:

    @_as_symbolic.register(xr.DataArray)
    def as_symbolic_xarray(x, **kwargs):
        return xtensor_constant(x, **kwargs)


def as_xtensor(x, name=None, dims: Sequence[str] | None = None):
    if isinstance(x, Apply):
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has to be fetched.",
                x,
            )
        else:
            x = x.outputs[0]

    if isinstance(x, Variable):
        if isinstance(x.type, XTensorType):
            return x
        if isinstance(x.type, TensorType):
            if x.type.ndim > 0 and dims is None:
                raise TypeError(
                    "non-scalar TensorVariable cannot be converted to XTensorVariable without dims."
                )
            return px.basic.xtensor_from_tensor(x, dims)
        else:
            raise TypeError(
                "Variable with type {x.type} cannot be converted to XTensorVariable."
            )
    try:
        return xtensor_constant(x, name=name, dims=dims)
    except TypeError as err:
        raise TypeError(f"Cannot convert {x} to XTensorType {type(x)}") from err
