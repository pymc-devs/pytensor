import warnings

from pytensor.tensor import TensorVariable, mul


try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ModuleNotFoundError:
    XARRAY_AVAILABLE = False

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np

from pytensor import _as_symbolic, config
from pytensor.graph import Apply, Constant
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.graph.type import HasDataType, HasShape, Type
from pytensor.tensor.utils import hash_from_ndarray
from pytensor.utils import hash_from_code


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
        # TODO: Implement this
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

    # DataArray-like attributes
    # https://docs.xarray.dev/en/latest/api.html#id1
    @property
    def values(self) -> TensorVariable:
        from pytensor.xtensor.basic import tensor_from_xtensor

        return tensor_from_xtensor(self)

    data = values

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
        from pytensor.xtensor.basic import tensor_from_xtensor

        return tuple(tensor_from_xtensor(self).shape)

    @property
    def size(self):
        return mul(*self.shape)

    @property
    def dtype(self):
        return self.type.dtype

    # DataArray contents
    # https://docs.xarray.dev/en/latest/api.html#dataarray-contents
    def rename(self, new_name_or_name_dict, **names):
        from pytensor.xtensor.basic import rename

        if isinstance(new_name_or_name_dict, str):
            # TODO: Should we make a symbolic copy?
            self.name = new_name_or_name_dict
            name_dict = None
        else:
            name_dict = new_name_or_name_dict
        return rename(name_dict, **names)

    # def swap_dims(self, *args, **kwargs):
    #     ...
    #
    # def expand_dims(self, *args, **kwargs):
    #     ...
    #
    # def squeeze(self):
    #     ...

    def copy(self):
        from pytensor.xtensor.math import identity

        return identity(self)

    def astype(self, dtype):
        from pytensor.xtensor.math import cast

        return cast(self, dtype)

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
        from pytensor.xtensor.indexing import index

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

        return index(self, *idx)

    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        drop: bool = False,  # Unused by PyTensor
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs,
    ):
        from pytensor.xtensor.indexing import index

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

        return index(self, *indices)

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
        from pytensor.xtensor.math import clip

        return clip(self, min, max)

    def conj(self):
        from pytensor.xtensor.math import conj

        return conj(self)

    @property
    def imag(self):
        from pytensor.xtensor.math import imag

        return imag(self)

    @property
    def real(self):
        from pytensor.xtensor.math import real

        return real(self)

    # @property
    # def T(self):
    #     ...


class XTensorConstantSignature(tuple):
    def __eq__(self, other):
        if type(self) is not type(other):
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


class XTensorConstant(XTensorVariable, Constant[_XTensorTypeType]):
    def __init__(self, type: _XTensorTypeType, data, name=None):
        # TODO: Add checks that type and data are compatible
        Constant.__init__(self, type, data, name)

    def signature(self):
        assert self.data is not None
        return XTensorConstantSignature((self.type, self.data))


XTensorType.variable_type = XTensorVariable
XTensorType.constant_type = XTensorConstant


def xtensor_constant(x, name=None):
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


if XARRAY_AVAILABLE:

    @_as_symbolic.register(xr.DataArray)
    def as_symbolic_xarray(x, **kwargs):
        return xtensor_constant(x, **kwargs)


def as_xtensor_variable(x, name=None):
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
        return xtensor_constant(x, name=name)
    except TypeError as err:
        raise TypeError(f"Cannot convert {x} to XTensorType {type(x)}") from err


as_xtensor = as_xtensor_variable
