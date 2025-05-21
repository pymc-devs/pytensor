import warnings


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

    def __setitem__(self, key, value):
        raise TypeError(
            "XTensorVariable does not support item assignment. Use the output of `x[idx].set` or `x[idx].inc` instead."
        )

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

    def sel(self, *args, **kwargs):
        raise NotImplementedError(
            "sel not implemented for XTensorVariable, use isel instead"
        )

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
