import typing

from pytensor.compile import (
    DeepCopyOp,
    ViewOp,
    register_deep_copy_op_c_code,
    register_view_op_c_code,
)
from pytensor.tensor import TensorType
from pytensor.tensor.math import variadic_mul


try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ModuleNotFoundError:
    XARRAY_AVAILABLE = False

from collections.abc import Sequence
from typing import TypeVar

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
        if len(set(dims)) < len(dims):
            raise ValueError(f"Dimensions must be unique. Found duplicates in {dims}: ")
        if shape is None:
            self.shape = (None,) * len(self.dims)
        else:
            self.shape = tuple(shape)
            if len(self.shape) != len(self.dims):
                raise ValueError(
                    f"Shape {self.shape} must have the same length as dims {self.dims}"
                )
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

    # DataArray-like attributes
    # https://docs.xarray.dev/en/latest/api.html#id1
    @property
    def values(self) -> TensorVariable:
        return typing.cast(TensorVariable, px.basic.tensor_from_xtensor(self))

    # Can't provide property data because that's already taken by Constants!
    # data = values

    @property
    def coords(self):
        raise NotImplementedError("coords not implemented for XTensorVariable")

    @property
    def dims(self) -> tuple[str, ...]:
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
    def shape(self) -> tuple[TensorVariable, ...]:
        return tuple(px.basic.tensor_from_xtensor(self).shape)  # type: ignore

    @property
    def size(self) -> TensorVariable:
        return typing.cast(TensorVariable, variadic_mul(*self.shape))

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
        raise NotImplementedError("Indexing not yet implemnented")


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


XTensorType.variable_type = XTensorVariable  # type: ignore
XTensorType.constant_type = XTensorConstant  # type: ignore


def xtensor_constant(x, name=None, dims: None | Sequence[str] = None):
    x_dims: tuple[str, ...]
    if isinstance(x, xr.DataArray):
        xarray_dims = x.dims
        if not all(isinstance(dim, str) for dim in xarray_dims):
            raise NotImplementedError(
                "DataArray can only be converted to xtensor_constant if all dims are of string type"
            )
        x_dims = tuple(typing.cast(typing.Iterable[str], xarray_dims))
        x_data = x.values

        if dims is not None and dims != x_dims:
            raise ValueError(
                f"xr.DataArray dims {x_dims} don't match requested specified {dims}. "
                "Use transpose or rename"
            )
    else:
        x_data = tensor_constant(x).data
        if dims is not None:
            x_dims = tuple(dims)
        else:
            if x_data.ndim == 0:
                x_dims = ()
            else:
                raise TypeError(
                    "Cannot convert TensorLike constant to XTensorConstant without specifying dims."
                )
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


register_view_op_c_code(
    XTensorType,
    # XTensorType is just TensorType under the hood
    *ViewOp.c_code_and_version[TensorType],
)

register_deep_copy_op_c_code(
    XTensorType,
    # XTensorType is just TensorType under the hood
    *DeepCopyOp.c_code_and_version[TensorType],
)
