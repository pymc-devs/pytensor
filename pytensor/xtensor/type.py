try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ModuleNotFoundError:
    XARRAY_AVAILABLE = False

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from pytensor import _as_symbolic, config
from pytensor import scalar as aes
from pytensor.graph import Apply, Constant
from pytensor.graph.basic import Variable
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
            if np.obj2sctype(dtype) is None:
                raise TypeError(f"Invalid dtype: {dtype}")

            self.dtype = np.dtype(dtype).name

        self.dims = tuple(dims)
        if shape is None:
            self.shape = (None,) * len(self.dims)
        else:
            self.shape = tuple(shape)
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
        return type(self)(format, dtype, shape=shape, dims=dims, **kwargs)

    def filter(self, value, strict=False, allow_downcast=None):
        # TODO: Implement this
        return value

        if isinstance(value, Variable):
            raise TypeError(
                "Expected an array-like object, but found a Variable: "
                "maybe you are trying to call a function on a (possibly "
                "shared) variable instead of a numeric array?"
            )

        if (
            isinstance(value, self.format_cls[self.format])
            and value.dtype == self.dtype
        ):
            return value

        if strict:
            raise TypeError(
                f"{value} is not sparse, or not the right dtype (is {value.dtype}, "
                f"expected {self.dtype})"
            )

        # The input format could be converted here
        if allow_downcast:
            sp = self.format_cls[self.format](value, dtype=self.dtype)
        else:
            data = self.format_cls[self.format](value)
            up_dtype = aes.upcast(self.dtype, data.dtype)
            if up_dtype != self.dtype:
                raise TypeError(f"Expected {self.dtype} dtype but got {data.dtype}")
            sp = data.astype(up_dtype)

        assert sp.format == self.format

        return sp

    def convert_variable(self, var):
        # TODO: Implement this
        return var
        res = super().convert_variable(var)

        if res is None:
            return res

        if not isinstance(res.type, type(self)):
            return None

        if res.dims != self.dims:
            # TODO: Does this make sense?
            return None

        return res

    def __hash__(self):
        return hash(super().__hash__(), self.shape, self.dims)

    def __repr__(self):
        # TODO: Add `?` for unknown shapes like `TensorType` does
        return f"XTensorType({self.dtype}, {self.dims}, {self.shape})"

    def __eq__(self, other):
        res = super().__eq__(other)

        if isinstance(res, bool):
            return res and self.dims == other.dims and self.shape == other.shape

        return res

    def is_super(self, otype):
        # TODO: Implement this
        return True

        if not super().is_super(otype):
            return False

        if self.dims == otype.dims:
            return True

        return False


def xtensor(
    name: str | None = None,
    *,
    dims: Sequence[str],
    shape: Sequence[int | None] | None = None,
    dtype: str | np.dtype = "floatX",
):
    return XTensorType(dtype, dims=dims, shape=shape)(name=name)


# class _x_tensor_py_operators


class XTensorVariable(Variable):
    pass

    # def __str__(self):
    #     return f"{self.__class__.__name__}{{{self.format},{self.dtype}}}"

    # def __repr__(self):
    #     return str(self)


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


_XTensorTypeType = TypeVar("_XTensorTypeType", bound=XTensorType)


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
