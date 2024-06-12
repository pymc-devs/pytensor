from typing import Iterable, Optional, Union, Sequence, TypeVar

import numpy as np

import pytensor
from pytensor import scalar as aes
from pytensor.graph.basic import Variable
from pytensor.graph.type import HasDataType
from pytensor.tensor.type import TensorType


_XTensorTypeType = TypeVar("_XTensorTypeType", bound=TensorType)


class XTensorType(TensorType, HasDataType):
    """A `Type` for sparse tensors.

    Notes
    -----
    Currently, sparse tensors can only be matrices (i.e. have two dimensions).

    """

    __props__ = ("dtype", "shape", "dims")

    def __init__(
        self,
        dtype: Union[str, np.dtype],
        *,
        dims: Sequence[str],
        shape: Optional[Iterable[Optional[Union[bool, int]]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(dtype, shape=shape, name=name)
        if not isinstance(dims, (list, tuple)):
            raise TypeError("dims must be a list or tuple")
        dims = tuple(dims)
        self.dims = dims

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
        return super().__hash__() ^ hash(self.dims)

    def __repr__(self):
        # TODO: Add `?` for unknown shapes like `TensorType` does
        return f"XTensorType({self.dtype}, {self.dims}, {self.shape})"

    def __eq__(self, other):
        res = super().__eq__(other)

        if isinstance(res, bool):
            return res and other.dims == self.dims

        return res

    def is_super(self, otype):
        # TODO: Implement this
        return True

        if not super().is_super(otype):
            return False

        if self.dims == otype.dims:
            return True

        return False


# TODO: Implement creater helper xtensor

pytensor.compile.register_view_op_c_code(
    XTensorType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)
