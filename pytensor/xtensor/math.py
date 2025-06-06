import inspect
import sys

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


def get_all_scalar_ops():
    """
    Find all scalar operations in the pytensor.scalar module that can be wrapped with XElemwise.

    Returns:
        dict: A dictionary mapping operation names to XElemwise instances
    """
    result = {}

    # Get all module members
    for name, obj in inspect.getmembers(ps):
        # Check if the object is a scalar op (has make_node method and is not an abstract class)
        if isinstance(obj, ScalarOp):
            result[name] = XElemwise(obj)

    return result


for name, op in get_all_scalar_ops().items():
    setattr(this_module, name, op)


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


class XDot(XOp):
    """Matrix multiplication between two XTensorVariables.

    This operation performs matrix multiplication between two tensors, automatically
    aligning and contracting dimensions. The behavior matches xarray's dot operation.

    Parameters
    ----------
    dims : tuple of str
        The dimensions to contract over. If None, will contract over all matching dimensions.
    """

    __props__ = ("dims",)

    def __init__(self, dims: tuple[str, ...] | None = None):
        self.dims = dims
        super().__init__()

    def make_node(self, x, y):
        x = as_xtensor(x)
        y = as_xtensor(y)

        # Get dimensions to contract
        if self.dims is None:
            # Contract over all matching dimensions
            x_dims = set(x.type.dims)
            y_dims = set(y.type.dims)
            contract_dims = tuple(x_dims & y_dims)
        else:
            contract_dims = self.dims

        # Determine output dimensions and shapes
        x_dims = list(x.type.dims)
        y_dims = list(y.type.dims)
        x_shape = list(x.type.shape)
        y_shape = list(y.type.shape)

        # Remove contracted dimensions
        for dim in contract_dims:
            x_idx = x_dims.index(dim)
            y_idx = y_dims.index(dim)
            x_dims.pop(x_idx)
            y_dims.pop(y_idx)
            x_shape.pop(x_idx)
            y_shape.pop(y_idx)

        # Combine remaining dimensions
        out_dims = tuple(x_dims + y_dims)
        out_shape = tuple(x_shape + y_shape)

        # Determine output dtype
        out_dtype = upcast(x.type.dtype, y.type.dtype)

        out = xtensor(dtype=out_dtype, shape=out_shape, dims=out_dims)
        return Apply(self, [x, y], [out])


def dot(x, y, dims: tuple[str, ...] | None = None):
    """Matrix multiplication between two XTensorVariables.

    This operation performs matrix multiplication between two tensors, automatically
    aligning and contracting dimensions. The behavior matches xarray's dot operation.

    Parameters
    ----------
    x : XTensorVariable
        First input tensor
    y : XTensorVariable
        Second input tensor
    dims : tuple of str, optional
        The dimensions to contract over. If None, will contract over all matching dimensions.

    Returns
    -------
    XTensorVariable
        The result of the matrix multiplication.

    Examples
    --------
    >>> x = xtensor(dtype="float64", dims=("a", "b"), shape=(2, 3))
    >>> y = xtensor(dtype="float64", dims=("b", "c"), shape=(3, 4))
    >>> z = dot(x, y)  # Result has dimensions ("a", "c")
    """
    x = as_xtensor(x)
    y = as_xtensor(y)

    # Validate dimensions if specified
    if dims is not None:
        if not isinstance(dims, tuple):
            dims = tuple(dims)
        for dim in dims:
            if dim not in x.type.dims:
                raise ValueError(
                    f"Dimension {dim} not found in first input {x.type.dims}"
                )
            if dim not in y.type.dims:
                raise ValueError(
                    f"Dimension {dim} not found in second input {y.type.dims}"
                )
            # Check for compatible shapes in contracted dimensions
            x_idx = x.type.dims.index(dim)
            y_idx = y.type.dims.index(dim)
            x_size = x.type.shape[x_idx]
            y_size = y.type.shape[y_idx]
            if x_size is not None and y_size is not None and x_size != y_size:
                raise ValueError(
                    f"Dimension {dim} has incompatible shapes: {x_size} and {y_size}"
                )

    return XDot(dims=dims)(x, y)
