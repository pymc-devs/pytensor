import inspect
import sys

import numpy as np

import pytensor.scalar as ps
from pytensor import config
from pytensor.scalar import ScalarOp
from pytensor.scalar.basic import _cast_mapping
from pytensor.xtensor.basic import as_xtensor
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
