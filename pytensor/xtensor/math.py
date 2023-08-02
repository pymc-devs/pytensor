import inspect
import sys

import pytensor.scalar as ps
from pytensor.scalar import ScalarOp
from pytensor.xtensor.basic import XElemwise


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
