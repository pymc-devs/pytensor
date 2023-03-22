"""
PyTensor is an optimizing compiler in Python, built to evaluate
complicated expressions (especially matrix-valued ones) as quickly as
possible.  PyTensor compiles expression graphs (see :doc:`graph` ) that
are built by Python code. The expressions in these graphs are called
`Apply` nodes and the variables in these graphs are called `Variable`
nodes.

You compile a graph by calling `function`, which takes a graph, and
returns a callable object.  One of pytensor's most important features is
that `function` can transform your graph before compiling it.  It can
replace simple expressions with faster or more numerically stable
implementations.

To learn more, check out:

- Op List (:doc:`oplist`)

"""


__docformat__ = "restructuredtext en"

# Set a default logger. It is important to do this before importing some other
# pytensor code, since this code may want to log some messages.
import logging
import os
import sys
from functools import singledispatch
from typing import Any, NoReturn, Optional

from pytensor.version import version as __version__


pytensor_logger = logging.getLogger("pytensor")
logging_default_handler = logging.StreamHandler()
logging_default_formatter = logging.Formatter(
    fmt="%(levelname)s (%(name)s): %(message)s"
)
logging_default_handler.setFormatter(logging_default_formatter)
pytensor_logger.setLevel(logging.WARNING)

if not pytensor_logger.hasHandlers():
    pytensor_logger.addHandler(logging_default_handler)


# Disable default log handler added to pytensor_logger when the module
# is imported.
def disable_log_handler(logger=pytensor_logger, handler=logging_default_handler):
    if logger.hasHandlers():
        logger.removeHandler(handler)


# Raise a meaningful warning/error if the pytensor directory is in the Python
# path.
rpath = os.path.realpath(__path__[0])
for p in sys.path:
    if os.path.realpath(p) != rpath:
        continue
    raise RuntimeError("You have the pytensor directory in your Python path.")

from pytensor.configdefaults import config


# This is the api version for ops that generate C code.  External ops
# might need manual changes if this number goes up.  An undefined
# __api_version__ can be understood to mean api version 0.
#
# This number is not tied to the release version and should change
# very rarely.
__api_version__ = 1

# isort: off
from pytensor.graph.basic import Variable
from pytensor.graph.replace import clone_replace, graph_replace

# isort: on


def as_symbolic(x: Any, name: Optional[str] = None, **kwargs) -> Variable:
    """Convert `x` into an equivalent PyTensor `Variable`.

    Parameters
    ----------
    x
        The object to be converted into a ``Variable`` type. A
        ``numpy.ndarray`` argument will not be copied, but a list of numbers
        will be copied to make an ``numpy.ndarray``.
    name
        If a new ``Variable`` instance is created, it will be named with this
        string.
    kwargs
        Options passed to the appropriate sub-dispatch functions.  For example,
        `ndim` and `dtype` can be passed when `x` is an `numpy.ndarray` or
        `Number` type.

    Raises
    ------
    TypeError
        If `x` cannot be converted to a `Variable`.

    """
    if isinstance(x, Variable):
        return x

    res = _as_symbolic(x, **kwargs)
    res.name = name
    return res


@singledispatch
def _as_symbolic(x, **kwargs) -> Variable:
    from pytensor.tensor import as_tensor_variable

    return as_tensor_variable(x, **kwargs)


# isort: off
from pytensor import scalar, tensor
from pytensor.compile import (
    In,
    Mode,
    Out,
    ProfileStats,
    predefined_linkers,
    predefined_modes,
    predefined_optimizers,
    shared,
)
from pytensor.compile.function import function, function_dump
from pytensor.compile.function.types import FunctionMaker
from pytensor.gradient import Lop, Rop, grad, subgraph_grad
from pytensor.printing import debugprint as dprint
from pytensor.printing import pp, pprint
from pytensor.updates import OrderedUpdates

# isort: on


def get_underlying_scalar_constant(v):
    """Return the constant scalar (i.e. 0-D) value underlying variable `v`.

    If `v` is the output of dim-shuffles, fills, allocs, cast, etc.
    this function digs through them.

    If ``pytensor.sparse`` is also there, we will look over CSM `Op`.

    If `v` is not some view of constant data, then raise a
    `NotScalarConstantError`.
    """
    # Is it necessary to test for presence of pytensor.sparse at runtime?
    sparse = globals().get("sparse")
    if sparse and isinstance(v.type, sparse.SparseTensorType):
        if v.owner is not None and isinstance(v.owner.op, sparse.CSM):
            data = v.owner.inputs[0]
            return tensor.get_underlying_scalar_constant_value(data)
    return tensor.get_underlying_scalar_constant_value(v)


# isort: off
import pytensor.tensor.random.var
import pytensor.sparse
from pytensor.scan import checkpoints
from pytensor.scan.basic import scan
from pytensor.scan.views import foldl, foldr, map, reduce

# isort: on


# Some config variables are registered by submodules. Only after all those
# imports were executed, we can warn about remaining flags provided by the user
# through PYTENSOR_FLAGS.
config.warn_unused_flags()
