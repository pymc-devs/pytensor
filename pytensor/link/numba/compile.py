import warnings
from collections.abc import Callable
from functools import singledispatch

import numba
import numpy as np
from numba import NumbaWarning
from numba import njit as _njit
from numba.core.extending import register_jitable

from pytensor import config
from pytensor.graph import Apply, FunctionGraph, Type
from pytensor.scalar import ScalarType
from pytensor.sparse import SparseTensorType
from pytensor.tensor import TensorType


def numba_njit(*args, fastmath=None, final_function: bool = False, **kwargs):
    if fastmath is None:
        if config.numba__fastmath:
            # Opinionated default on fastmath flags
            # https://llvm.org/docs/LangRef.html#fast-math-flags
            fastmath = {
                "arcp",  # Allow Reciprocal
                "contract",  # Allow floating-point contraction
                "afn",  # Approximate functions
                "reassoc",
                "nsz",  # no-signed zeros
            }
        else:
            fastmath = False

    if final_function:
        kwargs.setdefault("cache", True)
    else:
        kwargs.setdefault("no_cpython_wrapper", True)
        kwargs.setdefault("no_cfunc_wrapper", True)

    # Suppress cache warning for internal functions
    # We have to add an ansi escape code for optional bold text by numba
    warnings.filterwarnings(
        "ignore",
        message=(
            "(\x1b\\[1m)*"  # ansi escape code for bold text
            "Cannot cache compiled function "
            '"(numba_funcified_fgraph|store_core_outputs|cholesky|solve|solve_triangular|cho_solve|lu_factor)" '
            "as it uses dynamic globals"
        ),
        category=NumbaWarning,
    )

    func = _njit if final_function else register_jitable
    if len(args) > 0 and callable(args[0]):
        return func(*args[1:], fastmath=fastmath, **kwargs)(args[0])
    else:
        return func(*args, fastmath=fastmath, **kwargs)


@singledispatch
def numba_funcify(
    typ, node=None, storage_map=None, **kwargs
) -> Callable | tuple[Callable, str | int | None]:
    """Generate a numba function for a given op and apply node (or Fgraph).

    The resulting function will usually use the `no_cpython_wrapper`
    argument in numba, so it can not be called directly from python,
    but only from other jit functions.
    """
    raise NotImplementedError(f"Numba funcify not implemented for type {typ}")


def get_numba_type(
    pytensor_type: Type,
    layout: str = "A",
    force_scalar: bool = False,
    reduce_to_scalar: bool = False,
) -> numba.types.Type:
    r"""Create a Numba type object for a :class:`Type`.

    Parameters
    ----------
    pytensor_type
        The :class:`Type` to convert.
    layout
        The :class:`numpy.ndarray` layout to use.
    force_scalar
        Ignore dimension information and return the corresponding Numba scalar types.
    reduce_to_scalar
        Return Numba scalars for zero dimensional :class:`TensorType`\s.
    """

    if isinstance(pytensor_type, TensorType):
        dtype = pytensor_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if force_scalar or (
            reduce_to_scalar and getattr(pytensor_type, "ndim", None) == 0
        ):
            return numba_dtype
        return numba.types.Array(numba_dtype, pytensor_type.ndim, layout)
    elif isinstance(pytensor_type, ScalarType):
        dtype = np.dtype(pytensor_type.dtype)
        numba_dtype = numba.from_dtype(dtype)
        return numba_dtype
    elif isinstance(pytensor_type, SparseTensorType):
        from pytensor.link.numba.dispatch.sparse import CSCMatrixType, CSRMatrixType

        dtype = pytensor_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if pytensor_type.format == "csr":
            return CSRMatrixType(numba_dtype)
        if pytensor_type.format == "csc":
            return CSCMatrixType(numba_dtype)

        raise NotImplementedError()
    else:
        raise NotImplementedError(f"Numba type not implemented for {pytensor_type}")


def create_numba_signature(
    node_or_fgraph: FunctionGraph | Apply,
    force_scalar: bool = False,
    reduce_to_scalar: bool = False,
) -> numba.types.Type:
    """Create a Numba type for the signature of an `Apply` node or `FunctionGraph`."""
    input_types = [
        get_numba_type(
            inp.type, force_scalar=force_scalar, reduce_to_scalar=reduce_to_scalar
        )
        for inp in node_or_fgraph.inputs
    ]

    output_types = [
        get_numba_type(
            out.type, force_scalar=force_scalar, reduce_to_scalar=reduce_to_scalar
        )
        for out in node_or_fgraph.outputs
    ]

    if len(output_types) > 1:
        return numba.types.Tuple(output_types)(*input_types)
    elif len(output_types) == 1:
        return output_types[0](*input_types)
    else:
        return numba.types.void(*input_types)


def create_tuple_creator(f, n):
    """Construct a compile-time ``tuple``-comprehension-like loop.

    See https://github.com/numba/numba/issues/2771#issuecomment-414358902
    """
    assert n > 0

    f = numba_njit(f)

    @numba_njit
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba_njit
        def creator(args, creator=creator, i=i):
            return (*creator(args), f(i, *args))

    return numba_njit(lambda *args: creator(args))


def create_tuple_string(x):
    args = ", ".join(x + ([""] if len(x) == 1 else []))
    return f"({args})"


def create_arg_string(x):
    args = ", ".join(x)
    return args
