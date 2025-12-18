"""Symbolic tensor types and constructor functions."""

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import TYPE_CHECKING, Any, NoReturn, Optional, Union

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


TensorLike = Union[Variable, Sequence[Variable], "ArrayLike"]


def as_tensor_variable(
    x: TensorLike, name: str | None = None, ndim: int | None = None, **kwargs
) -> "TensorVariable":
    """Convert `x` into an equivalent `TensorVariable`.

    This function can be used to turn ndarrays, numbers, `ScalarType` instances,
    `Apply` instances and `TensorVariable` instances into valid input list
    elements.

    See `pytensor.as_symbolic` for a more general conversion function.

    Parameters
    ----------
    x
        The object to be converted into a `Variable` type. A
        `numpy.ndarray` argument will not be copied, but a list of numbers
        will be copied to make an `numpy.ndarray`.
    name
        If a new `Variable` instance is created, it will be named with this
        string.
    ndim
        Return a `Variable` with this many dimensions.
    dtype
        The dtype to use for the resulting `Variable`.  If `x` is already
        a `Variable` type, then the dtype will not be changed.

    Raises
    ------
    TypeError
        If `x` cannot be converted to a `TensorVariable`.

    """
    return _as_tensor_variable(x, name, ndim, **kwargs)


@singledispatch
def _as_tensor_variable(
    x: TensorLike, name: str | None, ndim: int | None, **kwargs
) -> "TensorVariable":
    raise NotImplementedError(f"Cannot convert {x!r} to a tensor variable.")


def get_vector_length(v: TensorLike) -> int:
    """Return the run-time length of a symbolic vector, when possible.

    Parameters
    ----------
    v
        A rank-1 `TensorType` variable.

    Raises
    ------
    TypeError
        `v` hasn't the proper type.
    ValueError
        No special case applies, the length is not known.
        In general this is not possible, but for a number of special cases
        the length can be determined at compile / graph-construction time.
        This function implements these special cases.

    """
    v = as_tensor_variable(v)

    if v.type.ndim != 1:
        raise TypeError(f"Argument must be a vector; got {v.type}")

    static_shape: int | None = v.type.shape[0]
    if static_shape is not None:
        return static_shape

    return _get_vector_length(getattr(v.owner, "op", v), v)


@singledispatch
def _get_vector_length(op: Op | Variable, var: Variable) -> int:
    """`Op`-based dispatch for `get_vector_length`."""
    raise ValueError(f"Length of {var} cannot be determined")


@_get_vector_length.register(Constant)
def _get_vector_length_Constant(op: Op | Variable, var: Constant) -> int:
    return len(var.data)


import pytensor.tensor.exceptions
import pytensor.tensor.rewriting
from pytensor.gradient import grad, hessian, jacobian

# adds shared-variable constructors
from pytensor.tensor import (
    blas,
    blas_c,
    sharedvar,
    xlogx,
)


# isort: off
import pytensor.tensor._linalg
from pytensor.tensor import linalg
from pytensor.tensor import special
from pytensor.tensor import signal
from pytensor.tensor import optimize

# For backward compatibility
from pytensor.tensor import nlinalg
from pytensor.tensor import slinalg

# isort: on
# Allow accessing numpy constants from pytensor.tensor
from numpy import e, euler_gamma, inf, nan, newaxis, pi

from pytensor.tensor.basic import *
from pytensor.tensor.blas import batched_dot, batched_tensordot
from pytensor.tensor.extra_ops import *
from pytensor.tensor.interpolate import interp, interpolate1d
from pytensor.tensor.math import *
from pytensor.tensor.pad import pad


# isort: off
# reshape needs to be imported before shape.reshape, otherwise the tensor.reshape imports fail
from pytensor.tensor.reshape import *
from pytensor.tensor.shape import (
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    specify_broadcastable,
    specify_shape,
)

# isort: on

# We import as `_shared` instead of `shared` to avoid confusion between
# `pytensor.shared` and `tensor._shared`.
from pytensor.tensor.sort import argsort, sort
from pytensor.tensor.subtensor import *
from pytensor.tensor.type import *
from pytensor.tensor.type_other import *
from pytensor.tensor.variable import TensorConstant, TensorVariable


# isort: off
from pytensor.tensor.einsum import einsum
from pytensor.tensor.functional import vectorize
# isort: on


__all__ = ["random"]  # noqa: F405
