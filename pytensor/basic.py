from functools import singledispatch
from typing import Any

from pytensor.graph import Variable


def as_symbolic(x: Any, name: str | None = None, **kwargs) -> Variable:
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
def _as_symbolic(x: Any, **kwargs) -> Variable:
    from pytensor.tensor import as_tensor_variable

    return as_tensor_variable(x, **kwargs)
