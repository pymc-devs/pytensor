import copy
from typing import TypeAlias

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.tensor.random.variable import (
    UNSET,
    RandomGeneratorSharedVariable,
    warn_reuse,
)
from pytensor.xtensor.random.type import (
    XRandomGeneratorType,
    xrandom_generator_type,
)
from pytensor.xtensor.type import XTensorVariable


XRNG_AND_DRAW: TypeAlias = tuple["XRandomGeneratorVariable", XTensorVariable]

__all__ = [
    "XRandomGeneratorSharedVariable",
    "XRandomGeneratorVariable",
    "rng",
    "shared_rng",
]


def _make_xrng_method(fn):
    """Create a method on XRandomGeneratorVariable that wraps an xtensor.random.basic function."""
    import inspect

    @warn_reuse
    def method(self, *args, **kwargs):
        return fn(*args, rng=self, return_next_rng=True, **kwargs)

    fn_name = getattr(fn, "__name__", None) or str(fn)
    method.__name__ = fn_name
    method.__qualname__ = f"XRandomGeneratorVariable.{fn_name}"
    method.__doc__ = fn.__doc__

    try:
        sig = inspect.signature(fn)
        filtered = [
            p
            for name, p in sig.parameters.items()
            if name not in ("rng", "return_next_rng", "kwargs")
        ]
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        method.__signature__ = sig.replace(parameters=[self_param, *filtered])
    except (ValueError, TypeError):
        pass

    return method


class _xrandom_generator_py_operators:
    """Mixin providing distribution methods that return (XRandomGeneratorVariable, XTensorVariable)."""

    pass


def _populate_xrng_methods():
    import pytensor.xtensor.random.basic as pxrb

    for name in pxrb.__all__:
        fn = getattr(pxrb, name)
        if callable(fn):
            setattr(_xrandom_generator_py_operators, name, _make_xrng_method(fn))


_populate_xrng_methods()


class XRandomGeneratorVariable(
    _xrandom_generator_py_operators,
    Variable[XRandomGeneratorType, OptionalApplyType],
):
    """The Variable type used for xtensor random number generator states."""


XRandomGeneratorType.variable_type = XRandomGeneratorVariable  # type: ignore[assignment]


def rng(name=None) -> XRandomGeneratorVariable:
    """Create a symbolic xtensor random number generator variable.

    This creates a root variable with no data attached,
    suitable for use as a function input. When compiling a function,
    use ``pytensor.In(rng, mutable=True)`` to allow in-place RNG updates.

    Parameters
    ----------
    name : str, optional
        Name for the variable.

    Returns
    -------
    XRandomGeneratorVariable
        A symbolic xtensor RNG variable.

    Examples
    --------
    >>> import numpy as np
    >>> import pytensor
    >>> import pytensor.xtensor.random as pxr

    >>> rng = pxr.rng("rng")
    >>> next_rng, x = rng.normal(0, 1, extra_dims={"a": 3})
    >>> fn = pytensor.function([pytensor.In(rng, mutable=True)], [next_rng, x])
    >>> rng_val = np.random.default_rng(153)
    >>> rng_val, draw = fn(rng_val)
    >>> draw.shape
    (3,)
    """
    return xrandom_generator_type(name=name)  # type: ignore[return-value]


class XRandomGeneratorSharedVariable(SharedVariable, XRandomGeneratorVariable):
    def __str__(self):
        return self.name or f"XRNG({self.container!r})"

    def set_value(self, new_value=None, *, seed=UNSET, borrow=False):
        new_value, borrow = RandomGeneratorSharedVariable._resolve_rng_value(
            new_value, seed=seed, borrow=borrow
        )
        super().set_value(new_value, borrow=borrow)


def shared_rng(
    value=None, *, seed=UNSET, name=None, borrow=False
) -> XRandomGeneratorSharedVariable:
    """Create a shared xtensor random number generator variable.

    The RNG state is stored internally and can be updated across function
    calls via the ``updates`` parameter of ``pytensor.function``.

    Parameters
    ----------
    value : numpy.random.Generator, optional
        The initial RNG state. If None, a new ``numpy.random.default_rng(seed)`` is used.
    seed : int, optional
        Seed for the default RNG. Only used when ``value`` is None.
    name : str, optional
        Name for the shared variable.
    borrow : bool
        If True, use the provided value directly without copying.

    Returns
    -------
    XRandomGeneratorSharedVariable

    Examples
    --------
    >>> import numpy as np
    >>> import pytensor
    >>> import pytensor.xtensor.random as pxr

    Create from an existing Generator:

    >>> rng = pxr.shared_rng(np.random.default_rng(153), name="rng")
    >>> next_rng, x = rng.normal(0, 1)
    >>> fn = pytensor.function([], x, updates={rng: next_rng})
    >>> fn()
    array(1.45769255)

    Or create from a seed directly:

    >>> rng2 = pxr.shared_rng(seed=153, name="rng2")
    >>> next_rng2, x2 = rng2.normal(0, 1)
    >>> fn2 = pytensor.function([], x2, updates={rng2: next_rng2})
    >>> fn2()
    array(1.45769255)

    Use ``set_value`` to reset the RNG state:

    >>> rng.set_value(np.random.default_rng(153))
    >>> fn()
    array(1.45769255)
    >>> rng.set_value(seed=153)
    >>> fn()
    array(1.45769255)
    """
    if value is None:
        if seed is UNSET:
            raise ValueError("Must set one of value or seed")
        if not borrow:
            seed = copy.deepcopy(seed)
        value = np.random.default_rng(seed)
    else:
        if seed is not UNSET:
            raise ValueError("Cannot specify both value and seed")
        if not isinstance(value, np.random.Generator):
            raise TypeError(f"Expected numpy.random.Generator, got {type(value)}")
        if not borrow:
            value = copy.deepcopy(value)

    return XRandomGeneratorSharedVariable(
        type=xrandom_generator_type,
        value=value,
        strict=False,
        allow_downcast=None,
        name=name,
    )
