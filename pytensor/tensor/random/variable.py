import copy
import warnings
from functools import wraps
from typing import TypeAlias

import numpy as np

import pytensor.tensor.random.basic as ptrb
from pytensor.compile.sharedvalue import SharedVariable, shared_constructor
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.tensor.random.type import RandomGeneratorType, random_generator_type
from pytensor.tensor.variable import TensorVariable


RNG_AND_DRAW: TypeAlias = tuple["RandomGeneratorVariable", TensorVariable]


def warn_reuse(func):
    # TODO: Extend this with a compile-time FunctionGraph pass that
    # detects RNG fan-out (multiple consumers of the same RNG variable).
    # That approach is purely structural, can't be bypassed, and has
    # full graph context for better error messages.
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self.tag, "used", False):
            warnings.warn(
                f"RandomGeneratorVariable {self} has already been used. "
                "You probably want to use the new RandomGeneratorVariable that was returned when you used it.",
                UserWarning,
            )
        self.tag.used = True
        return func(self, *args, **kwargs)

    return wrapper


def _make_rng_method(fn):
    """Create a method on RandomGeneratorVariable that wraps a basic.py function."""
    import inspect

    @warn_reuse
    def method(self, *args, **kwargs):
        return fn(*args, rng=self, return_next_rng=True, **kwargs)

    fn_name = getattr(fn, "__name__", None) or getattr(fn, "name", str(fn))
    method.__name__ = fn_name
    method.__qualname__ = f"RandomGeneratorVariable.{fn_name}"
    method.__doc__ = fn.__doc__

    # Copy the signature, removing rng/return_next_rng and adding self
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


class _random_generator_py_operators:
    pass


for _name in ptrb.__all__:  # type: ignore[has-type]
    _fn = getattr(ptrb, _name)
    if callable(_fn):
        setattr(_random_generator_py_operators, _name, _make_rng_method(_fn))


class RandomGeneratorVariable(
    _random_generator_py_operators,
    Variable[RandomGeneratorType, OptionalApplyType],
):
    """The Variable type used for random number generator states."""


RandomGeneratorType.variable_type = RandomGeneratorVariable  # type: ignore[assignment]


def rng(name=None) -> RandomGeneratorVariable:
    """Create a symbolic random number generator variable.

    This creates a root variable with no data attached,
    suitable for use as a function input. When compiling a function,
    use ``pytensor.In(rng, mutable=True)`` to allow in-place RNG updates.

    Parameters
    ----------
    name : str, optional
        Name for the variable.

    Returns
    -------
    RandomGeneratorVariable
        A symbolic random number generator variable.

    Examples
    --------
    >>> import numpy as np
    >>> import pytensor
    >>> import pytensor.tensor.random as ptr

    >>> rng = ptr.rng("rng")
    >>> next_rng, x = rng.normal()
    >>> fn = pytensor.function([pytensor.In(rng, mutable=True)], [next_rng, x])
    >>> rng_val = np.random.default_rng(153)
    >>> rng_val, draw = fn(rng_val)
    >>> draw
    array(1.45769255)
    >>> rng_val, draw = fn(rng_val)
    >>> draw
    array(0.44383835)
    """
    return random_generator_type(name=name)  # type: ignore[return-value]


UNSET = object()


class RandomGeneratorSharedVariable(SharedVariable, RandomGeneratorVariable):
    def __str__(self):
        return self.name or f"RNG({self.container!r})"

    @staticmethod
    def _resolve_rng_value(new_value, *, seed=UNSET, borrow=False):
        """Resolve new_value/seed into a (Generator, borrow) pair."""
        if new_value is None:
            if seed is UNSET:
                raise ValueError("Must set one of value or seed")
            if not borrow:
                seed = copy.deepcopy(seed)
                borrow = True
            new_value = np.random.default_rng(seed)
        elif seed is not UNSET:
            raise ValueError("Cannot specify both new_value and seed")
        return new_value, borrow

    def set_value(self, new_value=None, *, seed=UNSET, borrow=False):
        """Set the value of this shared Random Generator Variable

        Parameters
        ----------
        value : numpy.random.Generator, optional
            The initial RNG state. If None, a new ``numpy.random.default_rng(seed)`` is used.
        seed : optional
            Seed for the default RNG. Only used when ``value`` is None. Must define one of value or seed.
        borrow : bool
            If True, the shared variable will use the provided value directly without copying.

        Changes to this value will be visible to all functions using this SharedVariable.
        """
        new_value, borrow = self._resolve_rng_value(new_value, seed=seed, borrow=borrow)
        super().set_value(new_value, borrow=borrow)


def shared_rng(
    value=None, *, seed=UNSET, name=None, borrow=False
) -> RandomGeneratorSharedVariable:
    """Create a shared random number generator variable.

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
        If True, the shared variable will use the provided value directly
        without copying.

    Returns
    -------
    RandomGeneratorSharedVariable

    Examples
    --------

    >>> import numpy as np
    >>> import pytensor
    >>> import pytensor.tensor.random as ptr

    Create from an existing Generator:

    >>> rng = ptr.shared_rng(np.random.default_rng(153), name="rng")
    >>> next_rng, x = rng.normal()
    >>> fn = pytensor.function([], x, updates={rng: next_rng})
    >>> fn()
    array(1.45769255)

    Or create from a seed directly:

    >>> rng2 = ptr.shared_rng(seed=153, name="rng2")
    >>> next_rng2, x2 = rng2.normal()
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

    return RandomGeneratorSharedVariable(
        type=random_generator_type,
        value=value,
        strict=False,
        allow_downcast=None,
        name=name,
    )


@shared_constructor.register(np.random.RandomState)
@shared_constructor.register(np.random.Generator)
def randomgen_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False
):
    r"""`SharedVariable` constructor for NumPy's `Generator` and/or `RandomState`."""
    if isinstance(value, np.random.RandomState):
        raise TypeError(
            "`np.RandomState` is no longer supported in PyTensor. Use `np.random.Generator` instead."
        )

    return shared_rng(value, name=name, borrow=borrow)
