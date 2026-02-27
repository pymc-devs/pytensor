import copy
import warnings
from functools import wraps
from typing import TypeAlias

import numpy as np

from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable, shared_constructor
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.tensor.random.basic import normal
from pytensor.tensor.random.type import RandomGeneratorType, random_generator_type
from pytensor.tensor.variable import TensorVariable


RNG_AND_DRAW: TypeAlias = tuple["RandomGeneratorVariable", TensorVariable]


def warn_reuse(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self.tag, "used", False) and config.warn_rng.reuse:
            warnings.warn(
                f"RandomGeneratorVariable {self} has already been used. "
                "You probably want to use the new RandomGeneratorVariable that was returned when you used it.",
                UserWarning,
            )
        self.tag.used = True
        return func(self, *args, **kwargs)

    return wrapper


class _random_generator_py_operators:
    @warn_reuse
    def normal(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return normal(loc, scale, size=size, rng=self, return_next_rng=True)


class RandomGeneratorVariable(
    _random_generator_py_operators,
    Variable[RandomGeneratorType, OptionalApplyType],
):
    """The Variable type used for random number generator states."""


RandomGeneratorType.variable_type = RandomGeneratorVariable


def rng(name=None) -> RandomGeneratorVariable:
    """Create a new default random number generator variable.

    Returns
    -------
    RandomGeneratorVariable
        A new random number generator variable initialized with the default
        numpy random generator.
    """

    return random_generator_type(name=name)


class RandomGeneratorSharedVariable(SharedVariable, RandomGeneratorVariable):
    def __str__(self):
        return self.name or f"RNG({self.container!r})"


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

    rng_sv_type = RandomGeneratorSharedVariable
    rng_type = random_generator_type

    if not borrow:
        value = copy.deepcopy(value)

    return rng_sv_type(
        type=rng_type,
        value=value,
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )
