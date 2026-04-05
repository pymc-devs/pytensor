import numpy as np

from pytensor.graph.basic import Apply
from pytensor.tensor.random.type import (
    AbstractRandomGeneratorType,
    RandomGeneratorType,
    random_generator_type,
)
from pytensor.xtensor.basic import XTypeCastOp


class XRandomGeneratorType(AbstractRandomGeneratorType):
    r"""An xtensor-aware Type wrapper for `numpy.random.Generator`.

    Behaves identically to ``RandomGeneratorType`` at runtime, but lives
    in the xtensor type system so that methods on its variables return
    ``XTensorVariable`` outputs.
    """

    def __repr__(self):
        return "XRandomGeneratorType"

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def convert_variable(self, var):
        if isinstance(var.type, RandomGeneratorType):
            return rng_to_xrng(var)
        return super().convert_variable(var)


xrandom_generator_type = XRandomGeneratorType()


class RNGToXRNG(XTypeCastOp):
    """Cast a RandomGeneratorVariable to an XRandomGeneratorVariable."""

    def make_node(self, rng):
        if not isinstance(rng.type, RandomGeneratorType):
            raise TypeError(f"Expected RandomGeneratorType, got {rng.type}")
        rng.tag.used = True
        return Apply(self, [rng], [xrandom_generator_type()])


class XRNGToRNG(XTypeCastOp):
    """Cast an XRandomGeneratorVariable to a RandomGeneratorVariable."""

    def make_node(self, xrng):
        if not isinstance(xrng.type, XRandomGeneratorType):
            raise TypeError(f"Expected XRandomGeneratorType, got {xrng.type}")
        xrng.tag.used = True
        return Apply(self, [xrng], [random_generator_type()])


rng_to_xrng = RNGToXRNG()
xrng_to_rng = XRNGToRNG()


def as_rng(x):
    """Validate and cast a variable to an XRandomGeneratorVariable.

    Accepts XRandomGeneratorVariables (passthrough) and
    RandomGeneratorVariables (inserts RNGToXRNG cast).
    Raises on None and numpy Generators with informative messages.
    """
    if x is None:
        raise TypeError(
            "rng must not be None. Use xt.random.rng() for a symbolic input "
            "or xt.random.shared_rng() for a shared RNG."
        )

    if isinstance(x, np.random.Generator):
        raise TypeError(
            "as_rng does not accept numpy Generators directly. "
            "Use xt.random.shared_rng(x) to create a shared RNG variable, "
            "or xt.random.rng() for a symbolic input."
        )

    if isinstance(x.type, XRandomGeneratorType):
        return x

    if isinstance(x.type, RandomGeneratorType):
        return rng_to_xrng(x)

    raise TypeError(f"Expected an XRandomGeneratorVariable, got {type(x)}")
