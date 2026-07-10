from typing import TypeVar

import numpy as np
from numpy.random import Generator

import pytensor
from pytensor.graph.type import Type


T = TypeVar("T")


class RandomType(Type[T]):
    r"""A Type wrapper for `numpy.random.Generator."""

    is_backend_divergent = True


class AbstractRandomGeneratorType(RandomType[Generator]):
    r"""Abstract base for random generator types wrapping `numpy.random.Generator`.

    Provides shared ``filter``, ``values_eq``, and ``may_share_memory``
    implementations.  Concrete subclasses (``RandomGeneratorType``,
    ``XRandomGeneratorType``) add their own ``__eq__``/``__hash__``.
    """

    @staticmethod
    def may_share_memory(a: Generator, b: Generator):
        return a._bit_generator is b._bit_generator  # type: ignore[attr-defined]

    def filter(self, data, strict=False, allow_downcast=None):
        """
        XXX: This doesn't convert `data` to the same type of underlying RNG type
        as `self`.  It really only checks that `data` is of the appropriate type
        to be a valid `RandomGeneratorType`.

        In other words, it serves as a `Type.is_valid_value` implementation,
        but, because the default `Type.is_valid_value` depends on
        `Type.filter`, we need to have it here to avoid surprising circular
        dependencies in sub-classes.
        """
        if isinstance(data, Generator):
            return data

        raise TypeError()

    @staticmethod
    def values_eq(a, b):
        sa = a.bit_generator.state
        sb = b.bit_generator.state

        def _eq(sa, sb):
            for key in sa:
                if isinstance(sa[key], dict):
                    if not _eq(sa[key], sb[key]):
                        return False
                elif isinstance(sa[key], np.ndarray):
                    if not np.array_equal(sa[key], sb[key]):
                        return False
                else:
                    if sa[key] != sb[key]:
                        return False

            return True

        return _eq(sa, sb)


class RandomGeneratorType(AbstractRandomGeneratorType):
    r"""A Type wrapper for `numpy.random.Generator`.

    The reason this exists (and `Generic` doesn't suffice) is that
    `Generator` objects that would appear to be equal do not compare equal
    with the ``==`` operator.

    """

    def __repr__(self):
        return "RandomGeneratorType"

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def convert_variable(self, var):
        from pytensor.xtensor.random.type import XRandomGeneratorType, xrng_to_rng

        if isinstance(var.type, XRandomGeneratorType):
            return xrng_to_rng(var)
        return super().convert_variable(var)


# Register C code for `ViewOp` on the abstract base so all subclasses inherit it.
pytensor.compile.ops.register_view_op_c_code(
    AbstractRandomGeneratorType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)

random_generator_type = RandomGeneratorType()


def as_rng(x):
    """Validate and cast a variable to a RandomGeneratorVariable.

    Accepts RandomGeneratorVariables (passthrough) and
    XRandomGeneratorVariables (inserts XRNGToRNG cast).
    Raises on None and numpy Generators with informative messages.
    """
    if x is None:
        raise TypeError(
            "rng must not be None. Use pt.random.rng() for a symbolic input "
            "or pt.random.shared_rng() for a shared RNG."
        )

    if isinstance(x, np.random.Generator):
        raise TypeError(
            "as_rng does not accept numpy Generators directly. "
            "Use pt.random.shared_rng(x) to create a shared RNG variable, "
            "or pt.random.rng() for a symbolic input."
        )

    if isinstance(x.type, RandomGeneratorType):
        return x

    from pytensor.xtensor.random.type import XRandomGeneratorType, xrng_to_rng

    if isinstance(x.type, XRandomGeneratorType):
        return xrng_to_rng(x)

    raise TypeError(f"Expected a RandomGeneratorVariable, got {type(x)}")
