"""Defines Updates object for storing a (SharedVariable, new_value) mapping."""

import logging
import warnings
from collections import OrderedDict

from pytensor.compile.sharedvalue import SharedVariable


__docformat__ = "restructuredtext en"

logger = logging.getLogger("pytensor.updates")


# Must be an OrderedDict or updates will be applied in a non-deterministic
# order.
class OrderedUpdates(OrderedDict):
    """
    Dict-like mapping from SharedVariable keys to their new values.

    This mapping supports the use of the "+" operator for the union of updates.
    """

    def __init__(self, *key, **kwargs):
        if (
            len(key) >= 1
            and isinstance(key[0], dict)
            and len(key[0]) > 1
            and not isinstance(key[0], OrderedDict)
        ):
            # Warn when using as input a non-ordered dictionary.
            warnings.warn(
                "Initializing an `OrderedUpdates` from a "
                "non-ordered dictionary with 2+ elements could "
                "make your code non-deterministic. You can use "
                "an OrderedDict that is available at "
                "collections.OrderedDict for python 2.6+."
            )
        super().__init__(*key, **kwargs)
        for key in self:
            if not isinstance(key, SharedVariable):
                raise TypeError(
                    "OrderedUpdates keys must inherit from SharedVariable", key
                )

    def __setitem__(self, key, value):
        if isinstance(key, SharedVariable):
            # TODO: consider doing error-checking on value.
            # insist that it is an PyTensor variable? Have the right type?
            # This could have weird consequences - for example a

            return super().__setitem__(key, value)
        else:
            raise TypeError("OrderedUpdates keys must inherit from SharedVariable", key)

    def update(self, other=None):
        if other is None:
            return
        if (
            isinstance(other, dict)
            and len(other) > 1
            and not isinstance(other, OrderedDict)
        ):
            # Warn about non-determinism.
            warnings.warn(
                "Updating an `OrderedUpdates` with a "
                "non-ordered dictionary with 2+ elements could "
                "make your code non-deterministic",
                stacklevel=2,
            )
        for key, val in OrderedDict(other).items():
            if key in self:
                if self[key] == val:
                    continue
                raise KeyError("Collision", key)
            self[key] = val  # __setitem__ does type-checking

    def __add__(self, other):
        rval = OrderedUpdates()
        rval.update(self)
        rval.update(other)
        return rval

    def __radd__(other, self):
        rval = OrderedUpdates()
        rval.update(other)
        rval.update(self)
        return rval
