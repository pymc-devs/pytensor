"""Defines Updates object for storing a (SharedVariable, new_value) mapping."""

import logging

from pytensor.compile.sharedvalue import SharedVariable


__docformat__ = "restructuredtext en"

logger = logging.getLogger("pytensor.updates")


# Relies on the fact that dict is ordered, otherwise updates will be applied
# in a non-deterministic order.
class OrderedUpdates(dict):
    """
    Dict-like mapping from SharedVariable keys to their new values.

    This mapping supports the use of the "+" operator for the union of updates.
    """

    def __init__(self, *key, **kwargs):
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
        for key, val in dict(other).items():
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
