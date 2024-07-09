"""
Utility classes and methods to pickle parts of symbolic graph.

These pickled graphs can be used, for instance, as cases for
unit tests or regression tests.
"""

import pickle
import sys

import pytensor


__docformat__ = "restructuredtext en"
__authors__ = "Pascal Lamblin PyMC Developers PyTensor Developers "
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"


min_recursion = 3000
if sys.getrecursionlimit() < min_recursion:
    sys.setrecursionlimit(min_recursion)

Pickler = pickle.Pickler


class StripPickler(Pickler):
    """Subclass of `Pickler` that strips unnecessary attributes from PyTensor objects.

    Example
    -------

    ..code-block:: python

        fn_args = {
            "inputs": inputs,
            "outputs": outputs,
            "updates": updates,
        }
        dest_pkl = "my_test.pkl"
        with Path(dest_pkl).open("wb") as f:
            strip_pickler = StripPickler(f, protocol=-1)
            strip_pickler.dump(fn_args)
    """

    def __init__(self, file, protocol: int = 0, extra_tag_to_remove: str | None = None):
        # Can't use super as Pickler isn't a new style class
        super().__init__(file, protocol)
        self.tag_to_remove = ["trace", "test_value"]
        if extra_tag_to_remove:
            self.tag_to_remove.extend(extra_tag_to_remove)

    def save(self, obj):
        # Remove the tag.trace attribute from Variable and Apply nodes
        if isinstance(obj, pytensor.graph.utils.Scratchpad):
            for tag in self.tag_to_remove:
                if hasattr(obj, tag):
                    del obj.__dict__[tag]
        # Remove manually-added docstring of Elemwise ops
        elif isinstance(obj, pytensor.tensor.elemwise.Elemwise):
            if "__doc__" in obj.__dict__:
                del obj.__dict__["__doc__"]

        return Pickler.save(self, obj)
