import sys


__docformat__ = "restructuredtext en"


from pytensor import _version


__version__: str = _version.get_versions()["version"]

del _version


from pytensor.configdefaults import config


# isort: off
from pytensor import tensor
from pytensor.compile import (
    In,
    Mode,
    Out,
    shared,
    wrap_py,
)
from pytensor.compile.maker import function
from pytensor.compile.mode import get_mode
from pytensor.gradient import Lop, Rop, grad, pullback, pushforward
from pytensor.printing import debugprint as dprint

from pytensor.ifelse import ifelse
from pytensor.scan.basic import scan
from pytensor.scan.views import map
from pytensor.compile.builders import OpFromGraph
from pytensor.link.jax.ops import wrap_jax
from pytensor import _sparse_lazy
# isort: on


def __getattr__(name):
    if name == "sparse":
        # During pytensor.sparse's own import, submodules may do
        # `import pytensor.sparse.X as Y` which probes pytensor.sparse via
        # getattr before the parent attribute has been set. Return the
        # partially-loaded module from sys.modules to avoid re-entry.
        if "pytensor.sparse" in sys.modules:
            return sys.modules["pytensor.sparse"]
        import pytensor.sparse as sparse

        return sparse
    raise AttributeError(f"module 'pytensor' has no attribute {name!r}")


# Some config variables are registered by submodules. Only after all those
# imports were executed, we can warn about remaining flags provided by the user
# through PYTENSOR_FLAGS.
config.warn_unused_flags()
