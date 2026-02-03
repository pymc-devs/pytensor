__docformat__ = "restructuredtext en"


from pytensor import _version


__version__: str = _version.get_versions()["version"]

del _version


from pytensor.configdefaults import config


# isort: off
from pytensor import tensor
from pytensor import sparse
from pytensor.compile import (
    In,
    Mode,
    Out,
    shared,
    wrap_py,
    function,
)
from pytensor.gradient import Lop, Rop, grad
from pytensor.printing import debugprint as dprint

from pytensor.ifelse import ifelse
from pytensor.scan.basic import scan
from pytensor.scan.views import foldl, foldr, map, reduce
from pytensor.compile.builders import OpFromGraph
from pytensor.link.jax.ops import wrap_jax
# isort: on


# Some config variables are registered by submodules. Only after all those
# imports were executed, we can warn about remaining flags provided by the user
# through PYTENSOR_FLAGS.
config.warn_unused_flags()
