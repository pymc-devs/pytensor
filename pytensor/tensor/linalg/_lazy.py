"""Lazy scipy.linalg proxy.

scipy.linalg is one of the slowest scipy submodules to import (~250ms) and is
only needed at runtime in `Op.perform` methods, never at graph-construction
time. This module exposes a proxy that defers the import until first attribute
access.
"""

from pytensor.utils import lazy_scipy_module


scipy_linalg = lazy_scipy_module("linalg")
