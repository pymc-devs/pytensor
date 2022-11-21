import warnings


warnings.warn(
    "The module `pytensor.graph.unify` is deprecated; use `pytensor.graph.rewriting.unify` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.graph.rewriting.unify import *  # noqa: F401 E402 F403
