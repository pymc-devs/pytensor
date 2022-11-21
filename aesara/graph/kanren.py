import warnings


warnings.warn(
    "The module `pytensor.graph.kanren` is deprecated; use `pytensor.graph.rewriting.kanren` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.graph.rewriting.kanren import *  # noqa: F401 E402 F403
