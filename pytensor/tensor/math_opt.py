import warnings


warnings.warn(
    "The module `pytensor.tensor.math_opt` is deprecated; use `pytensor.tensor.rewriting.math` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.rewriting.math import *  # noqa: F401 E402 F403
