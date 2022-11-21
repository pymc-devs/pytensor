import warnings


warnings.warn(
    "The module `pytensor.tensor.opt_uncanonicalize` is deprecated; use `pytensor.tensor.rewriting.uncanonicalize` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.rewriting.uncanonicalize import *  # noqa: F401 E402 F403
