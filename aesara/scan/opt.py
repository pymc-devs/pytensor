import warnings


warnings.warn(
    "The module `pytensor.scan.opt` is deprecated; use `pytensor.scan.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.scan.rewriting import *  # noqa: F401 E402 F403
