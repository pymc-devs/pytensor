import warnings


warnings.warn(
    "The module `pytensor.tensor.random.opt` is deprecated; use `pytensor.tensor.random.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.random.rewriting import *  # noqa: F401 E402 F403
