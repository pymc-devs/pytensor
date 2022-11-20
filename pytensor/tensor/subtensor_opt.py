import warnings


warnings.warn(
    "The module `pytensor.tensor.subtensor_opt` is deprecated; use `pytensor.tensor.rewriting.subtensor` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.rewriting.subtensor import *  # noqa: F401 E402 F403
