import warnings


warnings.warn(
    "The module `pytensor.tensor.nnet.opt` is deprecated; use `pytensor.tensor.nnet.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.nnet.rewriting import *  # noqa: F401 E402 F403
