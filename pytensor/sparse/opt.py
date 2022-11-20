import warnings


warnings.warn(
    "The module `pytensor.sparse.opt` is deprecated; use `pytensor.sparse.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.sparse.rewriting import *  # noqa: F401 E402 F403
