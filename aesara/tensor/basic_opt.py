import warnings


warnings.warn(
    "The module `pytensor.tensor.basic_opt` is deprecated; use `pytensor.tensor.rewriting.basic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.tensor.rewriting.basic import *  # noqa: F401 E402 F403
from pytensor.tensor.rewriting.elemwise import *  # noqa: F401 E402 F403
from pytensor.tensor.rewriting.extra_ops import *  # noqa: F401 E402 F403
from pytensor.tensor.rewriting.shape import *  # noqa: F401 E402 F403
