import warnings


warnings.warn(
    "The module `pytensor.scalar.basic_scipy` is deprecated "
    "and has been renamed to `pytensor.scalar.math`",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.scalar.math import *
