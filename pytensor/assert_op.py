import warnings


warnings.warn(
    "The module `pytensor.assert_op` is deprecated "
    "and its `Op`s have been moved to `pytensor.raise_op`",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.raise_op import Assert, assert_op  # noqa: F401 E402
