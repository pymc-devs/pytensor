import warnings


from pytensor.tensor.variable import *  # noqa

warnings.warn(
    "The module 'pytensor.tensor.var' has been deprecated. "
    "Use 'pytensor.tensor.variable' instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
