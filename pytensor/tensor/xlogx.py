import warnings


def xlogx(x):
    """Compute x * log(x), returning 0 when x = 0.

    .. deprecated::
        Use ``pytensor.tensor.special.xlogy(x, x)`` instead.

    """
    warnings.warn(
        "pytensor.tensor.xlogx.xlogx is deprecated. "
        "Use pytensor.tensor.special.xlogy(x, x) instead.",
        FutureWarning,
        stacklevel=2,
    )
    from pytensor.tensor.special import xlogy

    return xlogy(x, x)


def xlogy0(x, y):
    """Compute x * log(y), returning 0 when x = 0.

    .. deprecated::
        Use ``pytensor.tensor.special.xlogy(x, y)`` instead.

    """
    warnings.warn(
        "pytensor.tensor.xlogx.xlogy0 is deprecated. "
        "Use pytensor.tensor.special.xlogy(x, y) instead.",
        FutureWarning,
        stacklevel=2,
    )
    from pytensor.tensor.special import xlogy

    return xlogy(x, y)
