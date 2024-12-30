from collections.abc import Callable
from difflib import get_close_matches
from typing import Literal, get_args

from pytensor import Variable
from pytensor.tensor.basic import as_tensor_variable, switch
from pytensor.tensor.extra_ops import searchsorted
from pytensor.tensor.functional import vectorize
from pytensor.tensor.math import clip, eq, le
from pytensor.tensor.sort import argsort


InterpolationMethod = Literal["linear", "nearest", "first", "last", "mean"]
valid_methods = get_args(InterpolationMethod)


def pad_or_return(x, idx, output, left_pad, right_pad, extrapolate):
    if extrapolate:
        return output

    n = x.shape[0]

    return switch(eq(idx, 0), left_pad, switch(eq(idx, n), right_pad, output))


def _linear_interp1d(x, y, x_hat, idx, left_pad, right_pad, extrapolate=True):
    clip_idx = clip(idx, 1, x.shape[0] - 1)

    slope = (x_hat - x[clip_idx - 1]) / (x[clip_idx] - x[clip_idx - 1])
    y_hat = y[clip_idx - 1] + slope * (y[clip_idx] - y[clip_idx - 1])

    return pad_or_return(x, idx, y_hat, left_pad, right_pad, extrapolate)


def _nearest_neighbor_interp1d(x, y, x_hat, idx, left_pad, right_pad, extrapolate=True):
    clip_idx = clip(idx, 1, x.shape[0] - 1)

    left_distance = x_hat - x[clip_idx - 1]
    right_distance = x[clip_idx] - x_hat
    y_hat = switch(le(left_distance, right_distance), y[clip_idx - 1], y[clip_idx])

    return pad_or_return(x, idx, y_hat, left_pad, right_pad, extrapolate)


def _stepwise_first_interp1d(x, y, x_hat, idx, left_pad, right_pad, extrapolate=True):
    clip_idx = clip(idx - 1, 0, x.shape[0] - 1)
    y_hat = y[clip_idx]

    return pad_or_return(x, idx, y_hat, left_pad, right_pad, extrapolate)


def _stepwise_last_interp1d(x, y, x_hat, idx, left_pad, right_pad, extrapolate=True):
    clip_idx = clip(idx, 0, x.shape[0] - 1)
    y_hat = y[clip_idx]

    return pad_or_return(x, idx, y_hat, left_pad, right_pad, extrapolate)


def _stepwise_mean_interp1d(x, y, x_hat, idx, left_pad, right_pad, extrapolate=True):
    clip_idx = clip(idx, 1, x.shape[0] - 1)
    y_hat = (y[clip_idx - 1] + y[clip_idx]) / 2

    return pad_or_return(x, idx, y_hat, left_pad, right_pad, extrapolate)


def interpolate1d(
    x: Variable,
    y: Variable,
    method: InterpolationMethod = "linear",
    left_pad: Variable | None = None,
    right_pad: Variable | None = None,
    extrapolate: bool = True,
) -> Callable[[Variable], Variable]:
    """
    Create a function to interpolate one-dimensional data.

    Parameters
    ----------
    x : TensorLike
        Input data used to create an interpolation function. Data will be sorted to be monotonically increasing.
    y: TensorLike
        Output data used to create an interpolation function. Must have the same shape as `x`.
    method : InterpolationMethod, optional
        Method for interpolation. The following methods are available:
        - 'linear': Linear interpolation
        - 'nearest': Nearest neighbor interpolation
        - 'first': Stepwise interpolation using the closest value to the left of the query point
        - 'last': Stepwise interpolation using the closest value to the right of the query point
        - 'mean': Stepwise interpolation using the mean of the two closest values to the query point
    left_pad: TensorLike, optional
        Value to return inputs `x_hat < x[0]`. Default is `y[0]`. Ignored if ``extrapolate == True``; in this
        case, values `x_hat < x[0]` will be extrapolated from the endpoints of `x` and `y`.
    right_pad: TensorLike, optional
        Value to return for inputs `x_hat > x[-1]`. Default is `y[-1]`. Ignored if ``extrapolate == True``; in this
        case, values `x_hat > x[-1]` will be extrapolated from the endpoints of `x` and `y`.
    extrapolate: bool
        Whether to extend the request interpolation function beyond the range of the input-output pairs specified in
        `x` and `y.` If False, constant values will be returned for such inputs.

    Returns
    -------
    interpolation_func: OpFromGraph
        A function that can be used to interpolate new data. The function takes a single input `x_hat` and returns
        the interpolated value `y_hat`. The input `x_hat` must be a 1d array.

    """
    x = as_tensor_variable(x)
    y = as_tensor_variable(y)

    sort_idx = argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    if left_pad is None:
        left_pad = y[0]  #  type: ignore
    else:
        left_pad = as_tensor_variable(left_pad)
    if right_pad is None:
        right_pad = y[-1]  # type: ignore
    else:
        right_pad = as_tensor_variable(right_pad)

    def _scalar_interpolate1d(x_hat):
        idx = searchsorted(x, x_hat)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Inputs must be 1d")

        if method == "linear":
            y_hat = _linear_interp1d(
                x, y, x_hat, idx, left_pad, right_pad, extrapolate=extrapolate
            )
        elif method == "nearest":
            y_hat = _nearest_neighbor_interp1d(
                x, y, x_hat, idx, left_pad, right_pad, extrapolate=extrapolate
            )
        elif method == "first":
            y_hat = _stepwise_first_interp1d(
                x, y, x_hat, idx, left_pad, right_pad, extrapolate=extrapolate
            )
        elif method == "mean":
            y_hat = _stepwise_mean_interp1d(
                x, y, x_hat, idx, left_pad, right_pad, extrapolate=extrapolate
            )
        elif method == "last":
            y_hat = _stepwise_last_interp1d(
                x, y, x_hat, idx, left_pad, right_pad, extrapolate=extrapolate
            )
        else:
            raise NotImplementedError(
                f"Unknown interpolation method: {method}. "
                f"Did you mean {get_close_matches(method, valid_methods)}?"
            )

        return y_hat

    return vectorize(_scalar_interpolate1d, signature="()->()")


def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation. Similar to ``pytensor.interpolate.interpolate1d``, but with a signature that
    matches ``np.interp``

    Parameters
    ----------
    x : TensorLike
        The x-coordinates at which to evaluate the interpolated values.

    xp : TensorLike
        The x-coordinates of the data points, must be increasing if argument `period` is not specified. Otherwise,
        `xp` is internally sorted after normalizing the periodic boundaries with ``xp = xp % period``.

    fp : TensorLike
        The y-coordinates of the data points, same length as `xp`.

    left : float, optional
        Value to return for `x < xp[0]`. Default is `fp[0]`.

    right : float, optional
        Value to return for `x > xp[-1]`. Default is `fp[-1]`.

    period : None
        Not supported. Included to ensure the signature of this function matches ``numpy.interp``.

    Returns
    -------
    y : Variable
        The interpolated values, same shape as `x`.
    """

    xp = as_tensor_variable(xp)
    fp = as_tensor_variable(fp)
    x = as_tensor_variable(x)

    f = interpolate1d(
        xp, fp, method="linear", left_pad=left, right_pad=right, extrapolate=False
    )

    return f(x)
