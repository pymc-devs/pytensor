import numpy as np
import pytest
from numpy.testing import assert_allclose

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.interpolate import (
    InterpolationMethod,
    interp,
    interpolate1d,
    valid_methods,
)


floatX = pytensor.config.floatX


def test_interp():
    xp = [1.0, 2.0, 3.0]
    fp = [3.0, 2.0, 0.0]

    x = [0, 1, 1.5, 2.72, 3.14]

    out = interp(x, xp, fp).eval()
    np_out = np.interp(x, xp, fp)

    assert_allclose(out, np_out)


def test_interp_padded():
    xp = [1.0, 2.0, 3.0]
    fp = [3.0, 2.0, 0.0]

    assert interp(3.14, xp, fp, right=-99.0).eval() == -99.0
    assert_allclose(
        interp([-1.0, -2.0, -3.0], xp, fp, left=1000.0).eval(), [1000.0, 1000.0, 1000.0]
    )
    assert_allclose(
        interp([-1.0, 10.0], xp, fp, left=-10, right=10).eval(), [-10, 10.0]
    )


@pytest.mark.parametrize("method", valid_methods, ids=str)
@pytest.mark.parametrize(
    "left_pad, right_pad", [(None, None), (None, 100), (-100, None), (-100, 100)]
)
def test_interpolate_scalar_no_extrapolate(
    method: InterpolationMethod, left_pad, right_pad
):
    x = np.linspace(-2, 6, 10)
    y = np.sin(x)

    f_op = interpolate1d(
        x, y, method, extrapolate=False, left_pad=left_pad, right_pad=right_pad
    )
    x_hat_pt = pt.dscalar("x_hat")
    f = pytensor.function([x_hat_pt], f_op(x_hat_pt), mode="FAST_RUN")

    # Data points should be returned exactly, except when method == mean
    if method not in ["mean", "first"]:
        assert f(x[3]) == y[3]
    elif method == "first":
        assert f(x[3]) == y[2]
    else:
        # method == 'mean
        assert f(x[3]) == (y[2] + y[3]) / 2

    # When extrapolate=False, points beyond the data envelope should be constant
    left_pad = y[0] if left_pad is None else left_pad
    right_pad = y[-1] if right_pad is None else right_pad

    assert f(-10) == left_pad
    assert f(100) == right_pad


@pytest.mark.parametrize("method", valid_methods, ids=str)
def test_interpolate_scalar_extrapolate(method: InterpolationMethod):
    x = np.linspace(-2, 6, 10)
    y = np.sin(x)

    f_op = interpolate1d(x, y, method)
    x_hat_pt = pt.dscalar("x_hat")
    f = pytensor.function([x_hat_pt], f_op(x_hat_pt), mode="FAST_RUN")

    left_test_point = -5
    right_test_point = 100
    if method == "linear":
        # Linear will compute a slope from the endpoints and continue it
        left_slope = (left_test_point - x[0]) / (x[1] - x[0])
        right_slope = (right_test_point - x[-2]) / (x[-1] - x[-2])
        assert f(left_test_point) == y[0] + left_slope * (y[1] - y[0])
        assert f(right_test_point) == y[-2] + right_slope * (y[-1] - y[-2])

    elif method == "mean":
        left_expected = (y[0] + y[1]) / 2
        right_expected = (y[-1] + y[-2]) / 2
        assert f(left_test_point) == left_expected
        assert f(right_test_point) == right_expected

    else:
        assert f(left_test_point) == y[0]
        assert f(right_test_point) == y[-1]

        # For interior points, "first" and "last" should disagree. First should take the left side of the interval,
        # and last should take the right.
        interior_point = x[3] + 0.1
        assert f(interior_point) == (y[4] if method == "last" else y[3])
