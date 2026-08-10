from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.scalar.math import I0
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


# The series splits at x = 8, so the ranges straddle it rather than sampling across it.
# float32 tolerances loosen with the argument because i0 is the scaled series times
# exp(x), and at float32 the exponent's own rounding is x * eps -- 5e-6 by x = 80. That
# is the dtype, not the approximation: the same ranges hold 1e-14 at float64, where the
# exponential goes through mx.power instead.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        (1e-8, 1.0, {"float32": 1e-6, "float64": 1e-13}),
        (1.0, 8.0, {"float32": 1e-6, "float64": 1e-13}),
        (8.0, 20.0, {"float32": 5e-6, "float64": 1e-13}),
        (20.0, 80.0, {"float32": 2e-5, "float64": 1e-13}),
    ],
    ids=["small", "below-split", "above-split", "large"],
)
def test_i0(low, high, tolerances, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(53).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.i0(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=tolerances[dtype]),
    )


@pytest.mark.parametrize(
    "low, high, rtol",
    [
        (1e-12, 1.0, 1e-14),
        (1.0, 8.0, 1e-14),
        (8.0, 100.0, 1e-13),
        # the exponent carries x * eps of its own, which is 1.5e-13 by x = 700 and has
        # nothing to do with the series
        (100.0, 700.0, 1e-12),
    ],
    ids=["tiny", "below-split", "above-split", "near-overflow"],
)
def test_i0_float64_precision(low, high, rtol):
    # The Chebyshev coefficients are only worth carrying if float64 survives to the end,
    # and the exponential is what decides that: mx.exp is a float32 kernel whatever it is
    # handed and overflows outright past exp(88), so this range would return inf through
    # it. This is also the test that fails if the coefficients weak-type to float32.
    x_test_value = np.exp(np.linspace(np.log(low), np.log(high), 401))

    with mlx.stream(mlx.cpu):
        res = np.asarray(mlx_funcify(I0())(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(res, scipy.special.i0(x_test_value), rtol=rtol)


def test_i0_edge_cases():
    # i0 is even and equals 1 at the origin. 713 is where i0 overflows float64, which is
    # not an exotic von Mises concentration, so the boundary is pinned rather than left
    # to chance.
    x = vector("x", dtype="float64")
    x_test_value = np.array([0.0, -0.0, -1.0, -30.0, 713.0, np.inf, -np.inf, np.nan])

    compare_mlx_and_py([x], [pt.i0(x)], [x_test_value])
