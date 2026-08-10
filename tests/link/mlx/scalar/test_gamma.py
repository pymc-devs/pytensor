from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.scalar.math import Gamma
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


# float32 loosens with the argument because gamma grows like a factorial and the
# exponential carries x * eps of its own; it overflows float32 outright past x = 35,
# which is why the ranges stop short of that.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        (1e-3, 1.0, {"float32": 1e-5, "float64": 1e-13}),
        (1.0, 10.0, {"float32": 1e-5, "float64": 1e-13}),
        (10.0, 34.0, {"float32": 1e-4, "float64": 1e-12}),
        (-0.9, -0.1, {"float32": 1e-5, "float64": 1e-13}),
        (-5.9, -5.1, {"float32": 1e-4, "float64": 1e-13}),
    ],
    ids=["tiny", "moderate", "large", "negative", "negative-deep"],
)
def test_gamma(low, high, tolerances, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(83).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.gamma(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=tolerances[dtype]),
    )


@pytest.mark.parametrize(
    "low, high",
    [(1e-3, 1.0), (1.0, 34.0), (34.0, 170.0), (-0.9, -0.1), (-20.9, -20.1)],
    ids=["tiny", "moderate", "near-overflow", "negative", "negative-deep"],
)
def test_gamma_float64_precision(low, high):
    # The negative ranges are the point. Below zero gamma goes through the reflection
    # formula, whose sin(pi z) is taken from tan by the half-angle identity: tan is
    # genuine at float64 where sin is a float32 kernel whatever it is handed, and going
    # through sin directly caps these ranges near 1e-6.
    x_test_value = np.linspace(low, high, 401)

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Gamma())(mlx.array(x_test_value, dtype=mlx.float64))
        )

    np.testing.assert_allclose(res, scipy.special.gamma(x_test_value), rtol=1e-12)


def test_gamma_edge_cases():
    # gamma is +/-inf at zero by the sign of the zero, undefined at every negative
    # integer -- where the reflection divides by a sin that rounding leaves tiny rather
    # than exactly zero -- and 171 is the last argument float64 can hold.
    x = vector("x", dtype="float64")
    x_test_value = np.array(
        [0.0, -0.0, -1.0, -3.0, 1.0, 171.0, np.inf, -np.inf, np.nan]
    )

    compare_mlx_and_py([x], [pt.gamma(x)], [x_test_value])


@pytest.mark.parametrize(
    "dtype, rtol", [("float32", 1e-4), ("float64", 1e-13)], ids=["float32", "float64"]
)
def test_gamma_grad(dtype, rtol):
    # d/dx gamma(x) is gamma(x) * psi(x), so this needs both dispatches
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(89).uniform(0.5, 8.0, 51).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.grad(pt.gamma(x).sum(), x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol),
    )


@pytest.mark.parametrize(
    "low, high",
    [(1e-300, 1e-100), (1e-100, 1e-38), (1e-38, 1e-3)],
    ids=["subnormal-scale", "below-float32", "small"],
)
def test_gamma_small_positive(low, high):
    # Gamma grows like 1 / z toward the origin, so anything that floors the argument
    # floors the whole region with it: clamping at float32's smallest normal returned
    # 8.5e37 for every argument below 1.2e-38, where the answers run out to 1e300. The
    # ranges are log-spaced because a linear sample here is all upper endpoint.
    x_test_value = np.exp(np.linspace(np.log(low), np.log(high), 201))

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Gamma())(mlx.array(x_test_value, dtype=mlx.float64))
        )

    np.testing.assert_allclose(res, scipy.special.gamma(x_test_value), rtol=1e-12)
