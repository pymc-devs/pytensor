from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.scalar.math import GammaInc, GammaIncC
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.gammainc import (
    _TEMME_MAX_RATIO,
    _TEMME_MIN_A,
    _TEMME_MIN_RATIO,
    _TINY,
)


def _sample(rng, shape_range, ratio_range, n):
    """Draw log-spaced ``(a, x)`` with ``x`` set as a multiple of ``a``.

    The expansion is chosen on ``x / a``, so a region is only reached by sampling that
    ratio rather than ``x`` on its own.
    """
    a = np.exp(rng.uniform(*np.log(shape_range), n))
    return a, a * np.exp(rng.uniform(*np.log(ratio_range), n))


# Each region selects a different expansion: Temme needs a >= 15 with x within [0.4a,
# 1.8a], the ascending series takes what is left below x = a + 1, the continued fraction
# what is left above it. float32 loosens where the series and the fraction reach their
# answer through a log prefactor that subtracts two numbers of order a log a.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "shape_range, ratio_range, tolerances",
    [
        ((1e-2, 5.0), (1e-2, 0.9), {"float32": 1e-4, "float64": 1e-12}),
        ((20.0, 5e4), (0.4, 1.8), {"float32": 3e-4, "float64": 1e-10}),
        ((1e-2, 10.0), (5.0, 50.0), {"float32": 1e-3, "float64": 1e-10}),
        ((50.0, 5e3), (0.05, 0.35), {"float32": 5e-4, "float64": 1e-11}),
    ],
    ids=["series", "temme", "continued-fraction", "series-far-left"],
)
@pytest.mark.parametrize(
    "op", [pt.gammainc, pt.gammaincc], ids=["gammainc", "gammaincc"]
)
def test_incomplete_gamma(op, shape_range, ratio_range, tolerances, dtype):
    a = vector("a", dtype=dtype)
    x = vector("x", dtype=dtype)
    a_test_value, x_test_value = _sample(
        np.random.default_rng(11), shape_range, ratio_range, 201
    )

    compare_mlx_and_py(
        [a, x],
        [op(a, x)],
        [a_test_value.astype(dtype), x_test_value.astype(dtype)],
        # Far enough into the tail the answer is a subnormal, where one ULP is percents
        # and no relative claim means anything; atol retires those elements at the
        # dtype's smallest normal rather than dropping them from the sample
        assert_fn=partial(
            np.testing.assert_allclose,
            rtol=tolerances[dtype],
            atol=np.finfo(dtype).tiny,
        ),
    )


@pytest.mark.parametrize(
    "shape_range, ratio_range",
    [
        ((1e-2, 5.0), (1e-2, 0.9)),
        ((20.0, 5e4), (0.4, 1.8)),
        ((1e-2, 10.0), (5.0, 50.0)),
        ((50.0, 5e3), (0.05, 0.35)),
    ],
    ids=["series", "temme", "continued-fraction", "series-far-left"],
)
def test_incomplete_gamma_float64_precision(shape_range, ratio_range):
    a_test_value, x_test_value = _sample(
        np.random.default_rng(23), shape_range, ratio_range, 2001
    )

    with mlx.stream(mlx.cpu):
        a_mlx = mlx.array(a_test_value, dtype=mlx.float64)
        x_mlx = mlx.array(x_test_value, dtype=mlx.float64)
        lower = np.asarray(mlx_funcify(GammaInc())(a_mlx, x_mlx))
        upper = np.asarray(mlx_funcify(GammaIncC())(a_mlx, x_mlx))

    # Past 1e-290 a tail has run out of exponent rather than of accuracy, so those
    # elements are dropped instead of carrying a relative claim about a denormal
    truth_lower = scipy.special.gammainc(a_test_value, x_test_value)
    truth_upper = scipy.special.gammaincc(a_test_value, x_test_value)
    for got, truth in ((lower, truth_lower), (upper, truth_upper)):
        keep = truth > 1e-290
        np.testing.assert_allclose(got[keep], truth[keep], rtol=1e-10)


def test_incomplete_gamma_small_tail():
    # The regression this exists for: taking one tail as one minus the other cancels
    # everything once that tail runs below the other's rounding. At a = 200, x = 0.5a
    # the true P is 9.3e-19, so a complement would report zero with 100% error. Both
    # tails come out of the same expansion directly instead.
    a_test_value = np.array([200.0, 200.0, 2000.0, 2000.0, 5000.0])
    x_test_value = a_test_value * np.array([0.5, 0.8, 0.8, 0.9, 0.85])

    with mlx.stream(mlx.cpu):
        lower = np.asarray(
            mlx_funcify(GammaInc())(
                mlx.array(a_test_value, dtype=mlx.float64),
                mlx.array(x_test_value, dtype=mlx.float64),
            )
        )

    truth = scipy.special.gammainc(a_test_value, x_test_value)
    assert truth.max() < 1e-2, "these points have to sit in the tail to test anything"
    np.testing.assert_allclose(lower, truth, rtol=1e-10)


def test_incomplete_gamma_branch_boundaries():
    # The three expansions meet along a = 15, x = 0.4a, x = 1.8a and x = a + 1, and the
    # answer has to agree from both sides of every seam. A wrong coefficient or trip count
    # in one expansion shows up here as a step across the seam it owns, which sampling
    # inside a region cannot see. The thresholds are imported rather than written out, so
    # moving a boundary moves the test with it.
    offsets = np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6])
    shapes, values = [], []
    for shape in (5.0, _TEMME_MIN_A - 1e-6, _TEMME_MIN_A, _TEMME_MIN_A + 1e-6, 3000.0):
        for ratio in (_TEMME_MIN_RATIO, _TEMME_MAX_RATIO):
            shapes.append(np.full(offsets.size, shape))
            values.append(shape * (ratio + offsets))
        shapes.append(np.full(offsets.size, shape))
        values.append(shape + 1.0 + offsets * shape)
    a_test_value = np.concatenate(shapes)
    x_test_value = np.concatenate(values)

    with mlx.stream(mlx.cpu):
        a_mlx = mlx.array(a_test_value, dtype=mlx.float64)
        x_mlx = mlx.array(x_test_value, dtype=mlx.float64)
        lower = np.asarray(mlx_funcify(GammaInc())(a_mlx, x_mlx))
        upper = np.asarray(mlx_funcify(GammaIncC())(a_mlx, x_mlx))

    np.testing.assert_allclose(
        lower, scipy.special.gammainc(a_test_value, x_test_value), rtol=1e-10
    )
    np.testing.assert_allclose(
        upper, scipy.special.gammaincc(a_test_value, x_test_value), rtol=1e-10
    )


def test_continued_fraction_floor_survives_float32():
    # The continued fraction seeds one Lentz variable at 1 / _TINY and guards both against
    # it. A floor below float32's smallest normal -- 1e-300 was -- rounds to zero there,
    # which seeds inf and turns each guard into |d| < 0, so neither can ever fire.
    assert np.float32(_TINY) > 0.0
    assert np.isfinite(np.float32(1.0) / np.float32(_TINY))


def test_incomplete_gamma_complement():
    # P + Q == 1 has to hold across every expansion and both sides of each boundary,
    # including where one tail is hundreds of orders below the other
    a_test_value, x_test_value = _sample(
        np.random.default_rng(37), (1e-2, 1e5), (1e-3, 50.0), 4001
    )

    with mlx.stream(mlx.cpu):
        a_mlx = mlx.array(a_test_value, dtype=mlx.float64)
        x_mlx = mlx.array(x_test_value, dtype=mlx.float64)
        total = np.asarray(mlx_funcify(GammaInc())(a_mlx, x_mlx)) + np.asarray(
            mlx_funcify(GammaIncC())(a_mlx, x_mlx)
        )

    np.testing.assert_allclose(total, 1.0, rtol=1e-12)


@pytest.mark.parametrize(
    "op", [pt.gammainc, pt.gammaincc], ids=["gammainc", "gammaincc"]
)
def test_incomplete_gamma_edge_cases(op):
    # x = 0 is exactly P = 0, which a log prefactor clamped away from zero rather than
    # substituted gets wrong by a whole answer: a * log(tiny) in the exponent lands on
    # 1e-150 for a = 0.5. The rest pin the saturated ends and nan propagation.
    a = vector("a", dtype="float64")
    x = vector("x", dtype="float64")
    a_test_value = np.array([1.0, 0.5, 1e-3, 200.0, 1.0, 200.0, 1.0, 1.0])
    x_test_value = np.array([0.0, 0.0, 0.0, 0.0, 1e30, 1e30, 1e-8, np.nan])

    compare_mlx_and_py([a, x], [op(a, x)], [a_test_value, x_test_value])


@pytest.mark.parametrize(
    "dtype, rtol", [("float32", 1e-4), ("float64", 1e-6)], ids=["float32", "float64"]
)
def test_incomplete_gamma_grad(dtype, rtol):
    # Only the gradient in the value has a dispatch: the one in the shape parameter is a
    # ScalarLoop, which the MLX backend does not convert yet.
    #
    # The gradient graph is pytensor's own exp(...) rather than anything in this module,
    # so it inherits mx.exp, a float32 kernel at every dtype that flushes to zero below
    # exp(-90). That sets the float64 tolerance, and it is why the sampled range keeps
    # the density above that floor rather than running out into the tail.
    a = vector("a", dtype=dtype)
    x = vector("x", dtype=dtype)
    a_test_value, x_test_value = _sample(
        np.random.default_rng(41), (0.5, 20.0), (0.2, 3.0), 101
    )

    compare_mlx_and_py(
        [a, x],
        [pt.grad(pt.gammainc(a, x).sum(), x), pt.grad(pt.gammaincc(a, x).sum(), x)],
        [a_test_value.astype(dtype), x_test_value.astype(dtype)],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol),
    )
