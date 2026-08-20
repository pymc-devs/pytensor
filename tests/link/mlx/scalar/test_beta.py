import os
from functools import partial

import numpy as np
import pytest
import scipy.special
import scipy.stats

import pytensor
import pytensor.tensor as pt
from pytensor.scalar.math import BetaInc
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.beta import (
    _TEMME_MAX_ETA,
    _TEMME_MIN_A,
    _TEMME_MIN_P,
    _metal_betainc_call,
)


# Both endpoints, then a very small parameter against a very large one in each order,
# then both large, then the endpoints again at parameters below one
_EDGE_CASES = (
    np.array([1.0, 1.0, 3.0, 3.0, 1e-3, 1e5, 1e-3, 1e5, 0.5, 2.0]),
    np.array([1.0, 1.0, 7.0, 7.0, 1e5, 1e-3, 1e-3, 1e5, 0.5, 2.0]),
    np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0]),
)

# Deep in the tail, where the answer goes like x**a and every digit of x matters
_DEEP_TAIL = (
    np.array([5.0, 5.0, 2.0, 50.0, 5.0, 200.0]),
    np.array([2.0, 2.0, 0.5, 2.0, 1e4, 3.0]),
    np.array([8e-7, 1e-8, 1e-9, 1e-4, 1e-9, 1e-3]),
)


def _sample(rng, shape_range, ratio_range, n):
    """Draw log-spaced ``(a, b, x)`` with ``x`` placed relative to the mode.

    The mode sits at ``a / (a + b)`` and the transition around it narrows like
    ``1 / sqrt(a + b)``, so ``x`` is drawn in units of that width rather than uniformly
    on the unit interval -- a uniform grid steps clean over the region both expansions
    are chosen for.
    """
    a = np.exp(rng.uniform(*np.log(shape_range), n))
    b = a * np.exp(rng.uniform(*np.log(ratio_range), n))
    mu = a + b
    p = a / mu
    x = p + rng.uniform(-6.0, 6.0, n) * np.sqrt(p * (1.0 - p) / mu)
    return a, b, np.clip(x, 1e-12, 1.0 - 1e-12)


def _assert_logcdf(got, want):
    """Compare two log-cdfs where the comparison says something.

    A log-cdf runs to zero as the cdf approaches one, where a relative claim is
    unbounded, and to -inf where the cdf underflows the dtype -- which is a limit of
    taking the log of a cdf rather than anything the dispatch did. ``atol`` covers the
    first, and the mask covers the second.
    """
    assert not np.isnan(got).any(), "the dispatch returned nan"
    keep = np.isfinite(got) & np.isfinite(want)
    assert keep.mean() > 0.5, "the sample has to keep enough points to test anything"
    np.testing.assert_allclose(got[keep], want[keep], rtol=1e-6, atol=1e-9)


def _direct(a, b, x, dtype):
    """The dispatch on its own, off the graph, so a tolerance can be tight."""
    with mlx.stream(mlx.cpu):
        return np.asarray(
            mlx_funcify(BetaInc())(
                mlx.array(a, dtype=dtype),
                mlx.array(b, dtype=dtype),
                mlx.array(x, dtype=dtype),
            )
        )


# The expansions divide on min(a, b) rather than on x: the continued fraction owns
# everything until its trip count runs short, and Temme takes the window around the mode
# from _TEMME_MIN_A up.
#
# The float32 tolerances are set by the reference rather than by the dispatch. scipy's
# single-precision betainc loop carries 1.4e-2 at the parameter sizes of the temme row,
# against the 4e-4 the dispatch delivers there, so a tighter number here would be
# asserting that we reproduce scipy's rounding. The dispatch's own float32 accuracy is
# claimed against a float64 reference in test_betainc_float32_precision instead.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "shape_range, ratio_range, tolerances",
    [
        ((1e-2, 5.0), (0.2, 5.0), {"float32": 1e-4, "float64": 1e-11}),
        ((5.0, 500.0), (0.2, 5.0), {"float32": 1e-4, "float64": 1e-11}),
        ((2e3, 5e5), (0.2, 5.0), {"float32": 3e-2, "float64": 1e-8}),
        ((5.0, 1e3), (20.0, 500.0), {"float32": 3e-3, "float64": 1e-9}),
    ],
    ids=[
        "fraction-small",
        "fraction-moderate",
        "temme",
        "asymmetric-small",
    ],
)
def test_betainc(shape_range, ratio_range, tolerances, dtype):
    a = vector("a", dtype=dtype)
    b = vector("b", dtype=dtype)
    x = vector("x", dtype=dtype)
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(11), shape_range, ratio_range, 201
    )

    compare_mlx_and_py(
        [a, b, x],
        [pt.betainc(a, b, x)],
        [
            a_test_value.astype(dtype),
            b_test_value.astype(dtype),
            x_test_value.astype(dtype),
        ],
        # Far enough into a tail the answer is a subnormal, where one ULP is percents and
        # no relative claim means anything; atol retires those elements at the dtype's
        # smallest normal rather than dropping them from the sample
        assert_fn=partial(
            np.testing.assert_allclose,
            rtol=tolerances[dtype],
            atol=np.finfo(dtype).tiny,
        ),
    )


@pytest.mark.parametrize(
    "shape_range, ratio_range, temme_share",
    [
        ((1e-2, 5.0), (0.2, 5.0), 0.0),
        ((5.0, 500.0), (0.2, 5.0), 0.0),
        ((2e3, 5e5), (0.2, 5.0), 0.5),
        ((5.0, 1e3), (20.0, 500.0), 0.0),
    ],
    ids=[
        "fraction-small",
        "fraction-moderate",
        "temme",
        "asymmetric-small",
    ],
)
def test_betainc_float64_precision(shape_range, ratio_range, temme_share):
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(23), shape_range, ratio_range, 2001
    )

    # The ids above name a branch, and moving either gate silently moves which one a row
    # exercises -- so the row asserts the split it claims rather than trusting it
    lower = np.minimum(a_test_value, b_test_value)
    in_temme = (lower >= _TEMME_MIN_A) & (
        lower / (a_test_value + b_test_value) >= _TEMME_MIN_P
    )
    if temme_share:
        assert in_temme.mean() >= temme_share, "this row no longer reaches Temme"
    else:
        assert not in_temme.any(), "this row was meant to stay on the fraction"

    got = _direct(a_test_value, b_test_value, x_test_value, mlx.float64)

    # Past 1e-290 a tail has run out of exponent rather than of accuracy, so those
    # elements are dropped instead of carrying a relative claim about a denormal
    truth = scipy.special.betainc(a_test_value, b_test_value, x_test_value)
    keep = truth > 1e-290
    np.testing.assert_allclose(got[keep], truth[keep], rtol=1e-8)


def test_betainc_uncovered_corner():
    # The corner both expansions find hardest: min(a, b) large and p small. The fraction
    # would need terms growing like sqrt(min(a, b)) there -- around 340 against the 120
    # the graph unrolls -- and Temme is leaning on its leading coefficient with c_1 to
    # c_3 extrapolating in s. It degrades gracefully rather than failing, and this pins
    # how far.
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(11), (2e3, 1e5), (20.0, 500.0), 501
    )

    # Both halves of "hardest", asserted rather than assumed: without this the sample
    # could drift into an easy region and the tolerance below would still pass
    lower = np.minimum(a_test_value, b_test_value)
    assert lower.min() > 1e3, "the corner needs min(a, b) large"
    assert (lower / (a_test_value + b_test_value)).max() < 0.05, "and p small"

    truth = scipy.special.betainc(a_test_value, b_test_value, x_test_value)
    keep = truth > 1e-290
    got = _direct(a_test_value, b_test_value, x_test_value, mlx.float64)
    np.testing.assert_allclose(got[keep], truth[keep], rtol=1e-5)


@pytest.mark.parametrize(
    "shape_range, ratio_range, tolerance",
    [
        ((1e-2, 5.0), (0.2, 5.0), 1e-4),
        ((5.0, 500.0), (0.2, 5.0), 1e-4),
        ((2e3, 5e5), (0.2, 5.0), 1e-3),
        ((5.0, 1e3), (20.0, 500.0), 1e-3),
    ],
    ids=["fraction-small", "fraction-moderate", "temme", "asymmetric-small"],
)
def test_betainc_float32_precision(shape_range, ratio_range, tolerance):
    # The float32 claim, made against a float64 reference rather than against scipy's
    # own single-precision loop. What sets the floor is the exponent both expansions
    # reach their answer through: it runs to order mu eta**2 / 2, and single precision
    # carries that to |exponent| * eps however the terms are arranged.
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(37), shape_range, ratio_range, 2001
    )
    a_test_value = a_test_value.astype("float32")
    b_test_value = b_test_value.astype("float32")
    x_test_value = x_test_value.astype("float32")
    got = _direct(a_test_value, b_test_value, x_test_value, mlx.float32)

    # float32 runs out of exponent long before float64 does, so the comparison stops at
    # its smallest normal rather than carrying a relative claim about a denormal
    truth = scipy.special.betainc(
        np.float64(a_test_value), np.float64(b_test_value), np.float64(x_test_value)
    )
    keep = truth > np.finfo("float32").tiny
    np.testing.assert_allclose(got[keep], truth[keep], rtol=tolerance)


def test_betainc_deep_tail():
    # The regression this exists for: the coefficient table is fitted for p <= 1/2, so
    # the expansion normalizes a <= b. Doing that by taking x to 1 - x throws away a
    # small x entirely -- and the answer there goes like x**a, so at a = 5, x = 8e-7 it
    # came out 17% high. eta is odd under the swap and is negated instead.
    truth = scipy.special.betainc(*_DEEP_TAIL)
    assert truth.max() < 1e-10, "these points have to sit in the tail to test anything"
    np.testing.assert_allclose(_direct(*_DEEP_TAIL, mlx.float64), truth, rtol=1e-9)


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_betainc_extreme_ratio(dtype):
    # The regression this exists for: once b / a falls below the working epsilon,
    # a / (a + b) rounds to exactly one, so a q taken as 1 - p is zero and the second
    # deficit term becomes 0 * inf -- nan, all the way out. Both parameters reach eta as
    # exact ratios instead. float32 loses the ratio from 1e-8, float64 from 1e-16, so
    # this sweeps past both.
    ratios = np.array([1e-7, 1e-8, 1e-12, 1e-16, 1e-20])
    a_test_value = np.concatenate([np.ones_like(ratios), ratios, [1e5, 1e-3]])
    b_test_value = np.concatenate([ratios, np.ones_like(ratios), [1e-3, 1e5]])
    x_test_value = np.full(a_test_value.size, 0.5)

    got = _direct(a_test_value, b_test_value, x_test_value, getattr(mlx, dtype))
    assert not np.isnan(got).any(), "an extreme parameter ratio returned nan"
    np.testing.assert_allclose(
        got,
        scipy.special.betainc(a_test_value, b_test_value, x_test_value),
        rtol=1e-5 if dtype == "float32" else 1e-9,
        atol=np.finfo(dtype).tiny,
    )


def test_betainc_temme_boundary():
    # The two expansions meet along min(a, b) = _TEMME_MIN_A, p = _TEMME_MIN_P and
    # |eta| = _TEMME_MAX_ETA, and the answer has to agree from both sides of each seam.
    # A wrong coefficient or trip count shows up here as a step across the seam it owns,
    # which sampling inside a region cannot see. The thresholds are imported rather than
    # written out, so moving a boundary moves the test with it.
    offsets = np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6])
    shapes, others, values = [], [], []
    for lower in _TEMME_MIN_A * (1.0 + offsets):
        for ratio in (1.0, 3.0, (1.0 - _TEMME_MIN_P) / _TEMME_MIN_P):
            upper = lower * ratio
            mu = lower + upper
            p = lower / mu
            # eta grows like (x - p) / sqrt(p q), so the window boundary is walked in
            # the units eta is measured in rather than in x
            width = np.sqrt(p * (1.0 - p)) * _TEMME_MAX_ETA
            for offset in offsets:
                shapes.append(lower)
                others.append(upper)
                values.append(p + width * (1.0 + offset))
    a_test_value = np.array(shapes)
    b_test_value = np.array(others)
    x_test_value = np.array(values)

    np.testing.assert_allclose(
        _direct(a_test_value, b_test_value, x_test_value, mlx.float64),
        scipy.special.betainc(a_test_value, b_test_value, x_test_value),
        rtol=1e-8,
    )


def test_betainc_edge_cases():
    np.testing.assert_allclose(
        _direct(*_EDGE_CASES, mlx.float64),
        scipy.special.betainc(*_EDGE_CASES),
        rtol=1e-9,
        atol=0.0,
    )


def test_betainc_reflection_identity():
    # I_x(a, b) + I_{1-x}(b, a) = 1, which is informative only where the two sides take
    # different branches. The sample straddles the mode so that they do.
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(5), (1e-1, 5e4), (0.05, 20.0), 1001
    )
    # Only where the complement round-trips. Where it does not the two sides are
    # different points rather than the same one, and at a = 0.1 the disagreement is the
    # representation of 1 - x rather than anything either side computed
    exact = (1.0 - (1.0 - x_test_value)) == x_test_value
    a_test_value = a_test_value[exact]
    b_test_value = b_test_value[exact]
    x_test_value = x_test_value[exact]
    assert exact.mean() > 0.5, "the sample has to keep enough points to test anything"

    left = _direct(a_test_value, b_test_value, x_test_value, mlx.float64)
    right = _direct(b_test_value, a_test_value, 1.0 - x_test_value, mlx.float64)
    np.testing.assert_allclose(left + right, 1.0, rtol=0.0, atol=1e-9)


def test_beta_logcdf():
    # pymc.Beta.logcdf, which is log(betainc(alpha, beta, value))
    alpha, beta = 2.7, 5.3
    value = np.linspace(1e-4, 1.0 - 1e-4, 501)
    got = np.log(
        _direct(
            np.full_like(value, alpha), np.full_like(value, beta), value, mlx.float64
        )
    )
    _assert_logcdf(got, scipy.stats.beta.logcdf(value, alpha, beta))


@pytest.mark.parametrize("nu", [2.5, 30.0, 4e3, 4e4], ids=lambda v: f"nu={v:g}")
def test_studentt_logcdf(nu):
    # pymc.StudentT.logcdf, which is log(betainc(nu / 2, nu / 2, x)) with x set from the
    # standardized value. nu / 2 on both sides is the symmetric corner, where the
    # continued fraction is slowest and Temme takes over
    z = np.linspace(-30.0, 30.0, 501)
    root = np.sqrt(z * z + nu)
    x = (z + root) / (2.0 * root)
    half = np.full_like(z, nu / 2.0)
    _assert_logcdf(
        np.log(_direct(half, half, x, mlx.float64)), scipy.stats.t.logcdf(z, nu)
    )


def test_skew_studentt_logcdf():
    # pymc.SkewStudentT.logcdf, the same expression with the two shape parameters free
    a, b = 3.0, 11.0
    z = np.linspace(-25.0, 25.0, 501)
    x = 0.5 * (1.0 + z / np.sqrt(z * z + a + b))
    _assert_logcdf(
        np.log(_direct(np.full_like(z, a), np.full_like(z, b), x, mlx.float64)),
        scipy.stats.jf_skew_t.logcdf(z, a, b),
    )


@pytest.mark.parametrize(
    "n, p", [(10.0, 0.3), (1e4, 0.5), (1e5, 0.01)], ids=lambda v: f"{v:g}"
)
def test_binomial_logcdf(n, p):
    # pymc.Binomial.logcdf, which is log(betainc(n - value, value + 1, 1 - p)). Its
    # arguments are built from counts, so they reach much further than a user would type
    value = np.unique(np.linspace(0.0, n - 1.0, 301).astype(np.int64)).astype(float)
    _assert_logcdf(
        np.log(
            _direct(n - value, value + 1.0, np.full_like(value, 1.0 - p), mlx.float64)
        ),
        scipy.stats.binom.logcdf(value, int(n), p),
    )


@pytest.mark.parametrize(
    "n, p", [(5.0, 0.4), (1e3, 0.5), (1e4, 0.9)], ids=lambda v: f"{v:g}"
)
def test_negative_binomial_logcdf(n, p):
    # pymc.NegativeBinomial.logcdf, which is log(betainc(n, floor(value) + 1, p))
    value = np.unique(np.linspace(0.0, 3.0 * n, 301).astype(np.int64)).astype(float)
    got = np.log(
        _direct(
            np.full_like(value, n), value + 1.0, np.full_like(value, p), mlx.float64
        )
    )
    _assert_logcdf(got, scipy.stats.nbinom.logcdf(value, n, p))


def test_betainc_grad():
    # Only the gradient with respect to the value argument, which is closed form. The
    # shape-parameter gradients go through ScalarLoop, which this backend does not
    # dispatch yet.
    a = vector("a", dtype="float64")
    b = vector("b", dtype="float64")
    x = vector("x", dtype="float64")
    a_test_value = np.array([0.7, 3.0, 40.0, 300.0])
    b_test_value = np.array([2.5, 3.0, 15.0, 90.0])
    x_test_value = np.array([0.2, 0.5, 0.8, 0.7])

    compare_mlx_and_py(
        [a, b, x],
        [pt.grad(pt.betainc(a, b, x).sum(), x)],
        [a_test_value, b_test_value, x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-7),
    )


@pytest.mark.parametrize("wrt", [0, 1], ids=["a", "b"])
def test_betainc_shape_gradient_raises(wrt):
    # The counterpart to test_betainc_grad: differentiating with respect to either shape
    # parameter goes through ScalarLoop, which this backend does not dispatch. It raises
    # rather than returning something wrong, and this pins that until step 6 lands -- at
    # which point this test is the one that should start failing.
    inputs = [vector(name, dtype="float64") for name in "abx"]
    graph = pt.grad(pt.betainc(*inputs).sum(), inputs[wrt])

    with pytest.raises(NotImplementedError, match="ScalarLoop"):
        pytensor.function(inputs, graph, mode="MLX")


@pytest.mark.skipif(
    not mlx.metal.is_available() or os.environ.get("PYTENSOR_MLX_SKIP_GPU") == "1",
    reason="needs a GPU that can run kernels; set PYTENSOR_MLX_SKIP_GPU=1 where it cannot",
)
def test_betainc_paths_agree():
    # The function is implemented twice: a Metal kernel, taken for float32 on the GPU
    # stream, and the vectorized fallback taken everywhere else -- which is every float64
    # graph, since Metal has no float64 at all. Only the fallback runs on CI, so this is
    # the only thing standing between the kernel and a silent drift.
    #
    # The kernel is not a transcription of the fallback: it takes one branch per element
    # where the fallback evaluates both, and it stops the fraction on convergence where
    # the fallback always runs the full trip count. The sample has to reach both branches
    # and both sides of the reflection for that to mean anything.
    a_test_value, b_test_value, x_test_value = _sample(
        np.random.default_rng(71), (1e-1, 5e5), (0.05, 20.0), 5001
    )

    # Enforce the claim above rather than trusting it: a later edit to the ranges could
    # empty a branch, and the test would keep passing while covering less than it says
    lower = np.minimum(a_test_value, b_test_value)
    mu = a_test_value + b_test_value
    in_temme = (lower >= _TEMME_MIN_A) & (lower / mu >= _TEMME_MIN_P)
    reflected = x_test_value > (a_test_value + 1.0) / (mu + 2.0)
    for name, reached in (
        ("temme", in_temme),
        ("fraction", ~in_temme),
        ("reflection", reflected),
        ("unreflected", ~reflected),
    ):
        assert reached.mean() > 0.05, f"the sample barely reaches the {name} branch"

    # The random sample stays within six sigma of the mode and caps the ratio at 20, so
    # the point sets carrying the endpoints, the deep tail and the extreme ratios are
    # appended: those are where the two implementations have any room to disagree
    extreme = np.array([1e-7, 1e-12, 1e-20])
    a_test_value, b_test_value, x_test_value = (
        np.concatenate([sampled, *edges])
        for sampled, *edges in zip(
            (a_test_value, b_test_value, x_test_value),
            _EDGE_CASES,
            _DEEP_TAIL,
            (
                np.concatenate([np.ones_like(extreme), extreme]),
                np.concatenate([extreme, np.ones_like(extreme)]),
                np.full(2 * extreme.size, 0.5),
            ),
            strict=True,
        )
    )

    a_test_value = a_test_value.astype("float32")
    b_test_value = b_test_value.astype("float32")
    x_test_value = x_test_value.astype("float32")

    with mlx.stream(mlx.gpu):
        a_mlx = mlx.array(a_test_value)
        b_mlx = mlx.array(b_test_value)
        x_mlx = mlx.array(x_test_value)
        # Without this the test compares the fallback against itself and passes whatever
        # the kernel does
        kernel = _metal_betainc_call(a_mlx, b_mlx, x_mlx)
        assert kernel is not None, "the kernel did not engage, so this test is vacuous"
        from_kernel = np.asarray(kernel).astype("float64")

    from_fallback = _direct(
        a_test_value, b_test_value, x_test_value, mlx.float32
    ).astype("float64")

    # A nan is how the two paths diverge when one of them loses a parameter, so it fails
    # here rather than being masked out below
    assert not np.isnan(from_kernel).any(), "the kernel returned nan"
    assert not np.isnan(from_fallback).any(), "the fallback returned nan"

    # The two take different routes to the same answer, so they agree to the precision
    # float32 carries rather than bitwise. Elements below the smallest normal are dropped:
    # there one ULP is percents and no relative claim means anything
    keep = from_fallback > np.finfo("float32").tiny
    np.testing.assert_allclose(
        from_kernel[keep], from_fallback[keep], rtol=1e-3, atol=0.0
    )
