import os
from functools import partial

import numpy as np
import pytest
import scipy.special
import scipy.stats

import pytensor
import pytensor.tensor as pt
from pytensor.scalar.math import I0, I1, Ive
from pytensor.tensor.type import scalar, vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.bessel import (
    _constant_order as constant_order,
)
from pytensor.link.mlx.dispatch.scalar.bessel import (
    _metal_ive_call as metal_ive_call,
)


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


# i1 splits at the same x = 8, and is odd rather than even
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        (1e-8, 1.0, {"float32": 1e-6, "float64": 1e-13}),
        (1.0, 8.0, {"float32": 5e-6, "float64": 1e-12}),
        (8.0, 20.0, {"float32": 5e-6, "float64": 1e-12}),
        (20.0, 80.0, {"float32": 2e-5, "float64": 1e-13}),
    ],
    ids=["small", "below-split", "above-split", "large"],
)
def test_i1(low, high, tolerances, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(59).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.i1(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=tolerances[dtype]),
    )


@pytest.mark.parametrize(
    "low, high, rtol",
    [
        (1e-12, 1.0, 1e-14),
        (1.0, 8.0, 1e-14),
        (8.0, 100.0, 1e-13),
        (100.0, 700.0, 1e-12),
    ],
    ids=["tiny", "below-split", "above-split", "near-overflow"],
)
def test_i1_float64_precision(low, high, rtol):
    # The small-interval series is fitted to i1e(x) / x, because i1 vanishes at the
    # origin and a series fitted to i1e itself holds absolute error there rather than
    # relative. This is the test that would catch dropping that division.
    x_test_value = np.exp(np.linspace(np.log(low), np.log(high), 401))

    with mlx.stream(mlx.cpu):
        res = np.asarray(mlx_funcify(I1())(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(res, scipy.special.i1(x_test_value), rtol=rtol)


def test_i1_edge_cases():
    # i1 is odd and zero at the origin, where i0 is even and one
    x = vector("x", dtype="float64")
    x_test_value = np.array([0.0, -0.0, -1.0, -30.0, 713.0, np.inf, -np.inf, np.nan])

    compare_mlx_and_py([x], [pt.i1(x)], [x_test_value])


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_i0_grad(dtype):
    # d/dx i0(x) is i1(x), so a graph with i0 in it fails at gradient time unless both
    # are dispatched -- which is why they ship together rather than i0 alone.
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(61).uniform(-8.0, 8.0, 51).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.grad(pt.i0(x).sum(), x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-6),
    )


@pytest.mark.parametrize("kappa_value", [0.5, 5.0, 50.0, 700.0], ids=str)
def test_vonmises_logp(kappa_value):
    # pymc's VonMises.logp (continuous.py:3178) reaches i0 from the logp itself, so the
    # distribution is unusable on this backend without it. The tolerance is set by
    # mx.cos, which is a float32 kernel whatever dtype it is handed -- see the float64
    # op-support table -- not by i0, which holds 1e-13 over this range.
    value, mu, kappa = pt.scalars("value", "mu", "kappa")
    logp = kappa * pt.cos(mu - value) - np.log(2.0 * np.pi) - pt.log(pt.i0(kappa))
    test_values = [0.3, 1.1, kappa_value]

    _, res = compare_mlx_and_py(
        [value, mu, kappa],
        logp,
        test_values,
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-6),
    )
    np.testing.assert_allclose(
        np.asarray(res),
        scipy.stats.vonmises.logpdf(0.3, kappa_value, loc=1.1),
        rtol=1e-6,
    )


@pytest.mark.parametrize("op", [pt.i0, pt.i1], ids=["i0", "i1"])
def test_bessel_i_continuous_across_split(op):
    # x = 8 is where the two Chebyshev intervals meet, and they are independent
    # approximations that have to agree there. Every other range stops at the boundary
    # rather than crossing it, and no uniform sample lands on it exactly, so a mismatched
    # clamp or a < where <= belongs would go unnoticed.
    x = vector("x", dtype="float64")
    x_test_value = np.array(
        [np.nextafter(8.0, 0.0), 8.0, np.nextafter(8.0, 9.0), 7.999999, 8.000001]
    )

    compare_mlx_and_py(
        [x],
        [op(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-14),
    )


# The order is sampled across the shapes that behave differently: integer (where the
# function continues to negative argument), half-integer (where the Hankel expansion
# terminates early), non-integer, and negative. One log-spaced range covers the series,
# the crossover and the asymptotic branch together, because the tolerance the dispatch
# holds does not vary across them -- only with the dtype.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize("order", [0.0, 0.5, 2.5, -0.5], ids=str)
def test_ive(order, dtype):
    rtol = {"float32": 1e-5, "float64": 1e-13}[dtype]
    x = vector("x", dtype=dtype)
    x_test_value = np.exp(
        np.random.default_rng(67).uniform(np.log(1e-6), np.log(2000.0), 201)
    ).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.ive(order, x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol),
    )


@pytest.mark.parametrize("order", [0.0, 0.5, 2.5, -0.5, 10.0, 20.0], ids=str)
def test_ive_float64_precision(order):
    # The split between the series and the Hankel expansion moves with the order, and so
    # does the trip count the series needs to reach it. This is the test that catches a
    # split left behind when the order grows: the expansion needs x large against v**2,
    # and applying it too early is wrong by whole percent rather than by an ulp.
    x_test_value = np.exp(np.linspace(np.log(1e-6), np.log(2000.0), 601))

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Ive(), node=pt.ive(order, vector("x", dtype="float64")).owner)(
                None, mlx.array(x_test_value, dtype=mlx.float64)
            )
        )

    np.testing.assert_allclose(res, scipy.special.ive(order, x_test_value), rtol=1e-12)


@pytest.mark.parametrize("order", [0.0, 1.0, 2.0, 0.5, -0.5], ids=str)
def test_ive_domain(order):
    # Only integer orders continue to negative argument, carrying the parity of the
    # order; a negative non-integer order also has a pole at zero. scipy is the contract
    # here, nan for nan.
    x = vector("x", dtype="float64")
    # ive decays to zero at either infinity, but scipy reports nan there and the op has
    # to agree with it across backends
    x_test_value = np.array([-2.0, -1.0, 0.0, 1.0, np.inf, -np.inf, np.nan])

    compare_mlx_and_py([x], [pt.ive(order, x)], [x_test_value])


def test_ive_symbolic_order_raises():
    # Both branches are polynomials whose coefficients come from the order, and the
    # domain rules above branch on whether it is an integer, so a symbolic order needs a
    # different implementation rather than a different constant.
    x = vector("x", dtype="float64")
    v = scalar("v", dtype="float64")

    with pytest.raises(NotImplementedError, match="constant order"):
        pytensor.function([v, x], pt.ive(v, x), mode="MLX")


def test_ive_large_order_raises():
    x = vector("x", dtype="float64")

    with pytest.raises(NotImplementedError, match="orders up to 20"):
        pytensor.function([x], pt.ive(25.0, x), mode="MLX")


@pytest.mark.skipif(
    not mlx.metal.is_available() or os.environ.get("PYTENSOR_MLX_SKIP_GPU") == "1",
    reason="needs a GPU that can run kernels; set PYTENSOR_MLX_SKIP_GPU=1 where it cannot",
)
@pytest.mark.parametrize("order", [0.0, 0.5, 1.0, 2.5, 5.0, -0.5, -1.0, -2.5], ids=str)
def test_ive_paths_agree(order):
    # ive is implemented twice: a Metal kernel, taken for float32 on the GPU stream, and
    # the vectorized fallback taken everywhere else -- which is every float64 graph,
    # since Metal has no float64 at all. Only the fallback runs on CI, so this is the
    # only thing standing between the kernel and a silent drift.
    #
    # The negative non-integer orders are the ones that matter. Their series terms
    # alternate in sign, and a convergence test written on the term rather than its
    # magnitude stops at the first one, which is wrong by a factor of two rather than by
    # an ulp. Every order with strictly positive terms passes such a test happily.
    x_test_value = np.exp(
        np.random.default_rng(71).uniform(np.log(1e-4), np.log(500.0), 1001)
    ).astype("float32")
    x = vector("x", dtype="float32")
    node = pt.ive(order, x).owner

    with mlx.stream(mlx.gpu):
        # Without this the test compares the fallback against itself and passes whatever
        # the kernel does. A scalar order reaches the dispatch with shape (1,) once it
        # has been broadcast against the argument, which is enough to disqualify it from
        # the kernel if the guard checks rank rather than size.
        graph_order = constant_order(node.inputs[0])
        assert metal_ive_call(graph_order, mlx.array(x_test_value)) is not None, (
            "the Metal kernel did not engage, so this comparison proves nothing"
        )
        metal = np.asarray(mlx_funcify(Ive(), node=node)(None, mlx.array(x_test_value)))
    with mlx.stream(mlx.cpu):
        vectorized = np.asarray(
            mlx_funcify(Ive(), node=node)(None, mlx.array(x_test_value))
        )

    np.testing.assert_allclose(metal, vectorized, rtol=1e-4, atol=0.0)


def test_ive_vector_order():
    # pymc's Periodic kernel asks for a whole vector of orders at once in its HSGP
    # approximation (gp/cov.py, power_spectral_density_approx): ive(J, a) with
    # J = arange(m). The order is constant there, just not a scalar, and the graph gets
    # one evaluation for all of them rather than one per order.
    a = scalar("a", dtype="float64")
    orders = np.arange(6)

    _, res = compare_mlx_and_py(
        [a],
        pt.ive(orders, a),
        [0.7],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-13),
    )
    np.testing.assert_allclose(
        np.asarray(res), scipy.special.ive(orders, 0.7), rtol=1e-13
    )


def test_ive_mixed_vector_order():
    # The domain rules are per-element once the order is a vector: the integer orders
    # continue to negative argument with their parity, the non-integer ones do not, and
    # a negative non-integer order has a pole at zero.
    x = vector("x", dtype="float64")
    orders = np.array([0.0, 1.0, -2.5, 2.0])
    x_test_value = np.array([-1.5, -1.5, -1.5, 0.0])

    compare_mlx_and_py([x], [pt.ive(orders, x)], [x_test_value])


@pytest.mark.parametrize("order", [0.0, 0.5, 1.0, 2.5, -0.5], ids=str)
def test_ive_small_argument(order):
    # The series takes its prefactor in log space, and the logarithm has to be kept away
    # from zero without being floored: clamping its argument at float32's smallest normal
    # left every x below 1.2e-38 sharing one value, which for ive(1, 1e-300) meant 5.9e-39
    # against a true 5.0e-301. Log-spaced because a linear sample here is all endpoint.
    x_test_value = np.exp(np.linspace(np.log(1e-300), np.log(1e-3), 301))

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Ive(), node=pt.ive(order, vector("x", dtype="float64")).owner)(
                None, mlx.array(x_test_value, dtype=mlx.float64)
            )
        )

    reference = scipy.special.ive(order, x_test_value)
    # scipy flushes the deepest subnormals to zero where the series still resolves them,
    # so the comparison runs where the reference is a normal number
    representable = reference > np.finfo(np.float64).tiny
    np.testing.assert_allclose(res[representable], reference[representable], rtol=1e-12)
