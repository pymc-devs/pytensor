from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.scalar.math import Erfc, Erfcx, GammaLn, Psi
from pytensor.tensor.math import (
    erf,
    erfc,
    erfcx,
    erfinv,
    log1mexp,
    sigmoid,
    softplus,
)
from pytensor.tensor.type import matrix, scalar, vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


# erf keeps the native mx.erf. It is a float32 kernel whatever dtype it is handed, so
# these tolerances are float32 tolerances at both precisions, but erf is well conditioned
# everywhere and never loses more than that. erfc and erfcx cannot be built on it -- both
# decay to where 1 - erf has no significant digits left -- so they carry their own kernel
# and are held to much tighter tolerances below.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, rtol, atol",
    [
        # erf vanishes at the origin, so an absolute tolerance carries the assertion for
        # any sample that lands near it
        (-1.0, 1.0, 3e-6, 1e-6),
        (1.0, 3.0, 3e-6, 0.0),
        (3.0, 6.0, 3e-7, 0.0),
    ],
    ids=["central", "moderate", "large"],
)
def test_erf(low, high, rtol, atol, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(17).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [erf(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, rtol",
    [
        (-3.0, 0.0, 3e-7),
        (0.0, 1.0, 1e-6),
        (1.0, 4.0, 1e-5),
        # erfc is exp(-y**2) * erfcx(y) out here. The exponential is the accuracy floor,
        # and squaring the argument amplifies its error in proportion to y**2
        (4.0, 9.0, 1e-4),
    ],
    ids=["negative", "small", "moderate", "tail"],
)
def test_erfc(low, high, rtol, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(19).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [erfc(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=0.0),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, rtol",
    [(-0.9, 0.9, 1e-6), (0.9, 0.999, 1e-5)],
    ids=["central", "edge"],
)
def test_erfinv(low, high, rtol, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(23).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [erfinv(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=0.0),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, rtol",
    [
        (-3.0, 0.0, 3e-6),
        (0.0, 0.47, 1e-6),
        (0.47, 4.0, 3e-6),
        (4.0, 27.0, 1e-6),
        # erfcx decays like 1 / (y sqrt(pi)) rather than vanishing, so the far tail is
        # no harder than the near one once the asymptotic rational takes over
        (27.0, 1e3, 1e-6),
    ],
    ids=["negative", "small", "moderate", "large", "far-tail"],
)
def test_erfcx(low, high, rtol, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(29).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [erfcx(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=0.0),
    )


@pytest.mark.parametrize(
    "low, high",
    [(0.47, 4.0), (4.0, 27.0), (27.0, 1e4)],
    ids=["mid-rational", "asymptotic", "far-tail"],
)
def test_erfcx_float64_precision(low, high):
    # The two upper Cody intervals are free of exp, which is what lets erfcx hold full
    # float64 accuracy where every other member of the family is capped near 1e-7. This
    # is also the test that fails if the coefficients are weak-typed to float32, since
    # MLX weak-types Python floats. float64 lives on the CPU stream, so the dispatch is
    # exercised directly rather than through a compiled function.
    x_test_value = np.linspace(low, high, 201)

    with mlx.stream(mlx.cpu):
        erfcx_fn = mlx_funcify(Erfcx())
        res = np.asarray(erfcx_fn(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(res, scipy.special.erfcx(x_test_value), rtol=1e-13)


def test_erfc_tail_is_not_truncated():
    # erfc reaches 1e-309 before it stops being representable. Deriving it from
    # 1 - erf(x) collapses it to exactly zero from x = 4, and a plain mx.exp scale
    # factor stops at x = 9.5, mx.exp having float32 range as well as float32 precision.
    x_test_value = np.array([4.0, 6.0, 9.0, 12.0, 20.0, 26.0])

    with mlx.stream(mlx.cpu):
        erfc_fn = mlx_funcify(Erfc())
        res = np.asarray(erfc_fn(mlx.array(x_test_value, dtype=mlx.float64)))

    assert (res > 0).all()
    np.testing.assert_allclose(res, scipy.special.erfc(x_test_value), rtol=1e-5)


def test_log_ndtr_via_erfcx():
    # log(erfcx(-z / sqrt(2)) / 2) - z**2 / 2 is the stable form for the log of the
    # Gaussian cdf in the left tail, and the clearest reason erfcx exists as its own op:
    # it stays finite for as long as the result is representable, where the erfc form
    # below underflows to -inf long before that
    z = vector("z", dtype="float64")
    lcdf = pt.switch(
        pt.lt(z, -1.0),
        pt.log(erfcx(-z / np.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-erfc(z / np.sqrt(2.0)) / 2.0),
    )
    z_test_value = np.array([2.0, 0.0, -1.0, -5.0, -10.0, -20.0, -40.0])

    compare_mlx_and_py(
        [z],
        [lcdf],
        [z_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5),
    )


def test_log_ndtr_via_erfc():
    # The same quantity through erfc alone. That form is exact as mathematics and bounded
    # in practice by erfc's own underflow, so it is worth pinning over the range where it
    # still has to agree with the erfcx form above
    x = vector("x", dtype="float64")
    x_test_value = np.array([-12.0, -8.0, -6.0, -3.0, 0.0, 3.0])

    compare_mlx_and_py(
        [x],
        [pt.log(0.5 * erfc(-x / np.sqrt(2.0)))],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5),
    )


@pytest.mark.parametrize("op", [erf, erfc, erfcx], ids=["erf", "erfc", "erfcx"])
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_erf_family_ranks(op, dtype):
    # A scalar is rank 0, and the fast path indexes a flat buffer, so it has to reshape
    # before dispatching: Metal binds a 0-d argument as a value rather than a pointer and
    # subscripting it does not compile. Matrices check that the flattening round-trips
    s = scalar("s", dtype=dtype)
    compare_mlx_and_py([s], [op(s)], [np.array(1.5, dtype=dtype)])

    m = matrix("m", dtype=dtype)
    m_test_value = np.random.default_rng(37).uniform(-3.0, 5.0, (4, 7)).astype(dtype)
    compare_mlx_and_py(
        [m],
        [op(m)],
        [m_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-7),
    )


@pytest.mark.parametrize("op", [erf, erfc, erfcx], ids=["erf", "erfc", "erfcx"])
def test_erf_family_edge_cases(op):
    # Both infinities, both signed zeros, and nan. erfcx(-inf) is +inf rather than a
    # finite limit, so this also pins the reflection erfcx(-y) = 2 exp(y**2) - erfcx(y)
    x = vector("x", dtype="float64")
    x_test_value = np.array([0.0, -0.0, np.inf, -np.inf, np.nan])

    compare_mlx_and_py([x], [op(x)], [x_test_value])


@pytest.mark.parametrize("op", [erf, erfc, erfcx], ids=["erf", "erfc", "erfcx"])
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_erf_family_grad(op, dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(31).uniform(-3.0, 3.0, 51).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.grad(op(x).sum(), x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-7),
    )


# float32 and float64 take deliberately different paths through both dispatches, so
# every range is checked at both precisions with tolerances the path actually supports
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        # gammaln has zeros at 1 and 2 and a minimum of -0.12 in between, so the
        # moderate and tiny ranges need an absolute tolerance to stay meaningful
        (0.5, 20.0, {"float32": (1e-5, 1e-4), "float64": (1e-11, 1e-11)}),
        (1e-3, 0.5, {"float32": (1e-5, 1e-4), "float64": (1e-11, 1e-11)}),
        (20.0, 1e4, {"float32": (1e-5, 0.0), "float64": (1e-12, 0.0)}),
        # Negative arguments go through the log|sin(pi x)| reflection formula, which
        # loses precision to cancellation near the poles. mx.sin is float32-accurate
        # whatever the dtype, so float64 gains nothing here
        (-5.4, -0.6, {"float32": (1e-3, 1e-3), "float64": (1e-3, 1e-3)}),
    ],
    ids=["moderate", "tiny", "large", "negative"],
)
def test_gammaln(low, high, tolerances, dtype):
    rtol, atol = tolerances[dtype]
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(11).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.gammaln(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_gammaln_edge_cases(dtype):
    x = vector("x", dtype=dtype)
    # Poles at the non-positive integers, plus inf and nan propagation. The finite
    # entries keep the test from passing on an implementation that returns inf for
    # everything
    x_test_value = np.array([0.0, -1.0, -2.0, np.inf, np.nan, 0.5, 4.0], dtype=dtype)

    compare_mlx_and_py([x], [pt.gammaln(x)], [x_test_value])


@pytest.mark.parametrize(
    "low, high",
    [
        (0.5, 20.0),
        # Below 0.5 the argument must be shifted up by the recurrence rather than
        # reflected: reflection would route it through mx.sin, which is float32-accurate
        # whatever the input dtype, and that alone costs seven orders of magnitude
        (1e-3, 0.5),
    ],
    ids=["above-half", "below-half"],
)
def test_gammaln_float64_precision(low, high):
    # MLX weak-types Python floats to float32, which would silently pin the Lanczos
    # coefficients (and so the whole result) to float32 accuracy. float64 lives on
    # the CPU stream, so the dispatch is exercised directly rather than through a
    # compiled function.
    x_test_value = np.linspace(low, high, 201)

    with mlx.stream(mlx.cpu):
        gammaln = mlx_funcify(GammaLn())
        res = np.asarray(gammaln(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(
        res, scipy.special.gammaln(x_test_value), rtol=1e-13, atol=1e-14
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_gammaln_grad(dtype):
    # d/dx gammaln is psi, which is why the two dispatches ship together
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(13).uniform(0.1, 20.0, 51).astype(dtype)

    compare_mlx_and_py([x], [pt.grad(pt.gammaln(x).sum(), x)], [x_test_value])


@pytest.mark.parametrize("op", [pt.gammaln, pt.psi], ids=["gammaln", "psi"])
def test_special_half_precision(op):
    # float16 has too few mantissa bits for the series coefficients, so the dispatch
    # widens to float32 internally. Evaluating the series in float16 throughout is
    # about a hundred times less accurate, which this tolerance rules out
    x = vector("x", dtype="float16")
    x_test_value = np.array([0.2, 0.5, 1.5, 3.0, 7.0, 12.0], dtype="float16")

    _, [res] = compare_mlx_and_py(
        [x],
        [op(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-3),
    )
    # the widening is internal; the result comes back at the dtype it went in as
    assert np.asarray(res).dtype == "float16"


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        # psi crosses zero at 1.4616, so this range needs an absolute tolerance
        (0.5, 20.0, {"float32": (1e-5, 1e-4), "float64": (1e-12, 1e-12)}),
        (1e-3, 0.5, {"float32": (1e-5, 0.0), "float64": (1e-12, 0.0)}),
        (20.0, 1e4, {"float32": (1e-5, 0.0), "float64": (1e-12, 0.0)}),
        # Negative arguments pick up the pi*cot(pi x) reflection term, which loses
        # precision to cancellation near the poles. mx.cos and mx.sin are
        # float32-accurate whatever the dtype, so float64 gains nothing here
        (-5.85, -5.15, {"float32": (1e-4, 1e-4), "float64": (1e-4, 1e-4)}),
    ],
    ids=["moderate", "tiny", "large", "negative"],
)
def test_psi(low, high, tolerances, dtype):
    rtol, atol = tolerances[dtype]
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(12).uniform(low, high, 101).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.psi(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_psi_edge_cases(dtype):
    # Negative integers are left out: PyTensor's own C implementation returns inf
    # there while scipy returns nan, so there is no agreed reference to compare to.
    # The finite entries keep the poles from being the only thing asserted
    x_test_value = np.array([0.0, np.inf, np.nan, 0.5, 4.0], dtype=dtype)
    x = vector("x", dtype=dtype)

    compare_mlx_and_py([x], [pt.psi(x)], [x_test_value])


@pytest.mark.parametrize(
    "low, high", [(0.5, 20.0), (1e-3, 0.5)], ids=["above-half", "below-half"]
)
def test_psi_float64_precision(low, high):
    # Guards the same weak-typing trap as test_gammaln_float64_precision
    x_test_value = np.linspace(low, high, 201)

    with mlx.stream(mlx.cpu):
        psi = mlx_funcify(Psi())
        res = np.asarray(psi(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(
        res, scipy.special.psi(x_test_value), rtol=1e-12, atol=1e-14
    )


def test_log1mexp():
    x = vector("x")
    out = log1mexp(x)

    compare_mlx_and_py([x], [out], [[-1.0, -0.75, -0.5, -0.25]])


def test_nnet():
    x = vector("x")
    x_test_value = np.r_[1.0, 2.0].astype(config.floatX)

    out = sigmoid(x)
    compare_mlx_and_py([x], [out], [x_test_value])

    out = softplus(x)
    compare_mlx_and_py([x], [out], [x_test_value])
