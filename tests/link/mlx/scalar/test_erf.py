import os
from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.scalar.math import Erfc, Erfcx
from pytensor.tensor.math import erf, erfc, erfcx, erfinv
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


# erfc is exp(-y**2) * erfcx(y), so the exponential sets the float32 tolerances and
# squaring the argument amplifies its error in proportion to y**2. float64 goes through
# mx.power instead, which is genuine, so it holds full precision across every range.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        (-3.0, 0.0, {"float32": 3e-7, "float64": 1e-14}),
        (0.0, 1.0, {"float32": 1e-6, "float64": 1e-14}),
        (1.0, 4.0, {"float32": 1e-5, "float64": 1e-14}),
        (4.0, 9.0, {"float32": 1e-4, "float64": 1e-14}),
    ],
    ids=["negative", "small", "moderate", "tail"],
)
def test_erfc(low, high, tolerances, dtype):
    rtol = tolerances[dtype]
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


# erfcx decays like 1 / (y sqrt(pi)) rather than vanishing, so the far tail is no harder
# than the near one once the asymptotic rational takes over
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high, tolerances",
    [
        (-3.0, 0.0, {"float32": 3e-6, "float64": 1e-14}),
        (0.0, 0.47, {"float32": 1e-6, "float64": 1e-14}),
        (0.47, 4.0, {"float32": 3e-6, "float64": 1e-14}),
        (4.0, 27.0, {"float32": 1e-6, "float64": 1e-14}),
        (27.0, 1e3, {"float32": 1e-6, "float64": 1e-14}),
    ],
    ids=["negative", "small", "moderate", "large", "far-tail"],
)
def test_erfcx(low, high, tolerances, dtype):
    rtol = tolerances[dtype]
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
    # The test that fails if the coefficients are weak-typed to float32, since MLX
    # weak-types Python floats. float64 lives on the CPU stream, so the dispatch is
    # exercised directly rather than through a compiled function.
    x_test_value = np.linspace(low, high, 201)

    with mlx.stream(mlx.cpu):
        erfcx_fn = mlx_funcify(Erfcx())
        res = np.asarray(erfcx_fn(mlx.array(x_test_value, dtype=mlx.float64)))

    np.testing.assert_allclose(res, scipy.special.erfcx(x_test_value), rtol=1e-13)


def test_erfc_tail_is_not_truncated():
    # erfc reaches 1e-309 before it stops being representable. Deriving it from
    # 1 - erf(x) collapses it to exactly zero from x = 4, and a plain mx.exp scale factor
    # stops at x = 9.5, mx.exp having float32 range as well as float32 precision.
    x_test_value = np.array([4.0, 6.0, 9.0, 12.0, 20.0, 26.0])

    with mlx.stream(mlx.cpu):
        erfc_fn = mlx_funcify(Erfc())
        res = np.asarray(erfc_fn(mlx.array(x_test_value, dtype=mlx.float64)))

    assert (res > 0).all()
    np.testing.assert_allclose(res, scipy.special.erfc(x_test_value), rtol=1e-13)


def test_erfc_float32_subnormal_tail():
    # erfc is still representable as a float32 subnormal past x = 9.5, where mx.exp
    # flushes to zero and would send log(erfc) to -inf against a true value near -93.
    x_test_value = np.array([9.3, 9.5], dtype="float32")

    with mlx.stream(mlx.cpu):
        res = np.asarray(mlx_funcify(Erfc())(mlx.array(x_test_value)))
        # x = 10 sits on the smallest float32 subnormal, carrying a bit or two, and by
        # x = 10.5 erfc is 7e-50 and genuinely underflows -- zero is the right answer
        bottom = np.asarray(
            mlx_funcify(Erfc())(mlx.array(np.array([10.0, 10.5], dtype="float32")))
        )

    assert (res > 0).all()
    np.testing.assert_allclose(
        res, scipy.special.erfc(x_test_value.astype("float64")), rtol=1e-5
    )
    assert bottom[0] > 0.0
    assert bottom[1] == 0.0


@pytest.mark.skipif(
    not mlx.metal.is_available() or os.environ.get("PYTENSOR_MLX_SKIP_GPU") == "1",
    reason="needs a GPU that can run kernels; set PYTENSOR_MLX_SKIP_GPU=1 where it cannot",
)
@pytest.mark.parametrize("op", [Erfc, Erfcx], ids=["erfc", "erfcx"])
def test_erf_family_paths_agree(op):
    # erfc and erfcx are implemented twice: a Metal kernel, taken for float32 on the GPU
    # stream, and the vectorized fallback taken everywhere else. The two share generated
    # coefficients but not their branch structure or their negative-argument reflection,
    # so this is what keeps them from drifting apart. The range spans every branch of
    # both, stopping short of where float32 underflows and the comparison goes trivial.
    #
    # This is the only test in the suite that *executes* on the GPU stream. Virtualized
    # machines abort outright when asked to, which is why conftest pins the CPU for the
    # session and why test_mlx_float64_downcast_on_gpu_warns opens that stream without
    # running a kernel on it. CI sets PYTENSOR_MLX_SKIP_GPU; a developer machine runs it.
    x_test_value = np.random.default_rng(41).uniform(-4.0, 9.0, 1001).astype("float32")

    with mlx.stream(mlx.gpu):
        metal = np.asarray(mlx_funcify(op())(mlx.array(x_test_value)))
    with mlx.stream(mlx.cpu):
        vectorized = np.asarray(mlx_funcify(op())(mlx.array(x_test_value)))

    np.testing.assert_allclose(metal, vectorized, rtol=1e-5, atol=0.0)


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
