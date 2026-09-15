from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.scalar.math import GammaLn, Psi
from pytensor.tensor.math import log1mexp, sigmoid, softplus
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


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
        # Negative arguments go through the log|sin(pi x)| reflection, which takes its
        # sine from the tangent by the half-angle identity. mx.tan is genuine at float64
        # where mx.sin is a float32 kernel whatever it is handed, so float64 holds here
        # rather than sharing single precision's ceiling
        (-5.4, -0.6, {"float32": (1e-3, 1e-3), "float64": (1e-11, 1e-11)}),
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
        # The reflection's pi * cot(pi x) is taken as a reciprocal tangent rather than
        # a ratio of cosine to sine, both of which are float32 kernels at any dtype. The
        # absolute tolerance carries float32, where psi's own zeros sit near its poles
        (-5.85, -5.15, {"float32": (1e-2, 1e-4), "float64": (1e-10, 1e-12)}),
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
    # The negative integers are left out. psi has a pole at each of them and the three
    # implementations disagree on what to report: scipy says nan, PyTensor's C code says
    # inf, and this dispatch says -inf, because reducing the cotangent's argument by the
    # nearest integer puts the tangent on a true zero there. All three are infinite or
    # undefined rather than a plausible finite number, which is what matters; there is
    # just no agreed reference to assert against. The finite entries keep the remaining
    # poles from being the only thing asserted
    x_test_value = np.array([0.0, np.inf, np.nan, 0.5, 4.0, -1.5], dtype=dtype)
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


@pytest.mark.parametrize(
    "scalar_op, reference",
    [(GammaLn, scipy.special.gammaln), (Psi, scipy.special.digamma)],
    ids=["gammaln", "psi"],
)
def test_reflection_near_the_poles(scalar_op, reference):
    # Both reflections divide by something that vanishes at every negative integer, and
    # both grow without bound approaching one. Every other range here samples uniformly
    # and so never lands near a pole, which is how a reflection that was 45% wrong
    # within 1e-12 of one went unnoticed. The tolerance is set by the argument
    # reduction, which loses a digit for each decade closer to the pole.
    poles = -np.array([1.0, 2.0, 5.0, 20.0])
    offsets = np.array([1e-1, 1e-3, 1e-5, 1e-7, 1e-9])
    x_test_value = np.unique(
        np.concatenate(
            [(poles[:, None] - offsets).ravel(), (poles[:, None] + offsets).ravel()]
        )
    )
    x_test_value = x_test_value[x_test_value < 0.0]

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(scalar_op())(mlx.array(x_test_value, dtype=mlx.float64))
        )

    np.testing.assert_allclose(res, reference(x_test_value), rtol=1e-5)
