import os
from functools import partial

import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.scalar.basic import Second
from pytensor.scalar.math import BetaInc, Erfc, Erfcx, GammaInc, GammaLn, Psi
from pytensor.tensor.math import log1mexp, sigmoid, softplus
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.math import _metal_math_call


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
        # Just below zero the reflection argument has to be reduced exactly, which
        # subtracting floor(x) does not do once x is far smaller than one
        (-1e-3, -1e-7, {"float32": (1e-5, 0.0), "float64": (1e-11, 0.0)}),
    ],
    ids=["moderate", "tiny", "large", "negative", "negative-near-zero"],
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


@pytest.mark.skipif(
    not mlx.metal.is_available() or os.environ.get("PYTENSOR_MLX_SKIP_GPU") == "1",
    reason="needs a GPU that can run kernels; set PYTENSOR_MLX_SKIP_GPU=1 where it cannot",
)
@pytest.mark.parametrize(
    "dtype, tolerance",
    [("float32", 1e-5), ("float16", 1e-3), ("bfloat16", 8e-3)],
    ids=["float32", "float16", "bfloat16"],
)
@pytest.mark.parametrize(
    "scalar_op, name", [(GammaLn, "gammaln"), (Psi, "psi")], ids=["gammaln", "psi"]
)
def test_gammaln_psi_paths_agree(scalar_op, name, dtype, tolerance):
    # Both functions are implemented twice: a Metal kernel, taken for single and half
    # precision on the GPU stream, and the vectorized fallback taken everywhere else --
    # which is every float64 graph, since Metal has no float64 at all. Only the fallback
    # runs on CI, so this is the only thing standing between the kernel and a silent drift.
    #
    # The uniform sample never lands near a pole, so points within 1e-1 and 1e-3 of the
    # first few are appended, and -1e-6 besides, where reducing the reflection argument
    # by hand is inexact. The edges are where the kernel handles infinities and poles
    # separately, and a 0-d input has to survive the kernel flattening its buffer.
    poles = -np.array([0.0, 1.0, 2.0, 5.0])
    offsets = np.array([1e-1, 1e-3])
    near_poles = np.concatenate(
        [
            (poles[:, None] + offsets).ravel(),
            (poles[:, None] - offsets).ravel(),
            [-1e-6],
        ]
    )
    x_test_value = np.concatenate(
        [np.random.default_rng(43).uniform(-9.9, 30.0, 1001), near_poles]
    ).astype("float32")
    edges = np.array(
        [0.0, -0.0, -1.0, -2.0, np.inf, -np.inf, np.nan, 0.5, 4.0, -1.5, 1e-30],
        dtype="float32",
    )
    scalar_value = np.float32(2.5)

    # NumPy has no bfloat16, so inputs are cast in MLX and results come back as float32
    def as_input(value):
        return mlx.array(value).astype(getattr(mlx, dtype))

    def as_result(value):
        return np.asarray(value.astype(mlx.float32))

    with mlx.stream(mlx.gpu):
        # Without this the test compares the fallback against itself and passes whatever
        # the kernel does
        kernel = _metal_math_call(name, as_input(x_test_value))
        assert kernel is not None, (
            "the Metal kernel did not engage, so this comparison proves nothing"
        )
        metal = as_result(kernel)
        metal_edges = as_result(_metal_math_call(name, as_input(edges)))
        metal_scalar = as_result(_metal_math_call(name, as_input(scalar_value)))
    with mlx.stream(mlx.cpu):
        fallback = mlx_funcify(scalar_op())
        vectorized = as_result(fallback(as_input(x_test_value)))
        vectorized_edges = as_result(fallback(as_input(edges)))
        vectorized_scalar = as_result(fallback(as_input(scalar_value)))

    # Both compute in float32 and round once to the dtype. At float32 they disagree by
    # 1.6e-6 at worst, and at half precision by at most one unit in the last place, so
    # each bound sits a little above that spread. The absolute term carries the zeros
    np.testing.assert_allclose(metal, vectorized, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(metal_edges, vectorized_edges, rtol=tolerance)
    assert metal_scalar.shape == ()
    np.testing.assert_allclose(metal_scalar, vectorized_scalar, rtol=tolerance)


@pytest.mark.skipif(
    not mlx.metal.is_available() or os.environ.get("PYTENSOR_MLX_SKIP_GPU") == "1",
    reason="needs a GPU that can run kernels; set PYTENSOR_MLX_SKIP_GPU=1 where it cannot",
)
@pytest.mark.parametrize(
    "scalar_op, name", [(GammaLn, "gammaln"), (Psi, "psi")], ids=["gammaln", "psi"]
)
def test_gammaln_psi_dispatch_reaches_kernel(scalar_op, name):
    # test_gammaln_psi_paths_agree calls the kernel directly, and everything that goes
    # through the registered dispatch runs on the CPU stream the conftest pins. So
    # removing the kernel from the dispatch would leave the whole suite green while every
    # GPU graph silently took the path that costs several launches.
    x_test_value = np.random.default_rng(7).uniform(-9.9, 30.0, 1001).astype("float32")

    with mlx.stream(mlx.gpu):
        dispatched = np.asarray(mlx_funcify(scalar_op())(mlx.array(x_test_value)))
        kernel = _metal_math_call(name, mlx.array(x_test_value))
        assert kernel is not None, "the Metal kernel refused this input"
        kernel = np.asarray(kernel)
    with mlx.stream(mlx.cpu):
        vectorized = np.asarray(mlx_funcify(scalar_op())(mlx.array(x_test_value)))

    # Bit equality is only a discriminator while the two paths actually differ; if they
    # ever agree exactly this test would pass without checking anything
    assert not np.array_equal(kernel, vectorized), (
        "the two paths agree bitwise, so this test cannot tell them apart"
    )
    np.testing.assert_array_equal(dispatched, kernel)


@pytest.mark.parametrize(
    "scalar_op, n_inputs",
    [
        (GammaLn, 1),
        (Psi, 1),
        (Erfc, 1),
        (Erfcx, 1),
        (GammaInc, 2),
        (BetaInc, 3),
        (Second, 2),
    ],
    ids=["gammaln", "psi", "erfc", "erfcx", "gammainc", "betainc", "second"],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_dispatch_does_not_copy_its_input(scalar_op, n_inputs, dtype, tmp_path):
    # Since MLX 0.32, mx.array copies an input that is already an mx.array, and the copy
    # survives mx.compile as an AsType node of its own. Every value lies in (0, 1), where
    # each of these functions is defined
    x_value = mlx.array(np.linspace(0.1, 0.9, 7, dtype=dtype))
    mlx.eval(x_value)
    graph = tmp_path / "graph.dot"

    # The assertion below searches MLX's graph dump for a primitive name, so first
    # confirm the dump still spells a cast that way
    mlx.export_to_dot(str(graph), mlx.compile(lambda x: x.astype(mlx.float16))(x_value))
    assert "AsType" in graph.read_text()

    outputs = mlx.compile(mlx_funcify(scalar_op()))(*[x_value] * n_inputs)
    mlx.export_to_dot(str(graph), outputs)

    assert "AsType" not in graph.read_text()


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
