from functools import partial

import numpy as np
import pytest
import scipy.special
import scipy.stats

import pytensor.tensor as pt
from pytensor.scalar.math import Erfcinv
from pytensor.tensor.type import vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


# erfcinv is sampled log-spaced rather than uniformly on purpose. The whole difficulty
# is at small argument, where erfinv(1 - x) cancels; a uniform draw over (0, 2) would
# put almost no points below 1e-3 and would pass against the implementation this
# replaces.
#
# The tolerances are per-dtype because erfcinv, alone in this family, has no float32
# ceiling to inherit: AS241 needs only log, sqrt and arithmetic, all of which MLX
# computes at the full working precision. One shared float32-sized tolerance would let a
# regression to mx.erfinv accuracy pass unnoticed, which is the failure this op exists
# to avoid. An absolute tolerance carries the upper range, where erfcinv crosses zero at
# x = 1.
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
@pytest.mark.parametrize(
    "low, high",
    [
        (1e-12, 1e-6),
        (1e-6, 1e-3),
        (1e-3, 0.5),
        (1.0, 2.0 - 1e-9),
    ],
    ids=["tiny", "small", "moderate", "upper"],
)
def test_erfcinv(low, high, dtype):
    rtol, atol = {"float32": (3e-6, 3e-6), "float64": (3e-15, 1e-14)}[dtype]
    rng = np.random.default_rng(43)
    x = vector("x", dtype=dtype)
    x_test_value = np.exp(rng.uniform(np.log(low), np.log(high), 101)).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.erfcinv(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
    )


@pytest.mark.parametrize(
    "low, high",
    [(1e-300, 1e-12), (1e-12, 1e-3), (1e-3, 1.0), (1.0, 2.0 - 1e-12)],
    ids=["far-tail", "tail", "lower", "upper"],
)
def test_erfcinv_float64_precision(low, high):
    # Nothing in this kernel needs exp, sin or cos -- only log, sqrt and arithmetic, all
    # of which MLX computes at the full working precision -- so unlike the rest of the
    # family it carries no float32 ceiling and is held to 1 ulp. This is also the test
    # that fails if the coefficients weak-type to float32, or if anyone reintroduces
    # mx.erfinv, which is float32 whatever it is handed.
    x_test_value = np.exp(np.linspace(np.log(low), np.log(high), 401))

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Erfcinv())(mlx.array(x_test_value, dtype=mlx.float64))
        )

    np.testing.assert_allclose(res, scipy.special.erfcinv(x_test_value), rtol=1e-14)


def test_erfcinv_edge_cases():
    # The poles at 0 and 2, the exact zero at 1, and both out-of-domain sides. The
    # out-of-domain entries also cover the compiled path, which aborts outright if the
    # NaN they select is written as a literal rather than computed.
    x = vector("x", dtype="float64")
    x_test_value = np.array([0.0, 1.0, 2.0, -0.5, 2.5, np.nan])

    compare_mlx_and_py([x], [pt.erfcinv(x)], [x_test_value])


def test_erfcinv_float32_subnormal():
    # AS241 needs -log(x / 2). Forming that product flushes the smallest float32
    # subnormals to exactly zero, which sends the tail variable to its saturation point
    # and returned 29.9 for an argument whose answer is 10.0. Folding the halving into
    # the logarithm as ln2 - log(x) keeps it exact.
    x_test_value = np.array([1.4e-45, 2.8e-45, 1e-44, 1e-40], dtype="float32")

    with mlx.stream(mlx.cpu):
        res = np.asarray(mlx_funcify(Erfcinv())(mlx.array(x_test_value)))

    np.testing.assert_allclose(
        res, scipy.special.erfcinv(x_test_value.astype("float64")), rtol=1e-6
    )


def test_erfcinv_upper_pole():
    # The other pole. Every range above samples log-spaced in x, which cannot approach 2
    # densely, so nothing else exercises the tail_x = 2 - x branch where the subtraction
    # is the whole computation -- the mirror of the small-x case this op exists for, and
    # where erfinv(1 - x) also fails.
    x_test_value = 2.0 - np.exp(np.linspace(np.log(1e-15), np.log(1e-3), 401))

    with mlx.stream(mlx.cpu):
        res = np.asarray(
            mlx_funcify(Erfcinv())(mlx.array(x_test_value, dtype=mlx.float64))
        )

    np.testing.assert_allclose(res, scipy.special.erfcinv(x_test_value), rtol=1e-14)


def test_erfcinv_half_precision():
    # float16 has too few mantissa bits for the AS241 coefficients, so the dispatch
    # widens to float32 internally and casts back. 1.0 is left out because erfcinv is
    # exactly zero there and a relative tolerance says nothing.
    x = vector("x", dtype="float16")
    x_test_value = np.array([1e-4, 0.01, 0.5, 1.5, 1.99], dtype="float16")

    _, [res] = compare_mlx_and_py(
        [x],
        [pt.erfcinv(x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-3),
    )
    # the widening is internal; the result comes back at the dtype it went in as
    assert np.asarray(res).dtype == "float16"


def test_normal_icdf():
    # mu + sigma * -sqrt(2) * erfcinv(2 * q) is Normal.icdf, and pymc's ErfcTransform
    # inverts through the same op, so any model applying erfc to a variable needs it
    q = vector("q", dtype="float64")
    icdf = -np.sqrt(2.0) * pt.erfcinv(2.0 * q)
    q_test_value = np.array([1e-12, 1e-6, 1e-3, 0.025, 0.5, 0.975, 1.0 - 1e-9])

    _, [res] = compare_mlx_and_py(
        [q],
        [icdf],
        [q_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-10, atol=1e-10),
    )
    np.testing.assert_allclose(
        np.asarray(res), scipy.stats.norm.ppf(q_test_value), rtol=1e-10
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids=str)
def test_erfcinv_grad(dtype):
    x = vector("x", dtype=dtype)
    x_test_value = np.random.default_rng(47).uniform(0.05, 1.95, 51).astype(dtype)

    compare_mlx_and_py(
        [x],
        [pt.grad(pt.erfcinv(x).sum(), x)],
        [x_test_value],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-6),
    )
