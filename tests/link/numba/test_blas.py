import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.blas import Gemm, Gemv


pytestmark = pytest.mark.filterwarnings("error")

pytest.importorskip("numba")

floatX = config.floatX


@pytest.mark.parametrize(
    "z_shape", [(6, 5), (1, 5), (6, 1), (1, 1)], ids=["full", "row", "column", "scalar"]
)
def test_gemm_broadcasts_its_accumulator(z_shape):
    """``Gemm`` takes a ``z`` that is only broadcast against the product -- a bias row is the common
    case -- so the buffer it accumulates into is the product's shape, not ``z``'s."""
    z = pt.tensor("z", shape=z_shape, dtype=floatX)
    x = pt.tensor("x", shape=(6, 3), dtype=floatX)
    y = pt.tensor("y", shape=(3, 5), dtype=floatX)
    alpha, beta = pt.scalar("alpha", dtype=floatX), pt.scalar("beta", dtype=floatX)

    rng = np.random.default_rng(sum(map(ord, f"gemm_broadcasts {z_shape}")))
    z_np = rng.normal(size=z_shape).astype(floatX)
    x_np = rng.normal(size=(6, 3)).astype(floatX)
    y_np = rng.normal(size=(3, 5)).astype(floatX)

    fn = pytensor.function(
        [z, alpha, x, y, beta], Gemm(inplace=False)(z, alpha, x, y, beta), mode="NUMBA"
    )
    np.testing.assert_allclose(
        fn(z_np, 2.0, x_np, y_np, 0.5), 0.5 * z_np + 2.0 * (x_np @ y_np), rtol=1e-5
    )


def test_gemm_no_inplace_leaves_its_accumulator_alone():
    """The non-inplace form must not write through ``z``; only the inplace form may."""
    z = pt.tensor("z", shape=(6, 5), dtype=floatX)
    x = pt.tensor("x", shape=(6, 3), dtype=floatX)
    y = pt.tensor("y", shape=(3, 5), dtype=floatX)
    alpha, beta = pt.scalar("alpha", dtype=floatX), pt.scalar("beta", dtype=floatX)

    rng = np.random.default_rng(sum(map(ord, "gemm_no_inplace")))
    z_np = rng.normal(size=(6, 5)).astype(floatX)
    original = z_np.copy()
    x_np = rng.normal(size=(6, 3)).astype(floatX)
    y_np = rng.normal(size=(3, 5)).astype(floatX)

    fn = pytensor.function(
        [z, alpha, x, y, beta], Gemm(inplace=False)(z, alpha, x, y, beta), mode="NUMBA"
    )
    result = fn(z_np, 2.0, x_np, y_np, 0.5)
    np.testing.assert_allclose(result, 0.5 * original + 2.0 * (x_np @ y_np), rtol=1e-5)
    np.testing.assert_array_equal(z_np, original)


def test_gemm_inplace_writes_through_its_accumulator():
    """The inplace form is chosen precisely to avoid the copy, so it has to destroy ``z``."""
    rng = np.random.default_rng(sum(map(ord, "gemm_inplace")))
    z_np = rng.normal(size=(6, 5)).astype(floatX)
    x_np = rng.normal(size=(6, 3)).astype(floatX)
    y_np = rng.normal(size=(3, 5)).astype(floatX)
    expected = 0.5 * z_np + 2.0 * (x_np @ y_np)

    z = pytensor.shared(z_np)
    x = pt.tensor("x", shape=(6, 3), dtype=floatX)
    y = pt.tensor("y", shape=(3, 5), dtype=floatX)
    alpha, beta = pt.scalar("alpha", dtype=floatX), pt.scalar("beta", dtype=floatX)

    # An inplace op cannot be built into a graph directly; only the rewrite that introduces one
    # may, so the test has to say it knows what it is asking for.
    fn = pytensor.function(
        [alpha, x, y, beta],
        Gemm(inplace=True)(z, alpha, x, y, beta),
        mode="NUMBA",
        accept_inplace=True,
    )
    np.testing.assert_allclose(fn(2.0, x_np, y_np, 0.5), expected, rtol=1e-5)
    np.testing.assert_allclose(z.get_value(), expected, rtol=1e-5)


def test_gemv_no_inplace_leaves_its_accumulator_alone():
    """The non-inplace form must not write through ``y``; only the inplace form may."""
    y = pt.tensor("y", shape=(6,), dtype=floatX)
    A = pt.tensor("A", shape=(6, 5), dtype=floatX)
    x = pt.tensor("x", shape=(5,), dtype=floatX)
    alpha, beta = pt.scalar("alpha", dtype=floatX), pt.scalar("beta", dtype=floatX)

    rng = np.random.default_rng(sum(map(ord, "gemv_no_inplace")))
    y_np = rng.normal(size=6).astype(floatX)
    original = y_np.copy()
    A_np = rng.normal(size=(6, 5)).astype(floatX)
    x_np = rng.normal(size=5).astype(floatX)

    fn = pytensor.function(
        [y, alpha, A, x, beta], Gemv(inplace=False)(y, alpha, A, x, beta), mode="NUMBA"
    )
    result = fn(y_np, 2.0, A_np, x_np, 0.5)
    np.testing.assert_allclose(result, 0.5 * original + 2.0 * (A_np @ x_np), rtol=1e-5)
    np.testing.assert_array_equal(y_np, original)


def test_gemv_inplace_writes_through_its_accumulator():
    """The inplace form is chosen precisely to avoid the copy, so it has to destroy ``y``."""
    rng = np.random.default_rng(sum(map(ord, "gemv_inplace")))
    y_np = rng.normal(size=6).astype(floatX)
    A_np = rng.normal(size=(6, 5)).astype(floatX)
    x_np = rng.normal(size=5).astype(floatX)
    expected = 0.5 * y_np + 2.0 * (A_np @ x_np)

    y = pytensor.shared(y_np)
    A = pt.tensor("A", shape=(6, 5), dtype=floatX)
    x = pt.tensor("x", shape=(5,), dtype=floatX)
    alpha, beta = pt.scalar("alpha", dtype=floatX), pt.scalar("beta", dtype=floatX)

    fn = pytensor.function(
        [alpha, A, x, beta],
        Gemv(inplace=True)(y, alpha, A, x, beta),
        mode="NUMBA",
        accept_inplace=True,
    )
    np.testing.assert_allclose(fn(2.0, A_np, x_np, 0.5), expected, rtol=1e-5)
    np.testing.assert_allclose(y.get_value(), expected, rtol=1e-5)
