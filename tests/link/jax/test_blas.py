import numpy as np
import pytest

from pytensor import tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.jax import JAXLinker
from pytensor.tensor import blas as pt_blas
from pytensor.tensor.type import tensor3
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a_test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b_test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = pt_blas.BatchedDot()(a, b)
    compare_jax_and_py([a, b], [out], [a_test_value, b_test_value])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [a_test_value[:-1], b_test_value]
    opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
    jax_mode = Mode(JAXLinker(), opts)
    pytensor_jax_fn = function([a, b], [out], mode=jax_mode)
    with pytest.raises(TypeError):
        pytensor_jax_fn(*inputs)


@pytest.mark.parametrize(
    "alpha, beta",
    [(None, None), (2.0, None), (None, 3.0), (2.0, 3.0)],
    ids=["plain", "alpha", "beta", "alpha_beta"],
)
def test_jax_Gemm(alpha, beta):
    # Gemm is what local_add_dot_to_gemm folds `beta * C + alpha * (A @ B)` into, so the
    # scales arrive as constants. JAX has no fused kernel to reach; XLA folds them
    # into the dot itself.
    z = pt.matrix("z", dtype=config.floatX)
    x = pt.matrix("x", dtype=config.floatX)
    y = pt.matrix("y", dtype=config.floatX)
    one = np.asarray(1.0, dtype=config.floatX)

    out = pt_blas.Gemm(inplace=False)(
        z,
        one if alpha is None else np.asarray(alpha, dtype=config.floatX),
        x,
        y,
        one if beta is None else np.asarray(beta, dtype=config.floatX),
    )

    rng = np.random.default_rng(418)
    test_values = [
        rng.normal(size=shape).astype(config.floatX)
        for shape in ((4, 6), (4, 5), (5, 6))
    ]
    compare_jax_and_py([z, x, y], [out], test_values)


def test_jax_Gemm_runtime_scales():
    z = pt.matrix("z", dtype=config.floatX)
    x = pt.matrix("x", dtype=config.floatX)
    y = pt.matrix("y", dtype=config.floatX)
    alpha = pt.scalar("alpha", dtype=config.floatX)
    beta = pt.scalar("beta", dtype=config.floatX)

    out = pt_blas.Gemm(inplace=False)(z, alpha, x, y, beta)

    rng = np.random.default_rng(418)
    test_values = [
        rng.normal(size=shape).astype(config.floatX)
        for shape in ((4, 6), (4, 5), (5, 6))
    ]
    compare_jax_and_py(
        [z, alpha, x, y, beta],
        [out],
        [
            test_values[0],
            np.asarray(2.0, dtype=config.floatX),
            test_values[1],
            test_values[2],
            np.asarray(3.0, dtype=config.floatX),
        ],
    )


def test_jax_Ger():
    # local_gemm_to_ger folds a rank-1 `A + alpha * outer(x, y)` into Ger.
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)
    out = pt_blas.Ger(inplace=False)(A, np.asarray(2.0, dtype=config.floatX), x, y)

    rng = np.random.default_rng(418)
    test_values = [
        rng.normal(size=shape).astype(config.floatX) for shape in ((4, 6), (4,), (6,))
    ]
    compare_jax_and_py([A, x, y], [out], test_values)


def test_jax_Ger_runtime_alpha():
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)
    alpha = pt.scalar("alpha", dtype=config.floatX)
    out = pt_blas.Ger(inplace=False)(A, alpha, x, y)

    rng = np.random.default_rng(418)
    A_val, x_val, y_val = (
        rng.normal(size=shape).astype(config.floatX) for shape in ((4, 6), (4,), (6,))
    )
    compare_jax_and_py(
        [A, alpha, x, y],
        [out],
        [A_val, np.asarray(2.0, dtype=config.floatX), x_val, y_val],
    )
