import numpy as np
import pytest

from pytensor import tensor as pt
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.mlx import MLXLinker
from pytensor.tensor import blas as pt_blas
from pytensor.tensor.type import tensor3
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_mlx_Gemv_static_scales():
    y = pt.vector("y", dtype=config.floatX)
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)

    out = pt_blas.Gemv(inplace=False)(
        y,
        np.asarray(0.5, dtype=config.floatX),
        A,
        x,
        np.asarray(2.0, dtype=config.floatX),
    )

    rng = np.random.default_rng(sum(map(ord, "test_mlx_Gemv_static_scales")))
    y_test = rng.normal(size=(3,)).astype(config.floatX)
    A_test = rng.normal(size=(3, 2)).astype(config.floatX)
    x_test = rng.normal(size=(2,)).astype(config.floatX)

    compare_mlx_and_py(
        [y, A, x],
        [out],
        [y_test, A_test, x_test],
    )


def test_mlx_Gemv_symbolic_scales():
    y = pt.vector("y", dtype=config.floatX)
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    alpha = pt.scalar("alpha", dtype=config.floatX)
    beta = pt.scalar("beta", dtype=config.floatX)

    out = pt_blas.Gemv(inplace=False)(y, alpha, A, x, beta)

    rng = np.random.default_rng(sum(map(ord, "test_mlx_Gemv_symbolic_scales")))
    y_test = rng.normal(size=(3,)).astype(config.floatX)
    A_test = rng.normal(size=(3, 2)).astype(config.floatX)
    x_test = rng.normal(size=(2,)).astype(config.floatX)
    alpha_test = np.asarray(0.5, dtype=config.floatX)
    beta_test = np.asarray(2.0, dtype=config.floatX)

    compare_mlx_and_py(
        [y, alpha, A, x, beta],
        [out],
        [y_test, alpha_test, A_test, x_test, beta_test],
    )


def test_mlx_Ger_static_scale():
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)

    out = pt_blas.Ger(inplace=False)(A, np.asarray(0.5, dtype=config.floatX), x, y)

    rng = np.random.default_rng(sum(map(ord, "test_mlx_Ger_static_scale")))
    A_test = rng.normal(size=(3, 2)).astype(config.floatX)
    x_test = rng.normal(size=(3,)).astype(config.floatX)
    y_test = rng.normal(size=(2,)).astype(config.floatX)

    compare_mlx_and_py(
        [A, x, y],
        [out],
        [A_test, x_test, y_test],
    )


def test_mlx_Ger_symbolic_scale():
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)
    alpha = pt.scalar("alpha", dtype=config.floatX)

    out = pt_blas.Ger(inplace=False)(A, alpha, x, y)

    rng = np.random.default_rng(sum(map(ord, "test_mlx_Ger_symbolic_scale")))
    A_test = rng.normal(size=(3, 2)).astype(config.floatX)
    x_test = rng.normal(size=(3,)).astype(config.floatX)
    y_test = rng.normal(size=(2,)).astype(config.floatX)
    alpha_test = np.asarray(0.5, dtype=config.floatX)

    compare_mlx_and_py(
        [A, alpha, x, y],
        [out],
        [A_test, alpha_test, x_test, y_test],
    )


def test_mlx_BatchedDot():
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

    opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
    mlx_mode = Mode(MLXLinker(), opts)
    pytensor_mlx_fn, _ = compare_mlx_and_py(
        [a, b], [out], [a_test_value, b_test_value], mlx_mode=mlx_mode
    )

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [a_test_value[:-1], b_test_value]
    with pytest.raises(TypeError):
        pytensor_mlx_fn(*inputs)


@pytest.mark.parametrize(
    "alpha, beta",
    [(None, None), (2.0, None), (None, 3.0), (2.0, 3.0)],
    ids=["plain", "alpha", "beta", "alpha_beta"],
)
def test_mlx_Gemm(alpha, beta):
    # Gemm is what local_add_dot_to_gemm folds `beta * C + alpha * (A @ B)` into, so the
    # scales arrive as constants. They reach MLX's fused kernel only in that form.
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
    compare_mlx_and_py([z, x, y], [out], test_values)


def test_mlx_Gemm_runtime_scales():
    # With alpha and beta as graph variables there is no constant to hand the fused
    # kernel, so this covers the arithmetic path the dispatch falls back to.
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
    compare_mlx_and_py(
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


def test_mlx_Ger():
    # local_gemm_to_ger folds a rank-1 `A + alpha * outer(x, y)` into Ger.
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)
    out = pt_blas.Ger(inplace=False)(A, np.asarray(2.0, dtype=config.floatX), x, y)

    rng = np.random.default_rng(418)
    test_values = [
        rng.normal(size=shape).astype(config.floatX) for shape in ((4, 6), (4,), (6,))
    ]
    compare_mlx_and_py([A, x, y], [out], test_values)


def test_mlx_Ger_runtime_alpha():
    A = pt.matrix("A", dtype=config.floatX)
    x = pt.vector("x", dtype=config.floatX)
    y = pt.vector("y", dtype=config.floatX)
    alpha = pt.scalar("alpha", dtype=config.floatX)
    out = pt_blas.Ger(inplace=False)(A, alpha, x, y)

    rng = np.random.default_rng(418)
    A_val, x_val, y_val = (
        rng.normal(size=shape).astype(config.floatX) for shape in ((4, 6), (4,), (6,))
    )
    compare_mlx_and_py(
        [A, alpha, x, y],
        [out],
        [A_val, np.asarray(2.0, dtype=config.floatX), x_val, y_val],
    )
