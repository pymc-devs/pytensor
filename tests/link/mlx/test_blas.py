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

    out = pt_blas.gemv_no_inplace(
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

    out = pt_blas.gemv_no_inplace(y, alpha, A, x, beta)

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
