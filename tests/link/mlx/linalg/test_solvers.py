import contextlib
from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


@pytest.mark.parametrize("assume_a", ["gen", "pos"])
def test_mlx_solve(assume_a):
    rng = np.random.default_rng(15)
    n = 3

    A = pt.tensor("A", shape=(n, n))
    b = pt.tensor("B", shape=(n, n))

    out = pt.linalg.solve(A, b, b_ndim=2, assume_a=assume_a)

    A_val = rng.normal(size=(n, n)).astype(config.floatX)
    A_val = A_val @ A_val.T

    b_val = rng.normal(size=(n, n)).astype(config.floatX)

    context = (
        contextlib.suppress()
        if assume_a == "gen"
        else pytest.warns(
            UserWarning, match=f"MLX solve does not support assume_a={assume_a}"
        )
    )

    with context:
        compare_mlx_and_py(
            [A, b],
            [out],
            [A_val, b_val],
            mlx_mode=mlx_mode,
            assert_fn=partial(
                np.testing.assert_allclose, atol=1e-6, rtol=1e-6, strict=True
            ),
        )


@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["full_diagonal", "unit_diagonal"]
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_mlx_SolveTriangular(lower, unit_diagonal):
    rng = np.random.default_rng(15)

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    # A diagonal far from one, so ignoring `unit_diagonal` gives a different answer
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    A_val[np.diag_indices(5)] = rng.uniform(3, 4, size=5).astype(config.floatX)
    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    out = pt.linalg.solve_triangular(
        A,
        b,
        trans=0,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    compare_mlx_and_py(
        [A, b],
        [out],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-6, rtol=1e-6, strict=True
        ),
    )


@pytest.mark.parametrize("batch_shape", [(), (3,)], ids=["core", "batched"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=["b_vec", "b_mat"])
def test_mlx_CholeskySolve(batch_shape, lower, b_ndim):
    rng = np.random.default_rng(15)
    n = 5
    b_shape = (*batch_shape, n) if b_ndim == 1 else (*batch_shape, n, 3)

    C = pt.tensor("C", shape=(*batch_shape, n, n))
    b = pt.tensor("b", shape=b_shape)

    out = pt.linalg.cho_solve((C, lower), b, b_ndim=b_ndim)

    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)
    A_val = A_val @ np.swapaxes(A_val, -1, -2) + n * np.eye(n, dtype=config.floatX)
    C_val = np.linalg.cholesky(A_val)
    if not lower:
        C_val = np.swapaxes(C_val, -1, -2).copy()

    b_val = rng.normal(size=b_shape).astype(config.floatX)

    compare_mlx_and_py(
        [C, b],
        [out],
        [C_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-6, rtol=1e-6, strict=True
        ),
    )


def test_mlx_CholeskySolve_mixed_dtypes():
    rng = np.random.default_rng(15)
    n = 5

    C = pt.tensor("C", shape=(n, n), dtype="float32")
    b = pt.tensor("b", shape=(n,), dtype="float64")

    out = pt.linalg.cho_solve((C, True), b, b_ndim=1)
    assert out.type.dtype == "float64"

    A_val = rng.normal(size=(n, n))
    A_val = A_val @ A_val.T + n * np.eye(n)
    C_val = np.linalg.cholesky(A_val).astype("float32")
    b_val = rng.normal(size=(n,))

    compare_mlx_and_py(
        [C, b],
        [out],
        [C_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-5, rtol=1e-5, strict=True
        ),
    )
