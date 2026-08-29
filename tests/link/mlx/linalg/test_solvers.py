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
    "unit_diagonal", [False, True], ids=["stored_diagonal", "unit_diagonal"]
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_mlx_SolveTriangular(lower, unit_diagonal):
    # `unit_diagonal=True` was dropped on the way to `mx.linalg.solve_triangular`,
    # which has no such argument, so the stored diagonal was used instead of ones
    # and a wrong answer was returned without raising (#2384).
    rng = np.random.default_rng(15)

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
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


def test_mlx_SolveTriangular_unit_diagonal_lu_solve():
    # Batched `lu_factor` is blocked on #2385, so this covers the core case only.
    batch_shape = ()
    # `lu_solve` packs a unit-triangular `L` into the LU array, with `U`'s
    # diagonal occupying those slots, so it relies on `unit_diagonal=True`
    # actually being honoured (#2384).
    rng = np.random.default_rng(15)
    n = 4

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    b = pt.tensor("b", shape=(*batch_shape, n))

    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)
    A_val = A_val @ np.swapaxes(A_val, -1, -2) + n * np.eye(n, dtype=config.floatX)
    b_val = rng.normal(size=(*batch_shape, n)).astype(config.floatX)

    out = pt.linalg.lu_solve(pt.linalg.lu_factor(A), b, b_ndim=1)

    compare_mlx_and_py(
        [A, b],
        [out],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-5, rtol=1e-5, strict=True
        ),
    )


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["batch", "batch_2d"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=["b_vec", "b_mat"])
def test_mlx_solve_batched(batch_shape, b_ndim):
    """`Blockwise` used to `vmap` `mx.linalg.solve`, which has no `LUF` rule (#2385)."""
    rng = np.random.default_rng(15)
    n = 3
    b_shape = (*batch_shape, n) if b_ndim == 1 else (*batch_shape, n, 2)

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    b = pt.tensor("b", shape=b_shape)

    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)
    A_val = A_val @ np.swapaxes(A_val, -1, -2) + n * np.eye(n, dtype=config.floatX)
    b_val = rng.normal(size=b_shape).astype(config.floatX)

    compare_mlx_and_py(
        [A, b],
        [pt.linalg.solve(A, b, b_ndim=b_ndim)],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-5, rtol=1e-5, strict=True
        ),
    )


@pytest.mark.parametrize(
    "A_batch, b_batch",
    [((4,), ()), ((), (4,)), ((1,), (4,))],
    ids=["A_batched", "b_batched", "A_broadcast"],
)
def test_mlx_solve_batch_broadcast(A_batch, b_batch):
    """`mx.linalg.solve` does not broadcast batch dims, so they must be aligned."""
    rng = np.random.default_rng(15)
    n = 3

    A = pt.tensor("A", shape=(*A_batch, n, n))
    b = pt.tensor("b", shape=(*b_batch, n))

    A_val = rng.normal(size=(*A_batch, n, n)).astype(config.floatX)
    A_val = A_val @ np.swapaxes(A_val, -1, -2) + n * np.eye(n, dtype=config.floatX)
    b_val = rng.normal(size=(*b_batch, n)).astype(config.floatX)

    compare_mlx_and_py(
        [A, b],
        [pt.linalg.solve(A, b, b_ndim=1)],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-5, rtol=1e-5, strict=True
        ),
    )


@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["stored_diagonal", "unit_diagonal"]
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_mlx_SolveTriangular_batched_ignores_other_triangle(lower, unit_diagonal):
    """A batched triangular solve must still ignore the opposite triangle.

    `mx.vmap`'s `solve_triangular` rule drops the `upper` flag, so a batched
    solve silently solved the *full* system. Nothing raised, and the answer was
    only correct when the input happened to be exactly triangular already --
    which `cho_solve` and `lu_solve` never are, since they pack `L` and `U`
    into a single array (#2385).
    """
    rng = np.random.default_rng(15)
    n, batch = 3, 4

    A = pt.tensor("A", shape=(batch, n, n))
    b = pt.tensor("b", shape=(batch, n))

    # Deliberately dense: both triangles are populated, so solving the full
    # system gives a different answer from the triangular one.
    A_val = rng.normal(size=(batch, n, n)).astype(config.floatX)
    A_val += n * np.eye(n, dtype=config.floatX)
    b_val = rng.normal(size=(batch, n)).astype(config.floatX)

    out = pt.linalg.solve_triangular(
        A, b, lower=lower, unit_diagonal=unit_diagonal, b_ndim=1
    )
    compare_mlx_and_py(
        [A, b],
        [out],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-5, rtol=1e-5, strict=True
        ),
    )


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["batch", "batch_2d"])
def test_mlx_lu_solve_batched(batch_shape):
    """Batched `lu_factor` + `lu_solve`: `LUF` vmap, the packed unit `L`, and
    the pivot-to-permutation scan all had to be batched (#2385)."""
    rng = np.random.default_rng(15)
    n = 4

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    b = pt.tensor("b", shape=(*batch_shape, n))

    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)
    A_val = A_val @ np.swapaxes(A_val, -1, -2) + n * np.eye(n, dtype=config.floatX)
    b_val = rng.normal(size=(*batch_shape, n)).astype(config.floatX)

    compare_mlx_and_py(
        [A, b],
        [pt.linalg.lu_solve(pt.linalg.lu_factor(A), b, b_ndim=1)],
        [A_val, b_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-4, rtol=1e-4, strict=True
        ),
    )
