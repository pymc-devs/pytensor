from functools import partial
from typing import Literal

import numpy as np
import pytest

import pytensor.tensor as pt
import tests.unittest_tools as utt
from pytensor.configdefaults import config
from pytensor.tensor._linalg.decomposition import lu
from pytensor.tensor._linalg.decomposition.cholesky import cholesky
from pytensor.tensor._linalg.solve import linear_control
from pytensor.tensor._linalg.solve.general import lu_solve, solve
from pytensor.tensor._linalg.solve.psd import cho_solve
from pytensor.tensor._linalg.solve.triangular import solve_triangular
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_solve():
    rng = np.random.default_rng(utt.fetch_seed())

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    out = solve(A, b, lower=False, transposed=False)

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    compare_jax_and_py(
        [A, b],
        [out],
        [A_val, b_val],
    )


@pytest.mark.parametrize(
    "A_size, b_size, b_ndim",
    [
        ((5, 5), (5,), 1),
        ((5, 5), (5, 1), 2),
        ((5, 5), (1, 5), 1),
        ((4, 5, 5), (4, 5, 5), 2),
    ],
    ids=["basic_vector", "basic_matrix", "vector_broadcasted", "fully_batched"],
)
def test_jax_tridiagonal_solve(A_size: tuple, b_size: tuple, b_ndim: int):
    A = pt.tensor("A", shape=A_size)
    b = pt.tensor("b", shape=b_size)

    out = pt.linalg.solve(A, b, assume_a="tridiagonal", b_ndim=b_ndim)

    A_val = np.zeros(A_size)
    N = A_size[-1]
    A_val[...] = np.eye(N)
    for i in range(N - 1):
        A_val[..., i, i + 1] = np.random.randn()
        A_val[..., i + 1, i] = np.random.randn()

    b_val = np.random.randn(*b_size)

    compare_jax_and_py(
        [A, b],
        [out],
        [A_val, b_val],
    )


def test_jax_SolveTriangular():
    rng = np.random.default_rng(utt.fetch_seed())

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    out = solve_triangular(
        A,
        b,
        trans=0,
        lower=True,
        unit_diagonal=False,
    )
    compare_jax_and_py([A, b], [out], [A_val, b_val])


@pytest.mark.parametrize("b_shape", [(5,), (5, 5)])
def test_jax_lu_solve(b_shape):
    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    b_val = rng.normal(size=b_shape).astype(config.floatX)

    A = pt.tensor(name="A", shape=(5, 5))
    b = pt.tensor(name="b", shape=b_shape)
    lu_and_pivots = lu.lu_factor(A)
    out = lu_solve(lu_and_pivots, b)

    compare_jax_and_py([A, b], [out], [A_val, b_val])


@pytest.mark.parametrize("b_shape, lower", [((5,), True), ((5, 5), False)])
def test_jax_cho_solve(b_shape, lower):
    rng = np.random.default_rng(utt.fetch_seed())
    L_val = rng.normal(size=(5, 5)).astype(config.floatX)
    A_val = (L_val @ L_val.T).astype(config.floatX)

    b_val = rng.normal(size=b_shape).astype(config.floatX)

    A = pt.tensor(name="A", shape=(5, 5))
    b = pt.tensor(name="b", shape=b_shape)
    c = cholesky(A, lower=lower)
    out = cho_solve((c, lower), b, b_ndim=len(b_shape))

    compare_jax_and_py([A, b], [out], [A_val, b_val])


@pytest.mark.parametrize("method", ["direct", "bilinear"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
def test_jax_solve_discrete_lyapunov(
    method: Literal["direct", "bilinear"], shape: tuple[int]
):
    A = pt.tensor(name="A", shape=shape)
    B = pt.tensor(name="B", shape=shape)
    out = linear_control.solve_discrete_lyapunov(A, B, method=method)

    atol = rtol = 1e-8 if config.floatX == "float64" else 1e-3
    compare_jax_and_py(
        [A, B],
        [out],
        [
            np.random.normal(size=shape).astype(config.floatX),
            np.random.normal(size=shape).astype(config.floatX),
        ],
        jax_mode="JAX",
        assert_fn=partial(np.testing.assert_allclose, atol=atol, rtol=rtol),
    )


def test_bilinear_to_direct_rewrite(monkeypatch):
    mock_called = []

    def mock_load_solve_sylvester():
        mock_called.append(True)
        raise ImportError("Simulated ImportError for testing.")

    monkeypatch.setattr(
        "pytensor.tensor.rewriting.linalg.solve._load_solve_sylvester",
        mock_load_solve_sylvester,
    )

    A = pt.tensor(name="A", shape=(3, 3))
    B = pt.tensor(name="B", shape=(3, 3))

    out = linear_control.solve_discrete_lyapunov(A, B, method="bilinear")

    atol = rtol = 1e-8 if config.floatX == "float64" else 1e-3
    compare_jax_and_py(
        [A, B],
        [out],
        [
            np.random.normal(size=(3, 3)).astype(config.floatX),
            np.random.normal(size=(3, 3)).astype(config.floatX),
        ],
        jax_mode="JAX",
        assert_fn=partial(np.testing.assert_allclose, atol=atol, rtol=rtol),
    )
    assert mock_called


def test_jax_solve_sylvester():
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=(3, 3))
    B = pt.tensor(name="B", shape=(3, 3))
    C = pt.tensor(name="C", shape=(3, 3))

    A_val = rng.normal(size=(3, 3)).astype(config.floatX)
    B_val = rng.normal(size=(3, 3)).astype(config.floatX)
    C_val = rng.normal(size=(3, 3)).astype(config.floatX)

    out = linear_control.solve_sylvester(A, B, C)

    compare_jax_and_py([A, B, C], [out], [A_val, B_val, C_val])
