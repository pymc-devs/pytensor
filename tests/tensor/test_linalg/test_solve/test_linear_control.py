import functools
from typing import Literal

import numpy as np
import pytest
from scipy import linalg as scipy_linalg

from pytensor import function
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor.linalg import (
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
    solve_sylvester,
)
from tests import unittest_tools as utt


@pytest.mark.parametrize(
    "shape, use_complex",
    [((5, 5), False), ((5, 5), True), ((5, 5, 5), False)],
    ids=["float", "complex", "batch_float"],
)
def test_solve_continuous_sylvester(shape: tuple[int], use_complex: bool):
    rng = np.random.default_rng()

    dtype = config.floatX
    if use_complex:
        dtype = "complex128" if dtype == "float64" else "complex64"

    A1, A2 = rng.normal(size=(2, *shape))
    B1, B2 = rng.normal(size=(2, *shape))
    Q1, Q2 = rng.normal(size=(2, *shape))

    if use_complex:
        A_val = A1 + 1j * A2
        B_val = B1 + 1j * B2
        Q_val = Q1 + 1j * Q2
    else:
        A_val = A1
        B_val = B1
        Q_val = Q1

    A = pt.tensor("A", shape=shape, dtype=dtype)
    B = pt.tensor("B", shape=shape, dtype=dtype)
    Q = pt.tensor("Q", shape=shape, dtype=dtype)

    X = solve_sylvester(A, B, Q)
    Q_recovered = A @ X + X @ B

    fn = function([A, B, Q], [X, Q_recovered])
    X_val, Q_recovered_val = fn(A_val, B_val, Q_val)

    vec_sylvester = np.vectorize(
        scipy_linalg.solve_sylvester, signature="(m,m),(m,m),(m,m)->(m,m)"
    )
    np.testing.assert_allclose(Q_recovered_val, Q_val, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        X_val, vec_sylvester(A_val, B_val, Q_val), atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batched"])
@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
def test_solve_continuous_sylvester_grad(shape: tuple[int], use_complex):
    if config.floatX == "float32":
        pytest.skip(reason="Not enough precision in float32 to get a good gradient")
    if use_complex:
        pytest.skip(reason="Complex numbers are not supported in the gradient test")

    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.normal(size=shape).astype(config.floatX)
    B = rng.normal(size=shape).astype(config.floatX)
    Q = rng.normal(size=shape).astype(config.floatX)

    utt.verify_grad(solve_sylvester, pt=[A, B, Q], rng=rng)


def recover_Q(A, X, continuous=True):
    if continuous:
        return A @ X + X @ A.conj().T
    else:
        return X - A @ X @ A.conj().T


vec_recover_Q = np.vectorize(recover_Q, signature="(m,m),(m,m),()->(m,m)")


@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
@pytest.mark.parametrize("method", ["direct", "bilinear"])
def test_solve_discrete_lyapunov(
    use_complex, shape: tuple[int], method: Literal["direct", "bilinear"]
):
    rng = np.random.default_rng(utt.fetch_seed())
    dtype = config.floatX
    if use_complex:
        precision = int(dtype[-2:])
        dtype = f"complex{int(2 * precision)}"

    A1, A2 = rng.normal(size=(2, *shape))
    Q1, Q2 = rng.normal(size=(2, *shape))

    if use_complex:
        A = A1 + 1j * A2
        Q = Q1 + 1j * Q2
    else:
        A = A1
        Q = Q1

    A, Q = A.astype(dtype), Q.astype(dtype)

    a = pt.tensor(name="a", shape=shape, dtype=dtype)
    q = pt.tensor(name="q", shape=shape, dtype=dtype)

    x = solve_discrete_lyapunov(a, q, method=method)
    f = function([a, q], x)

    X = f(A, Q)
    Q_recovered = vec_recover_Q(A, X, continuous=False)

    atol = rtol = 1e-4 if config.floatX == "float32" else 1e-8
    np.testing.assert_allclose(Q_recovered, Q, atol=atol, rtol=rtol)


@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
@pytest.mark.parametrize("method", ["direct", "bilinear"])
def test_solve_discrete_lyapunov_gradient(
    use_complex, shape: tuple[int], method: Literal["direct", "bilinear"]
):
    if config.floatX == "float32":
        pytest.skip(reason="Not enough precision in float32 to get a good gradient")
    if use_complex:
        pytest.skip(reason="Complex numbers are not supported in the gradient test")

    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.normal(size=shape).astype(config.floatX)
    Q = rng.normal(size=shape).astype(config.floatX)

    utt.verify_grad(
        functools.partial(solve_discrete_lyapunov, method=method),
        pt=[A, Q],
        rng=rng,
    )


def test_solve_continuous_lyapunov():
    A = pt.tensor("A", shape=(3, 5, 5))
    Q = pt.tensor("Q", shape=(3, 5, 5))

    X = solve_continuous_lyapunov(A, Q)
    Q_recovered = A @ X + X @ A.conj().mT

    fn = function([A, Q], [X, Q_recovered])

    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.normal(size=(3, 5, 5)).astype(config.floatX)
    Q_val = rng.normal(size=(3, 5, 5)).astype(config.floatX)
    _, Q_recovered_val = fn(A_val, Q_val)

    atol = rtol = 1e-2 if config.floatX == "float32" else 1e-8
    np.testing.assert_allclose(Q_recovered_val, Q_val, atol=atol, rtol=rtol)
    utt.verify_grad(solve_continuous_lyapunov, pt=[A_val, Q_val], rng=rng)


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_solve_discrete_are_forward(add_batch_dim):
    a, b, q, r = (
        np.array([[4, 3], [-4.5, -3.5]]),
        np.array([[1], [-1]]),
        np.array([[9, 6], [6, 4]]),
        np.array([[1]]),
    )
    if add_batch_dim:
        a, b, q, r = (np.stack([x] * 5) for x in [a, b, q, r])

    a, b, q, r = (pt.as_tensor_variable(x).astype(config.floatX) for x in [a, b, q, r])

    x = solve_discrete_are(a, b, q, r)

    def eval_fun(a, b, q, r, x):
        term_1 = a.T @ x @ a
        term_2 = a.T @ x @ b
        term_3 = pt.linalg.solve(r + b.T @ x @ b, b.T) @ x @ a

        return term_1 - x - term_2 @ term_3 + q

    res = pt.vectorize(eval_fun, "(m,m),(m,n),(m,m),(n,n),(m,m)->(m,m)")(a, b, q, r, x)
    res_np = res.eval()

    atol = 1e-4 if config.floatX == "float32" else 1e-12
    np.testing.assert_allclose(res_np, np.zeros_like(res_np), atol=atol)


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_solve_discrete_are_grad(add_batch_dim):
    a, b, q, r = (
        np.array([[4, 3], [-4.5, -3.5]]),
        np.array([[1], [-1]]),
        np.array([[9, 6], [6, 4]]),
        np.array([[1]]),
    )
    if add_batch_dim:
        a, b, q, r = (np.stack([x] * 5) for x in [a, b, q, r])

    a, b, q, r = (x.astype(config.floatX) for x in [a, b, q, r])
    rng = np.random.default_rng(utt.fetch_seed())

    # TODO: Is there a "theoretically motivated" value to use here? I pulled 1e-4 out of a hat
    atol = 1e-4 if config.floatX == "float32" else 1e-12

    utt.verify_grad(
        solve_discrete_are,
        pt=[a, b, q, r],
        rng=rng,
        abs_tol=atol,
    )
