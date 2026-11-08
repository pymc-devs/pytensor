"""Native (faer) Cholesky and SolveTriangular."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.llvm.linker import LLVM_REWRITES, LLVMLinker
from pytensor.tensor.linalg import cholesky, solve_triangular


pytest.importorskip("pytensor_llvm")

LLVM = Mode(linker=LLVMLinker(numba_fallback=False), optimizer=RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]))
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(7)


def spd(n, dtype="float64"):
    a = rng.normal(size=(n, n))
    return (a @ a.T + n * np.eye(n)).astype(dtype)


def compare(inputs, outputs, values, rtol=1e-10):
    f = pytensor.function(inputs, outputs, mode=LLVM)
    sched = f.maker.linker.compiled.info()["schedule"]
    assert any("faer" in d for d in sched), sched
    ref = pytensor.function(inputs, outputs, mode=REF)(*values)
    for r, e in zip(f(*values), ref, strict=True):
        assert r.dtype == e.dtype and r.shape == e.shape
        np.testing.assert_allclose(r, e, rtol=rtol, atol=rtol)
    return f


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [1, 5, 40])
def test_cholesky(lower, n):
    A = pt.matrix("A")
    compare([A], [cholesky(A, lower=lower)], [spd(n)])


def test_cholesky_strided_input_and_not_pd():
    A = pt.matrix("A")
    f = compare([A], [cholesky(A.T[::-1, ::-1])], [spd(6)])
    out = f(-np.eye(4))[0]  # not positive definite -> NaN like LAPACK
    assert np.isnan(out).all()


def test_cholesky_float32():
    A = pt.matrix("A", dtype="float32")
    compare([A], [cholesky(A)], [spd(8, "float32")], rtol=1e-4)


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("unit_diagonal", [True, False])
@pytest.mark.parametrize("b_ndim", [1, 2])
def test_solve_triangular(lower, unit_diagonal, b_ndim):
    A = pt.matrix("A")
    b = pt.tensor("b", shape=(None,) * b_ndim)
    L = np.linalg.cholesky(spd(7))
    L = L if lower else L.T
    bv = rng.normal(size=(7,) if b_ndim == 1 else (7, 3))
    out = solve_triangular(A, b, lower=lower, unit_diagonal=unit_diagonal, b_ndim=b_ndim)
    compare([A, b], [out], [L, bv])


def test_singular_triangular_gives_nan():
    A = pt.matrix("A")
    b = pt.vector("b")
    f = compare([A, b], [solve_triangular(A, b, lower=True, b_ndim=1)], [np.tril(np.ones((3, 3))), np.ones(3)])
    assert np.isnan(f(np.diag([1.0, 0.0, 1.0]), np.ones(3))[0]).all()


def test_mvnormal_logp_and_gradient():
    # the pattern of an MvNormal logp: chol, triangular solve, log-det; and its gradient
    S = pt.matrix("S")
    x = pt.matrix("x")
    L = cholesky(S)
    z = solve_triangular(L, x.T, lower=True, b_ndim=2)
    logp = -0.5 * (z**2).sum() - x.shape[0] * pt.log(pt.diag(L)).sum()
    with_fallback = Mode(linker=LLVMLinker(), optimizer=RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]))
    ins, outs = [S, x], [logp, *pytensor.grad(logp, [S, x])]
    vals = [spd(5), rng.normal(size=(30, 5))]
    f = pytensor.function(ins, outs, mode=with_fallback)
    ref = pytensor.function(ins, outs, mode=REF)(*vals)
    for r, e in zip(f(*vals), ref, strict=True):
        np.testing.assert_allclose(r, e, rtol=1e-9, atol=1e-10)
