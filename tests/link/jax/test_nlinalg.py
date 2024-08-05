import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import nlinalg as pt_nlinalg
from pytensor.tensor.type import matrix
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_basic_multiout():
    rng = np.random.default_rng(213234)

    M = rng.normal(size=(3, 3))
    X = M.dot(M.T)

    x = matrix("x")

    outs = pt_nlinalg.eig(x)
    out_fg = FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(config.floatX), y, rtol=1e-3)

    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = pt_nlinalg.eigh(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = pt_nlinalg.qr(x, mode="full")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = pt_nlinalg.qr(x, mode="reduced")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = pt_nlinalg.svd(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = pt_nlinalg.slogdet(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)


def test_pinv():
    x = matrix("x")
    x_inv = pt_nlinalg.pinv(x)

    fgraph = FunctionGraph([x], [x_inv])
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    compare_jax_and_py(fgraph, [x_np])


def test_pinv_hermitian():
    A = matrix("A", dtype="complex128")
    A_h_test = np.c_[[3, 3 + 2j], [3 - 2j, 2]]
    A_not_h_test = A_h_test + 0 + 1j

    A_inv = pt_nlinalg.pinv(A, hermitian=False)
    jax_fn = function([A], A_inv, mode="JAX")

    assert np.allclose(jax_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=False))
    assert np.allclose(jax_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=True))
    assert np.allclose(
        jax_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=False)
    )
    assert not np.allclose(
        jax_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=True)
    )

    A_inv = pt_nlinalg.pinv(A, hermitian=True)
    jax_fn = function([A], A_inv, mode="JAX")

    assert np.allclose(jax_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=False))
    assert np.allclose(jax_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=True))
    assert not np.allclose(
        jax_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=False)
    )
    # Numpy fails differently than JAX when hermitian assumption is violated
    assert not np.allclose(
        jax_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=True)
    )


def test_kron():
    x = matrix("x")
    y = matrix("y")
    z = pt_nlinalg.kron(x, y)

    fgraph = FunctionGraph([x, y], [z])
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    y_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)

    compare_jax_and_py(fgraph, [x_np, y_np])
