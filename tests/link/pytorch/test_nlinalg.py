import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor import blas as pt_blas
from pytensor.tensor import nlinalg as pt_nla
from pytensor.tensor.math import argmax, dot, max
from pytensor.tensor.type import matrix, tensor3, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.fixture
def matrix_test():
    rng = np.random.default_rng(213234)

    M = rng.normal(size=(3, 3))
    test_value = M.dot(M.T).astype(config.floatX)

    x = matrix("x")
    return (x, test_value)


def test_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a.tag.test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b.tag.test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = pt_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [get_test_value(a)[:-1], get_test_value(b)]
    pytensor_jax_fn = function(fgraph.inputs, fgraph.outputs, mode="PYTORCH")
    with pytest.raises(TypeError):
        pytensor_jax_fn(*inputs)


@pytest.mark.parametrize(
    "func",
    (
        pt_nla.eig,
        pt_nla.eigh,
        pt_nla.slogdet,
        pytest.param(
            pt_nla.inv, marks=pytest.mark.xfail(reason="Blockwise not implemented")
        ),
        pytest.param(
            pt_nla.det, marks=pytest.mark.xfail(reason="Blockwise not implemented")
        ),
    ),
)
def test_lin_alg_no_params(func, matrix_test):
    x, test_value = matrix_test

    outs = func(x)
    out_fg = FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x, y, rtol=1e-3)

    compare_pytorch_and_py(out_fg, [test_value], assert_fn=assert_fn)


@pytest.mark.parametrize(
    "mode",
    (
        "complete",
        "reduced",
        "r",
        pytest.param("raw", marks=pytest.mark.xfail(raises=NotImplementedError)),
    ),
)
def test_qr(mode, matrix_test):
    x, test_value = matrix_test
    outs = pt_nla.qr(x, mode=mode)
    out_fg = FunctionGraph([x], [outs] if mode == "r" else outs)
    compare_pytorch_and_py(out_fg, [test_value])


@pytest.mark.xfail(reason="Blockwise not implemented")
@pytest.mark.parametrize("compute_uv", [False, True])
@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd(compute_uv, full_matrices, matrix_test):
    x, test_value = matrix_test

    outs = pt_nla.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
    out_fg = FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x, y, rtol=1e-3)

    compare_pytorch_and_py(out_fg, [test_value], assert_fn=assert_fn)


def test_pinv():
    x = matrix("x")
    x_inv = pt_nla.pinv(x)

    fgraph = FunctionGraph([x], [x_inv])
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    compare_pytorch_and_py(fgraph, [x_np])


@pytest.mark.parametrize("hermitian", [False, True])
def test_pinv_hermitian(hermitian):
    A = matrix("A", dtype="complex128")
    A_h_test = np.c_[[3, 3 + 2j], [3 - 2j, 2]]
    A_not_h_test = A_h_test + 0 + 1j

    A_inv = pt_nla.pinv(A, hermitian=hermitian)
    torch_fn = function([A], A_inv, mode="PYTORCH")

    assert np.allclose(torch_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=False))
    assert np.allclose(torch_fn(A_h_test), np.linalg.pinv(A_h_test, hermitian=True))

    assert (
        np.allclose(
            torch_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=False)
        )
        is not hermitian
    )

    assert (
        np.allclose(
            torch_fn(A_not_h_test), np.linalg.pinv(A_not_h_test, hermitian=True)
        )
        is hermitian
    )


def test_kron():
    x = matrix("x")
    y = matrix("y")
    z = pt_nla.kron(x, y)

    fgraph = FunctionGraph([x, y], [z])
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    y_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)

    compare_pytorch_and_py(fgraph, [x_np, y_np])


@pytest.mark.parametrize("func", (max, argmax))
@pytest.mark.parametrize("axis", [None, [0], [0, 1], [0, 2], [0, 1, 2]])
def test_max_and_argmax(func, axis):
    x = tensor3("x")
    np.random.seed(42)
    test_value = np.random.randint(0, 20, (4, 3, 2))

    out = func(x, axis=axis)
    out_fg = FunctionGraph([x], [out])
    compare_pytorch_and_py(out_fg, [test_value])


def test_dot():
    x = vector("x")
    test_value = np.array([1, 2, 3])

    out = dot(x, x)
    out_fg = FunctionGraph([x], [out])
    compare_pytorch_and_py(out_fg, [test_value])
