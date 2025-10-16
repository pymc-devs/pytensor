import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.tensor import nlinalg as pt_nla
from pytensor.tensor.type import matrix
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def assert_fn(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-3)


@pytest.mark.parametrize(
    "func",
    (pt_nla.eigh, pt_nla.SLogDet(), pt_nla.inv, pt_nla.det),
)
def test_lin_alg_no_params(func, matrix_test):
    x, test_value = matrix_test

    outs = func(x)

    compare_pytorch_and_py([x], outs, [test_value], assert_fn=assert_fn)


def test_eig(matrix_test):
    x, test_value = matrix_test
    out = pt_nla.eig(x)

    compare_pytorch_and_py([x], out, [test_value], assert_fn=assert_fn)


@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_svd(compute_uv, full_matrices, matrix_test):
    x, test_value = matrix_test

    out = pt_nla.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    compare_pytorch_and_py([x], out, [test_value])


def test_pinv():
    x = matrix("x")
    x_inv = pt_nla.pinv(x)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    compare_pytorch_and_py([x], [x_inv], [x_np])


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

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    y_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)

    compare_pytorch_and_py([x, y], [z], [x_np, y_np])
