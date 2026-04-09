import numpy as np
import pytest

from pytensor.compile.maker import function
from pytensor.configdefaults import config
from pytensor.tensor import nlinalg as pt_nla
from pytensor.tensor.type import matrix
from tests.link.pytorch.test_basic import compare_pytorch_and_py


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
