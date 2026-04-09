import numpy as np
import pytest

from pytensor.tensor import nlinalg as pt_nla
from pytensor.tensor._linalg.decomposition import qr, svd
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

    out = svd.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    compare_pytorch_and_py([x], out, [test_value])


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
    outs = qr.qr(x, mode=mode)

    compare_pytorch_and_py([x], outs, [test_value])
