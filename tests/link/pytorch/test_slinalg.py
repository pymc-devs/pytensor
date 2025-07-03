import pytest

import pytensor
from tests.link.pytorch.test_basic import compare_pytorch_and_py


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
    outs = pytensor.tensor.slinalg.qr(x, mode=mode)

    compare_pytorch_and_py([x], outs, [test_value])
