import numpy as np
import pytest

from pytensor.signal.conv import Conv1d
from tests import unittest_tools as utt


@pytest.mark.parametrize("data_shape", [3, 5, 8])
@pytest.mark.parametrize("kernel_shape", [3, 5, 8])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_conv1d_grad(mode, data_shape, kernel_shape):
    rng = np.random.default_rng()

    data_val = rng.normal(size=data_shape)
    kernel_val = rng.normal(size=kernel_shape)

    op = Conv1d(mode=mode)

    utt.verify_grad(op=op, pt=[data_val, kernel_val])
