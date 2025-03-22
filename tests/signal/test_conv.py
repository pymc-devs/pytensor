from functools import partial

import numpy as np
import pytest
from scipy.signal import convolve as scipy_convolve

from pytensor import function
from pytensor.signal.conv import convolve
from pytensor.tensor import vector
from tests import unittest_tools as utt


@pytest.mark.parametrize("kernel_shape", [3, 5, 8], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("data_shape", [3, 5, 8], ids=lambda x: f"data_shape={x}")
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_convolve(mode, data_shape, kernel_shape):
    rng = np.random.default_rng()

    data = vector("data")
    kernel = vector("kernel")
    op = partial(convolve, mode=mode)

    rng = np.random.default_rng()
    data_val = rng.normal(size=data_shape)
    kernel_val = rng.normal(size=kernel_shape)

    fn = function([data, kernel], op(data, kernel))
    np.testing.assert_allclose(
        fn(data_val, kernel_val),
        scipy_convolve(data_val, kernel_val, mode=mode),
    )
    utt.verify_grad(op=lambda x: op(x, kernel_val), pt=[data_val])
