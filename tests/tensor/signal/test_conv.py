from functools import partial

import numpy as np
import pytest
from scipy.signal import convolve as scipy_convolve

from pytensor import config, function
from pytensor.tensor import matrix, vector
from pytensor.tensor.signal.conv import convolve1d
from tests import unittest_tools as utt


@pytest.mark.parametrize("kernel_shape", [3, 5, 8], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("data_shape", [3, 5, 8], ids=lambda x: f"data_shape={x}")
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_convolve1d(mode, data_shape, kernel_shape):
    data = vector("data")
    kernel = vector("kernel")
    op = partial(convolve1d, mode=mode)

    rng = np.random.default_rng((26, kernel_shape, data_shape, sum(map(ord, mode))))
    data_val = rng.normal(size=data_shape).astype(data.dtype)
    kernel_val = rng.normal(size=kernel_shape).astype(kernel.dtype)

    fn = function([data, kernel], op(data, kernel))
    np.testing.assert_allclose(
        fn(data_val, kernel_val),
        scipy_convolve(data_val, kernel_val, mode=mode),
        rtol=1e-6 if config.floatX == "float32" else 1e-15,
    )
    utt.verify_grad(op=lambda x: op(x, kernel_val), pt=[data_val])


def test_convolve1d_batch():
    x = matrix("data")
    y = matrix("kernel")
    out = convolve1d(x, y)

    rng = np.random.default_rng(38)
    x_test = rng.normal(size=(2, 8)).astype(x.dtype)
    y_test = x_test[::-1]

    res = out.eval({x: x_test, y: y_test})
    # Second entry of x, y are just y, x respectively,
    # so res[0] and res[1] should be identical.
    rtol = 1e-6 if config.floatX == "float32" else 1e-15
    res_np = np.convolve(x_test[0], y_test[0])
    np.testing.assert_allclose(res[0], res_np, rtol=rtol)
    np.testing.assert_allclose(res[1], res_np, rtol=rtol)


def test_convolve1d_same():
    x = matrix("data")
    y = matrix("kernel")
    out = convolve1d(x, y, mode="same")

    rng = np.random.default_rng(38)
    x_test = rng.normal(size=(2, 8)).astype(x.dtype)
    y_test = rng.normal(size=(2, 8)).astype(x.dtype)

    res = out.eval({x: x_test, y: y_test})
    assert res.shape == (2, 8)
