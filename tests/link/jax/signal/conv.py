import numpy as np
import pytest
from link.jax.test_basic import compare_jax_and_py

from pytensor.tensor import matrix
from pytensor.tensor.signal import convolve


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_conv(mode):
    x = matrix("x")
    y = matrix("y")
    out = convolve(x[None], y[:, None], mode=mode)

    rng = np.random.default_rng()
    test_x = rng.normal(size=(3, 5))
    test_y = rng.normal(size=(7, 11))
    compare_jax_and_py([x, y], out, [test_x, test_y])
