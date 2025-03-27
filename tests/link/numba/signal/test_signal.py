import numpy as np
import pytest
from link.numba.test_basic import compare_numba_and_py

from pytensor.tensor import matrix
from pytensor.tensor.signal import convolve


pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_conv(mode):
    x = matrix("x")
    y = matrix("y")
    out = convolve(x[None], y[:, None], mode=mode)

    rng = np.random.default_rng()
    test_x = rng.normal(size=(3, 5))
    test_y = rng.normal(size=(7, 11))
    # Object mode is not supported for numba
    compare_numba_and_py([x, y], out, [test_x, test_y], eval_obj_mode=False)
