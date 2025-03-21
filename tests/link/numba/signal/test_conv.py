import numpy as np
import pytest

from pytensor.tensor import dmatrix
from pytensor.tensor.signal import convolve1d
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_convolve1d(mode):
    x = dmatrix("x")
    y = dmatrix("y")
    out = convolve1d(x[None], y[:, None], mode=mode)

    rng = np.random.default_rng()
    test_x = rng.normal(size=(3, 5))
    test_y = rng.normal(size=(7, 11))
    # Blockwise dispatch for numba can't be run on object mode
    compare_numba_and_py([x, y], out, [test_x, test_y], eval_obj_mode=False)
