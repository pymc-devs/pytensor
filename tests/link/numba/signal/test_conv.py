import numpy as np
import pytest

from pytensor.tensor import dmatrix
from pytensor.tensor.signal import convolve1d
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
    r"ignore:Cannot cache compiled function \"numba_funcified_fgraph.*:numba.NumbaWarning",
)


@pytest.mark.parametrize("bcast_order", (1, 0))
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_convolve1d(mode, bcast_order):
    x = dmatrix("x")
    y = dmatrix("y")
    # Testing two orders because this revealed a bug in the past
    if bcast_order == 0:
        out = convolve1d(x[:, None], y[None, :], mode=mode)
    else:
        out = convolve1d(x[None], y[:, None], mode=mode)

    rng = np.random.default_rng()
    test_x = rng.normal(size=(3, 5))
    test_y = rng.normal(size=(7, 11))
    # Blockwise dispatch for numba can't be run on object mode
    numba_fn, res = compare_numba_and_py(
        [x, y], out, [test_x, test_y], eval_obj_mode=False
    )

    # Try other order of inputs, as implementation depends on it
    # Result should be the same, just in different order, except for 'same' mode
    if mode != "same":
        np.testing.assert_allclose(
            np.swapaxes(numba_fn(test_y, test_x), 0, 1),
            res,
        )
