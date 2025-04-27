from functools import partial

import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import dmatrix, tensor
from pytensor.tensor.signal import convolve1d
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("x_smaller", (False, True))
def test_convolve1d(x_smaller, mode):
    x = dmatrix("x")
    y = dmatrix("y")
    if x_smaller:
        out = convolve1d(x[None], y[:, None], mode=mode)
    else:
        out = convolve1d(y[:, None], x[None], mode=mode)

    rng = np.random.default_rng()
    test_x = rng.normal(size=(3, 5))
    test_y = rng.normal(size=(7, 11))
    # Blockwise dispatch for numba can't be run on object mode
    compare_numba_and_py([x, y], out, [test_x, test_y], eval_obj_mode=False)


@pytest.mark.parametrize("mode", ("full", "valid"), ids=lambda x: f"mode={x}")
@pytest.mark.parametrize("batch", (False, True), ids=lambda x: f"batch={x}")
def test_convolve1d_benchmark(batch, mode, benchmark):
    x = tensor(
        shape=(
            7,
            183,
        )
        if batch
        else (183,)
    )
    y = tensor(shape=(7, 6) if batch else (6,))
    out = convolve1d(x, y, mode=mode)
    fn = function([x, y], out, mode="NUMBA", trust_input=True)

    rng = np.random.default_rng()
    x_test = rng.normal(size=(x.type.shape)).astype(x.type.dtype)
    y_test = rng.normal(size=(y.type.shape)).astype(y.type.dtype)

    np_convolve1d = np.vectorize(
        partial(np.convolve, mode=mode), signature="(x),(y)->(z)"
    )

    np.testing.assert_allclose(
        fn(x_test, y_test),
        np_convolve1d(x_test, y_test),
    )
    benchmark(fn, x_test, y_test)
