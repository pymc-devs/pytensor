from functools import partial

import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import dmatrix, tensor
from pytensor.tensor.signal import convolve1d
from tests.link.numba.test_basic import compare_numba_and_py
from tests.tensor.signal.test_conv import convolve1d_grad_benchmarker


pytestmark = pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
    r"ignore:Cannot cache compiled function \"numba_funcified_fgraph\".*:numba.NumbaWarning",
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


@pytest.mark.parametrize("mode", ("full", "valid"), ids=lambda x: f"mode={x}")
@pytest.mark.parametrize("batch", (False, True), ids=lambda x: f"batch={x}")
def test_convolve1d_benchmark_numba(batch, mode, benchmark):
    x = tensor(shape=(7, 183) if batch else (183,))
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


@pytest.mark.parametrize("convolve_mode", ["full", "valid"])
def test_convolve1d_grad_benchmark_numba(convolve_mode, benchmark):
    convolve1d_grad_benchmarker(convolve_mode, "NUMBA", benchmark)
