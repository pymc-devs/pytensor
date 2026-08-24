import numpy as np
import pytest

from pytensor import function
from pytensor.gradient import grad
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor import dmatrix, scalar, vector
from pytensor.tensor.blockwise import Blockwise
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


@pytest.mark.parametrize("x_shape", [(10,), (None,)], ids=["static", "dynamic"])
@pytest.mark.parametrize("mode", ["valid", "full"])
def test_grad_chained_vectorized_convolve1d(mode, x_shape):
    # Regression test for https://github.com/pymc-devs/pytensor/issues/2360
    # Gradient of chained convolutions with a vectorized kernel and an
    # unbatched signal used to fall back to object mode (and then crash on the
    # scalar full_mode input) in the numba backend.
    x = vector("x", shape=x_shape)
    alpha = scalar("alpha")
    kernel = alpha ** np.arange(4, dtype="float64")
    y = convolve1d(convolve1d(x, kernel, mode=mode), kernel, mode=mode)

    alpha_batch = vector("alpha_batch", shape=(5,))
    y_batch = vectorize_graph(y, replace={alpha: alpha_batch})
    grads = grad(y_batch.sum(), wrt=[x, alpha_batch])

    rng = np.random.default_rng(2360)
    x_test = rng.uniform(size=(10,))
    alpha_test = rng.uniform(0.1, 0.9, size=(5,))
    # The minimal rewrites of the default test mode leave the boolean mode
    # inputs symbolic and batched, exercising the object-mode Blockwise.perform
    # fallback with a ScalarType core input
    compare_numba_and_py(
        [x, alpha_batch], grads, [x_test, alpha_test], eval_obj_mode=False
    )

    # Under the full NUMBA mode rewrites, every Blockwise must be lowered to
    # BlockwiseWithCoreShape; a plain Blockwise would mean an object-mode fallback
    fn = function([x, alpha_batch], grads, mode="NUMBA")
    assert not any(
        isinstance(node.op, Blockwise) for node in fn.maker.fgraph.apply_nodes
    )
    np.testing.assert_allclose(
        fn(x_test, alpha_test)[0],
        function([x, alpha_batch], grads, mode="FAST_COMPILE")(x_test, alpha_test)[0],
    )
