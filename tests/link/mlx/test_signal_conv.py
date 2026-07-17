import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor.type import matrix, vector
from tests.link.mlx.test_basic import compare_mlx_and_py


pytest.importorskip("mlx.core")


@pytest.mark.parametrize("mode", ["full", "valid"])
def test_convolve1d_vector(mode):
    x = vector("x", dtype="float32")
    k = vector("k", dtype="float32")
    out = pt.signal.conv.convolve1d(x, k, mode=mode)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(32).astype("float32")
    k_np = rng.standard_normal(5).astype("float32")

    compare_mlx_and_py([x, k], [out], [x_np, k_np])


@pytest.mark.parametrize("mode", ["full", "valid"])
def test_convolve1d_batched_kernel_broadcast(mode):
    """A vector kernel shared across a batch of signals is wrapped in a
    Blockwise that broadcasts it to a leading-1 dim, so the MLX core thunk
    must flatten it back to 1-D before calling ``mx.convolve``.

    Regression test for #2092.
    """
    x = matrix("x", dtype="float32")
    k = vector("k", dtype="float32")
    out = pt.signal.conv.convolve1d(x, k, mode=mode)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 32)).astype("float32")
    k_np = rng.standard_normal(5).astype("float32")

    compare_mlx_and_py([x, k], [out], [x_np, k_np])
