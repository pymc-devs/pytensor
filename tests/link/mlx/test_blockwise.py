import numpy as np

import pytensor.tensor as pt
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot
from tests.link.mlx.test_basic import compare_mlx_and_py


# Equivalent blockwise to matmul but with dumb signature
odd_matmul = Blockwise(Dot(), signature="(i00,i01),(i10,i11)->(o00,o01)")


def test_blockwise_conv1d():
    rng = np.random.default_rng(14)
    a = tensor("a", shape=(2, 100))
    b = tensor("b", shape=(2, 8))

    a_test = rng.normal(size=(2, 100))
    b_test = rng.normal(size=(2, 8))

    test_values = [a_test, b_test]

    out = pt.signal.convolve1d(a, b, mode="valid")

    # assert isinstance(out.owner.op, Blockwise)
    compare_mlx_and_py([a, b], [out], test_values, must_be_device_array=True)


def test_blockwise_no_batch_dimensions():
    """Test that Blockwise returns the core function when there are no batch dimensions.

    This verifies the fix for the vmap dispatcher issue where mx.vmap should not
    be called when there are no batch dimensions to vectorize over.
    """
    rng = np.random.default_rng(42)

    # Create a blockwise matmul with no batch dimensions (core operation only)
    x = pt.matrix("x")
    y = pt.matrix("y")

    blockwise_matmul = Blockwise(Dot(), signature="(i,j),(j,k)->(i,k)")
    z = blockwise_matmul(x, y)

    x_test = rng.normal(size=(2, 3))
    y_test = rng.normal(size=(3, 4))

    compare_mlx_and_py([x, y], [z], [x_test, y_test], must_be_device_array=True)


def test_blockwise_all_broadcastable_batch_dims():
    """Test that Blockwise returns the core function when all batch dims are broadcastable.

    When all batch dimensions are size-1 (broadcastable), vmap should not be called
    since there's no actual vectorization needed.
    """
    rng = np.random.default_rng(43)

    # Create inputs with size-1 batch dimensions
    x = tensor("x", shape=(1, 2, 3))
    y = tensor("y", shape=(1, 3, 4))

    blockwise_matmul = Blockwise(Dot(), signature="(i,j),(j,k)->(i,k)")
    z = blockwise_matmul(x, y)

    x_test = rng.normal(size=(1, 2, 3))
    y_test = rng.normal(size=(1, 3, 4))

    compare_mlx_and_py([x, y], [z], [x_test, y_test], must_be_device_array=True)
