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
