import mlx.core as mx
import numpy as np

import pytensor
from pytensor.tensor.type import matrix


def test_mlx_dot():
    x = matrix("x")
    y = matrix("y")

    out = x.dot(y)
    fn = pytensor.function([x, y], out, mode="MLX")

    test_x = mx.array(np.random.normal(size=(3, 2)))
    test_y = mx.array(np.random.normal(size=(2, 4)))
    np.testing.assert_allclose(
        fn(test_x, test_y),
        np.dot(test_x, test_y),
    )
