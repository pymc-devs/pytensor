import numpy as np

import pytensor
from pytensor.tensor.type import matrix


def test_mlx_dot():
    x = matrix("x")
    y = matrix("y")

    out = x.dot(y)
    fn = pytensor.function([x, y], out, mode="MLX")

    seed = sum(map(ord, "test_mlx_dot"))
    rng = np.random.default_rng(seed)

    test_x = rng.normal(size=(3, 2))
    test_y = rng.normal(size=(2, 4))

    actual = fn(test_x, test_y)
    expected = np.dot(test_x, test_y)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
