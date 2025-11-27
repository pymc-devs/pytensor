import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.math import Argmax, Max
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_dot():
    x = pt.matrix("x")
    y = pt.matrix("y")

    out = x.dot(y)
    fn = pytensor.function([x, y], out, mode="MLX")

    seed = sum(map(ord, "test_mlx_dot"))
    rng = np.random.default_rng(seed)

    test_x = rng.normal(size=(3, 2))
    test_y = rng.normal(size=(2, 4))

    actual = fn(test_x, test_y)
    assert isinstance(actual, mx.array)
    expected = np.dot(test_x, test_y)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_mlx_max_and_argmax():
    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = pt.dvector()
    mx = Max([0])(x)
    amx = Argmax([0])(x)
    out = mx * amx
    compare_mlx_and_py([x], [out], [np.r_[1, 2]])
