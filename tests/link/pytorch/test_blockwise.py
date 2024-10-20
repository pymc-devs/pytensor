import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt


torch = pytest.importorskip("torch")


def test_blockwise_broadcast():
    _x = np.random.rand(5, 1, 2, 3)
    _y = np.random.rand(3, 3, 2)

    x = pt.tensor4("x", shape=(5, 1, 2, 3))
    y = pt.tensor3("y", shape=(3, 3, 2))

    f = pytensor.function([x, y], x @ y, mode="PYTORCH")
    res = f(_x, _y)
    assert tuple(res.shape) == (5, 3, 2, 2)
    np.testing.assert_allclose(res, _x @ _y)
