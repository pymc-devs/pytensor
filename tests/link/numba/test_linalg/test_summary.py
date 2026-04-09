import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor._linalg.summary import Det, SLogDet
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize("dtype", ("float64", "int64"))
@pytest.mark.parametrize("op", (Det(), SLogDet()), ids=["det", "slogdet"])
def test_Det_SLogDet(op, dtype):
    x = pt.matrix(dtype=dtype)

    rng = np.random.default_rng([50, sum(map(ord, dtype))])
    x_ = rng.random(size=(3, 3)).astype(dtype)
    test_x = x_.T.dot(x_)

    g = op(x)

    compare_numba_and_py([x], g, [test_x])
