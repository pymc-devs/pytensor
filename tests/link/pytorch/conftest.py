import numpy as np
import pytest

from pytensor import config
from pytensor.tensor.type import matrix


@pytest.fixture
def matrix_test():
    rng = np.random.default_rng(213234)

    M = rng.normal(size=(3, 3))
    test_value = M.dot(M.T).astype(config.floatX)

    x = matrix("x")
    return x, test_value
