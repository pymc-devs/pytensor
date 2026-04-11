import numpy as np

from pytensor.configdefaults import config
from pytensor.tensor.linalg.products import kron
from pytensor.tensor.type import matrix
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_kron():
    x = matrix("x")
    y = matrix("y")
    z = kron(x, y)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    y_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)

    compare_pytorch_and_py([x, y], [z], [x_np, y_np])
