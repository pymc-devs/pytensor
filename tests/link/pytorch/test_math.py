import numpy as np

from pytensor.configdefaults import config
from pytensor.tensor.type import matrix, scalar, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_dot():
    y = vector("y")
    y_test = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x_test = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A_test = np.array([[6, 3], [3, 0]], dtype=config.floatX)
    alpha = scalar("alpha")
    alpha_test = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta_test = np.array(5.0, dtype=config.floatX)

    # 2D * 2D
    out = A.dot(A * alpha) + beta * A

    compare_pytorch_and_py([A, alpha, beta], [out], [A_test, alpha_test, beta_test])

    # 1D * 2D and 1D * 1D
    out = y.dot(alpha * A).dot(x) + beta * y

    compare_pytorch_and_py(
        [y, x, A, alpha, beta], [out], [y_test, x_test, A_test, alpha_test, beta_test]
    )
