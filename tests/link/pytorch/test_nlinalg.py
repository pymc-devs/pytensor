import numpy as np

from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor.type import matrix, scalar, tensor3, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_dot():
    a = tensor3("a")
    a.tag.test_value = np.zeros((3, 2, 4)).astype(config.floatX)
    b = tensor3("b")
    b.tag.test_value = np.zeros((3, 4, 1)).astype(config.floatX)
    y = vector("y")
    y.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A.tag.test_value = np.array([[6, 3], [3, 0]], dtype=config.floatX)
    alpha = scalar("alpha")
    alpha.tag.test_value = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta.tag.test_value = np.array(5.0, dtype=config.floatX)

    # 3D * 3D
    out = a.dot(b * alpha) + beta * b
    fgraph = FunctionGraph([a, b, alpha, beta], [out])
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # 2D * 2D
    out = A.dot(A * alpha) + beta * A
    fgraph = FunctionGraph([A, alpha, beta], [out])
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # 1D * 2D and 1D * 1D
    out = y.dot(alpha * A).dot(x) + beta * y
    fgraph = FunctionGraph([y, x, A, alpha, beta], [out])
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
