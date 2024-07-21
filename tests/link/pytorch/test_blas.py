import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import blas as pt_blas
from pytensor.tensor.type import tensor3
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a_test = np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    b = tensor3("b")
    b_test = np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    out = pt_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    pytensor_pytorch_fn, _ = compare_pytorch_and_py(fgraph, [a_test, b_test])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [a_test[:-1], b_test]
    with pytest.raises(TypeError):
        pytensor_pytorch_fn(*inputs)
