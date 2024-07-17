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
    A = np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    b = tensor3("b")
    B = np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    out = pt_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    pytensor_pytorch_fn, _ = compare_pytorch_and_py(fgraph, [A, B])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [A[:-1], B]
    with pytest.raises(TypeError):
        pytensor_pytorch_fn(*inputs)
