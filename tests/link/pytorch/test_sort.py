import numpy as np
import pytest

from pytensor.graph import FunctionGraph
from pytensor.tensor import matrix
from pytensor.tensor.sort import argsort, sort
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize("func", (sort, argsort))
@pytest.mark.parametrize("axis", [0, 1, None])
def test_sort(func, axis):
    x = matrix("x", shape=(2, 2), dtype="float64")
    out = func(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    arr = np.array([[1.0, 4.0], [5.0, 2.0]])
    compare_pytorch_and_py(fgraph, [arr])
