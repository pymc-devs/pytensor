import numpy as np
import pytest

from pytensor.graph import FunctionGraph
from pytensor.tensor import matrix
from pytensor.tensor.sort import argsort, sort
from tests.link.jax.test_basic import compare_jax_and_py


@pytest.mark.parametrize("axis", [None, -1])
@pytest.mark.parametrize("func", (sort, argsort))
def test_sort(func, axis):
    x = matrix("x", shape=(2, 2), dtype="float64")
    out = func(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    arr = np.array([[1.0, 4.0], [5.0, 2.0]])
    compare_jax_and_py(fgraph, [arr])
