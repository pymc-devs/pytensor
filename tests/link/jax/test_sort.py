import numpy as np
import pytest

from pytensor.graph import FunctionGraph
from pytensor.tensor import matrix
from pytensor.tensor.sort import sort
from tests.link.jax.test_basic import compare_jax_and_py


@pytest.mark.parametrize("axis", [None, -1])
def test_sort(axis):
    x = matrix("x", shape=(2, 2), dtype="float64")
    out = sort(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    arr = np.array([[1.0, 4.0], [5.0, 2.0]])
    compare_jax_and_py(fgraph, [arr])
