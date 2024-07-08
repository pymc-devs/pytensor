import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor.math import argmax
from pytensor.tensor.type import matrix
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize(
    "keepdims",
    [True, False],
)
@pytest.mark.parametrize(
    "axis",
    [None, 1, (0,)],
)
def test_pytorch_argmax(axis, keepdims):
    a = matrix("a", dtype=config.floatX)
    a.tag.test_value = np.random.randn(4, 4).astype(config.floatX)
    amx = argmax(a, axis=axis, keepdims=keepdims)
    fgraph = FunctionGraph([a], amx)
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
