import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize(
    "axis",
    [
        None,
        1,
    ],
)
def test_pytorch_CumOp(axis):
    """Test PyTorch conversion of the `CumOp` `Op`."""

    # Create a symbolic input for the first input of `CumOp`
    a = pt.matrix("a")

    # Create test value tag for a
    test_value = np.arange(9, dtype=config.floatX).reshape((3, 3))

    # Create the output variable
    out = pt.cumsum(a, axis=axis)

    # Create a PyTensor `FunctionGraph`
    fgraph = FunctionGraph([a], [out])

    # Pass the graph and inputs to the testing function
    compare_pytorch_and_py(fgraph, [test_value])

    # For the second mode of CumOp
    out = pt.cumprod(a, axis=axis)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])
