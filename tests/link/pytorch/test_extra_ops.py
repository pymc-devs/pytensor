import numpy as np

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph
from pytensor.graph.op import get_test_value
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_CumOp():
    """Test PyTorch conversion of the `CumOp` `Op`."""

    # Create a symbolic input for the first input of `CumOp`
    a = pt.matrix("a")

    # Create test value tag for a
    a.tag.test_value = np.arange(9, dtype=config.floatX).reshape((3, 3))

    # Create the output variable
    out = pt.cumsum(a, axis=0)

    # Create a PyTensor `FunctionGraph`
    fgraph = FunctionGraph([a], [out])

    # Pass the graph and inputs to the testing function
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # For the second mode of CumOp
    out = pt.cumprod(a, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
