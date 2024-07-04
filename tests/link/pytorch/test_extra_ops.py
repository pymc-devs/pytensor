import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.graph import FunctionGraph
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize(
    "dtype",
    ["float64", "int64"],
)
@pytest.mark.parametrize(
    "axis",
    [None, 1, (0,)],
)
def test_pytorch_CumOp(axis, dtype):
    """Test PyTorch conversion of the `CumOp` `Op`."""

    # Create a symbolic input for the first input of `CumOp`
    a = pt.matrix("a", dtype=dtype)

    # Create test value
    test_value = np.arange(9, dtype=dtype).reshape((3, 3))

    # Create the output variable
    if isinstance(axis, tuple):
        with pytest.raises(TypeError, match="axis must be an integer or None."):
            out = pt.cumsum(a, axis=axis)
        with pytest.raises(TypeError, match="axis must be an integer or None."):
            out = pt.cumprod(a, axis=axis)
    else:
        out = pt.cumsum(a, axis=axis)
        # Create a PyTensor `FunctionGraph`
        fgraph = FunctionGraph([a], [out])

        # Pass the graph and inputs to the testing function
        compare_pytorch_and_py(fgraph, [test_value])

        # For the second mode of CumOp
        out = pt.cumprod(a, axis=axis)
        fgraph = FunctionGraph([a], [out])
        compare_pytorch_and_py(fgraph, [test_value])
