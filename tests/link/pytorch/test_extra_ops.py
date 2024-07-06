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


def test_pytorch_Repeat():
    a = pt.matrix("a", dtype="float64")

    test_value = np.arange(6, dtype="float64").reshape((3, 2))

    # Test along axis 0
    out = pt.repeat(a, (1, 2, 3), axis=0)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])

    # Test along axis 1
    out = pt.repeat(a, (3, 3), axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])


def test_pytorch_Unique():
    a = pt.matrix("a", dtype="float64")

    test_value = np.array(
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 0.0]], dtype="float64"
    )

    # Test along axis 0
    out = pt.unique(a, axis=0)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])

    # Test along axis 1
    out = pt.unique(a, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])

    # Test with params
    out = pt.unique(a, return_inverse=True, return_counts=True, axis=0)
    fgraph = FunctionGraph([a], [out[0]])
    compare_pytorch_and_py(fgraph, [test_value])

    # Test with return_index=True
    out = pt.unique(a, return_index=True, axis=0)
    fgraph = FunctionGraph([a], [out[0]])

    with pytest.raises(NotImplementedError):
        compare_pytorch_and_py(fgraph, [test_value])
