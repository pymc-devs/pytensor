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


@pytest.mark.parametrize("axis, repeats", [(0, (1, 2, 3)), (1, (3, 3)), (None, 3)])
def test_pytorch_Repeat(axis, repeats):
    a = pt.matrix("a", dtype="float64")

    test_value = np.arange(6, dtype="float64").reshape((3, 2))

    out = pt.repeat(a, repeats, axis=axis)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_pytorch_Unique_axis(axis):
    a = pt.matrix("a", dtype="float64")

    test_value = np.array(
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 0.0]], dtype="float64"
    )

    out = pt.unique(a, axis=axis)
    fgraph = FunctionGraph([a], [out])
    compare_pytorch_and_py(fgraph, [test_value])


@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
@pytest.mark.parametrize(
    "return_index",
    (False, pytest.param(True, marks=pytest.mark.xfail(raises=NotImplementedError))),
)
def test_pytorch_Unique_params(return_index, return_inverse, return_counts):
    a = pt.matrix("a", dtype="float64")
    test_value = np.array(
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 0.0]], dtype="float64"
    )

    out = pt.unique(
        a,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=0,
    )
    fgraph = FunctionGraph([a], [out[0] if isinstance(out, list) else out])
    compare_pytorch_and_py(fgraph, [test_value])
