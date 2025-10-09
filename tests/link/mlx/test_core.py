import numpy as np
import pytest

import pytensor
from pytensor import tensor as pt
from pytensor.tensor.basic import Alloc
from tests.link.mlx.test_basic import compile_mode, mlx_mode_no_compile, mx


def test_alloc_with_different_shape_types():
    """Test Alloc works with different types of shape parameters.

    This addresses the TypeError that occurred when shape parameters
    contained MLX arrays instead of Python integers.
    """
    from pytensor.link.mlx.dispatch.core import (
        mlx_funcify_Alloc,
    )

    # Create a mock node (we don't need a real node for this test)
    class MockNode:
        def __init__(self):
            self.op = Alloc()
            self.inputs = None
            self.outputs = None

    alloc_func = mlx_funcify_Alloc(Alloc(), MockNode())
    x = mx.array(5.0)

    # Test with Python ints
    result = alloc_func(x, 3, 4)
    assert result.shape == (3, 4)
    assert float(result[0, 0]) == 5.0

    # Test with MLX arrays (this used to fail)
    result = alloc_func(x, mx.array(3), mx.array(4))
    assert result.shape == (3, 4)
    assert float(result[0, 0]) == 5.0

    # Test with mixed types
    result = alloc_func(x, 3, mx.array(4))
    assert result.shape == (3, 4)
    assert float(result[0, 0]) == 5.0


def test_alloc_pytensor_integration():
    """Test Alloc in a PyTensor graph context."""
    # Test basic constant shape allocation
    x = pt.scalar("x", dtype="float32")
    result = pt.alloc(x, 3, 4)

    f = pytensor.function([x], result, mode="MLX")
    output = f(5.0)

    assert output.shape == (3, 4)
    assert float(output[0, 0]) == 5.0


def test_alloc_compilation_limitation():
    """Test that Alloc operations with dynamic shapes provide helpful error in compiled contexts."""

    # Create variables
    x = pt.scalar("x", dtype="float32")
    s1 = pt.scalar("s1", dtype="int64")
    s2 = pt.scalar("s2", dtype="int64")

    # Create Alloc operation with dynamic shapes
    result = pt.alloc(x, s1, s2)

    # Create function with non-compiled MLX mode
    f = pytensor.function([x, s1, s2], result, mode=mlx_mode_no_compile)

    # Test that it works with concrete values (non-compiled context)
    output = f(5.0, 3, 4)
    assert output.shape == (3, 4)
    assert np.allclose(output, 5.0)

    # Test that compilation fails with helpful error
    compiled_f = pytensor.function([x, s1, s2], result, mode=compile_mode)

    with pytest.raises(ValueError) as exc_info:
        compiled_f(5.0, 3, 4)

    error_msg = str(exc_info.value)
    assert "MLX compilation limitation" in error_msg
    assert "Alloc operations with dynamic shapes" in error_msg
    assert "cannot be used inside compiled functions" in error_msg
    assert "Workarounds:" in error_msg
    assert "Avoid using Alloc with dynamic shapes in compiled contexts" in error_msg
    assert "Use static shapes when possible" in error_msg
    assert "Move Alloc operations outside compiled functions" in error_msg


def test_alloc_static_shapes_compilation():
    """Test that Alloc operations with static shapes work fine in compiled contexts."""
    # Create a scenario with static shapes that should work
    x = pt.scalar("x", dtype="float32")

    # Use constant shape - this should work even in compilation
    result = pt.alloc(x, 3, 4)  # Static shapes

    # Test both compiled and non-compiled modes
    f_normal = pytensor.function([x], result, mode=mlx_mode_no_compile)
    f_compiled = pytensor.function([x], result, mode=compile_mode)

    # Both should work
    output_normal = f_normal(5.0)
    output_compiled = f_compiled(5.0)

    assert output_normal.shape == (3, 4)
    assert output_compiled.shape == (3, 4)
    assert np.allclose(output_normal, 5.0)
    assert np.allclose(output_compiled, 5.0)
    assert np.allclose(output_normal, output_compiled)
