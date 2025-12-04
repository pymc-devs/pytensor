import numpy as np
import pytest

import pytensor
from pytensor import config
from pytensor import tensor as pt
from pytensor.tensor.basic import Alloc
from tests.link.mlx.test_basic import (
    compare_mlx_and_py,
    compile_mode,
    mlx_mode_no_compile,
    mx,
)


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
    np.testing.assert_allclose(output, 5.0)

    # Test that compilation fails with helpful error
    compiled_f = pytensor.function([x, s1, s2], result, mode=compile_mode)

    with pytest.raises(
        ValueError,
        match="MLX compilation limitation: Alloc operations with dynamic shapes cannot be "
        "used inside compiled functions",
    ):
        compiled_f(5.0, 3, 4)


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
    np.testing.assert_allclose(output_normal, 5.0)
    np.testing.assert_allclose(output_compiled, 5.0)
    np.testing.assert_allclose(output_normal, output_compiled)


def test_empty_static_shape():
    result = pt.empty((3, 4), dtype="float32")

    f = pytensor.function([], result, mode="MLX")
    output = f()

    assert output.shape == (3, 4)
    np.testing.assert_allclose(output, 0.0)


def test_empty_dynamic_shape():
    s1 = pt.scalar("s1", dtype="int64")
    s2 = pt.scalar("s2", dtype="int64")
    result = pt.empty((s1, s2), dtype="float32")

    f = pytensor.function([s1, s2], result, mode=mlx_mode_no_compile)
    output = f(3, 4)

    assert output.shape == (3, 4)
    np.testing.assert_allclose(output, 0.0)

    f_compiled = pytensor.function([s1, s2], result, mode=compile_mode)
    with pytest.raises(
        ValueError,
        match="MLX compilation limitation: Alloc operations with dynamic shapes cannot be "
        "used inside compiled functions",
    ):
        f_compiled(3, 4)


def test_split_const_axis_const_splits_compiled():
    x = pt.vector("x")
    splits = [2, 3]
    outs = pt.split(x, splits, n_splits=len(splits), axis=0)
    compare_mlx_and_py([x], outs, [np.arange(5, dtype="float32")])


def test_split_dynamic_axis_const_splits():
    x = pt.matrix("x")
    axis = pt.scalar("axis", dtype="int64")
    splits = [1, 2, 3]
    outs = pt.split(x, splits, n_splits=len(splits), axis=axis)

    test_input = np.arange(12).astype(config.floatX).reshape(2, 6)

    with pytest.raises(
        ValueError, match="Symbolic axis is not supported in MLX Split implementation"
    ):
        compare_mlx_and_py([x, axis], outs, [test_input, np.array(1)])
