import numpy as np
import pytest

import pytensor
from pytensor import tensor as pt
from pytensor.tensor.basic import Alloc, arange
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
    from pytensor.link.mlx.dispatch.tensor_basic import (
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


def test_split_symbolic_axis_rejected():
    x = pt.matrix("x")
    axis = pt.scalar("axis", dtype="int64")
    splits = [1, 2, 3]

    with pytest.raises(TypeError, match="Symbolic axes are no longer supported"):
        pt.split(x, splits, n_splits=len(splits), axis=axis)


def test_arange():
    out = arange(1, 10, 2)

    compare_mlx_and_py([], [out], [])


def test_arange_dynamic_shape():
    # Shape-derived bounds are concrete under mx.compile even when the static
    # shape is unknown, so a genuinely dynamic length must work (regression: this
    # used to raise NotImplementedError because of an over-aggressive constant
    # check). Exercises every position (start/stop/step) being shape-derived, an
    # offset, and an empty result.
    x = pt.vector("x")
    y = pt.vector("y")
    outs = [
        arange(x.shape[0]),  # dynamic stop
        arange(x.shape[0] + 2),  # shape-derived expression
        arange(x.shape[0], y.shape[0]),  # dynamic start and stop
        arange(0, y.shape[0], x.shape[0]),  # dynamic step
        arange(y.shape[0], x.shape[0]),  # start > stop -> empty
    ]
    compare_mlx_and_py(
        [x, y],
        outs,
        [np.zeros(3, dtype="float32"), np.zeros(7, dtype="float32")],
    )


def test_arange_dynamic_advanced_index():
    # The motivating case: a vectorized gather lowers to advanced indexing that
    # internally builds arange(idx.shape[0]) with a runtime-dynamic length.
    logp = pt.matrix("logp")
    targets = pt.lvector("targets")
    out = logp[arange(targets.shape[0]), targets]
    compare_mlx_and_py(
        [logp, targets],
        [out],
        [np.arange(12, dtype="float32").reshape(3, 4), np.array([0, 2, 3])],
    )


def test_arange_data_dependent_raises():
    # A genuinely data-dependent length has a runtime-only output shape, which MLX
    # cannot compile. This must fail loudly (at compile time) rather than silently.
    x = pt.vector("x")
    out = arange(pt.sum(x > 0).astype("int64"))
    with pytest.raises(NotImplementedError, match="data-dependent length"):
        pytensor.function([x], out, mode=compile_mode)
