from collections.abc import Callable, Iterable
from functools import partial

import mlx.core as mx
import numpy as np

import pytensor
from pytensor import tensor as pt
from pytensor.compile.function import function
from pytensor.compile.mode import MLX, Mode
from pytensor.graph import RewriteDatabaseQuery
from pytensor.graph.basic import Variable
from pytensor.link.mlx import MLXLinker
from pytensor.link.mlx.dispatch.core import (
    mlx_funcify_Alloc,
    mlx_funcify_ScalarFromTensor,
)
from pytensor.tensor.basic import Alloc


optimizer = RewriteDatabaseQuery(include=["mlx"], exclude=MLX._optimizer.exclude)
mlx_mode = Mode(linker=MLXLinker(), optimizer=optimizer)
mlx_mode_no_compile = Mode(linker=MLXLinker(use_compile=False), optimizer=optimizer)
compile_mode = Mode(linker=MLXLinker(use_compile=True), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_mlx_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    mlx_mode=mlx_mode,
    py_mode=py_mode,
):
    """Function to compare python function output and mlx compiled output for testing equality

    The inputs and outputs are then passed to this function which then compiles the given function in both
    mlx and python, runs the calculation in both and checks if the results are the same

    Parameters
    ----------
    graph_inputs:
        Symbolic inputs to the graph
    outputs:
        Symbolic outputs of the graph
    test_inputs: iter
        Numerical inputs for testing the function.
    assert_fn: func, opt
        Assert function used to check for equality between python and mlx. If not
        provided uses np.testing.assert_allclose
    must_be_device_array: Bool
        Checks for instance of jax.interpreters.xla.DeviceArray. For testing purposes
        if this device array is found it indicates if the result was computed by jax

    Returns
    -------
    mlx_res

    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables")

    pytensor_mlx_fn = function(graph_inputs, graph_outputs, mode=mlx_mode)
    mlx_res = pytensor_mlx_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(mlx_res, list):
            assert all(isinstance(res, mx.array) for res in mlx_res)
        else:
            assert isinstance(mlx_res, mx.array)

    pytensor_py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if isinstance(graph_outputs, list | tuple):
        for j, p in zip(mlx_res, py_res, strict=True):
            assert_fn(j, p)
    else:
        assert_fn(mlx_res, py_res)

    return pytensor_mlx_fn, mlx_res


def test_scalar_from_tensor_with_scalars():
    """Test ScalarFromTensor works with both MLX arrays and Python/NumPy scalars.

    This addresses the AttributeError that occurred when Python integers were
    passed to ScalarFromTensor instead of MLX arrays.
    """
    scalar_from_tensor_func = mlx_funcify_ScalarFromTensor(None)

    # Test with MLX array
    mlx_array = mx.array([42])
    result = scalar_from_tensor_func(mlx_array)
    assert result == 42

    # Test with Python int (this used to fail)
    python_int = 42
    result = scalar_from_tensor_func(python_int)
    assert result == 42

    # Test with Python float
    python_float = 3.14
    result = scalar_from_tensor_func(python_float)
    assert abs(result - 3.14) < 1e-6

    # Test with NumPy scalar
    numpy_scalar = np.int32(123)
    result = scalar_from_tensor_func(numpy_scalar)
    assert result == 123

    # Test with NumPy float scalar
    numpy_float = np.float32(2.71)
    result = scalar_from_tensor_func(numpy_float)
    assert abs(result - 2.71) < 1e-6


def test_scalar_from_tensor_pytensor_integration():
    """Test ScalarFromTensor in a PyTensor graph context."""
    # Create a symbolic scalar input to actually test MLX execution
    x = pt.scalar("x", dtype="int64")

    # Apply ScalarFromTensor
    scalar_result = pt.scalar_from_tensor(x)

    # Create function and test
    f = pytensor.function([x], scalar_result, mode="MLX")
    result = f(42)

    assert result == 42


def test_alloc_with_different_shape_types():
    """Test Alloc works with different types of shape parameters.

    This addresses the TypeError that occurred when shape parameters
    contained MLX arrays instead of Python integers.
    """

    # Create a mock node (we don't need a real node for this test)
    class MockNode:
        pass

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

    # Use MLX mode
    from pytensor.compile import mode

    mlx_mode = mode.get_mode("MLX")

    f = pytensor.function([x], result, mode=mlx_mode)
    output = f(5.0)

    assert output.shape == (3, 4)
    assert float(output[0, 0]) == 5.0


def test_alloc_compilation_limitation():
    """Test that Alloc operations with dynamic shapes provide helpful error in compiled contexts."""
    import pytest

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


def test_mlx_float64_auto_casting():
    """Test MLX automatic casting of float64 to float32 with warnings."""
    import warnings

    # Test 1: Direct Cast operation with warning
    x = pt.scalar("x", dtype="float32")
    y = pt.cast(x, "float64")

    # Capture warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        f = pytensor.function([x], y, mode=mlx_mode, allow_input_downcast=True)
        result = f(3.14)

        # Check that the operation succeeded
        assert result.dtype == mx.float32  # Should be auto-cast to float32
        assert abs(float(result) - 3.14) < 1e-6

        # Check that a warning was issued
        warning_messages = [str(w.message) for w in warning_list]
        dtype_warnings = [
            msg for msg in warning_messages if "float64" in msg and "float32" in msg
        ]
        assert (
            len(dtype_warnings) > 0
        ), f"Expected dtype warning, got warnings: {warning_messages}"


def test_mlx_float64_complex_operations():
    """Test float64 casting in more complex operations."""
    import warnings

    # Test with vector operations
    x = pt.vector("x", dtype="float32")
    y = pt.cast(x, "float64")
    z = pt.exp(y) + pt.sin(y)  # Multiple operations on float64

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        f = pytensor.function([x], z, mode=mlx_mode, allow_input_downcast=True)
        result = f([1.0, 2.0, 3.0])

        # Should work and return float32 results
        assert result.dtype == mx.float32
        assert result.shape == (3,)

        # Should have issued warnings
        warning_messages = [str(w.message) for w in warning_list]
        dtype_warnings = [
            msg
            for msg in warning_messages
            if "float64" in msg or "MLX GPU limitation" in msg
        ]
        assert len(dtype_warnings) > 0


def test_mlx_float64_no_warning_when_disabled():
    """Test that auto-casting can be controlled."""
    import warnings

    from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx

    # Test that we can disable auto-casting
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # This should not issue warnings when auto_cast_unsupported=False
        dtype = convert_dtype_to_mlx("float64", auto_cast_unsupported=False)
        assert dtype == mx.float64  # Should return the original dtype

        # No warnings should be issued for proactive conversion when disabled
        dtype_warnings = [
            str(w.message) for w in warning_list if "float64" in str(w.message)
        ]
        assert len(dtype_warnings) == 0


def test_mlx_complex128_auto_casting():
    """Test automatic casting of complex128 to complex64."""
    import warnings

    from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # This should trigger a warning and return complex64
        dtype = convert_dtype_to_mlx("complex128", auto_cast_unsupported=True)
        assert dtype == mx.complex64

        # Should have issued a warning
        warning_messages = [str(w.message) for w in warning_list]
        complex_warnings = [
            msg
            for msg in warning_messages
            if "complex128" in msg and "complex64" in msg
        ]
        assert len(complex_warnings) > 0
