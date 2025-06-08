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
from pytensor.link.mlx.dispatch.core import mlx_funcify_ScalarFromTensor


optimizer = RewriteDatabaseQuery(include=["mlx"], exclude=MLX._optimizer.exclude)
mlx_mode = Mode(linker=MLXLinker(), optimizer=optimizer)
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
    # Create a 0-d tensor (scalar tensor)
    x = pt.as_tensor_variable(42)

    # Apply ScalarFromTensor
    scalar_result = pt.scalar_from_tensor(x)

    # Create function and test
    f = pytensor.function([], scalar_result, mode="MLX")
    result = f()

    assert result == 42
