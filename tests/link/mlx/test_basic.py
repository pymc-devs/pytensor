from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Variable
from pytensor.link.mlx import MLXLinker


mx = pytest.importorskip("mlx.core")

mlx_mode = Mode(linker=MLXLinker())
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
