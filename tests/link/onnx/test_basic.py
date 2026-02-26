"""Core testing utilities for ONNX backend."""

from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Variable


# These will be imported once the ONNX backend is implemented
# For now, we'll set up the structure so tests can use them
try:
    from pytensor.link.onnx import ONNXLinker

    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    py_mode = Mode(linker="py", optimizer=None)
except ImportError:
    # ONNX backend not yet implemented
    onnx_mode = None
    py_mode = Mode(linker="py", optimizer=None)


def compare_onnx_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_validate: bool = True,
    onnx_mode=onnx_mode,
    py_mode=py_mode,
):
    """Compare ONNX Runtime output to Python reference.

    This is the core testing utility that:
    1. Compiles graph with ONNX backend
    2. Compiles graph with Python backend
    3. Executes both with test_inputs
    4. Asserts outputs match
    5. Validates ONNX model

    Parameters
    ----------
    graph_inputs : Iterable[Variable]
        Symbolic inputs to the graph
    graph_outputs : Variable | Iterable[Variable]
        Symbolic outputs of the graph
    test_inputs : Iterable
        Numerical inputs for testing the function
    assert_fn : Callable, optional
        Assert function used to check for equality between ONNX and Python.
        If not provided, uses np.testing.assert_allclose with rtol=1e-4
    must_validate : bool, optional
        If True, validates the ONNX model with onnx.checker.check_model
    onnx_mode : Mode, optional
        Mode to use for ONNX compilation
    py_mode : Mode, optional
        Mode to use for Python reference compilation

    Returns
    -------
    tuple
        (onnx_function, onnx_result)

    Raises
    ------
    AssertionError
        If ONNX output doesn't match Python output
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables")

    # Compile with ONNX backend
    onnx_fn = function(graph_inputs, graph_outputs, mode=onnx_mode)
    onnx_res = onnx_fn(*test_inputs)

    # Compile with Python reference
    py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = py_fn(*test_inputs)

    # Compare outputs
    if isinstance(graph_outputs, list | tuple):
        for o, p in zip(onnx_res, py_res, strict=True):
            assert_fn(o, p)
    else:
        assert_fn(onnx_res, py_res)

    # Validate ONNX model
    if must_validate and hasattr(onnx_fn.maker.linker, "onnx_model"):
        import onnx

        onnx.checker.check_model(onnx_fn.maker.linker.onnx_model)

    return onnx_fn, onnx_res


def get_onnx_node_types(fn):
    """Get list of ONNX node types in compiled function.

    Parameters
    ----------
    fn : Function
        Compiled PyTensor function with ONNX linker

    Returns
    -------
    list of str
        List of ONNX operation types (e.g., ['Add', 'Mul', 'Sub'])
    """
    if not hasattr(fn.maker.linker, "onnx_model"):
        raise ValueError("Function was not compiled with ONNX linker")

    return [node.op_type for node in fn.maker.linker.onnx_model.graph.node]


# Meta-test: test the test utilities themselves
def test_compare_onnx_and_py_simple():
    """Test that compare_onnx_and_py works for a simple identity operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    import pytensor.tensor as pt

    # Simple identity
    x = pt.vector("x", dtype="float32")
    y = x

    # Test data
    x_val = np.array([1, 2, 3], dtype="float32")

    # Should not raise
    try:
        _fn, result = compare_onnx_and_py([x], y, [x_val])
        np.testing.assert_array_equal(result, x_val)
    except Exception as e:
        pytest.fail(f"compare_onnx_and_py raised unexpectedly: {e}")


def test_get_onnx_node_types():
    """Test that get_onnx_node_types utility works."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    import pytensor
    import pytensor.tensor as pt
    from pytensor.link.onnx.linker import ONNXLinker

    # Create a graph with Add operation
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    # Compile
    fn = pytensor.function([x, y], z, mode=Mode(linker=ONNXLinker()))

    # Get node types
    node_types = get_onnx_node_types(fn)

    assert "Add" in node_types, f"Expected 'Add' in node types, got {node_types}"
