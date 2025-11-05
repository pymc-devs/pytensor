"""Tests for ONNXLinker."""

import numpy as np
import pytest

from pytensor.compile.mode import Mode


def test_linker_instantiation():
    """Test that ONNXLinker can be instantiated."""
    from pytensor.link.onnx.linker import ONNXLinker

    linker = ONNXLinker(opset_version=18)

    assert linker is not None, "Linker instantiation returned None"
    assert linker.opset_version == 18, f"Expected opset 18, got {linker.opset_version}"


def test_linker_empty_graph():
    """Test that linker can convert a trivial passthrough graph."""
    import pytensor
    import pytensor.tensor as pt

    from pytensor.link.onnx.linker import ONNXLinker

    # Create identity graph
    x = pt.scalar("x", dtype="float32")
    y = x  # Passthrough

    # Compile with ONNX linker
    fn = pytensor.function([x], y, mode=Mode(linker=ONNXLinker()))

    # Test execution
    result = fn(5.0)
    assert result == 5.0, f"Expected 5.0, got {result}"

    # Verify ONNX model exists
    assert hasattr(
        fn.maker.linker, "onnx_model"
    ), "Linker should have onnx_model attribute"
    assert (
        fn.maker.linker.onnx_model is not None
    ), "onnx_model should not be None"


def test_linker_constant_graph():
    """Test that linker correctly handles constants as initializers."""
    import pytensor
    import pytensor.tensor as pt

    from pytensor.link.onnx.linker import ONNXLinker

    # Create graph with constant
    x = pt.scalar("x", dtype="float32")
    c = pt.constant(2.0, dtype="float32")
    y = x * c

    # Compile
    fn = pytensor.function([x], y, mode=Mode(linker=ONNXLinker()))

    # Test
    result = fn(3.0)
    expected = 6.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify ONNX model has initializer for constant
    model = fn.maker.linker.onnx_model
    assert (
        len(model.graph.initializer) > 0
    ), "Model should have at least one initializer for the constant"
