"""Tests for ONNX export API."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt


def test_export_onnx_basic(tmp_path):
    """Test that export_onnx creates a valid ONNX file."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    import onnx

    from pytensor.link.onnx import export_onnx

    # Define graph
    x = pt.vector("x", dtype="float32")
    y = x * 2

    # Export
    output_path = tmp_path / "test_model.onnx"
    model = export_onnx([x], y, str(output_path))

    # Verify file exists
    assert output_path.exists(), f"ONNX file not created at {output_path}"

    # Verify model is valid
    onnx.checker.check_model(model)

    # Verify model can be loaded
    loaded_model = onnx.load(str(output_path))
    assert loaded_model is not None


def test_compile_onnx_basic():
    """Test that compile_onnx returns an executable function."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from pytensor.link.onnx import compile_onnx

    x = pt.vector("x", dtype="float32")
    y = x + 1

    # Compile
    fn = compile_onnx([x], y)

    # Test execution
    x_val = np.array([1, 2, 3], dtype="float32")
    result = fn(x_val)

    expected = np.array([2, 3, 4], dtype="float32")
    np.testing.assert_array_equal(result, expected)


def test_export_function_onnx(tmp_path):
    """Test exporting a compiled PyTensor function to ONNX."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    import onnx

    from pytensor.link.onnx import export_function_onnx

    # Create and compile function
    x = pt.vector("x", dtype="float32")
    y = pt.sqrt(x)
    fn = pytensor.function([x], y)

    # Export
    output_path = tmp_path / "function.onnx"
    model = export_function_onnx(fn, str(output_path))

    # Verify
    assert output_path.exists()
    onnx.checker.check_model(model)
