"""Tests for ONNX backend module structure and imports."""

import pytest


def test_onnx_module_exists():
    """Test that pytensor.link.onnx module exists and is importable."""
    try:
        import pytensor.link.onnx  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import pytensor.link.onnx: {e}")


def test_onnx_public_api():
    """Test that ONNX backend exports expected public API."""
    from pytensor.link.onnx import (
        ONNX_OPSET_VERSION,
        ONNXLinker,
        compile_onnx,
        export_onnx,
        onnx_funcify,
    )

    assert ONNXLinker is not None, "ONNXLinker not exported"
    assert export_onnx is not None, "export_onnx not exported"
    assert compile_onnx is not None, "compile_onnx not exported"
    assert onnx_funcify is not None, "onnx_funcify not exported"
    assert ONNX_OPSET_VERSION == 18, f"Expected opset 18, got {ONNX_OPSET_VERSION}"


def test_dispatch_module_structure():
    """Test that dispatch module has expected structure."""
    from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify

    # Check they're singledispatch functions
    assert hasattr(onnx_funcify, "register"), (
        "onnx_funcify should be a singledispatch function"
    )
    assert hasattr(onnx_typify, "register"), (
        "onnx_typify should be a singledispatch function"
    )
