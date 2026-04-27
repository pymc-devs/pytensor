"""Tests for ONNX dispatch system."""

import numpy as np
import pytest


def test_onnx_funcify_unregistered_op():
    """Test that onnx_funcify raises informative error for unregistered ops."""
    from pytensor.link.onnx.dispatch import onnx_funcify

    # Create a fake op
    class FakeOp:
        pass

    fake_op = FakeOp()

    with pytest.raises(NotImplementedError) as exc_info:
        onnx_funcify(fake_op)

    error_msg = str(exc_info.value)
    assert "No ONNX conversion available" in error_msg, (
        f"Error should mention no conversion available, got: {error_msg}"
    )
    assert "FakeOp" in error_msg, f"Error should mention the op type, got: {error_msg}"


def test_onnx_typify_ndarray():
    """Test that onnx_typify converts numpy arrays to ONNX tensors."""
    pytest.importorskip("onnx")

    import onnx
    from onnx import numpy_helper

    from pytensor.link.onnx.dispatch import onnx_typify

    # Test data
    arr = np.array([1, 2, 3], dtype="float32")

    # Convert
    result = onnx_typify(arr, name="test_tensor")

    # Verify it's a TensorProto
    assert isinstance(result, onnx.TensorProto), (
        f"Expected TensorProto, got {type(result)}"
    )

    # Verify data is correct
    result_arr = numpy_helper.to_array(result)
    np.testing.assert_array_equal(result_arr, arr)


def test_make_value_info_basic():
    """Test that make_value_info creates correct ONNX ValueInfo."""
    pytest.importorskip("onnx")

    import onnx

    import pytensor.tensor as pt
    from pytensor.link.onnx.dispatch.basic import make_value_info

    # Create a PyTensor variable
    x = pt.vector("x", dtype="float32")

    # Create ValueInfo
    value_info = make_value_info(x, "x")

    # Verify type
    assert isinstance(value_info, onnx.ValueInfoProto), (
        f"Expected ValueInfoProto, got {type(value_info)}"
    )

    # Verify name
    assert value_info.name == "x", f"Expected name 'x', got {value_info.name}"

    # Verify dtype
    assert value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, (
        f"Expected FLOAT dtype, got {value_info.type.tensor_type.elem_type}"
    )
