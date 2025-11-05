"""Tests for ONNX elemwise operations."""

import numpy as np
import pytest

import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Test binary arithmetic operations
def test_add_vectors():
    """Test that vector addition exports correctly to ONNX."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    # Define graph
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    # Test data
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    # Compare outputs
    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Add" in node_types, f"Expected 'Add' node in ONNX graph, got {node_types}"


def test_mul_vectors():
    """Test that vector multiplication exports correctly to ONNX."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x * y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    assert "Mul" in get_onnx_node_types(fn)


def test_sub_vectors():
    """Test vector subtraction."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x - y

    x_val = np.array([5, 6, 7], dtype="float32")
    y_val = np.array([1, 2, 3], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Sub" in get_onnx_node_types(fn)


def test_div_vectors():
    """Test vector division."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x / y

    x_val = np.array([6, 8, 10], dtype="float32")
    y_val = np.array([2, 4, 5], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Div" in get_onnx_node_types(fn)


def test_chained_arithmetic():
    """Test that chained arithmetic operations work correctly."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype="float32")

    fn, result = compare_onnx_and_py([x], z, [x_val])

    # Should have multiple operation nodes
    node_types = get_onnx_node_types(fn)
    assert "Mul" in node_types
    assert "Add" in node_types
    assert "Div" in node_types


# Test unary operations
def test_neg():
    """Test negation operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = -x

    x_val = np.array([1, -2, 3], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert "Neg" in get_onnx_node_types(fn)


def test_abs():
    """Test absolute value operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.abs(x)

    x_val = np.array([1, -2, 3, -4], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert "Abs" in get_onnx_node_types(fn)


def test_exp():
    """Test exponential operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.exp(x)

    x_val = np.array([0, 1, 2], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert "Exp" in get_onnx_node_types(fn)


def test_log():
    """Test natural logarithm operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.log(x)

    x_val = np.array([1, 2, np.e], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert "Log" in get_onnx_node_types(fn)


def test_sqrt():
    """Test square root operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.sqrt(x)

    x_val = np.array([1, 4, 9, 16], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert "Sqrt" in get_onnx_node_types(fn)


def test_pow():
    """Test power operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x**y

    x_val = np.array([2, 3, 4], dtype="float32")
    y_val = np.array([2, 2, 3], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Pow" in get_onnx_node_types(fn)


@pytest.mark.parametrize(
    "op_name,op_func,expected_node",
    [
        ("floor", pt.floor, "Floor"),
        ("ceil", pt.ceil, "Ceil"),
        ("round", pt.round, "Round"),
    ],
)
def test_rounding_operations(op_name, op_func, expected_node):
    """Test floor, ceil, and round operations."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = op_func(x)

    x_val = np.array([1.2, 2.5, 3.7, -1.5], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert (
        expected_node in get_onnx_node_types(fn)
    ), f"Expected {expected_node} node for {op_name}"


def test_maximum():
    """Test element-wise maximum operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.maximum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([4, 2, 6], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Max" in get_onnx_node_types(fn)


def test_minimum():
    """Test element-wise minimum operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.minimum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([4, 2, 6], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Min" in get_onnx_node_types(fn)
