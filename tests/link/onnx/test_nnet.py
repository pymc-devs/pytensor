"""Tests for ONNX backend neural network operations (Tier 5)."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor.special import log_softmax, softmax
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Softmax Tests


@pytest.mark.parametrize("axis", [None, -1, 0, 1])
def test_softmax(axis):
    """Test softmax activation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from scipy.special import softmax as scipy_softmax

    x = pt.matrix("x", dtype="float32")
    y = softmax(x, axis=axis)

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Compute expected with scipy
    # Note: axis=None applies to the entire flattened array
    expected = scipy_softmax(x_val, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert "Softmax" in node_types


def test_logsoftmax():
    """Test log-softmax activation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from scipy.special import log_softmax as scipy_log_softmax

    x = pt.matrix("x", dtype="float32")
    # Explicitly specify axis=1 to match typical neural network usage
    y = log_softmax(x, axis=1)

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = scipy_log_softmax(x_val, axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert "LogSoftmax" in node_types


# Switch Test


def test_switch():
    """Test Switch operation (element-wise conditional).

    Switch(condition, then_value, else_value) returns:
    - then_value where condition is True
    - else_value where condition is False

    In ONNX this maps to Where operator.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")

    result = pt.switch(condition, x, y)

    cond_val = np.array([True, False, True, False, True], dtype=bool)
    x_val = np.array([1, 2, 3, 4, 5], dtype="float32")
    y_val = np.array([10, 20, 30, 40, 50], dtype="float32")

    fn, output = compare_onnx_and_py(
        [condition, x, y], result, [cond_val, x_val, y_val]
    )

    expected = np.where(cond_val, x_val, y_val)
    np.testing.assert_array_equal(output, expected)

    node_types = get_onnx_node_types(fn)
    assert "Where" in node_types, f"Expected 'Where' node, got {node_types}"
