"""Tests for ONNX backend extra operations (Tier 5)."""

import numpy as np
import pytest

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# CumSum Tests


@pytest.mark.parametrize("axis", [0, 1])
def test_cumsum(axis):
    """Test cumulative sum operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.matrix("x", dtype="float32")
    y = pt.cumsum(x, axis=axis)

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.cumsum(x_val, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert "CumSum" in node_types


# Repeat Tests


def test_repeat():
    """Test repeat operation (repeat elements)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.repeat(x, repeats=3, axis=0)

    x_val = np.array([1, 2, 3], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.repeat(x_val, repeats=3, axis=0)
    np.testing.assert_array_equal(result, expected)

    # Repeat in ONNX can be done with Tile or Expand


# Unique Tests


def test_unique():
    """Test unique operation (find unique elements).

    Note: ONNX Unique has different semantics than NumPy.
    May need special handling.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="int64")
    y = pt.unique(x)

    x_val = np.array([1, 2, 3, 2, 1, 4, 3], dtype="int64")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.unique(x_val)

    # Result may be sorted differently
    np.testing.assert_array_equal(sorted(result), sorted(expected))

    node_types = get_onnx_node_types(fn)
    assert "Unique" in node_types


# Pad Tests


def test_pad():
    """Test pad operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.matrix("x", dtype="float32")
    # Pad with 1 zero on each side
    y = pt.pad(x, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0)

    x_val = np.array([[1, 2], [3, 4]], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.pad(x_val, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert "Pad" in node_types
