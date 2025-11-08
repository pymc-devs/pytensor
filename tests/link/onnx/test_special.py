"""Tests for ONNX backend special operations (Tier 5)."""

import numpy as np
import pytest

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Trigonometric Functions


@pytest.mark.parametrize(
    "pt_op,np_op,onnx_op",
    [
        (pt.sin, np.sin, "Sin"),
        (pt.cos, np.cos, "Cos"),
        (pt.tan, np.tan, "Tan"),
        (pt.arcsin, np.arcsin, "Asin"),
        (pt.arccos, np.arccos, "Acos"),
        (pt.arctan, np.arctan, "Atan"),
    ],
)
def test_trigonometric_functions(pt_op, np_op, onnx_op):
    """Test trigonometric functions."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt_op(x)

    # Use values in appropriate domain
    if pt_op in [pt.arcsin, pt.arccos]:
        # Domain [-1, 1]
        x_val = np.linspace(-0.9, 0.9, 10).astype("float32")
    else:
        x_val = np.linspace(-3, 3, 10).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types, f"Expected '{onnx_op}' node, got {node_types}"


# Hyperbolic Functions


@pytest.mark.parametrize(
    "pt_op,np_op,onnx_op",
    [
        (pt.sinh, np.sinh, "Sinh"),
        (pt.cosh, np.cosh, "Cosh"),
        (pt.tanh, np.tanh, "Tanh"),
        (pt.arcsinh, np.arcsinh, "Asinh"),
        (pt.arccosh, np.arccosh, "Acosh"),
        (pt.arctanh, np.arctanh, "Atanh"),
    ],
)
def test_hyperbolic_functions(pt_op, np_op, onnx_op):
    """Test hyperbolic functions."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt_op(x)

    # Use values in appropriate domain
    if pt_op == pt.arccosh:
        # Domain [1, inf)
        x_val = np.linspace(1.1, 3, 10).astype("float32")
    elif pt_op == pt.arctanh:
        # Domain (-1, 1)
        x_val = np.linspace(-0.9, 0.9, 10).astype("float32")
    else:
        x_val = np.linspace(-2, 2, 10).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types


# Comparison Operations


@pytest.mark.parametrize(
    "pt_op,np_op,onnx_op",
    [
        (pt.lt, np.less, "Less"),
        (pt.gt, np.greater, "Greater"),
        (pt.le, np.less_equal, "LessOrEqual"),
        (pt.ge, np.greater_equal, "GreaterOrEqual"),
        (pt.eq, np.equal, "Equal"),
        (pt.neq, np.not_equal, "Not"),  # Not + Equal
    ],
)
def test_comparison_ops(pt_op, np_op, onnx_op):
    """Test comparison operations."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt_op(x, y)

    x_val = np.array([1, 2, 3, 4, 5], dtype="float32")
    y_val = np.array([2, 2, 2, 2, 2], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np_op(x_val, y_val)
    np.testing.assert_array_equal(result, expected)

    # Result should be boolean
    assert result.dtype == bool or result.dtype == np.bool_


# Logical Operations


@pytest.mark.parametrize(
    "pt_op,np_op,onnx_op",
    [
        (pt.and_, np.logical_and, "And"),
        (pt.or_, np.logical_or, "Or"),
        (pt.xor, np.logical_xor, "Xor"),
        (pt.invert, np.logical_not, "Not"),
    ],
)
def test_logical_ops(pt_op, np_op, onnx_op):
    """Test logical operations."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    if pt_op == pt.invert:
        # Unary operation
        x = pt.vector("x", dtype="bool")
        y = pt_op(x)

        x_val = np.array([True, False, True, False, True], dtype=bool)

        fn, result = compare_onnx_and_py([x], y, [x_val])

        expected = np_op(x_val)
        np.testing.assert_array_equal(result, expected)
    else:
        # Binary operation
        x = pt.vector("x", dtype="bool")
        y_tensor = pt.vector("y", dtype="bool")
        z = pt_op(x, y_tensor)

        x_val = np.array([True, True, False, False], dtype=bool)
        y_val = np.array([True, False, True, False], dtype=bool)

        fn, result = compare_onnx_and_py([x, y_tensor], z, [x_val, y_val])

        expected = np_op(x_val, y_val)
        np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types


# Special Math Functions


@pytest.mark.parametrize(
    "pt_op,onnx_op",
    [
        (pt.sigmoid, "Sigmoid"),
        (pt.softplus, "Softplus"),
    ],
)
def test_sigmoid_softplus(pt_op, onnx_op):
    """Test sigmoid and softplus activations."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt_op(x)

    x_val = np.linspace(-5, 5, 20).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Verify with manual computation
    if pt_op == pt.sigmoid:
        expected = 1 / (1 + np.exp(-x_val))
    else:  # softplus
        expected = np.log(1 + np.exp(x_val))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types


def test_erf():
    """Test error function."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from scipy import special

    x = pt.vector("x", dtype="float32")
    y = pt.erf(x)

    x_val = np.linspace(-3, 3, 20).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = special.erf(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert "Erf" in node_types


@pytest.mark.parametrize(
    "pt_op,np_op",
    [
        (pt.log1p, np.log1p),
        (pt.expm1, np.expm1),
    ],
)
def test_log1p_expm1(pt_op, np_op):
    """Test log1p and expm1 functions.

    These may not have direct ONNX ops, but can be composed:
    - log1p(x) = log(1 + x) using Add + Log
    - expm1(x) = exp(x) - 1 using Exp + Sub
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt_op(x)

    x_val = np.linspace(-0.5, 2, 20).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_clip():
    """Test clip operation (clamp values to range)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.clip(x, -1.0, 1.0)

    x_val = np.array([-2, -0.5, 0, 0.5, 2], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.clip(x_val, -1.0, 1.0)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert "Clip" in node_types, f"Expected 'Clip' node, got {node_types}"
