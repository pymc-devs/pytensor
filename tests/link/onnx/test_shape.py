"""Tests for ONNX shape operations."""

import pytest
import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.shape import Shape_i

# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


def test_shape_basic():
    """Test Shape operation (single node return)."""
    x = pt.matrix('x', dtype='float32')
    y = x.shape

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([3, 4], dtype='int64')
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types


def test_shape_i_dim0():
    """Test Shape_i getting dimension 0 (multi-node return)."""
    x = pt.matrix('x', dtype='float32')
    # Use Shape_i directly to test the multi-node return pattern
    shape_i_op = Shape_i(0)
    y = shape_i_op(x)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result == 3

    # Verify multi-node pattern: Constant + Shape + Gather
    node_types = get_onnx_node_types(fn)
    assert 'Constant' in node_types
    assert 'Shape' in node_types
    assert 'Gather' in node_types


def test_shape_i_dim1():
    """Test Shape_i getting dimension 1 (multi-node return)."""
    x = pt.matrix('x', dtype='float32')
    # Use Shape_i directly
    shape_i_op = Shape_i(1)
    y = shape_i_op(x)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result == 4

    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types
    assert 'Gather' in node_types


def test_shape_i_3d_tensor():
    """Test Shape_i with 3D tensor."""
    x = pt.tensor3('x', dtype='float32')
    # Use Shape_i directly for each dimension
    dim0 = Shape_i(0)(x)
    dim1 = Shape_i(1)(x)
    dim2 = Shape_i(2)(x)

    x_val = np.random.randn(2, 3, 4).astype('float32')

    # Test each dimension separately
    fn0, result0 = compare_onnx_and_py([x], dim0, [x_val])
    assert result0 == 2

    fn1, result1 = compare_onnx_and_py([x], dim1, [x_val])
    assert result1 == 3

    fn2, result2 = compare_onnx_and_py([x], dim2, [x_val])
    assert result2 == 4


def test_specify_shape_passthrough():
    """Test that SpecifyShape creates no ONNX nodes (None return)."""
    from pytensor.tensor.shape import specify_shape

    x = pt.vector('x', dtype='float32')
    # SpecifyShape should pass through without creating ONNX nodes
    x_specified = specify_shape(x, (4,))
    y = x_specified * 2.0

    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # SpecifyShape should not appear in ONNX graph
    node_types = get_onnx_node_types(fn)
    assert 'SpecifyShape' not in node_types
    assert 'Mul' in node_types

    expected = x_val * 2.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_concatenate_axis0():
    """Test concatenate operation along axis 0."""
    x = pt.matrix('x', dtype='float32')
    y = pt.matrix('y', dtype='float32')
    z = pt.concatenate([x, y], axis=0)

    x_val = np.random.randn(2, 3).astype('float32')
    y_val = np.random.randn(4, 3).astype('float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.concatenate([x_val, y_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert 'Concat' in node_types


def test_concatenate_axis1():
    """Test concatenate operation along axis 1."""
    x = pt.matrix('x', dtype='float32')
    y = pt.matrix('y', dtype='float32')
    z = pt.concatenate([x, y], axis=1)

    x_val = np.random.randn(3, 2).astype('float32')
    y_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.concatenate([x_val, y_val], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert 'Concat' in node_types


def test_stack_axis0():
    """Test stack operation along axis 0."""
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = pt.stack([x, y], axis=0)

    x_val = np.array([1.0, 2.0, 3.0], dtype='float32')
    y_val = np.array([4.0, 5.0, 6.0], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.stack([x_val, y_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    # Stack uses Join which maps to Concat, along with Unsqueeze
    assert 'Concat' in node_types or 'Unsqueeze' in node_types


def test_split_equal():
    """Test split operation with equal sizes."""
    from pytensor.tensor.basic import split

    x = pt.vector('x', dtype='float32')
    splits_var = pt.constant([2, 2, 2], dtype='int64')
    a, b, c = split(x, splits_var, n_splits=3, axis=0)

    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32')

    fn, results = compare_onnx_and_py([x], [a, b, c], [x_val])

    expected_a = x_val[:2]
    expected_b = x_val[2:4]
    expected_c = x_val[4:]

    np.testing.assert_allclose(results[0], expected_a, rtol=1e-5)
    np.testing.assert_allclose(results[1], expected_b, rtol=1e-5)
    np.testing.assert_allclose(results[2], expected_c, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert 'Split' in node_types


def test_split_unequal():
    """Test split operation with unequal sizes."""
    from pytensor.tensor.basic import split

    x = pt.vector('x', dtype='float32')
    splits_var = pt.constant([3, 2, 1], dtype='int64')
    a, b, c = split(x, splits_var, n_splits=3, axis=0)

    x_val = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')

    fn, results = compare_onnx_and_py([x], [a, b, c], [x_val])

    expected_a = x_val[:3]
    expected_b = x_val[3:5]
    expected_c = x_val[5:]

    np.testing.assert_allclose(results[0], expected_a, rtol=1e-5)
    np.testing.assert_allclose(results[1], expected_b, rtol=1e-5)
    np.testing.assert_allclose(results[2], expected_c, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert 'Split' in node_types
