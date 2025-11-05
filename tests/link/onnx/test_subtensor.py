"""Tests for ONNX subtensor (slicing) operations."""

import numpy as np
import pytest

# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


class TestSubtensorBasic:
    """Test basic slicing operations."""

    def test_slice_1d_basic(self):
        """Test basic 1D slicing: x[2:5]"""
        x = pt.vector('x', dtype='float32')
        y = x[2:5]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])

        # Verify correct output
        expected = x_val[2:5]
        np.testing.assert_array_equal(result, expected)

        # Verify ONNX uses Slice operation
        node_types = get_onnx_node_types(fn)
        assert 'Slice' in node_types, f"Expected 'Slice' in {node_types}"

    def test_slice_1d_from_start(self):
        """Test slicing from start: x[:5]"""
        x = pt.vector('x', dtype='float32')
        y = x[:5]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[:5]
        np.testing.assert_array_equal(result, expected)

    def test_slice_1d_to_end(self):
        """Test slicing to end: x[3:]"""
        x = pt.vector('x', dtype='float32')
        y = x[3:]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[3:]
        np.testing.assert_array_equal(result, expected)

    def test_slice_1d_with_step(self):
        """Test slicing with step: x[::2]"""
        x = pt.vector('x', dtype='float32')
        y = x[::2]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[::2]
        np.testing.assert_array_equal(result, expected)

    def test_slice_1d_with_step_range(self):
        """Test slicing with step and range: x[1:8:2]"""
        x = pt.vector('x', dtype='float32')
        y = x[1:8:2]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[1:8:2]
        np.testing.assert_array_equal(result, expected)

    def test_slice_2d_basic(self):
        """Test 2D slicing: x[1:3, 2:4]"""
        x = pt.matrix('x', dtype='float32')
        y = x[1:3, 2:4]

        x_val = np.arange(20, dtype='float32').reshape(4, 5)

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[1:3, 2:4]
        np.testing.assert_array_equal(result, expected)

    def test_slice_2d_one_axis(self):
        """Test 2D slicing on one axis: x[1:3, :]"""
        x = pt.matrix('x', dtype='float32')
        y = x[1:3, :]

        x_val = np.arange(20, dtype='float32').reshape(4, 5)

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[1:3, :]
        np.testing.assert_array_equal(result, expected)

    def test_slice_3d(self):
        """Test 3D slicing: x[0:2, 1:3, 2:4]"""
        x = pt.tensor3('x', dtype='float32')
        y = x[0:2, 1:3, 2:4]

        x_val = np.arange(60, dtype='float32').reshape(3, 4, 5)

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[0:2, 1:3, 2:4]
        np.testing.assert_array_equal(result, expected)


class TestSubtensorNegativeIndices:
    """Test slicing with negative indices (when implemented)."""

    @pytest.mark.skip(reason="Negative indices not yet implemented")
    def test_slice_negative_start(self):
        """Test slicing with negative start: x[-3:]"""
        x = pt.vector('x', dtype='float32')
        y = x[-3:]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[-3:]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.skip(reason="Negative indices not yet implemented")
    def test_slice_negative_end(self):
        """Test slicing with negative end: x[:-2]"""
        x = pt.vector('x', dtype='float32')
        y = x[:-2]

        x_val = np.arange(10, dtype='float32')

        fn, result = compare_onnx_and_py([x], y, [x_val])
        expected = x_val[:-2]
        np.testing.assert_array_equal(result, expected)


class TestAdvancedSubtensor:
    """Test advanced indexing."""

    def test_integer_array_indexing(self):
        """Test integer array indexing: x[indices]"""
        x = pt.vector('x', dtype='float32')
        indices = pt.vector('indices', dtype='int64')
        y = x[indices]

        x_val = np.arange(10, dtype='float32')
        indices_val = np.array([0, 2, 5], dtype='int64')

        fn, result = compare_onnx_and_py([x, indices], y, [x_val, indices_val])
        expected = x_val[indices_val]
        np.testing.assert_array_equal(result, expected)

        # Verify ONNX uses Gather operation
        node_types = get_onnx_node_types(fn)
        assert 'Gather' in node_types, f"Expected 'Gather' in {node_types}"

    def test_integer_array_indexing_2d(self):
        """Test integer array indexing on 2D array: x[indices, :]"""
        x = pt.matrix('x', dtype='float32')
        indices = pt.vector('indices', dtype='int64')
        y = x[indices]

        x_val = np.arange(20, dtype='float32').reshape(4, 5)
        indices_val = np.array([0, 2], dtype='int64')

        fn, result = compare_onnx_and_py([x, indices], y, [x_val, indices_val])
        expected = x_val[indices_val]
        np.testing.assert_array_equal(result, expected)

        # Verify ONNX uses Gather operation
        node_types = get_onnx_node_types(fn)
        assert 'Gather' in node_types, f"Expected 'Gather' in {node_types}"


class TestIncSubtensor:
    """Test set_subtensor and inc_subtensor (when implemented)."""

    @pytest.mark.skip(reason="IncSubtensor not yet implemented")
    def test_set_subtensor(self):
        """Test set_subtensor: x[2:5] = values"""
        x = pt.vector('x', dtype='float32')
        values = pt.vector('values', dtype='float32')
        y = pt.set_subtensor(x[2:5], values)

        x_val = np.arange(10, dtype='float32')
        values_val = np.array([100, 200, 300], dtype='float32')

        fn, result = compare_onnx_and_py([x, values], y, [x_val, values_val])

        expected = x_val.copy()
        expected[2:5] = values_val
        np.testing.assert_array_equal(result, expected)
