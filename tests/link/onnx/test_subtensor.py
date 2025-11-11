"""Tests for ONNX subtensor (slicing) operations.

Test Strategy:
- Property-based tests provide primary coverage (40+ scenarios)
- Individual property test per operation type (4 operations)
- Manual tests retained for specific patterns and edge cases

Operations: Subtensor (slicing), AdvancedSubtensor (integer indexing),
            set_subtensor, inc_subtensor

Known Limitations:
- Negative indices NOT supported (limitation documented in subtensor.py:122-127)
- Property tests explicitly exclude negative indices
- Manual tests for negative indices are skipped (will be enabled when supported)
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types
from tests.link.onnx.strategies import SUBTENSOR_OPERATIONS, INCSUBTENSOR_OPERATIONS


# ============================================================================
# PROPERTY-BASED TESTS (Primary Coverage)
# ============================================================================


@given(
    op_name=st.sampled_from(['slice_basic', 'slice_multidim', 'slice_with_step']),
    data=st.data(),
)
@settings(max_examples=20, deadline=None)  # Higher count for slicing edge cases
def test_subtensor_basic_slicing_correctness(op_name, data):
    """
    Property test: Basic subtensor slicing operations produce correct results.

    This test verifies:
    - Basic slicing (x[2:5]) works correctly
    - Multi-dimensional slicing (x[1:3, 2:4]) works correctly
    - Slicing with step (x[::2], x[1:8:2]) works correctly
    - ONNX output matches Python reference
    - Correct ONNX node type (Slice)

    Operations tested: slice_basic, slice_multidim, slice_with_step
    Total: 3 patterns × 20 examples = 60 test scenarios

    Note: This test does NOT cover negative indices (not yet supported in ONNX backend)
    """
    op_config = SUBTENSOR_OPERATIONS[op_name]

    # Generate test data (tensor with valid size for slicing)
    x_val = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected one of {expected_ops}, got {node_types}"

    # Additional validation: verify result shape is reasonable
    assert result.ndim <= x_val.ndim, \
        f"Result should not have more dimensions than input"
    assert result.size <= x_val.size, \
        f"Slice result should not be larger than input"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_advanced_subtensor_indexing_correctness(data):
    """
    Property test: Advanced subtensor indexing produces correct results.

    This test verifies:
    - Integer array indexing (x[indices]) works correctly
    - Selected elements match Python reference
    - ONNX output matches PyTensor
    - Correct ONNX node type (Gather)

    Note: Uses advanced_index_strategy to generate valid indices
          (all indices are non-negative and within bounds)
    """
    op_config = SUBTENSOR_OPERATIONS['advanced_index']

    # Generate test data (tensor and valid integer indices)
    test_data = data.draw(op_config['strategy'])
    x_val, indices_val = test_data

    # Verify indices are valid (strategy constraint)
    assert np.all(indices_val >= 0), \
        "Indices should be non-negative (negative indices not supported)"
    assert np.all(indices_val < x_val.shape[0]), \
        "Indices should be within bounds"

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, indices_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, indices_val])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"Expected one of {expected_ops}, got {node_types}"

    # Validate result shape
    expected_shape = (indices_val.shape[0],)
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_set_subtensor_operation_correctness(data):
    """
    Property test: set_subtensor correctly replaces slice with values.

    This test verifies:
    - set_subtensor replaces slice with provided values
    - Other elements remain unchanged
    - ONNX output matches PyTensor
    - Correct ONNX node types (ScatterElements/ScatterND)

    Note: Uses set_subtensor_strategy to generate compatible shapes
    """
    op_config = INCSUBTENSOR_OPERATIONS['set_subtensor']

    # Generate test data (tensor and replacement values)
    x_val, values_val = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, values_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, values_val])

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"Expected one of {expected_ops}, got {node_types}"

    # Use Hypothesis assume() to filter edge case where new values equal old
    # This avoids false failures when values_val happens to equal x_val[2:5]
    assume(not np.array_equal(values_val, x_val[2:5]))

    # Validate that slice was modified
    # (This assertion is now guaranteed to be meaningful)
    assert not np.array_equal(result[2:5], x_val[2:5]), \
        "Slice should have been modified"

    # Validate that values were set correctly
    np.testing.assert_array_equal(result[2:5], values_val)

    # Validate that other elements unchanged
    np.testing.assert_array_equal(result[:2], x_val[:2])
    np.testing.assert_array_equal(result[5:], x_val[5:])


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_inc_subtensor_operation_correctness(data):
    """
    Property test: inc_subtensor correctly increments slice values.

    This test verifies:
    - inc_subtensor adds values to existing slice
    - Other elements remain unchanged
    - ONNX output matches PyTensor
    - Correct ONNX node types (Gather, Add, ScatterElements)

    Note: inc_subtensor is more complex than set_subtensor
          (requires gather, add, then scatter)
    """
    op_config = INCSUBTENSOR_OPERATIONS['inc_subtensor']

    # Generate test data (tensor and increment values)
    x_val, values_val = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, values_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, values_val])

    # Verify ONNX node types (should include Gather, Add, ScatterElements)
    node_types = get_onnx_node_types(fn)
    # Note: inc_subtensor requires multiple operations
    assert 'Gather' in node_types or 'Slice' in node_types, \
        "Expected gather/slice operation"
    assert 'Add' in node_types, \
        "Expected Add operation (for increment)"
    assert 'ScatterElements' in node_types or 'ScatterND' in node_types, \
        "Expected scatter operation"

    # Use Hypothesis assume() to filter edge case where increment values are zero
    # This avoids false failures when values_val is all zeros
    assume(not np.allclose(values_val, 0))

    # Validate that slice was modified
    # (This assertion is now guaranteed to be meaningful)
    assert not np.array_equal(result[2:5], x_val[2:5]), \
        "Slice should have been modified"

    # Validate that values were incremented correctly
    expected_slice = x_val[2:5] + values_val
    np.testing.assert_allclose(result[2:5], expected_slice, rtol=1e-5)

    # Validate that other elements unchanged
    np.testing.assert_array_equal(result[:2], x_val[:2])
    np.testing.assert_array_equal(result[5:], x_val[5:])


# ============================================================================
# MANUAL EDGE CASE TESTS
# ============================================================================
# These tests complement the property-based tests above by:
# - Testing specific edge cases and patterns
# - Providing readable examples for documentation
# - Validating 3D operations (more complex than property tests cover)
# ============================================================================


class TestSubtensorBasic:
    """Test specific basic slicing patterns.

    Note: Many of these patterns are also covered by property-based tests above,
    but are retained for:
    - Explicit documentation of supported patterns
    - Quick debugging when property tests fail
    - Testing specific slice boundaries
    """

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
    """Test slicing with negative indices (when implemented).

    IMPORTANT: These tests are currently skipped because negative indices are NOT
    yet supported in the ONNX backend. This is a known limitation documented at:
    pytensor/link/onnx/dispatch/subtensor.py:122-127

    These tests document the expected behavior when the feature is implemented.
    Remove @pytest.mark.skip decorators when negative index support is added.
    """

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
    """Test advanced indexing with integer arrays.

    These tests verify that integer array indexing (fancy indexing) works correctly.
    Also covered by test_advanced_subtensor_indexing_correctness property test.
    """

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
    """Test set_subtensor and inc_subtensor operations.

    These tests verify that setting and incrementing subtensor slices works correctly.
    They also document the expected ONNX node patterns (ScatterElements for both,
    plus Gather and Add for inc_subtensor).

    Also covered by property tests: test_set_subtensor_operation_correctness and
    test_inc_subtensor_operation_correctness.
    """

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

        # Verify ONNX uses ScatterElements operation
        node_types = get_onnx_node_types(fn)
        assert 'ScatterElements' in node_types, f"Expected 'ScatterElements' in {node_types}"

    def test_inc_subtensor(self):
        """Test inc_subtensor: x[2:5] += values"""
        x = pt.vector('x', dtype='float32')
        values = pt.vector('values', dtype='float32')
        y = pt.inc_subtensor(x[2:5], values)

        x_val = np.arange(10, dtype='float32')
        values_val = np.array([1, 2, 3], dtype='float32')

        fn, result = compare_onnx_and_py([x, values], y, [x_val, values_val])

        expected = x_val.copy()
        expected[2:5] += values_val
        np.testing.assert_array_equal(result, expected)

        # Verify ONNX uses Gather, Add, and ScatterElements operations
        node_types = get_onnx_node_types(fn)
        assert 'Gather' in node_types, f"Expected 'Gather' in {node_types}"
        assert 'Add' in node_types, f"Expected 'Add' in {node_types}"
        assert 'ScatterElements' in node_types, f"Expected 'ScatterElements' in {node_types}"
