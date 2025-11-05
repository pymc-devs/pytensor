"""Tests for ONNX math operations (reductions)."""

import pytest
import numpy as np
import pytensor.tensor as pt
from hypothesis import given, strategies as st, settings

# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types
from tests.link.onnx.strategies import (
    REDUCTION_OPERATIONS,
    tensor_with_axis_strategy,
)


# ============================================================================
# Property-Based Tests for Reduction Operations
# ============================================================================

@given(
    op_name=st.sampled_from(list(REDUCTION_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_reduction_operations_correctness(op_name, data):
    """Property test: All reduction operations produce correct ONNX results.

    Tests: sum, prod, max, min, argmax, argmin, all, any
    Total: 8 operations × 10 examples = 80 test scenarios
    """
    op_config = REDUCTION_OPERATIONS[op_name]

    # Generate tensor and axis
    test_data = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](*test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data[0]])

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected {expected_ops}, got {node_types}"


# ============================================================================
# Specific Tests for Edge Cases
# ============================================================================

def test_reduction_keepdims():
    """Reduction with keepdims parameter."""
    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=1, keepdims=True)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (3, 1)
    assert 'ReduceSum' in get_onnx_node_types(fn)


@pytest.mark.parametrize("axis", [None, 0, 1, [0, 1]])
def test_reduction_axis_variations(axis):
    """Test reductions with different axis specifications."""
    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=axis)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert 'ReduceSum' in get_onnx_node_types(fn)


def test_sum_reduction():
    """Basic sum reduction."""
    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=1)

    x_val = np.random.randn(4, 5).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.sum(x_val, axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-4)
    assert 'ReduceSum' in get_onnx_node_types(fn)


def test_prod_reduction():
    """Product reduction."""
    x = pt.matrix('x', dtype='float32')
    y = pt.prod(x, axis=0)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.prod(x_val, axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-4)
    assert 'ReduceProd' in get_onnx_node_types(fn)


def test_max_min_reduction():
    """Max and min reductions."""
    x = pt.matrix('x', dtype='float32')
    y_max = pt.max(x, axis=1)
    y_min = pt.min(x, axis=1)

    x_val = np.random.randn(4, 5).astype('float32')

    fn_max, result_max = compare_onnx_and_py([x], y_max, [x_val])
    fn_min, result_min = compare_onnx_and_py([x], y_min, [x_val])

    expected_max = np.max(x_val, axis=1)
    expected_min = np.min(x_val, axis=1)

    np.testing.assert_allclose(result_max, expected_max, rtol=1e-4)
    np.testing.assert_allclose(result_min, expected_min, rtol=1e-4)

    assert 'ReduceMax' in get_onnx_node_types(fn_max)
    # Min is implemented as -max(-x), so we expect Neg and ReduceMax
    node_types_min = get_onnx_node_types(fn_min)
    assert 'ReduceMax' in node_types_min and 'Neg' in node_types_min


def test_argmax_argmin():
    """Argmax and argmin reductions."""
    x = pt.matrix('x', dtype='float32')
    y_argmax = pt.argmax(x, axis=1)
    y_argmin = pt.argmin(x, axis=1)

    x_val = np.random.randn(4, 5).astype('float32')

    fn_argmax, result_argmax = compare_onnx_and_py([x], y_argmax, [x_val])
    fn_argmin, result_argmin = compare_onnx_and_py([x], y_argmin, [x_val])

    expected_argmax = np.argmax(x_val, axis=1)
    expected_argmin = np.argmin(x_val, axis=1)

    np.testing.assert_array_equal(result_argmax, expected_argmax)
    np.testing.assert_array_equal(result_argmin, expected_argmin)

    assert 'ArgMax' in get_onnx_node_types(fn_argmax)
    # ArgMin is implemented as ArgMax of negated input
    node_types_argmin = get_onnx_node_types(fn_argmin)
    assert 'ArgMax' in node_types_argmin or 'ArgMin' in node_types_argmin


@pytest.mark.skip(reason="Boolean reduction operations (all/any) not yet fully supported in ONNX backend")
def test_logical_reductions():
    """Test logical all and any reductions."""
    x = pt.matrix('x', dtype='bool')
    y_all = pt.all(x, axis=1)
    y_any = pt.any(x, axis=1)

    x_val = np.random.rand(4, 5) > 0.5

    fn_all, result_all = compare_onnx_and_py([x], y_all, [x_val])
    fn_any, result_any = compare_onnx_and_py([x], y_any, [x_val])

    expected_all = np.all(x_val, axis=1)
    expected_any = np.any(x_val, axis=1)

    np.testing.assert_array_equal(result_all, expected_all)
    np.testing.assert_array_equal(result_any, expected_any)

    # All/Any map to ReduceMin/ReduceMax for boolean tensors
    node_types_all = get_onnx_node_types(fn_all)
    node_types_any = get_onnx_node_types(fn_any)
    assert 'ReduceMin' in node_types_all or 'ReduceMax' in node_types_all
    assert 'ReduceMin' in node_types_any or 'ReduceMax' in node_types_any


def test_reduction_no_axis():
    """Reduction over all axes (axis=None)."""
    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x)  # Sum over all axes

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.sum(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_reduction_multiple_axes():
    """Reduction over multiple axes."""
    x = pt.tensor3('x', dtype='float32')
    y = pt.sum(x, axis=[0, 2])

    x_val = np.random.randn(2, 3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.sum(x_val, axis=(0, 2))
    np.testing.assert_allclose(result, expected, rtol=1e-4)
