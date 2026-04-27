"""Tests for ONNX shape operations."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes

import pytensor.tensor as pt
from pytensor.tensor.shape import Shape_i
from tests.link.onnx.strategies import SHAPE_OPERATIONS
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


# ============================================================================
# PROPERTY-BASED TESTS - Shape Inspection
# ============================================================================


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_shape_operation_correctness(data):
    """
    Property test: Shape operation returns correct tensor shape.

    This test verifies:
    - Shape operation returns correct dimensions
    - Output is int64 array
    - Correct ONNX node type (Shape) is generated
    - Works with tensors of various dimensionalities (1D-4D)
    """
    op_config = SHAPE_OPERATIONS["shape"]

    # Generate test tensor
    test_data = data.draw(op_config["strategy"])

    # Build graph
    x = pt.tensor("x", dtype="float32", shape=(None,) * test_data.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate result
    expected_shape = np.array(test_data.shape, dtype="int64")
    np.testing.assert_array_equal(result, expected_shape)

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Shape" in node_types, f"Expected 'Shape' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_shape_i_operation_correctness(data):
    """
    Property test: Shape_i operation returns correct dimension.

    This test verifies:
    - Shape_i returns correct dimension value
    - Output is scalar integer
    - Correct ONNX node pattern (Constant + Shape + Gather)
    - Works with valid dimension indices
    """
    op_config = SHAPE_OPERATIONS["shape_i"]

    # Generate test data (tensor and valid dimension index)
    test_data = data.draw(op_config["strategy"])
    x_val, dim_index = test_data

    # Build graph
    x = pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x, dim_index)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Validate result
    expected_dim = x_val.shape[dim_index]
    assert result == expected_dim, (
        f"Expected dimension {dim_index} to be {expected_dim}, got {result}"
    )

    # Verify ONNX node pattern (multi-node return)
    node_types = get_onnx_node_types(fn)
    assert "Shape" in node_types, "Expected 'Shape' node"
    assert "Gather" in node_types, "Expected 'Gather' node"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_specify_shape_passthrough_correctness(data):
    """
    Property test: SpecifyShape passes through without creating ONNX nodes.

    This test verifies:
    - SpecifyShape doesn't appear in ONNX graph
    - Computation continues correctly after SpecifyShape
    - Numerical correctness maintained
    - Return pattern: None (pass-through)
    """
    from pytensor.tensor.shape import specify_shape

    # Generate random tensor
    shape = data.draw(array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10))
    x_val = np.random.randn(*shape).astype("float32")

    # Build graph with SpecifyShape in the middle
    x = pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
    x_specified = specify_shape(x, x_val.shape)
    y = x_specified * 2.0  # Some computation after SpecifyShape

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Validate numerical correctness
    expected = x_val * 2.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify SpecifyShape doesn't appear in ONNX
    node_types = get_onnx_node_types(fn)
    assert "SpecifyShape" not in node_types, (
        "SpecifyShape should not appear in ONNX graph (it's a pass-through)"
    )


# ============================================================================
# PROPERTY-BASED TESTS - Reshape Operations
# ============================================================================


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_reshape_operation_correctness(data):
    """
    Property test: Reshape operation correctly transforms tensor shape.

    This test verifies:
    - Reshape produces correct output shape
    - Element values preserved (same data, different shape)
    - Total element count preserved
    - Correct ONNX node type (Reshape)
    """
    op_config = SHAPE_OPERATIONS["reshape"]

    # Generate tensor and compatible reshape target
    test_data = data.draw(op_config["strategy"])
    x_val, new_shape = test_data

    # Build graph
    x = pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x, new_shape)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Validate shape transformation
    expected = x_val.reshape(new_shape)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == new_shape, f"Expected shape {new_shape}, got {result.shape}"

    # Verify total elements preserved
    assert result.size == x_val.size, (
        f"Element count changed: {x_val.size} -> {result.size}"
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Reshape" in node_types, f"Expected 'Reshape' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_transpose_operation_correctness(data):
    """
    Property test: Transpose operation correctly transposes matrices.

    This test verifies:
    - Transpose swaps axes (shape becomes (cols, rows))
    - Element values correctly repositioned
    - Correct ONNX node type (Transpose)
    - Works with various matrix sizes
    """
    op_config = SHAPE_OPERATIONS["transpose"]

    # Generate 2D matrix
    test_data = data.draw(op_config["strategy"])

    # Build graph
    x = pt.tensor("x", dtype="float32", shape=(None,) * test_data.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate transposition
    expected = test_data.T
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (test_data.shape[1], test_data.shape[0]), (
        f"Expected shape {test_data.T.shape}, got {result.shape}"
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Transpose" in node_types, f"Expected 'Transpose' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_dimshuffle_add_dim_correctness(data):
    """
    Property test: DimShuffle correctly adds dimensions.

    This test verifies:
    - DimShuffle adds dimension at correct position
    - Shape changes correctly (e.g., (5,) -> (1, 5))
    - Element values unchanged
    - Correct ONNX node type (Unsqueeze)
    """
    op_config = SHAPE_OPERATIONS["dimshuffle_add_dim"]

    # Generate vector
    test_data = data.draw(op_config["strategy"])

    # Build graph (adds dimension at position 0)
    x = pt.tensor("x", dtype="float32", shape=(None,) * test_data.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate dimension addition
    expected = test_data[np.newaxis, :]  # Add dimension at position 0
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (1, test_data.shape[0]), (
        f"Expected shape (1, {test_data.shape[0]}), got {result.shape}"
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Unsqueeze" in node_types, f"Expected 'Unsqueeze' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_dimshuffle_squeeze_correctness(data):
    """
    Property test: DimShuffle correctly removes singleton dimensions.

    This test verifies:
    - DimShuffle removes dimension of size 1
    - Shape changes correctly (e.g., (3, 1, 4) -> (3, 4))
    - Element values unchanged
    - Correct ONNX node type (Squeeze)
    """
    op_config = SHAPE_OPERATIONS["dimshuffle_squeeze"]

    # Generate tensor with singleton dimension
    test_data = data.draw(op_config["strategy"])

    # Build graph (removes dimension at position 1)
    x = pt.tensor("x", dtype="float32", shape=(None,) * test_data.ndim)
    graph_inputs, graph_output = op_config["build_graph"](x)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate dimension removal
    expected = test_data.squeeze(axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.ndim == test_data.ndim - 1, (
        f"Expected {test_data.ndim - 1} dimensions, got {result.ndim}"
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Squeeze" in node_types, f"Expected 'Squeeze' node, got {node_types}"


# ============================================================================
# PROPERTY-BASED TESTS - Join/Split Operations
# ============================================================================


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_concatenate_operation_correctness(data):
    """
    Property test: Concatenate correctly joins tensors.

    This test verifies:
    - Concatenate joins tensors along specified axis
    - Output shape is correct (sum of input dimensions)
    - Element values correctly positioned
    - Correct ONNX node type (Concat)
    """
    op_config = SHAPE_OPERATIONS["concatenate"]

    # Generate two compatible tensors and axis
    test_data = data.draw(op_config["strategy"])
    a_val, b_val, axis = test_data

    # Build graph
    a = pt.tensor("a", dtype="float32", shape=(None,) * a_val.ndim)
    b = pt.tensor("b", dtype="float32", shape=(None,) * b_val.ndim)
    graph_inputs, graph_output = op_config["build_graph"](a, b, axis)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [a_val, b_val])

    # Validate concatenation
    expected = np.concatenate([a_val, b_val], axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify shape along concatenation axis
    expected_shape = list(a_val.shape)
    expected_shape[axis] = a_val.shape[axis] + b_val.shape[axis]
    assert result.shape == tuple(expected_shape), (
        f"Expected shape {tuple(expected_shape)}, got {result.shape}"
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Concat" in node_types, f"Expected 'Concat' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_stack_operation_correctness(data):
    """
    Property test: Stack correctly stacks tensors with new dimension.

    This test verifies:
    - Stack adds new dimension for stacking
    - Output shape is correct (adds 1 to ndim)
    - Element values correctly positioned
    - Correct ONNX node types (Unsqueeze + Concat)
    """
    op_config = SHAPE_OPERATIONS["stack"]

    # Generate two tensors with same shape
    test_data = data.draw(op_config["strategy"])
    a_val, b_val = test_data

    # Build graph (stack along axis 0)
    a = pt.tensor("a", dtype="float32", shape=(None,) * a_val.ndim)
    b = pt.tensor("b", dtype="float32", shape=(None,) * b_val.ndim)
    graph_inputs, graph_output = op_config["build_graph"](a, b)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [a_val, b_val])

    # Validate stacking
    expected = np.stack([a_val, b_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify shape (added dimension)
    assert result.ndim == a_val.ndim + 1, (
        f"Expected {a_val.ndim + 1} dimensions, got {result.ndim}"
    )
    assert result.shape[0] == 2, f"Expected size 2 along axis 0, got {result.shape[0]}"

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    assert "Concat" in node_types or "Unsqueeze" in node_types, (
        f"Expected 'Concat' or 'Unsqueeze' nodes, got {node_types}"
    )


# ============================================================================
# MANUAL EDGE CASE TESTS
# ============================================================================


def test_shape_basic():
    """Test Shape operation (single node return)."""
    x = pt.matrix("x", dtype="float32")
    y = x.shape

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([3, 4], dtype="int64")
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert "Shape" in node_types


def test_shape_i_dim0():
    """Test Shape_i getting dimension 0 (multi-node return)."""
    x = pt.matrix("x", dtype="float32")
    # Use Shape_i directly to test the multi-node return pattern
    shape_i_op = Shape_i(0)
    y = shape_i_op(x)

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result == 3

    # Verify multi-node pattern: Constant + Shape + Gather
    node_types = get_onnx_node_types(fn)
    assert "Constant" in node_types
    assert "Shape" in node_types
    assert "Gather" in node_types


def test_shape_i_dim1():
    """Test Shape_i getting dimension 1 (multi-node return)."""
    x = pt.matrix("x", dtype="float32")
    # Use Shape_i directly
    shape_i_op = Shape_i(1)
    y = shape_i_op(x)

    x_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result == 4

    node_types = get_onnx_node_types(fn)
    assert "Shape" in node_types
    assert "Gather" in node_types


def test_shape_i_3d_tensor():
    """Test Shape_i with 3D tensor."""
    x = pt.tensor3("x", dtype="float32")
    # Use Shape_i directly for each dimension
    dim0 = Shape_i(0)(x)
    dim1 = Shape_i(1)(x)
    dim2 = Shape_i(2)(x)

    x_val = np.random.randn(2, 3, 4).astype("float32")

    # Test each dimension separately
    _fn0, result0 = compare_onnx_and_py([x], dim0, [x_val])
    assert result0 == 2

    _fn1, result1 = compare_onnx_and_py([x], dim1, [x_val])
    assert result1 == 3

    _fn2, result2 = compare_onnx_and_py([x], dim2, [x_val])
    assert result2 == 4


def test_specify_shape_passthrough():
    """Test that SpecifyShape creates no ONNX nodes (None return)."""
    from pytensor.tensor.shape import specify_shape

    x = pt.vector("x", dtype="float32")
    # SpecifyShape should pass through without creating ONNX nodes
    x_specified = specify_shape(x, (4,))
    y = x_specified * 2.0

    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # SpecifyShape should not appear in ONNX graph
    node_types = get_onnx_node_types(fn)
    assert "SpecifyShape" not in node_types
    assert "Mul" in node_types

    expected = x_val * 2.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_concatenate_axis0():
    """Test concatenate operation along axis 0."""
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.concatenate([x, y], axis=0)

    x_val = np.random.randn(2, 3).astype("float32")
    y_val = np.random.randn(4, 3).astype("float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.concatenate([x_val, y_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert "Concat" in node_types


def test_concatenate_axis1():
    """Test concatenate operation along axis 1."""
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.concatenate([x, y], axis=1)

    x_val = np.random.randn(3, 2).astype("float32")
    y_val = np.random.randn(3, 4).astype("float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.concatenate([x_val, y_val], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert "Concat" in node_types


def test_stack_axis0():
    """Test stack operation along axis 0."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.stack([x, y], axis=0)

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([4.0, 5.0, 6.0], dtype="float32")

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np.stack([x_val, y_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    # Stack uses Join which maps to Concat, along with Unsqueeze
    assert "Concat" in node_types or "Unsqueeze" in node_types


def test_split_equal():
    """Test split operation with equal sizes."""
    from pytensor.tensor.basic import split

    x = pt.vector("x", dtype="float32")
    splits_var = pt.constant([2, 2, 2], dtype="int64")
    a, b, c = split(x, splits_var, n_splits=3, axis=0)

    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32")

    fn, results = compare_onnx_and_py([x], [a, b, c], [x_val])

    expected_a = x_val[:2]
    expected_b = x_val[2:4]
    expected_c = x_val[4:]

    np.testing.assert_allclose(results[0], expected_a, rtol=1e-5)
    np.testing.assert_allclose(results[1], expected_b, rtol=1e-5)
    np.testing.assert_allclose(results[2], expected_c, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert "Split" in node_types


def test_split_unequal():
    """Test split operation with unequal sizes."""
    from pytensor.tensor.basic import split

    x = pt.vector("x", dtype="float32")
    splits_var = pt.constant([3, 2, 1], dtype="int64")
    a, b, c = split(x, splits_var, n_splits=3, axis=0)

    x_val = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")

    fn, results = compare_onnx_and_py([x], [a, b, c], [x_val])

    expected_a = x_val[:3]
    expected_b = x_val[3:5]
    expected_c = x_val[5:]

    np.testing.assert_allclose(results[0], expected_a, rtol=1e-5)
    np.testing.assert_allclose(results[1], expected_b, rtol=1e-5)
    np.testing.assert_allclose(results[2], expected_c, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert "Split" in node_types
