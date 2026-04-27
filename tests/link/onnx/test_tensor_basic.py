"""Tests for ONNX tensor basic operations (allocation, etc.)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pytensor.tensor as pt
from tests.link.onnx.strategies import ALLOCATION_OPERATIONS
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


# ============================================================================
# Property-Based Tests for Allocation Operations
# ============================================================================


@given(
    op_name=st.sampled_from(list(ALLOCATION_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_allocation_operations_correctness(op_name, data):
    """Property test: All allocation operations produce correct ONNX results.

    Tests: alloc, alloc_empty, make_vector, arange
    Total: 4 operations x 10 examples = 40 test scenarios
    """
    op_config = ALLOCATION_OPERATIONS[op_name]

    # Generate test data
    test_data = data.draw(op_config["strategy"])
    inputs_tuple = test_data if isinstance(test_data, tuple) else (test_data,)

    # Build graph
    graph_inputs, graph_output = op_config["build_graph"](*inputs_tuple)

    # Prepare test inputs (many allocation ops have no inputs)
    test_inputs = []

    # Special handling for AllocEmpty (only check shape/dtype)
    if op_name == "alloc_empty":

        def assert_shape_dtype(a, b):
            assert a.shape == b.shape
            assert a.dtype == b.dtype

        fn, _result = compare_onnx_and_py(
            graph_inputs, graph_output, test_inputs, assert_fn=assert_shape_dtype
        )
    else:
        fn, _result = compare_onnx_and_py(graph_inputs, graph_output, test_inputs)

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config["expected_onnx_ops"]
    assert any(op in node_types for op in expected_ops), (
        f"{op_name}: Expected {expected_ops}, got {node_types}"
    )


# ============================================================================
# Specific Tests for Edge Cases
# ============================================================================


def test_arange_requires_constants():
    """ARange requires constant inputs (ONNX limitation)."""
    x = pt.arange(0, 10, 2, dtype="int64")

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.arange(0, 10, 2, dtype="int64")
    np.testing.assert_array_equal(result, expected)
    assert "Range" in get_onnx_node_types(fn)


def test_alloc_constant_shape():
    """Alloc with constant shape."""
    val = 5.0
    x = pt.alloc(val, 3, 4)

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.full((3, 4), val, dtype="float32")
    np.testing.assert_allclose(result, expected)
    assert "Expand" in get_onnx_node_types(fn)


def test_alloc_dynamic_shape():
    """Alloc with dynamic shape from scalar inputs."""
    val = pt.scalar("val", dtype="float32")
    s1 = pt.scalar("s1", dtype="int64")
    s2 = pt.scalar("s2", dtype="int64")
    x = pt.alloc(val, s1, s2)

    val_data = np.array(3.5, dtype="float32")
    s1_data = np.array(4, dtype="int64")
    s2_data = np.array(5, dtype="int64")

    fn, result = compare_onnx_and_py([val, s1, s2], x, [val_data, s1_data, s2_data])

    expected = np.full((4, 5), 3.5, dtype="float32")
    np.testing.assert_allclose(result, expected)
    assert "Expand" in get_onnx_node_types(fn)


def test_make_vector_from_scalars():
    """MakeVector creates vector from scalar values."""
    a = 1.0
    b = 2.0
    c = 3.0
    vec = pt.stack([a, b, c])

    fn, result = compare_onnx_and_py([], vec, [])

    expected = np.array([1.0, 2.0, 3.0], dtype="float32")
    np.testing.assert_allclose(result, expected)

    node_types = get_onnx_node_types(fn)
    # MakeVector uses Unsqueeze + Concat
    assert "Concat" in node_types


def test_alloc_empty_shape_dtype():
    """AllocEmpty creates tensor with correct shape and dtype."""
    x = pt.empty((3, 4), dtype="float32")

    fn, result = compare_onnx_and_py(
        [],
        x,
        [],
        assert_fn=lambda a, b: (a.shape == b.shape and a.dtype == b.dtype)
        or (_ for _ in ()).throw(
            AssertionError(
                f"Shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
            )
        ),
    )

    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    assert "ConstantOfShape" in get_onnx_node_types(fn)


def test_arange_with_different_dtypes():
    """ARange works with different dtypes."""
    # int64
    x_int = pt.arange(0, 10, 1, dtype="int64")
    _fn_int, result_int = compare_onnx_and_py([], x_int, [])
    expected_int = np.arange(0, 10, 1, dtype="int64")
    np.testing.assert_array_equal(result_int, expected_int)

    # float32
    x_float = pt.arange(0.0, 5.0, 0.5, dtype="float32")
    _fn_float, result_float = compare_onnx_and_py([], x_float, [])
    expected_float = np.arange(0.0, 5.0, 0.5, dtype="float32")
    np.testing.assert_allclose(result_float, expected_float)
