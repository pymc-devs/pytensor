"""Tests for ONNX elemwise operations.

Test Strategy:
- Property-based tests provide primary coverage (180+ scenarios)
- Main property test covers 13 unconstrained operations
- Separate property tests for constrained operations (log, sqrt, pow, clip)
- Manual tests retained for edge cases and compositions

Coverage: 18 elemwise operations total
"""

from functools import partial

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pytensor.tensor as pt
from tests.link.onnx.strategies import ELEMWISE_OPERATIONS
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# ============================================================================
# NUMERICAL TOLERANCE CONSTANTS
# ============================================================================
# These tolerances account for numerical precision differences between
# PyTensor and ONNX implementations. Documented rationale for each:

# Standard tolerance for stable operations (add, mul, sub, etc.)
STANDARD_TOLERANCE = {"rtol": 1e-5, "atol": 1e-8}

# Relaxed tolerance for numerically unstable operations
# Used for: pow (negative base + fractional exponent), exp (large values)
# Rationale: These operations amplify floating-point errors
RELAXED_TOLERANCE = {"rtol": 1e-3, "atol": 1e-5}

# Log-specific tolerance (between standard and relaxed)
# Used for: log (values near zero are numerically sensitive)
# Rationale: log(x) for small x has larger relative error
LOG_TOLERANCE = {"rtol": 1e-4, "atol": 1e-6}


# ============================================================================
# PROPERTY-BASED TESTS (Primary Coverage)
# ============================================================================


@given(
    op_name=st.sampled_from(
        [
            # Binary arithmetic (5)
            "add",
            "mul",
            "sub",
            "div",
            "int_div",
            # Binary min/max (2)
            "maximum",
            "minimum",
            # Unary (3)
            "neg",
            "abs",
            "exp",
            # Rounding (3)
            "floor",
            "ceil",
            "round",
            # Total: 13 unconstrained operations
        ]
    ),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_elemwise_operations_correctness(op_name, data):
    """
    Property test: All unconstrained elemwise operations produce correct ONNX results.

    This test verifies:
    - ONNX output matches Python reference implementation
    - Correct ONNX node types are generated
    - Operations handle diverse inputs correctly

    Operations tested (13 unconstrained Tier 1 operations):
    - Binary arithmetic: add, mul, sub, div, int_div (5)
    - Binary min/max: maximum, minimum (2)
    - Unary: neg, abs, exp (3)
    - Rounding: floor, ceil, round (3)

    Total: 13 operations x 10 examples = 130 test scenarios

    Constrained operations tested separately:
    - pow, log, sqrt, clip (separate tests with constrained strategies)
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    # Get operation configuration from registry
    op_config = ELEMWISE_OPERATIONS[op_name]

    # Generate test data using operation's strategy
    test_data = data.draw(op_config["strategy"])

    # Handle both tuple and single value returns
    if isinstance(test_data, tuple):
        inputs_tuple = test_data
    else:
        inputs_tuple = (test_data,)

    # Build PyTensor graph
    graph_inputs, graph_output = op_config["build_graph"](*inputs_tuple)

    # Prepare test inputs for execution
    if isinstance(test_data, tuple):
        test_inputs = list(test_data)
    else:
        test_inputs = [test_data]

    # Compare ONNX vs PyTensor
    fn, _result = compare_onnx_and_py(graph_inputs, graph_output, test_inputs)

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config["expected_onnx_ops"]

    # Check that at least one expected operation is present
    assert any(op in node_types for op in expected_ops), (
        f"{op_name}: Expected one of {expected_ops}, got {node_types}"
    )


@given(data=st.data())
@settings(max_examples=50, deadline=None)  # Higher count for critical operation
def test_log_operation_correctness(data):
    """
    Property test: Log operation produces correct ONNX results.

    This test verifies:
    - Log operation works with positive inputs
    - ONNX output matches Python reference
    - Correct ONNX node type (Log) is generated

    Note: Uses positive_float32_array_strategy to ensure valid inputs
          (log requires x > 0). Uses 50 examples (vs standard 10) due to
          numerical sensitivity.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    op_config = ELEMWISE_OPERATIONS["log"]

    # Generate positive test data
    test_data = data.draw(op_config["strategy"])

    # Verify inputs are positive (strategy constraint)
    assert np.all(test_data > 0), "Log operation requires positive inputs"

    # Build graph
    graph_inputs, graph_output = op_config["build_graph"](test_data)

    # Compare ONNX vs PyTensor with log-specific tolerance
    # Uses LOG_TOLERANCE (rtol=1e-4, atol=1e-6) - see tolerance constants
    fn, _result = compare_onnx_and_py(
        graph_inputs,
        graph_output,
        [test_data],
        assert_fn=partial(np.testing.assert_allclose, **LOG_TOLERANCE),
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Log" in node_types, f"Expected 'Log' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_sqrt_operation_correctness(data):
    """
    Property test: Sqrt operation produces correct ONNX results.

    This test verifies:
    - Sqrt operation works with non-negative inputs
    - ONNX output matches Python reference
    - Correct ONNX node type (Sqrt) is generated

    Note: Uses non_negative_float32_array_strategy to ensure valid inputs
          (sqrt requires x >= 0)
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    op_config = ELEMWISE_OPERATIONS["sqrt"]

    # Generate non-negative test data
    test_data = data.draw(op_config["strategy"])

    # Verify inputs are non-negative (strategy constraint)
    assert np.all(test_data >= 0), "Sqrt operation requires non-negative inputs"

    # Build graph
    graph_inputs, graph_output = op_config["build_graph"](test_data)

    # Compare ONNX vs PyTensor
    fn, _result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Sqrt" in node_types, f"Expected 'Sqrt' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=50, deadline=None)  # Higher count for critical operation
def test_pow_operation_correctness(data):
    """
    Property test: Pow operation produces correct ONNX results.

    This test verifies:
    - Pow operation works with float inputs
    - ONNX output matches Python reference
    - Correct ONNX node type (Pow) is generated

    Note: May have numerical precision issues with negative bases
          and fractional exponents. Using relaxed tolerance. Uses
          50 examples (vs standard 10) due to numerical complexity.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    op_config = ELEMWISE_OPERATIONS["pow"]

    # Generate test data (two arrays)
    test_data = data.draw(op_config["strategy"])
    x_val, y_val = test_data

    # Build graph
    graph_inputs, graph_output = op_config["build_graph"](x_val, y_val)

    # Compare ONNX vs PyTensor with relaxed tolerance
    # Uses RELAXED_TOLERANCE (rtol=1e-3, atol=1e-5) - see tolerance constants
    # Rationale: Pow with negative base + fractional exponent amplifies errors
    fn, _result = compare_onnx_and_py(
        graph_inputs,
        graph_output,
        [x_val, y_val],
        assert_fn=partial(np.testing.assert_allclose, **RELAXED_TOLERANCE),
    )

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Pow" in node_types, f"Expected 'Pow' node, got {node_types}"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_clip_operation_correctness(data):
    """
    Property test: Clip operation produces correct ONNX results.

    This test verifies:
    - Clip operation correctly bounds values
    - ONNX output matches Python reference
    - Correct ONNX node type (Clip) is generated
    - Min/max bounds are respected
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    op_config = ELEMWISE_OPERATIONS["clip"]

    # Generate test data (array, min, max)
    test_data = data.draw(op_config["strategy"])
    x_val, min_val, max_val = test_data

    # Build graph
    graph_inputs, graph_output = op_config["build_graph"](x_val, min_val, max_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert "Clip" in node_types, f"Expected 'Clip' node, got {node_types}"

    # Additional validation: verify bounds are respected
    assert np.all(result >= min_val), f"Result contains values below min_val={min_val}"
    assert np.all(result <= max_val), f"Result contains values above max_val={max_val}"


# ============================================================================
# MANUAL EDGE CASE TESTS
# ============================================================================


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
    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])

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

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])

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

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])
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

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Div" in get_onnx_node_types(fn)


def test_chained_arithmetic():
    """Test that chained arithmetic operations work correctly."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype="float32")

    fn, _result = compare_onnx_and_py([x], z, [x_val])

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

    fn, _result = compare_onnx_and_py([x], y, [x_val])
    assert "Neg" in get_onnx_node_types(fn)


def test_abs():
    """Test absolute value operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.abs(x)

    x_val = np.array([1, -2, 3, -4], dtype="float32")

    fn, _result = compare_onnx_and_py([x], y, [x_val])
    assert "Abs" in get_onnx_node_types(fn)


def test_exp():
    """Test exponential operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.exp(x)

    x_val = np.array([0, 1, 2], dtype="float32")

    fn, _result = compare_onnx_and_py([x], y, [x_val])
    assert "Exp" in get_onnx_node_types(fn)


def test_log():
    """Test natural logarithm operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.log(x)

    x_val = np.array([1, 2, np.e], dtype="float32")

    fn, _result = compare_onnx_and_py([x], y, [x_val])
    assert "Log" in get_onnx_node_types(fn)


def test_sqrt():
    """Test square root operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.sqrt(x)

    x_val = np.array([1, 4, 9, 16], dtype="float32")

    fn, _result = compare_onnx_and_py([x], y, [x_val])
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

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])
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

    fn, _result = compare_onnx_and_py([x], y, [x_val])
    assert expected_node in get_onnx_node_types(fn), (
        f"Expected {expected_node} node for {op_name}"
    )


def test_maximum():
    """Test element-wise maximum operation."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.maximum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([4, 2, 6], dtype="float32")

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])
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

    fn, _result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert "Min" in get_onnx_node_types(fn)
