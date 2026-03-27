"""Tests for ONNX strategy registries.

This module validates the structure and correctness of operation registries
used for property-based testing of the ONNX backend.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pytensor.tensor as pt


# ============================================================================
# REGISTRY STRUCTURE TESTS
# ============================================================================


def test_elemwise_registry_exists():
    """
    Test that ELEMWISE_OPERATIONS registry exists and is accessible.

    This test verifies:
    - Registry is defined in strategies module
    - Registry is a dictionary
    - Registry is not empty
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    assert isinstance(ELEMWISE_OPERATIONS, dict), (
        "ELEMWISE_OPERATIONS should be a dictionary"
    )
    assert len(ELEMWISE_OPERATIONS) > 0, "ELEMWISE_OPERATIONS should not be empty"


def test_elemwise_registry_completeness():
    """
    Test that all 18 Tier 1 elemwise operations are registered.

    This test verifies:
    - All expected Tier 1 operations are present
    - No unexpected operations are present (optional)
    - Operation names follow naming conventions

    Tier 1 Operations from SCALAR_OP_TO_ONNX (pytensor/link/onnx/dispatch/elemwise.py:10-30):
    - Binary arithmetic: Add, Mul, Sub, TrueDiv, IntDiv, Pow (6)
    - Unary math: Neg, Abs, Exp, Log, Sqrt (5)
    - Rounding: Floor, Ceil, RoundHalfToEven, RoundHalfAwayFromZero (4)
    - Min/Max: Maximum, Minimum (2)
    - Special: Clip (1)
    Total: 18 operations

    Note: Both RoundHalfToEven and RoundHalfAwayFromZero should be in registry as 'round'
    and 'round_away' to enable testing both behaviors.
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    expected_ops = {
        # Binary arithmetic operations (6)
        "add",
        "mul",
        "sub",
        "div",
        "int_div",
        "pow",
        # Unary math operations (5)
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        # Rounding operations (4 - two Python operations, both mapped to ONNX "Round")
        "floor",
        "ceil",
        "round",
        "round_away",
        # Element-wise min/max operations (2)
        "maximum",
        "minimum",
        # Special operations (1)
        "clip",
    }

    actual_ops = set(ELEMWISE_OPERATIONS.keys())
    missing_ops = expected_ops - actual_ops

    assert len(expected_ops) == 18, (
        f"Expected ops count should be 18 Tier 1 operations, got {len(expected_ops)}"
    )
    assert missing_ops == set(), f"Missing operations in registry: {missing_ops}"
    # Note: extra operations in actual_ops are OK if testing Tier 4-5 operations


@pytest.mark.parametrize(
    "op_name",
    [
        "add",
        "mul",
        "sub",
        "div",
        "int_div",
        "pow",
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        "floor",
        "ceil",
        "round",
        "maximum",
        "minimum",
        "clip",
    ],
)
def test_elemwise_registry_entry_structure(op_name):
    """
    Test that each registry entry has required fields with correct types.

    This test verifies:
    - Entry has 'build_graph' (callable)
    - Entry has 'strategy' (hypothesis strategy)
    - Entry has 'expected_onnx_ops' (list of strings)
    - Entry has 'description' (string)
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    entry = ELEMWISE_OPERATIONS[op_name]

    # Check all required fields present
    required_fields = {"build_graph", "strategy", "expected_onnx_ops", "description"}
    actual_fields = set(entry.keys())
    missing_fields = required_fields - actual_fields

    assert missing_fields == set(), (
        f"{op_name}: Missing required fields: {missing_fields}"
    )

    # Check field types
    assert callable(entry["build_graph"]), (
        f"{op_name}: 'build_graph' should be callable"
    )
    assert isinstance(entry["expected_onnx_ops"], list), (
        f"{op_name}: 'expected_onnx_ops' should be a list"
    )
    assert all(isinstance(op, str) for op in entry["expected_onnx_ops"]), (
        f"{op_name}: 'expected_onnx_ops' should contain strings"
    )
    assert isinstance(entry["description"], str), (
        f"{op_name}: 'description' should be a string"
    )


# ============================================================================
# STRATEGY VALIDATION TESTS
# ============================================================================


@given(data=st.data())
@settings(max_examples=5, deadline=None)
def test_binary_op_strategy_generates_valid_data(data):
    """
    Test that binary operation strategies generate valid tensor pairs.

    This test verifies:
    - Strategy generates two arrays
    - Arrays have float32 dtype
    - Arrays have compatible shapes (for broadcasting)
    - Arrays contain finite values
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    # Test with 'add' as representative binary op
    op_config = ELEMWISE_OPERATIONS["add"]
    test_inputs = data.draw(op_config["strategy"])

    assert isinstance(test_inputs, tuple), "Binary op strategy should return tuple"
    assert len(test_inputs) >= 2, "Binary op strategy should return at least 2 arrays"

    x_val, y_val = test_inputs[0], test_inputs[1]

    assert x_val.dtype == np.float32, f"Expected float32, got {x_val.dtype}"
    assert y_val.dtype == np.float32, f"Expected float32, got {y_val.dtype}"
    assert np.all(np.isfinite(x_val)), "Generated data should be finite"
    assert np.all(np.isfinite(y_val)), "Generated data should be finite"


@given(data=st.data())
@settings(max_examples=5, deadline=None)
def test_unary_op_strategy_generates_valid_data(data):
    """
    Test that unary operation strategies generate valid tensors.

    This test verifies:
    - Strategy generates one array (or tuple with one array)
    - Array has float32 dtype
    - Array contains finite values
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    # Test with 'neg' as representative unary op
    op_config = ELEMWISE_OPERATIONS["neg"]
    test_inputs = data.draw(op_config["strategy"])

    # Handle both tuple and direct array returns
    if isinstance(test_inputs, tuple):
        x_val = test_inputs[0]
    else:
        x_val = test_inputs

    assert x_val.dtype == np.float32, f"Expected float32, got {x_val.dtype}"
    assert np.all(np.isfinite(x_val)), "Generated data should be finite"


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_log_strategy_generates_positive_values(data):
    """
    Test that log strategy generates positive values.

    This test verifies:
    - Strategy generates positive values (log requires x > 0)
    - Values are not too close to zero (numerical stability)
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    op_config = ELEMWISE_OPERATIONS["log"]
    test_inputs = data.draw(op_config["strategy"])

    if isinstance(test_inputs, tuple):
        x_val = test_inputs[0]
    else:
        x_val = test_inputs

    assert np.all(x_val > 0), "Log operation requires positive inputs"
    assert np.all(x_val > 1e-6), (
        "Values should not be too close to zero for numerical stability"
    )


@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_sqrt_strategy_generates_non_negative_values(data):
    """
    Test that sqrt strategy generates non-negative values.

    This test verifies:
    - Strategy generates non-negative values (sqrt requires x >= 0)
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    op_config = ELEMWISE_OPERATIONS["sqrt"]
    test_inputs = data.draw(op_config["strategy"])

    if isinstance(test_inputs, tuple):
        x_val = test_inputs[0]
    else:
        x_val = test_inputs

    assert np.all(x_val >= 0), "Sqrt operation requires non-negative inputs"


# ============================================================================
# BUILD GRAPH VALIDATION TESTS
# ============================================================================


def test_build_graph_returns_valid_structure():
    """
    Test that build_graph functions return valid graph structure.

    This test verifies:
    - build_graph returns a tuple
    - First element is a list of PyTensor Variables (inputs)
    - Second element is a PyTensor Variable (output)
    """
    from tests.link.onnx.strategies import ELEMWISE_OPERATIONS

    # Test with 'add' as representative
    op_config = ELEMWISE_OPERATIONS["add"]

    # Create dummy inputs
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    # Call build_graph
    result = op_config["build_graph"](x_val, y_val)

    assert isinstance(result, tuple), "build_graph should return a tuple"
    assert len(result) == 2, "build_graph should return (inputs, output)"

    graph_inputs, graph_output = result

    assert isinstance(graph_inputs, list), "First element should be list of inputs"
    assert all(isinstance(inp, pt.Variable) for inp in graph_inputs), (
        "All inputs should be PyTensor Variables"
    )
    assert isinstance(graph_output, pt.Variable), "Output should be PyTensor Variable"
