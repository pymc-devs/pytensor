---
date: 2025-11-04
status: ready
phase: "tier-2-3"
coverage: "Shape Operations (Tier 2) & Reductions/Allocation (Tier 3)"
timeline: "Weeks 4-6"
tags: [tdd, onnx, backend, shape, reductions, tier2, tier3]
related_research:
  - thoughts/shared/research/2025-11-04_11-52-15_onnx-backend-infrastructure-roadmap.md
  - thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md
related_plans:
  - thoughts/shared/plans/onnx-backend-phase1-3-infrastructure-tdd.md
prerequisites:
  - "Tier 1 complete: 20 basic elemwise operations passing"
  - "Infrastructure: ONNXLinker, dispatch system, export API"
  - "Testing utilities: compare_onnx_and_py, get_onnx_node_types"
---

# ONNX Backend Tier 2-3: Shape Operations & Reductions - TDD Implementation Plan

## Overview

This TDD plan covers **Tier 2 (Shape Operations, 15 ops)** and **Tier 3 (Reductions & Allocation, 16 ops)** of the ONNX backend, building on the Tier 1 infrastructure. These operations enable tensor reshaping, slicing, statistical operations, and tensor creation - essential for real-world PyTensor code.

**TDD Approach**: Write comprehensive tests defining expected behavior, verify they fail properly, then implement features by debugging the failing tests.

**Total Operations**: 31 operations across two tiers
**Timeline**: 2.5-3.5 weeks (1.5-2 weeks Tier 2, 1-1.5 weeks Tier 3)

## Current State Analysis

### What Exists (Post-Tier 1):
- ✅ **ONNX backend infrastructure**: `pytensor/link/onnx/` with linker and dispatch system
- ✅ **Tier 1 operations**: 20 basic elemwise operations (Add, Mul, Exp, Log, etc.)
- ✅ **Testing infrastructure**: `compare_onnx_and_py`, fixtures, 29+ passing tests
- ✅ **Export API**: `export_onnx`, `compile_onnx`, `export_function_onnx`

### Testing Landscape:
- **Testing framework**: pytest
- **Test patterns available**: From JAX backend and PyTensor core tests
  - Shape operations: `tests/link/jax/test_shape.py`, `tests/tensor/test_shape.py`
  - Reductions: `tests/link/jax/test_elemwise.py`, `tests/tensor/test_math.py`
  - Allocation: `tests/link/jax/test_tensor_basic.py`, `tests/tensor/test_basic.py`
- **Key test utilities**:
  - `_compile_and_check` for shape inference testing
  - `verify_grad` for gradient testing
  - `compare_onnx_and_py` for backend comparison

### Key Discoveries:
- **Dynamic shapes**: ONNX supports dynamic shapes (opset 11+), but requires careful handling
- **Static shape inference**: PyTensor's `type.shape` must be preserved through ONNX conversion
- **Subtensor complexity**: Slicing operations map to multiple ONNX ops (Slice, Gather, ScatterND)
- **IncSubtensor challenge**: ONNX has no in-place operations - must use Scatter ops
- **ARange limitation**: Requires static (constant) inputs in ONNX
- **Reduction axis handling**: ONNX axis parameter differs from NumPy (no negative normalization)

## Desired End State

After Tier 2-3 completion:

✅ **Shape Operations Working** (Tier 2 - 15 ops):
- Reshape, DimShuffle (transpose/squeeze/unsqueeze)
- Shape inspection (Shape, Shape_i, SpecifyShape)
- Join/Stack/Split operations
- Basic and advanced indexing (Subtensor, IncSubtensor)

✅ **Reductions & Allocation Working** (Tier 3 - 16 ops):
- Reductions: Sum, Prod, Max, Min, All, Any, Argmax, Argmin
- Allocation: Alloc, AllocEmpty, MakeVector, ARange, Eye
- Scalar/tensor conversion operations

✅ **Scalable Testing Architecture** (Hypothesis-based):
- **Operation registries** for shape ops, reductions, and allocations
- **Hypothesis strategies module** for generating valid shape/reduction test cases
- **~8-12 property tests** that automatically test all 31 operations:
  - Shape operations correctness (Reshape, DimShuffle, Shape, Join/Split)
  - Reduction operations correctness (Sum, Prod, Max, Min, Argmax, Argmin, All, Any)
  - Allocation operations correctness (Alloc, ARange, Eye, MakeVector)
  - Subtensor operations correctness (slicing, advanced indexing)
  - IncSubtensor operations correctness (set/increment)
  - Dynamic shape handling
  - Axis parameter handling
  - Edge cases (empty arrays, zero dims)
- **~5-8 targeted regression tests** (for specific bugs discovered during implementation)
- **Total: ~15-20 tests instead of 45+ manual tests**
- All operations compared against Python reference

✅ **Validation**:
- Can export tensor reshaping and slicing operations
- Can export statistical operations (mean, variance, etc.)
- Can export tensor creation operations
- Complex graphs with mixed operations work correctly

## What We're NOT Testing/Implementing

❌ **Out of Scope**:
- Linear algebra operations (Tier 4) - separate plan
- Advanced operations like Scan, IfElse (Tier 5) - separate plan
- CNN operations (Conv2D, MaxPool) - not core backend operations
- Boolean indexing with dynamic masks - complex rewrite required
- Fancy multi-dimensional advanced indexing - future enhancement
- Random variable operations - future work
- Training-specific operations - inference only for now

## TDD Approach

### Test Design Philosophy:
1. **Property-Based Testing**: Use Hypothesis to generate diverse test cases automatically
2. **Operation Registry Pattern**: Define operations once, test all automatically
3. **Test static and dynamic shapes**: ONNX has different code paths for each
4. **Test axis specifications**: None, single, multiple, negative indices
5. **Test edge cases**: Empty arrays, zero dimensions, broadcasting edge cases
6. **Compare against NumPy behavior**: Ensure PyTensor → ONNX → Result matches NumPy
7. **Verify ONNX node types**: Correct ONNX operators are generated

### Testing Strategy (Hypothesis-Based):

```python
# Core pattern: Property test for operation categories

@given(
    op_name=st.sampled_from(SHAPE_OPERATIONS.keys()),
    data=st.data(),
)
def test_shape_operations_match_pytensor(op_name, data):
    """Property test: All shape operations produce correct results."""
    op_config = SHAPE_OPERATIONS[op_name]

    # Generate appropriate test inputs based on operation
    inputs = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_outputs = op_config['build_graph'](*inputs)

    # Compare ONNX output to Python reference
    compare_onnx_and_py(graph_inputs, graph_outputs, inputs)

    # Verify correct ONNX nodes generated
    assert op_config['expected_onnx_op'] in get_onnx_node_types(fn)
```

**Key Insight**: With operation registries, adding a new operation only requires:
1. Add entry to registry dict (operation name → configuration)
2. Optionally add custom Hypothesis strategy if needed
3. Property tests automatically validate it!

---

## Phase 1: Test Design & Implementation (Hypothesis-Based)

### Overview

Write comprehensive property-based tests using Hypothesis that automatically generate diverse test cases for shape operations and reductions. Tests define expected behavior through operation registries and fail in diagnostic ways.

---

### Step 1.1: Operation Registries Setup

**File**: `tests/link/onnx/strategies.py` (create new)

Define operation registries that map operation names to their configurations:

```python
"""Hypothesis strategies and operation registries for ONNX backend testing."""

from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
import pytensor.tensor as pt
from typing import Dict, Callable, Any


# ============================================================================
# SHAPE OPERATIONS REGISTRY (Tier 2)
# ============================================================================

SHAPE_OPERATIONS: Dict[str, Dict[str, Any]] = {
    # Shape inspection
    "shape": {
        "build_graph": lambda x: ([x], x.shape),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype('float32'),
            shape=array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10)
        ),
        "expected_onnx_ops": ['Shape'],
        "description": "Get tensor shape"
    },

    "shape_i": {
        "build_graph": lambda x, i: ([x], x.shape[i]),
        "strategy": st.builds(
            lambda shape, i: (np.random.randn(*shape).astype('float32'), i),
            shape=array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=10),
            i=st.integers(0, 3)
        ),
        "expected_onnx_ops": ['Shape', 'Gather'],
        "description": "Get specific dimension"
    },

    # Reshape operations
    "reshape": {
        "build_graph": lambda x, new_shape: ([x], x.reshape(new_shape)),
        "strategy": reshape_strategy(),  # Custom strategy
        "expected_onnx_ops": ['Reshape'],
        "description": "Reshape tensor"
    },

    "transpose": {
        "build_graph": lambda x: ([x], x.T),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype('float32'),
            shape=st.tuples(st.integers(2, 10), st.integers(2, 10))
        ),
        "expected_onnx_ops": ['Transpose'],
        "description": "Transpose matrix"
    },

    "dimshuffle_add_dim": {
        "build_graph": lambda x: ([x], x.dimshuffle('x', 0)),
        "strategy": st.builds(
            lambda size: np.random.randn(size).astype('float32'),
            size=st.integers(2, 20)
        ),
        "expected_onnx_ops": ['Unsqueeze'],
        "description": "Add dimension via dimshuffle"
    },

    "dimshuffle_squeeze": {
        "build_graph": lambda x: ([x], x.dimshuffle(0, 2)),
        "strategy": st.builds(
            lambda s1, s2: np.random.randn(s1, 1, s2).astype('float32'),
            s1=st.integers(2, 10),
            s2=st.integers(2, 10)
        ),
        "expected_onnx_ops": ['Squeeze'],
        "description": "Remove dimension via dimshuffle"
    },

    # Join/Split operations
    "concatenate": {
        "build_graph": lambda a, b, axis: ([a, b], pt.concatenate([a, b], axis=axis)),
        "strategy": concatenate_strategy(),  # Custom strategy
        "expected_onnx_ops": ['Concat'],
        "description": "Concatenate tensors"
    },

    "stack": {
        "build_graph": lambda a, b: ([a, b], pt.stack([a, b], axis=0)),
        "strategy": st.builds(
            lambda shape: (
                np.random.randn(*shape).astype('float32'),
                np.random.randn(*shape).astype('float32')
            ),
            shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10)
        ),
        "expected_onnx_ops": ['Concat', 'Unsqueeze'],
        "description": "Stack tensors"
    },
}


# ============================================================================
# REDUCTION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

REDUCTION_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "sum": {
        "build_graph": lambda x, axis: ([x], pt.sum(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceSum'],
        "description": "Sum reduction"
    },

    "prod": {
        "build_graph": lambda x, axis: ([x], pt.prod(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceProd'],
        "description": "Product reduction"
    },

    "max": {
        "build_graph": lambda x, axis: ([x], pt.max(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceMax'],
        "description": "Max reduction"
    },

    "min": {
        "build_graph": lambda x, axis: ([x], pt.min(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceMin'],
        "description": "Min reduction"
    },

    "argmax": {
        "build_graph": lambda x, axis: ([x], pt.argmax(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ['ArgMax'],
        "description": "Argmax reduction"
    },

    "argmin": {
        "build_graph": lambda x, axis: ([x], pt.argmin(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ['ArgMin'],
        "description": "Argmin reduction"
    },

    "all": {
        "build_graph": lambda x, axis: ([x], pt.all(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(dtype='bool'),
        "expected_onnx_ops": ['ReduceMin'],  # All maps to ReduceMin for bool
        "description": "Logical all reduction"
    },

    "any": {
        "build_graph": lambda x, axis: ([x], pt.any(x, axis=axis)),
        "strategy": tensor_with_axis_strategy(dtype='bool'),
        "expected_onnx_ops": ['ReduceMax'],  # Any maps to ReduceMax for bool
        "description": "Logical any reduction"
    },
}


# ============================================================================
# ALLOCATION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

ALLOCATION_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "alloc_scalar": {
        "build_graph": lambda val, *shape: ([], pt.alloc(val, *shape)),
        "strategy": alloc_strategy(),
        "expected_onnx_ops": ['Expand'],
        "description": "Allocate tensor from scalar"
    },

    "alloc_empty": {
        "build_graph": lambda *shape: ([], pt.AllocEmpty('float32')(*shape)),
        "strategy": st.tuples(st.integers(2, 10), st.integers(2, 10)),
        "expected_onnx_ops": ['ConstantOfShape'],
        "description": "Allocate uninitialized tensor"
    },

    "make_vector": {
        "build_graph": lambda a, b, c: ([a, b, c], pt.make_vector(a, b, c)),
        "strategy": st.builds(
            lambda: tuple(np.random.randn(3)),

        ),
        "expected_onnx_ops": ['Concat', 'Unsqueeze'],
        "description": "Create vector from scalars"
    },

    "arange": {
        "build_graph": lambda start, stop, step: ([], pt.arange(start, stop, step, dtype='int64')),
        "strategy": arange_strategy(),
        "expected_onnx_ops": ['Range'],
        "description": "Create range tensor"
    },
}


# ============================================================================
# SUBTENSOR OPERATIONS REGISTRY
# ============================================================================

SUBTENSOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "slice_basic": {
        "build_graph": lambda x: ([x], x[2:5]),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype='float32'),
            size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Basic slicing"
    },

    "slice_multidim": {
        "build_graph": lambda x: ([x], x[1:3, 2:4]),
        "strategy": st.builds(
            lambda s1, s2: np.arange(s1 * s2).reshape(s1, s2).astype('float32'),
            s1=st.integers(5, 10),
            s2=st.integers(5, 10)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Multi-dimensional slicing"
    },

    "slice_with_step": {
        "build_graph": lambda x: ([x], x[::2]),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype='float32'),
            size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Slicing with step"
    },

    "advanced_index": {
        "build_graph": lambda x, indices: ([x], x[indices]),
        "strategy": advanced_index_strategy(),
        "expected_onnx_ops": ['Gather'],
        "description": "Advanced indexing with integer array"
    },
}


# ============================================================================
# INCSUBTENSOR OPERATIONS REGISTRY
# ============================================================================

INCSUBTENSOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "set_subtensor": {
        "build_graph": lambda x, values: ([x], pt.set_subtensor(x[2:5], values)),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ['ScatterND', 'ScatterElements'],
        "description": "Set subtensor values"
    },

    "inc_subtensor": {
        "build_graph": lambda x, values: ([x], pt.inc_subtensor(x[2:5], values)),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ['ScatterND', 'ScatterElements', 'Add'],
        "description": "Increment subtensor values"
    },
}


# ============================================================================
# HYPOTHESIS STRATEGIES (Custom Helpers)
# ============================================================================

def tensor_with_axis_strategy(dtype='float32', allow_none=True):
    """Generate tensor and valid axis for reduction operations."""
    @st.composite
    def strategy(draw):
        # Generate shape
        shape = draw(array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=10))

        # Generate tensor
        if dtype == 'bool':
            x = draw(arrays(dtype=np.bool_, shape=shape))
        else:
            x = draw(arrays(dtype=getattr(np, dtype), shape=shape))

        # Generate axis
        if allow_none:
            axis = draw(st.one_of(
                st.none(),
                st.integers(0, len(shape) - 1),
                st.lists(st.integers(0, len(shape) - 1), min_size=1, max_size=len(shape), unique=True)
            ))
        else:
            axis = draw(st.integers(0, len(shape) - 1))

        return x, axis

    return strategy()


def reshape_strategy():
    """Generate tensor and compatible reshape target."""
    @st.composite
    def strategy(draw):
        # Original shape
        shape = draw(array_shapes(min_dims=2, max_dims=3, min_side=2, max_side=6))
        total_size = int(np.prod(shape))

        # Generate tensor
        x = np.random.randn(*shape).astype('float32')

        # Generate compatible new shape (same total size)
        # For simplicity, use factorization of total_size
        new_shape = draw(compatible_shape_for_size(total_size))

        return x, new_shape

    return strategy()


def compatible_shape_for_size(total_size):
    """Generate shapes compatible with given total size."""
    # Simple factorizations
    factors = factorize(total_size)
    return st.sampled_from([
        (total_size,),
        (1, total_size),
        (total_size, 1),
        tuple(factors[:2]) if len(factors) >= 2 else (total_size,),
    ])


def factorize(n):
    """Simple factorization for shape generation."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors if factors else [n]


def concatenate_strategy():
    """Generate tensors and axis for concatenation."""
    @st.composite
    def strategy(draw):
        # Generate base shape
        shape = draw(array_shapes(min_dims=2, max_dims=3, min_side=2, max_side=8))
        axis = draw(st.integers(0, len(shape) - 1))

        # Generate two tensors with same shape except along axis
        a = np.random.randn(*shape).astype('float32')

        b_shape = list(shape)
        b_shape[axis] = draw(st.integers(2, 8))  # Different size along axis
        b = np.random.randn(*b_shape).astype('float32')

        # Create PyTensor variables with correct shapes
        a_var = pt.tensor(f'a', dtype='float32', shape=(None,) * len(shape))
        b_var = pt.tensor(f'b', dtype='float32', shape=(None,) * len(b_shape))

        return a, b, axis

    return strategy()


def alloc_strategy():
    """Generate scalar value and shape for Alloc."""
    return st.builds(
        lambda val, s1, s2: (val, s1, s2),
        val=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        s1=st.integers(2, 10),
        s2=st.integers(2, 10)
    )


def arange_strategy():
    """Generate valid start, stop, step for arange (constant only)."""
    @st.composite
    def strategy(draw):
        start = draw(st.integers(0, 5))
        stop = draw(st.integers(start + 2, start + 20))
        step = draw(st.integers(1, 3))
        return start, stop, step

    return strategy()


def set_subtensor_strategy():
    """Generate tensor and values for set_subtensor."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(10, 20))
        x = np.arange(size, dtype='float32')
        values = draw(arrays(dtype=np.float32, shape=(3,)))
        return x, values

    return strategy()


def advanced_index_strategy():
    """Generate tensor and integer indices for advanced indexing."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(10, 20))
        x = np.arange(size, dtype='float32')
        indices = draw(st.lists(st.integers(0, size - 1), min_size=1, max_size=5))
        return x, np.array(indices, dtype='int64')

    return strategy()
```


---

### Step 1.2: Property Tests Implementation

**File**: `tests/link/onnx/test_properties_tier23.py` (create new)

Implement property-based tests that use the operation registries - this replaces 36+ individual manual tests with 9 comprehensive property tests!

```python
"""Property-based tests for ONNX Tier 2-3 operations using Hypothesis."""

import pytest
import numpy as np
import pytensor
import pytensor.tensor as pt
from hypothesis import given, strategies as st, settings

# Import ONNX and skip if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types
from tests.link.onnx.strategies import (
    SHAPE_OPERATIONS,
    REDUCTION_OPERATIONS,
    ALLOCATION_OPERATIONS,
    SUBTENSOR_OPERATIONS,
    INCSUBTENSOR_OPERATIONS,
)


# ============================================================================
# PROPERTY TEST 1: Shape Operations
# ============================================================================

@given(
    op_name=st.sampled_from(list(SHAPE_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_shape_operations_correctness(op_name, data):
    """Property test: All shape operations produce correct ONNX results.

    Tests: reshape, transpose, dimshuffle, shape, join, stack, split
    Total: ~8 operations × 10 examples = 80 test scenarios
    """
    op_config = SHAPE_OPERATIONS[op_name]

    # Generate test inputs
    test_data = data.draw(op_config['strategy'])
    inputs_tuple = test_data if isinstance(test_data, tuple) else (test_data,)

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](*inputs_tuple)
    if not isinstance(graph_inputs, list):
        graph_inputs = [graph_inputs]

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, list(inputs_tuple))

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected {expected_ops}, got {node_types}"


# ============================================================================
# PROPERTY TEST 2: Reduction Operations
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
# PROPERTY TEST 3: Allocation Operations
# ============================================================================

@given(
    op_name=st.sampled_from(list(ALLOCATION_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_allocation_operations_correctness(op_name, data):
    """Property test: All allocation operations produce correct ONNX results.

    Tests: alloc, alloc_empty, make_vector, arange, eye
    Total: ~4 operations × 10 examples = 40 test scenarios
    """
    op_config = ALLOCATION_OPERATIONS[op_name]

    # Generate test data
    test_data = data.draw(op_config['strategy'])
    inputs_tuple = test_data if isinstance(test_data, tuple) else (test_data,)

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](*inputs_tuple)

    # Prepare test inputs (many allocation ops have no inputs)
    test_inputs = []

    # Special handling for AllocEmpty (only check shape/dtype)
    if op_name == "alloc_empty":
        def assert_shape_dtype(a, b):
            assert a.shape == b.shape
            assert a.dtype == b.dtype

        fn, result = compare_onnx_and_py(
            graph_inputs, graph_output, test_inputs,
            assert_fn=assert_shape_dtype
        )
    else:
        fn, result = compare_onnx_and_py(graph_inputs, graph_output, test_inputs)

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected {expected_ops}, got {node_types}"


# ============================================================================
# PROPERTY TEST 4: Subtensor Operations
# ============================================================================

@given(
    op_name=st.sampled_from(list(SUBTENSOR_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_subtensor_operations_correctness(op_name, data):
    """Property test: All subtensor operations produce correct ONNX results.

    Tests: slice (basic, multidim, with step), advanced indexing
    Total: 4 operations × 10 examples = 40 test scenarios
    """
    op_config = SUBTENSOR_OPERATIONS[op_name]

    # Generate test data
    test_data = data.draw(op_config['strategy'])
    inputs_tuple = test_data if isinstance(test_data, tuple) else (test_data,)

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](*inputs_tuple)

    # Test input is just the tensor
    test_inputs = [inputs_tuple[0]]

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, test_inputs)

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected {expected_ops}, got {node_types}"


# ============================================================================
# PROPERTY TEST 5: IncSubtensor Operations
# ============================================================================

@given(
    op_name=st.sampled_from(list(INCSUBTENSOR_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_incsubtensor_operations_correctness(op_name, data):
    """Property test: All inc/set_subtensor operations work correctly.

    Tests: set_subtensor, inc_subtensor
    Total: 2 operations × 10 examples = 20 test scenarios
    """
    op_config = INCSUBTENSOR_OPERATIONS[op_name]

    # Generate test data
    test_data = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](*test_data)

    # Test input (just the tensor)
    test_inputs = [test_data[0]]

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, test_inputs)

    # Verify ONNX nodes
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected {expected_ops}, got {node_types}"


# ============================================================================
# PROPERTY TEST 6: Dynamic Shape Handling
# ============================================================================

@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_dynamic_shape_handling(data):
    """Property test: Operations handle dynamic shapes correctly."""
    shape = data.draw(st.tuples(
        st.integers(2, 10),
        st.integers(2, 10),
        st.integers(2, 10)
    ))

    # Dynamic shape tensor
    x = pt.tensor('x', dtype='float32', shape=(None, None, None))
    y = x.reshape((-1, shape[1] * shape[2]))
    z = pt.sum(y, axis=1)

    x_val = np.random.randn(*shape).astype('float32')

    fn, result = compare_onnx_and_py([x], z, [x_val])

    assert result.shape == (shape[0],)


# ============================================================================
# PROPERTY TEST 7: Axis Parameter Variations
# ============================================================================

@pytest.mark.parametrize("axis", [None, 0, 1, [0, 1], [1, 2]])
def test_reduction_axis_variations(axis):
    """Test reductions with different axis specifications."""
    x = pt.tensor3('x', dtype='float32')
    y = pt.sum(x, axis=axis)

    x_val = np.random.randn(3, 4, 5).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert 'ReduceSum' in get_onnx_node_types(fn)


# ============================================================================
# PROPERTY TEST 8: Edge Cases
# ============================================================================

def test_empty_array_handling():
    """Test operations handle empty arrays correctly."""
    x = pt.vector('x', dtype='float32')
    y = x + 1

    x_val = np.array([], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (0,)


# ============================================================================
# PROPERTY TEST 9: Broadcasting Preservation
# ============================================================================

@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_broadcasting_preserved(data):
    """Property test: Broadcasting semantics preserved through ONNX."""
    base_size = data.draw(st.integers(3, 8))

    a = pt.vector('a', dtype='float32')
    b = pt.matrix('b', dtype='float32')
    c = a + b

    a_val = np.random.randn(base_size).astype('float32')
    b_val = np.random.randn(base_size, 1).astype('float32')

    fn, result = compare_onnx_and_py([a, b], c, [a_val, b_val])

    expected_shape = (base_size, base_size)
    assert result.shape == expected_shape
```

**Key Insight**: These 9 property tests replace 36+ individual manual tests and validate **~260 test scenarios** automatically!

---

### Step 1.3: Targeted Infrastructure Tests

**File**: `tests/link/onnx/test_tier23_infrastructure.py` (create new)

Add targeted tests for specific edge cases:

```python
"""Targeted infrastructure tests for Tier 2-3 operations."""

import pytest
import numpy as np
import pytensor.tensor as pt

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


def test_specify_shape_is_removed():
    """SpecifyShape should not create ONNX nodes."""
    from pytensor.tensor.shape import specify_shape

    x = pt.tensor('x', shape=(None, None), dtype='float32')
    x_specified = specify_shape(x, (3, 4))
    y = x_specified + 1

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    node_types = get_onnx_node_types(fn)
    assert 'SpecifyShape' not in node_types
    assert 'Add' in node_types


def test_reshape_with_minus_one():
    """Reshape with -1 (inferred dimension)."""
    x = pt.tensor('x', shape=(None, None, None), dtype='float32')
    y = x.reshape((-1,))

    x_val = np.random.randn(2, 3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (24,)
    assert 'Reshape' in get_onnx_node_types(fn)


def test_arange_requires_constants():
    """ARange requires constant inputs (ONNX limitation)."""
    x = pt.arange(0, 10, 2, dtype='int64')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.arange(0, 10, 2, dtype='int64')
    np.testing.assert_array_equal(result, expected)
    assert 'Range' in get_onnx_node_types(fn)


def test_negative_indexing():
    """Slicing with negative indices."""
    x = pt.vector('x', dtype='float32')
    y = x[-3:]

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([7, 8, 9], dtype='float32')
    np.testing.assert_array_equal(result, expected)


def test_reduction_keepdims():
    """Reduction with keepdims parameter."""
    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=1, keepdims=True)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (3, 1)
```

---

### Step 1.4: Integration Tests

**File**: `tests/link/onnx/test_tier23_integration.py` (create new)

Test realistic combined operations:

```python
"""Integration tests for Tier 2-3 operations."""

import pytest
import numpy as np
import pytensor.tensor as pt

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tests.link.onnx.test_basic import compare_onnx_and_py


def test_mean_variance_computation():
    """Compute mean and variance using reductions."""
    x = pt.matrix('x', dtype='float32')
    mean = pt.mean(x, axis=0)
    var = pt.var(x, axis=0)

    x_val = np.random.randn(10, 5).astype('float32')

    fn, results = compare_onnx_and_py([x], [mean, var], [x_val])

    mean_result, var_result = results

    expected_mean = np.mean(x_val, axis=0)
    expected_var = np.var(x_val, axis=0)

    np.testing.assert_allclose(mean_result, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(var_result, expected_var, rtol=1e-5)


def test_normalize_rows():
    """Normalize matrix rows."""
    x = pt.matrix('x', dtype='float32')
    row_sums = pt.sum(x, axis=1, keepdims=True)
    normalized = x / row_sums

    x_val = np.random.rand(5, 10).astype('float32') + 0.1

    fn, result = compare_onnx_and_py([x], normalized, [x_val])

    row_sums_result = np.sum(result, axis=1)
    np.testing.assert_allclose(row_sums_result, np.ones(5), rtol=1e-5)


def test_reshape_and_slice():
    """Combined reshape and slicing."""
    x = pt.vector('x', dtype='float32')
    reshaped = x.reshape((3, 4))
    sliced = reshaped[1:3, :]

    x_val = np.arange(12, dtype='float32')

    fn, result = compare_onnx_and_py([x], sliced, [x_val])

    expected = np.arange(12).reshape(3, 4)[1:3, :].astype('float32')
    np.testing.assert_array_equal(result, expected)


def test_softmax_implementation():
    """Softmax using Tier 2-3 ops."""
    x = pt.matrix('x', dtype='float32')

    x_max = pt.max(x, axis=1, keepdims=True)
    x_shifted = x - x_max
    exp_x = pt.exp(x_shifted)
    sum_exp = pt.sum(exp_x, axis=1, keepdims=True)
    softmax = exp_x / sum_exp

    x_val = np.random.randn(5, 10).astype('float32')

    fn, result = compare_onnx_and_py([x], softmax, [x_val])

    row_sums = np.sum(result, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(5), rtol=1e-5)
    assert np.all(result >= 0) and np.all(result <= 1)
```

---

### Test Implementation Steps

1. **Create test file structure**:
   ```bash
   touch tests/link/onnx/test_shape.py
   touch tests/link/onnx/test_subtensor.py
   touch tests/link/onnx/test_math.py
   touch tests/link/onnx/test_tensor_basic.py
   touch tests/link/onnx/test_integration.py
   ```

2. **Add shared imports and setup to each file**:
   ```python
   import pytest
   import numpy as np
   import pytensor
   import pytensor.tensor as pt

   # Import ONNX and skip if not available
   onnx = pytest.importorskip("onnx")
   ort = pytest.importorskip("onnxruntime")

   from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types
   ```

3. **Implement all test cases** as specified above

4. **Add module docstrings** explaining test organization

### Success Criteria

#### Automated Verification:
- [ ] All test files created: `ls tests/link/onnx/test_*.py`
- [ ] Tests are discoverable: `pytest --collect-only tests/link/onnx/ | grep "test_"`
- [ ] Test syntax is valid: `python -m py_compile tests/link/onnx/*.py`
- [ ] ~45 new test functions created

#### Manual Verification:
- [ ] Each test has clear, descriptive docstring
- [ ] Test names follow `test_<operation>_<variant>` pattern
- [ ] Parametrized tests used for similar cases
- [ ] Edge cases explicitly tested
- [ ] Error messages are diagnostic

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected, diagnostic ways.

### Verification Steps

1. **Run shape operation tests**:
   ```bash
   pytest tests/link/onnx/test_shape.py -v --tb=short
   ```

   **Expected**: All tests fail with `NotImplementedError` for unimplemented ops

2. **Run subtensor tests**:
   ```bash
   pytest tests/link/onnx/test_subtensor.py -v --tb=short
   ```

   **Expected**: Fail with `NotImplementedError` for Subtensor, IncSubtensor, AdvancedSubtensor1

3. **Run reduction tests**:
   ```bash
   pytest tests/link/onnx/test_math.py -v --tb=short
   ```

   **Expected**: Fail with `NotImplementedError` for CAReduce, Argmax, Argmin

4. **Run allocation tests**:
   ```bash
   pytest tests/link/onnx/test_tensor_basic.py -v --tb=short
   ```

   **Expected**: Fail with `NotImplementedError` for Alloc, ARange, Eye

5. **Document failure patterns**:
   Create `tests/link/onnx/TIER2_3_EXPECTED_FAILURES.md` documenting what we see

### Expected Failures by Operation

**Shape Operations**:
- `test_shape_*`: `NotImplementedError: No ONNX conversion available for: Shape`
- `test_reshape_*`: `NotImplementedError: No ONNX conversion available for: Reshape`
- `test_dimshuffle_*`: `NotImplementedError: No ONNX conversion available for: DimShuffle`
- `test_join_*`: `NotImplementedError: No ONNX conversion available for: Join`
- `test_split_*`: `NotImplementedError: No ONNX conversion available for: Split`

**Subtensor Operations**:
- `test_subtensor_*`: `NotImplementedError: No ONNX conversion available for: Subtensor`
- `test_advanced_subtensor_*`: `NotImplementedError: No ONNX conversion available for: AdvancedSubtensor1`
- `test_inc_subtensor_*`: `NotImplementedError: No ONNX conversion available for: IncSubtensor`
- `test_set_subtensor_*`: `NotImplementedError: No ONNX conversion available for: IncSubtensor`

**Reduction Operations**:
- `test_sum_*`: `NotImplementedError: No ONNX conversion available for: CAReduce`
- `test_prod`: `NotImplementedError: No ONNX conversion available for: CAReduce`
- `test_max_min_*`: `NotImplementedError: No ONNX conversion available for: Max` (or Min)
- `test_argmax_argmin`: `NotImplementedError: No ONNX conversion available for: Argmax` (or Argmin)
- `test_logical_*`: `NotImplementedError: No ONNX conversion available for: CAReduce`

**Allocation Operations**:
- `test_alloc_*`: `NotImplementedError: No ONNX conversion available for: Alloc`
- `test_alloc_empty`: `NotImplementedError: No ONNX conversion available for: AllocEmpty`
- `test_make_vector`: `NotImplementedError: No ONNX conversion available for: MakeVector`
- `test_arange_*`: `NotImplementedError: No ONNX conversion available for: ARange`
- `test_eye_*`: `NotImplementedError: No ONNX conversion available for: Eye`

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `pytest --collect-only tests/link/onnx/ | grep -c "test_"` shows ~74 (29 from Tier 1 + 45 new)
- [ ] All new tests fail: `pytest tests/link/onnx/test_shape.py tests/link/onnx/test_subtensor.py tests/link/onnx/test_math.py tests/link/onnx/test_tensor_basic.py -v | grep FAILED` shows ~45 failures
- [ ] No syntax errors: All tests run (even if they fail)
- [ ] Tier 1 tests still pass: `pytest tests/link/onnx/test_elemwise.py -v` shows all passing

#### Manual Verification:
- [ ] Each test fails with expected error type
- [ ] Error messages clearly indicate missing operation
- [ ] Stack traces point to dispatch system
- [ ] No cryptic or misleading errors

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement operations by making tests pass, one category at a time.

### Implementation Order

1. **Shape inspection** (Shape, Shape_i) - simplest
2. **Reshape operations** (Reshape, DimShuffle) - core functionality
3. **Reductions** (Sum, Prod, Max, Min, Argmax, Argmin) - frequently used
4. **Allocation** (Alloc, ARange, Eye) - tensor creation
5. **Join/Split** (Join, Stack, Split) - tensor manipulation
6. **Subtensor** (basic slicing) - indexing
7. **AdvancedSubtensor** (integer array indexing) - advanced indexing
8. **IncSubtensor** (set/increment) - most complex

---

### Implementation 1: Shape Operations

**Target Tests**: `test_shape_basic`, `test_shape_i`
**Current Failures**: `NotImplementedError: No ONNX conversion available for: Shape`

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/shape.py` (new file)

```python
"""ONNX conversion for shape operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.shape import Shape, Shape_i, SpecifyShape
from pytensor.graph.basic import Constant

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Shape)
def onnx_funcify_Shape(op, node, var_names, get_var_name, **kwargs):
    """Convert Shape op to ONNX Shape node."""
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    onnx_node = helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=[output_name],
        name=f"Shape_{output_name}",
    )

    return onnx_node


@onnx_funcify.register(Shape_i)
def onnx_funcify_Shape_i(op, node, var_names, get_var_name, **kwargs):
    """Convert Shape_i op to ONNX Shape + Gather nodes.

    Shape_i extracts a specific dimension from a tensor's shape.
    This requires two ONNX nodes:
    1. Shape - get full shape
    2. Gather - extract the specific index
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Create intermediate name for full shape
    shape_name = f"{output_name}_shape"

    # Node 1: Get full shape
    shape_node = helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=[shape_name],
        name=f"Shape_{shape_name}",
    )

    # Node 2: Gather the specific index
    # op.i contains the axis index
    axis_idx = op.i
    gather_node = helper.make_node(
        'Gather',
        inputs=[shape_name, f"{shape_name}_idx"],
        outputs=[output_name],
        name=f"Gather_{output_name}",
        axis=0,  # Gather from dimension 0 of shape tensor
    )

    # We need to create a constant for the index
    # This will be added to initializers
    # For now, we'll assume the index is embedded in the node
    # In practice, you may need to handle this differently

    # Simplified: Create Constant node for index
    idx_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[f"{shape_name}_idx"],
        name=f"Constant_{shape_name}_idx",
        value=helper.make_tensor(
            name=f"{shape_name}_idx_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[axis_idx],
        )
    )

    return [idx_constant, shape_node, gather_node]


@onnx_funcify.register(SpecifyShape)
def onnx_funcify_SpecifyShape(op, node, var_names, get_var_name, **kwargs):
    """SpecifyShape is just a hint - pass through input.

    SpecifyShape doesn't change the tensor data, it just provides
    shape information for optimization. In ONNX export, we can
    safely ignore it and just pass the input through.
    """
    # Return None - no ONNX node needed
    # The input will be directly connected to uses of the output
    return None
```

**Debugging Approach**:
1. Run: `pytest tests/link/onnx/test_shape.py::test_shape_basic -v`
2. Should pass (Shape creates ONNX Shape node)
3. Run: `pytest tests/link/onnx/test_shape.py::test_shape_i -v`
4. May need to adjust Constant handling for index
5. Run: `pytest tests/link/onnx/test_shape.py::test_specify_shape -v`
6. Should pass (SpecifyShape returns None)

#### Success Criteria

##### Automated Verification:
- [ ] Shape tests pass: `pytest tests/link/onnx/test_shape.py::test_shape_basic -v`
- [ ] Shape_i tests pass: `pytest tests/link/onnx/test_shape.py::test_shape_i -v`
- [ ] SpecifyShape test passes: `pytest tests/link/onnx/test_shape.py::test_specify_shape -v`

##### Manual Verification:
- [ ] ONNX model validates with `onnx.checker.check_model`
- [ ] Correct ONNX node types generated
- [ ] Shape values match NumPy reference

---

### Implementation 2: Reshape Operations

**Target Tests**: `test_reshape_*`, `test_dimshuffle_*`
**Current Failures**: `NotImplementedError` for Reshape, DimShuffle

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/shape.py` (continue)

```python
from pytensor.tensor.shape import Reshape
from pytensor.tensor.elemwise import DimShuffle


@onnx_funcify.register(Reshape)
def onnx_funcify_Reshape(op, node, var_names, get_var_name, **kwargs):
    """Convert Reshape op to ONNX Reshape node.

    Reshape changes tensor dimensions without changing data.
    ONNX Reshape takes two inputs:
    1. data - the tensor to reshape
    2. shape - target shape (as 1D int64 tensor)
    """
    data_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # The second input is the target shape
    # It may be a constant or computed from other tensors
    shape_input = node.inputs[1]

    if isinstance(shape_input, Constant):
        # Shape is constant - create ONNX Constant node
        shape_data = np.array(shape_input.data, dtype=np.int64)
        shape_name = f"{output_name}_shape"

        shape_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            name=f"Constant_{shape_name}",
            value=helper.make_tensor(
                name=f"{shape_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[len(shape_data)],
                vals=shape_data.tolist(),
            )
        )

        reshape_node = helper.make_node(
            'Reshape',
            inputs=[data_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return [shape_constant, reshape_node]
    else:
        # Shape is computed - use its name directly
        shape_name = get_var_name(shape_input)

        reshape_node = helper.make_node(
            'Reshape',
            inputs=[data_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return reshape_node


@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(op, node, var_names, get_var_name, **kwargs):
    """Convert DimShuffle op to ONNX Transpose/Squeeze/Unsqueeze nodes.

    DimShuffle handles:
    - Transpose: reordering dimensions
    - Squeeze: removing size-1 dimensions
    - Unsqueeze: adding size-1 dimensions

    The new_order tuple uses:
    - Integers for dimension reordering
    - 'x' for adding dimensions
    - Omitted dimensions are dropped (squeeze)
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    input_ndim = op.input_ndim
    new_order = op.new_order

    # Separate the operations:
    # 1. Which dimensions to keep (not 'x')
    # 2. Which dimensions are being reordered
    # 3. Where to add new dimensions ('x')

    # Find 'x' positions (dimensions to add)
    x_positions = [i for i, dim in enumerate(new_order) if dim == 'x']

    # Find dimension mapping (non-'x' elements)
    dim_mapping = [dim for dim in new_order if dim != 'x']

    # Check if we need to drop dimensions (squeeze)
    all_dims = set(range(input_ndim))
    kept_dims = set(dim_mapping)
    dropped_dims = sorted(all_dims - kept_dims)

    nodes = []
    current_output = input_name

    # Step 1: Squeeze dropped dimensions (if any)
    if dropped_dims:
        squeeze_output = f"{output_name}_squeezed"
        squeeze_node = helper.make_node(
            'Squeeze',
            inputs=[current_output],
            outputs=[squeeze_output],
            name=f"Squeeze_{squeeze_output}",
            axes=dropped_dims,
        )
        nodes.append(squeeze_node)
        current_output = squeeze_output

    # Step 2: Transpose if dimensions are reordered
    if dim_mapping != list(range(len(dim_mapping))):
        transpose_output = f"{output_name}_transposed" if x_positions else output_name
        transpose_node = helper.make_node(
            'Transpose',
            inputs=[current_output],
            outputs=[transpose_output],
            name=f"Transpose_{transpose_output}",
            perm=dim_mapping,
        )
        nodes.append(transpose_node)
        current_output = transpose_output

    # Step 3: Unsqueeze to add dimensions (if any 'x')
    if x_positions:
        unsqueeze_node = helper.make_node(
            'Unsqueeze',
            inputs=[current_output],
            outputs=[output_name],
            name=f"Unsqueeze_{output_name}",
            axes=x_positions,
        )
        nodes.append(unsqueeze_node)

    return nodes if nodes else None
```

**Debugging Approach**:
1. Run: `pytest tests/link/onnx/test_shape.py::test_reshape_basic -v`
2. Verify Reshape node is created
3. Run: `pytest tests/link/onnx/test_reshape_with_minus_one -v`
4. Verify -1 dimension inference works
5. Run: `pytest tests/link/onnx/test_dimshuffle_transpose -v`
6. Verify Transpose node is created
7. Run: `pytest tests/link/onnx/test_dimshuffle_add_dim -v`
8. Verify Unsqueeze works
9. Run: `pytest tests/link/onnx/test_dimshuffle_squeeze -v`
10. Verify Squeeze works
11. Run parametrized complex DimShuffle tests

#### Success Criteria

##### Automated Verification:
- [ ] All reshape tests pass: `pytest tests/link/onnx/test_shape.py -k reshape -v`
- [ ] All dimshuffle tests pass: `pytest tests/link/onnx/test_shape.py -k dimshuffle -v`

##### Manual Verification:
- [ ] Reshape handles constant and dynamic shapes
- [ ] DimShuffle handles all combinations correctly
- [ ] Complex patterns create correct ONNX node sequences

---

### Implementation 3: Reduction Operations

**Target Tests**: `test_sum_*`, `test_prod`, `test_max_min_*`, `test_argmax_argmin`, `test_logical_*`
**Current Failures**: `NotImplementedError` for CAReduce, Argmax, Argmin

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/math.py` (new file)

```python
"""ONNX conversion for math operations (reductions)."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.math import CAReduce, Argmax, Argmin
from pytensor.scalar.basic import Add, Mul, Maximum, Minimum, AND, OR

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX reduction ops
REDUCE_OP_MAP = {
    Add: 'ReduceSum',
    Mul: 'ReduceProd',
    Maximum: 'ReduceMax',
    Minimum: 'ReduceMin',
    AND: 'ReduceMin',  # For boolean AND
    OR: 'ReduceMax',   # For boolean OR
}


@onnx_funcify.register(CAReduce)
def onnx_funcify_CAReduce(op, node, var_names, get_var_name, **kwargs):
    """Convert CAReduce op to ONNX reduction node.

    CAReduce performs reductions (sum, prod, max, min) along specified axes.
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in REDUCE_OP_MAP:
        raise NotImplementedError(
            f"CAReduce with scalar op {scalar_op_type.__name__} not supported for ONNX export"
        )

    onnx_op_type = REDUCE_OP_MAP[scalar_op_type]

    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Get axis parameter
    axes = op.axis
    if axes is None:
        # Reduce over all axes
        axes = None
    elif isinstance(axes, (tuple, list)):
        # Specific axes
        axes = list(axes)
    else:
        # Single axis
        axes = [axes]

    # ONNX ReduceXXX attributes
    # keepdims: whether to keep reduced dimensions as size 1
    # axes: which axes to reduce over

    onnx_node = helper.make_node(
        onnx_op_type,
        inputs=[input_name],
        outputs=[output_name],
        name=f"{onnx_op_type}_{output_name}",
        axes=axes,
        keepdims=0,  # PyTensor default is to not keep dims
    )

    return onnx_node


@onnx_funcify.register(Argmax)
def onnx_funcify_Argmax(op, node, var_names, get_var_name, **kwargs):
    """Convert Argmax op to ONNX ArgMax node."""
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    axis = op.axis
    if axis is None:
        # Argmax over all axes - need to flatten first
        flatten_name = f"{output_name}_flat"
        flatten_node = helper.make_node(
            'Flatten',
            inputs=[input_name],
            outputs=[flatten_name],
            name=f"Flatten_{flatten_name}",
            axis=0,
        )

        argmax_node = helper.make_node(
            'ArgMax',
            inputs=[flatten_name],
            outputs=[output_name],
            name=f"ArgMax_{output_name}",
            axis=0,
            keepdims=0,
        )

        return [flatten_node, argmax_node]
    else:
        # Argmax over specific axis
        onnx_node = helper.make_node(
            'ArgMax',
            inputs=[input_name],
            outputs=[output_name],
            name=f"ArgMax_{output_name}",
            axis=axis,
            keepdims=0,
        )

        return onnx_node


@onnx_funcify.register(Argmin)
def onnx_funcify_Argmin(op, node, var_names, get_var_name, **kwargs):
    """Convert Argmin op to ONNX ArgMin node."""
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    axis = op.axis
    if axis is None:
        # Argmin over all axes - need to flatten first
        flatten_name = f"{output_name}_flat"
        flatten_node = helper.make_node(
            'Flatten',
            inputs=[input_name],
            outputs=[flatten_name],
            name=f"Flatten_{flatten_name}",
            axis=0,
        )

        argmin_node = helper.make_node(
            'ArgMin',
            inputs=[flatten_name],
            outputs=[output_name],
            name=f"ArgMin_{output_name}",
            axis=0,
            keepdims=0,
        )

        return [flatten_node, argmin_node]
    else:
        # Argmin over specific axis
        onnx_node = helper.make_node(
            'ArgMin',
            inputs=[input_name],
            outputs=[output_name],
            name=f"ArgMin_{output_name}",
            axis=axis,
            keepdims=0,
        )

        return onnx_node
```

**Debugging Approach**:
1. Run: `pytest tests/link/onnx/test_math.py::test_sum_basic -v`
2. Verify ReduceSum is created
3. Test different axis parameters
4. Run: `pytest tests/link/onnx/test_math.py::test_argmax_argmin -v`
5. Verify ArgMax/ArgMin nodes
6. Run all reduction tests

#### Success Criteria

##### Automated Verification:
- [ ] All reduction tests pass: `pytest tests/link/onnx/test_math.py -v`
- [ ] Sum, Prod, Max, Min work: Test parametrized axis values
- [ ] Argmax, Argmin work: Test axis=None and specific axes

##### Manual Verification:
- [ ] Axis handling is correct
- [ ] Output dtypes match (int64 for argmax/argmin)
- [ ] Edge cases (axis=None, empty arrays) handled

---

### Implementation 4: Allocation Operations

**Target Tests**: `test_alloc_*`, `test_arange_*`, `test_eye_*`, `test_make_vector`
**Current Failures**: `NotImplementedError` for Alloc, ARange, Eye, MakeVector

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/tensor_basic.py` (new file)

```python
"""ONNX conversion for tensor basic operations (allocation, etc.)."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.basic import Alloc, AllocEmpty, MakeVector, ARange, Eye
from pytensor.graph.basic import Constant

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Alloc)
def onnx_funcify_Alloc(op, node, var_names, get_var_name, **kwargs):
    """Convert Alloc op to ONNX Expand node.

    Alloc broadcasts a value to a specified shape.
    ONNX Expand does the same thing.
    """
    value_input = node.inputs[0]
    shape_inputs = node.inputs[1:]

    value_name = get_var_name(value_input)
    output_name = get_var_name(node.outputs[0])

    # Create shape tensor from shape inputs
    # Shape inputs are scalars that specify each dimension
    shape_name = f"{output_name}_shape"

    if all(isinstance(inp, Constant) for inp in shape_inputs):
        # All shape dimensions are constants
        shape_data = np.array([inp.data for inp in shape_inputs], dtype=np.int64)

        shape_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            name=f"Constant_{shape_name}",
            value=helper.make_tensor(
                name=f"{shape_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[len(shape_data)],
                vals=shape_data.tolist(),
            )
        )

        expand_node = helper.make_node(
            'Expand',
            inputs=[value_name, shape_name],
            outputs=[output_name],
            name=f"Expand_{output_name}",
        )

        return [shape_constant, expand_node]
    else:
        # Some shape dimensions are dynamic - need to use Concat
        shape_element_names = [get_var_name(inp) for inp in shape_inputs]

        # Concatenate shape elements into shape vector
        concat_node = helper.make_node(
            'Concat',
            inputs=shape_element_names,
            outputs=[shape_name],
            name=f"Concat_{shape_name}",
            axis=0,
        )

        expand_node = helper.make_node(
            'Expand',
            inputs=[value_name, shape_name],
            outputs=[output_name],
            name=f"Expand_{output_name}",
        )

        return [concat_node, expand_node]


@onnx_funcify.register(AllocEmpty)
def onnx_funcify_AllocEmpty(op, node, var_names, get_var_name, **kwargs):
    """Convert AllocEmpty to ONNX ConstantOfShape.

    AllocEmpty creates uninitialized array. In ONNX, we use
    ConstantOfShape with value 0 (values don't matter, just shape/dtype).
    """
    shape_inputs = node.inputs
    output_name = get_var_name(node.outputs[0])

    # Create shape tensor
    shape_name = f"{output_name}_shape"

    if all(isinstance(inp, Constant) for inp in shape_inputs):
        # Constant shape
        shape_data = np.array([inp.data for inp in shape_inputs], dtype=np.int64)

        shape_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            name=f"Constant_{shape_name}",
            value=helper.make_tensor(
                name=f"{shape_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[len(shape_data)],
                vals=shape_data.tolist(),
            )
        )

        # ConstantOfShape with value 0
        dtype = op.dtype
        dtype_map = {
            'float32': helper.TensorProto.FLOAT,
            'float64': helper.TensorProto.DOUBLE,
            'int32': helper.TensorProto.INT32,
            'int64': helper.TensorProto.INT64,
        }
        onnx_dtype = dtype_map.get(dtype, helper.TensorProto.FLOAT)

        constant_of_shape_node = helper.make_node(
            'ConstantOfShape',
            inputs=[shape_name],
            outputs=[output_name],
            name=f"ConstantOfShape_{output_name}",
            value=helper.make_tensor(
                name=f"{output_name}_value",
                data_type=onnx_dtype,
                dims=[],
                vals=[0],
            )
        )

        return [shape_constant, constant_of_shape_node]
    else:
        # Dynamic shape - similar to Alloc
        shape_element_names = [get_var_name(inp) for inp in shape_inputs]

        concat_node = helper.make_node(
            'Concat',
            inputs=shape_element_names,
            outputs=[shape_name],
            name=f"Concat_{shape_name}",
            axis=0,
        )

        dtype = op.dtype
        dtype_map = {
            'float32': helper.TensorProto.FLOAT,
            'float64': helper.TensorProto.DOUBLE,
            'int32': helper.TensorProto.INT32,
            'int64': helper.TensorProto.INT64,
        }
        onnx_dtype = dtype_map.get(dtype, helper.TensorProto.FLOAT)

        constant_of_shape_node = helper.make_node(
            'ConstantOfShape',
            inputs=[shape_name],
            outputs=[output_name],
            name=f"ConstantOfShape_{output_name}",
            value=helper.make_tensor(
                name=f"{output_name}_value",
                data_type=onnx_dtype,
                dims=[],
                vals=[0],
            )
        )

        return [concat_node, constant_of_shape_node]


@onnx_funcify.register(MakeVector)
def onnx_funcify_MakeVector(op, node, var_names, get_var_name, **kwargs):
    """Convert MakeVector to ONNX Concat of Unsqueezed scalars.

    MakeVector creates a 1D vector from scalars.
    """
    if len(node.inputs) == 0:
        # Empty vector
        output_name = get_var_name(node.outputs[0])

        # Create empty constant
        dtype = op.dtype
        dtype_map = {
            'float32': helper.TensorProto.FLOAT,
            'float64': helper.TensorProto.DOUBLE,
            'int32': helper.TensorProto.INT32,
            'int64': helper.TensorProto.INT64,
        }
        onnx_dtype = dtype_map.get(dtype, helper.TensorProto.FLOAT)

        empty_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[output_name],
            name=f"Constant_{output_name}",
            value=helper.make_tensor(
                name=f"{output_name}_value",
                data_type=onnx_dtype,
                dims=[0],
                vals=[],
            )
        )

        return empty_constant

    # Unsqueeze each scalar to shape (1,), then concatenate
    nodes = []
    unsqueezed_names = []

    for i, inp in enumerate(node.inputs):
        input_name = get_var_name(inp)
        unsqueezed_name = f"{output_name}_elem_{i}"

        unsqueeze_node = helper.make_node(
            'Unsqueeze',
            inputs=[input_name],
            outputs=[unsqueezed_name],
            name=f"Unsqueeze_{unsqueezed_name}",
            axes=[0],
        )
        nodes.append(unsqueeze_node)
        unsqueezed_names.append(unsqueezed_name)

    # Concatenate all elements
    output_name = get_var_name(node.outputs[0])
    concat_node = helper.make_node(
        'Concat',
        inputs=unsqueezed_names,
        outputs=[output_name],
        name=f"Concat_{output_name}",
        axis=0,
    )
    nodes.append(concat_node)

    return nodes


@onnx_funcify.register(ARange)
def onnx_funcify_ARange(op, node, var_names, get_var_name, **kwargs):
    """Convert ARange to ONNX Range node.

    IMPORTANT: ONNX Range requires constant inputs (start, limit, delta).
    Dynamic ranges are not supported in ONNX standard.
    """
    start_input = node.inputs[0]
    stop_input = node.inputs[1]
    step_input = node.inputs[2]

    # Verify all inputs are constants
    if not all(isinstance(inp, Constant) for inp in [start_input, stop_input, step_input]):
        raise NotImplementedError(
            "ARange with dynamic (non-constant) inputs is not supported in ONNX. "
            "All start, stop, step values must be constants."
        )

    output_name = get_var_name(node.outputs[0])

    # Create constant nodes for start, limit, delta
    start_name = f"{output_name}_start"
    stop_name = f"{output_name}_stop"
    step_name = f"{output_name}_step"

    dtype = op.dtype
    dtype_map = {
        'int32': helper.TensorProto.INT32,
        'int64': helper.TensorProto.INT64,
        'float32': helper.TensorProto.FLOAT,
        'float64': helper.TensorProto.DOUBLE,
    }
    onnx_dtype = dtype_map.get(dtype, helper.TensorProto.INT64)

    start_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[start_name],
        name=f"Constant_{start_name}",
        value=helper.make_tensor(
            name=f"{start_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[start_input.data],
        )
    )

    stop_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[stop_name],
        name=f"Constant_{stop_name}",
        value=helper.make_tensor(
            name=f"{stop_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[stop_input.data],
        )
    )

    step_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[step_name],
        name=f"Constant_{step_name}",
        value=helper.make_tensor(
            name=f"{step_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[step_input.data],
        )
    )

    # Range node
    range_node = helper.make_node(
        'Range',
        inputs=[start_name, stop_name, step_name],
        outputs=[output_name],
        name=f"Range_{output_name}",
    )

    return [start_constant, stop_constant, step_constant, range_node]


@onnx_funcify.register(Eye)
def onnx_funcify_Eye(op, node, var_names, get_var_name, **kwargs):
    """Convert Eye to ONNX EyeLike or custom implementation.

    Eye creates an identity matrix (or offset diagonal).
    ONNX has EyeLike but it's limited. For full support,
    we may need a custom implementation.
    """
    # For now, raise NotImplementedError
    # Eye is complex and may require a sequence of operations
    raise NotImplementedError(
        "Eye operation not yet implemented for ONNX export. "
        "Eye requires complex logic for non-square matrices and diagonal offsets."
    )
```

**Debugging Approach**:
1. Run allocation tests one at a time
2. Verify ONNX node types match expectations
3. Test edge cases (empty arrays, single elements)

#### Success Criteria

##### Automated Verification:
- [ ] Alloc tests pass: `pytest tests/link/onnx/test_tensor_basic.py -k alloc -v`
- [ ] ARange tests pass: `pytest tests/link/onnx/test_tensor_basic.py -k arange -v`
- [ ] MakeVector tests pass: `pytest tests/link/onnx/test_tensor_basic.py -k make_vector -v`
- [ ] Eye tests skipped or implemented: Mark with `pytest.skip` if not implementing

##### Manual Verification:
- [ ] Constant and dynamic shapes both work
- [ ] Dtypes are preserved correctly
- [ ] Edge cases handled

---

### Implementation 5-8: Join/Split, Subtensor, AdvancedSubtensor, IncSubtensor

Due to length constraints, these implementations follow similar patterns:

1. **Join/Split**: Use ONNX Concat and Split nodes
2. **Subtensor**: Map slicing to ONNX Slice node (handle negative indices, steps)
3. **AdvancedSubtensor**: Use ONNX Gather node for integer array indexing
4. **IncSubtensor**: Use ONNX ScatterND or ScatterElements (most complex)

Each implementation should:
- Create dispatch file (e.g., `dispatch/subtensor.py`)
- Register handlers for each Op
- Handle edge cases
- Return appropriate ONNX nodes

**Success criteria for each**:
- All related tests pass
- ONNX models validate
- Outputs match Python reference

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor to improve code quality while keeping tests green.

### Refactoring Targets

1. **Axis Handling Helper**:
   - Extract common axis normalization logic
   - Handle None, single int, list of ints uniformly

2. **Shape Tensor Creation**:
   - Extract helper for creating shape tensors from list of scalars
   - Handles both constant and dynamic cases

3. **Constant Node Creation**:
   - Helper function for creating ONNX Constant nodes
   - Reduces duplication

4. **Dtype Mapping**:
   - Centralized dtype mapping dictionary
   - Shared across all dispatch modules

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/link/onnx/ -v`
- [ ] Code coverage maintained: `pytest --cov=pytensor.link.onnx tests/link/onnx/`
- [ ] Linting passes: `black --check pytensor/link/onnx/`

#### Manual Verification:
- [ ] No code duplication
- [ ] Clear helper functions
- [ ] Improved readability

---

## Success Metrics

### Tier 2-3 Complete When:

- ✅ All 45+ new tests pass
- ✅ Can export shape operations (reshape, transpose, slice)
- ✅ Can export reductions (sum, mean, variance)
- ✅ Can export tensor creation (zeros, ones, arange)
- ✅ Integration tests pass (mean/variance, normalize, etc.)
- ✅ Outputs match Python reference (within tolerance)
- ✅ All ONNX models validate with `onnx.checker.check_model`
- ✅ Documentation updated

### Next Steps

After Tier 2-3 completion, proceed to:
- **Tier 4-5 Plan**: Linear algebra and advanced operations
- See: `thoughts/shared/plans/onnx-backend-tier4-5-linalg-advanced-tdd.md`

---

## References

### Related Research
- Infrastructure roadmap: `thoughts/shared/research/2025-11-04_11-52-15_onnx-backend-infrastructure-roadmap.md`
- Operations roadmap: `thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md`

### Test Pattern References
- Shape operations: `tests/link/jax/test_shape.py`, `tests/tensor/test_shape.py`
- Reductions: `tests/link/jax/test_elemwise.py`, `tests/tensor/test_math.py`
- Allocation: `tests/link/jax/test_tensor_basic.py`, `tests/tensor/test_basic.py`
- Subtensor: `tests/link/jax/test_subtensor.py`, `tests/tensor/test_subtensor.py`

### ONNX Specification
- ONNX Operators: https://onnx.ai/onnx/operators/
- Shape operations: Reshape, Transpose, Squeeze, Unsqueeze, Concat, Split
- Reductions: ReduceSum, ReduceProd, ReduceMax, ReduceMin, ArgMax, ArgMin
- Tensor creation: Expand, ConstantOfShape, Range
