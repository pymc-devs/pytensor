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

✅ **Comprehensive Testing**:
- 45+ new tests (15 for Tier 2, 15 for Tier 3, plus integration tests)
- Dynamic shape handling validated
- Static shape inference preserved
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
1. **Test static and dynamic shapes separately**: ONNX has different code paths
2. **Test axis specifications thoroughly**: None, single, multiple, negative indices
3. **Test edge cases explicitly**: Empty arrays, zero dimensions, out of bounds
4. **Compare against NumPy behavior**: Ensure PyTensor → ONNX → Result matches NumPy
5. **Test ONNX node types**: Verify correct ONNX operators are generated

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive, informative tests that define shape operations and reductions completely. Tests should fail in expected, diagnostic ways.

---

### Test Category 1: Shape Inspection Operations

**Test File**: `tests/link/onnx/test_shape.py`
**Purpose**: Test Shape, Shape_i, and SpecifyShape operations

#### Test: `test_shape_basic`
**Purpose**: Test Shape op returns tensor shape

**Test Data**: Matrix with known shape (3, 4)

**Expected Behavior**: Shape operation returns [3, 4]

```python
def test_shape_basic():
    """Test that Shape operation returns correct shape tensor."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    s = x.shape

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], s, [x_val])

    # Verify ONNX node type
    from tests.link.onnx.test_basic import get_onnx_node_types
    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types, \
        f"Expected 'Shape' node in ONNX graph, got {node_types}"

    # Verify shape is correct
    assert tuple(result) == (3, 4), \
        f"Expected shape (3, 4), got {tuple(result)}"
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: `No ONNX conversion available for: Shape`

#### Test: `test_shape_i`
**Purpose**: Test Shape_i extracts specific dimension

```python
def test_shape_i():
    """Test that Shape_i extracts specific dimension."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    s0 = x.shape[0]  # First dimension
    s1 = x.shape[1]  # Second dimension

    x_val = np.random.randn(3, 4).astype('float32')

    # Test first dimension
    fn0, result0 = compare_onnx_and_py([x], s0, [x_val])
    assert result0 == 3, f"Expected dimension 0 to be 3, got {result0}"

    # Test second dimension
    fn1, result1 = compare_onnx_and_py([x], s1, [x_val])
    assert result1 == 4, f"Expected dimension 1 to be 4, got {result1}"

    # Verify ONNX uses Shape + Gather
    node_types = get_onnx_node_types(fn0)
    assert 'Shape' in node_types and 'Gather' in node_types, \
        f"Expected 'Shape' and 'Gather' nodes, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Shape_i

#### Test: `test_specify_shape`
**Purpose**: Test SpecifyShape for optimization hints

```python
def test_specify_shape():
    """Test that SpecifyShape is handled (typically removed in ONNX export)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.shape import specify_shape

    x = pt.tensor('x', shape=(None, None), dtype='float32')
    # Specify that x has shape (3, 4)
    x_specified = specify_shape(x, (3, 4))
    y = x_specified + 1  # Use in computation

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # SpecifyShape should not create ONNX nodes (it's just a hint)
    node_types = get_onnx_node_types(fn)
    # Should only have Add (for +1), no SpecifyShape node
    assert 'Add' in node_types, f"Expected 'Add' node, got {node_types}"
```

**Expected Failure Mode**: May pass if SpecifyShape is already handled by graph rewrites

---

### Test Category 2: Reshape Operations

**Test File**: `tests/link/onnx/test_shape.py` (continued)
**Purpose**: Test Reshape and DimShuffle operations

#### Test: `test_reshape_basic`
**Purpose**: Test basic reshape operation

```python
def test_reshape_basic():
    """Test basic reshape operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    # Reshape from (2, 6) to (3, 4)
    y = x.reshape((3, 4))

    x_val = np.arange(12).reshape(2, 6).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (3, 4), \
        f"Expected shape (3, 4), got {result.shape}"

    # Verify ONNX uses Reshape
    node_types = get_onnx_node_types(fn)
    assert 'Reshape' in node_types, \
        f"Expected 'Reshape' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Reshape

#### Test: `test_reshape_with_minus_one`
**Purpose**: Test reshape with inferred dimension (-1)

```python
def test_reshape_with_minus_one():
    """Test reshape with inferred dimension using -1."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor('x', shape=(None, None, None), dtype='float32')

    # Flatten to 1D (infer size)
    y1 = x.reshape((-1,))

    # Reshape to (6, -1) - infer second dimension
    y2 = x.reshape((6, -1))

    x_val = np.random.randn(2, 3, 4).astype('float32')

    # Test flatten
    fn1, result1 = compare_onnx_and_py([x], y1, [x_val])
    assert result1.shape == (24,), \
        f"Expected shape (24,), got {result1.shape}"

    # Test inferred dimension
    fn2, result2 = compare_onnx_and_py([x], y2, [x_val])
    assert result2.shape == (6, 4), \
        f"Expected shape (6, 4), got {result2.shape}"
```

**Expected Failure Mode**: May fail with handling of -1 dimension

#### Test: `test_reshape_dynamic_shape`
**Purpose**: Test reshape using another tensor's shape

```python
def test_reshape_dynamic_shape():
    """Test reshape using dynamic shape from another tensor."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    target = pt.matrix('target', dtype='float32')

    # Reshape x to match target's shape
    y = x.reshape(target.shape)

    x_val = np.arange(12).astype('float32')
    target_val = np.zeros((3, 4), dtype='float32')

    fn, result = compare_onnx_and_py([x, target], y, [x_val, target_val])

    assert result.shape == (3, 4), \
        f"Expected shape (3, 4), got {result.shape}"
```

**Expected Failure Mode**: May fail with dynamic shape handling

#### Test: `test_dimshuffle_transpose`
**Purpose**: Test DimShuffle for transpose

```python
def test_dimshuffle_transpose():
    """Test DimShuffle transpose operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    # Transpose
    y = x.T

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (4, 3), \
        f"Expected shape (4, 3), got {result.shape}"

    # Verify ONNX uses Transpose
    node_types = get_onnx_node_types(fn)
    assert 'Transpose' in node_types, \
        f"Expected 'Transpose' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for DimShuffle

#### Test: `test_dimshuffle_add_dim`
**Purpose**: Test DimShuffle adding dimensions

```python
def test_dimshuffle_add_dim():
    """Test DimShuffle adding dimensions (unsqueeze)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    # Add dimension at start
    y = x.dimshuffle('x', 0)

    x_val = np.random.randn(5).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (1, 5), \
        f"Expected shape (1, 5), got {result.shape}"

    # Verify ONNX uses Unsqueeze
    node_types = get_onnx_node_types(fn)
    assert 'Unsqueeze' in node_types, \
        f"Expected 'Unsqueeze' node, got {node_types}"
```

**Expected Failure Mode**: May fail with 'x' notation handling

#### Test: `test_dimshuffle_squeeze`
**Purpose**: Test DimShuffle removing dimensions

```python
def test_dimshuffle_squeeze():
    """Test DimShuffle removing dimensions (squeeze)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Tensor with known broadcastable dimension
    x = pt.tensor('x', shape=(None, 1, None), dtype='float32')
    # Drop the middle dimension
    y = x.dimshuffle(0, 2)

    x_val = np.random.randn(3, 1, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == (3, 4), \
        f"Expected shape (3, 4), got {result.shape}"

    # Verify ONNX uses Squeeze
    node_types = get_onnx_node_types(fn)
    assert 'Squeeze' in node_types, \
        f"Expected 'Squeeze' node, got {node_types}"
```

**Expected Failure Mode**: May fail with broadcastable dimension handling

#### Test: `test_dimshuffle_complex`
**Purpose**: Test complex DimShuffle (transpose + add/remove dims)

```python
@pytest.mark.parametrize("shuffle,input_shape,expected_shape", [
    ((1, 'x', 0), (2, 3), (3, 1, 2)),           # Transpose + add dim
    ((2, 1, 0), (2, 3, 4), (4, 3, 2)),          # Full transpose
    (('x', 2, 1, 0, 'x'), (2, 3, 4), (1, 4, 3, 2, 1)),  # Complex
])
def test_dimshuffle_complex(shuffle, input_shape, expected_shape):
    """Test complex DimShuffle patterns."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.elemwise import DimShuffle

    x = pt.tensor('x', shape=input_shape, dtype='float32')
    y = DimShuffle(len(input_shape), shuffle)(x)

    x_val = np.random.randn(*input_shape).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"
```

**Expected Failure Mode**: May fail with complex pattern handling

---

### Test Category 3: Join/Split Operations

**Test File**: `tests/link/onnx/test_shape.py` (continued)
**Purpose**: Test Join, Stack, and Split operations

#### Test: `test_join_vectors`
**Purpose**: Test joining vectors

```python
def test_join_vectors():
    """Test joining vectors along axis 0."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    a = pt.vector('a', dtype='float32')
    b = pt.vector('b', dtype='float32')
    c = pt.concatenate([a, b], axis=0)

    a_val = np.array([1, 2, 3], dtype='float32')
    b_val = np.array([4, 5, 6], dtype='float32')

    fn, result = compare_onnx_and_py([a, b], c, [a_val, b_val])

    expected = np.array([1, 2, 3, 4, 5, 6], dtype='float32')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Concat
    node_types = get_onnx_node_types(fn)
    assert 'Concat' in node_types, \
        f"Expected 'Concat' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Join

#### Test: `test_join_matrices`
**Purpose**: Test joining matrices along different axes

```python
@pytest.mark.parametrize("axis", [0, 1])
def test_join_matrices(axis):
    """Test joining matrices along different axes."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    a = pt.matrix('a', dtype='float32')
    b = pt.matrix('b', dtype='float32')
    c = pt.concatenate([a, b], axis=axis)

    if axis == 0:
        # Join vertically
        a_val = np.array([[1, 2], [3, 4]], dtype='float32')
        b_val = np.array([[5, 6]], dtype='float32')
        expected_shape = (3, 2)
    else:
        # Join horizontally
        a_val = np.array([[1, 2], [3, 4]], dtype='float32')
        b_val = np.array([[5], [6]], dtype='float32')
        expected_shape = (2, 3)

    fn, result = compare_onnx_and_py([a, b], c, [a_val, b_val])

    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"
```

**Expected Failure Mode**: May fail with axis handling

#### Test: `test_stack`
**Purpose**: Test stacking tensors (adds new dimension)

```python
def test_stack():
    """Test stacking tensors to create new dimension."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    a = pt.vector('a', dtype='float32')
    b = pt.vector('b', dtype='float32')
    c = pt.stack([a, b], axis=0)

    a_val = np.array([1, 2, 3], dtype='float32')
    b_val = np.array([4, 5, 6], dtype='float32')

    fn, result = compare_onnx_and_py([a, b], c, [a_val, b_val])

    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3), \
        f"Expected shape (2, 3), got {result.shape}"
```

**Expected Failure Mode**: May fail - Stack may use Join + Reshape

#### Test: `test_split`
**Purpose**: Test splitting tensor

```python
def test_split():
    """Test splitting tensor into parts."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    # Split into 3 parts
    splits = pt.split(x, [2, 4], 3, axis=0)

    x_val = np.array([1, 2, 3, 4, 5, 6], dtype='float32')

    fn, results = compare_onnx_and_py([x], splits, [x_val])

    # Should have 3 outputs
    assert len(results) == 3, \
        f"Expected 3 outputs, got {len(results)}"

    expected_0 = np.array([1, 2], dtype='float32')
    expected_1 = np.array([3, 4], dtype='float32')
    expected_2 = np.array([5, 6], dtype='float32')

    np.testing.assert_array_equal(results[0], expected_0)
    np.testing.assert_array_equal(results[1], expected_1)
    np.testing.assert_array_equal(results[2], expected_2)

    # Verify ONNX uses Split
    node_types = get_onnx_node_types(fn)
    assert 'Split' in node_types, \
        f"Expected 'Split' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Split

---

### Test Category 4: Subtensor (Indexing) Operations

**Test File**: `tests/link/onnx/test_subtensor.py`
**Purpose**: Test basic and advanced indexing operations

#### Test: `test_subtensor_simple_slice`
**Purpose**: Test basic slicing

```python
def test_subtensor_simple_slice():
    """Test basic slicing operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = x[2:5]  # Simple slice

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([2, 3, 4], dtype='float32')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Slice
    node_types = get_onnx_node_types(fn)
    assert 'Slice' in node_types, \
        f"Expected 'Slice' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Subtensor

#### Test: `test_subtensor_multi_dim_slice`
**Purpose**: Test multi-dimensional slicing

```python
def test_subtensor_multi_dim_slice():
    """Test multi-dimensional slicing."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = x[1:3, 2:4]  # Slice both dimensions

    x_val = np.arange(20).reshape(4, 5).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val[1:3, 2:4]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 2), \
        f"Expected shape (2, 2), got {result.shape}"
```

**Expected Failure Mode**: May fail with multi-dim slicing

#### Test: `test_subtensor_with_step`
**Purpose**: Test slicing with step

```python
def test_subtensor_with_step():
    """Test slicing with step parameter."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = x[::2]  # Every other element

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([0, 2, 4, 6, 8], dtype='float32')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail with step handling

#### Test: `test_subtensor_negative_indices`
**Purpose**: Test negative indexing

```python
def test_subtensor_negative_indices():
    """Test negative indexing (from end)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = x[-3:]  # Last 3 elements

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([7, 8, 9], dtype='float32')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail with negative index handling

#### Test: `test_advanced_subtensor_list`
**Purpose**: Test advanced indexing with list

```python
def test_advanced_subtensor_list():
    """Test advanced indexing with integer list."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.subtensor import advanced_subtensor1

    x = pt.vector('x', dtype='float32')
    indices = [1, 3, 5]
    y = advanced_subtensor1(x, indices)

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.array([1, 3, 5], dtype='float32')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Gather
    node_types = get_onnx_node_types(fn)
    assert 'Gather' in node_types, \
        f"Expected 'Gather' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for AdvancedSubtensor1

---

### Test Category 5: IncSubtensor Operations

**Test File**: `tests/link/onnx/test_subtensor.py` (continued)
**Purpose**: Test set/increment subtensor operations

#### Test: `test_set_subtensor_slice`
**Purpose**: Test set_subtensor with slice

```python
def test_set_subtensor_slice():
    """Test set_subtensor operation with slice."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.subtensor import set_subtensor

    x = pt.vector('x', dtype='float32')
    y = set_subtensor(x[2:5], np.array([10, 20, 30], dtype='float32'))

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val.copy()
    expected[2:5] = [10, 20, 30]
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses ScatterND or ScatterElements
    node_types = get_onnx_node_types(fn)
    assert any(op in node_types for op in ['ScatterND', 'ScatterElements']), \
        f"Expected 'ScatterND' or 'ScatterElements' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for IncSubtensor

#### Test: `test_inc_subtensor_slice`
**Purpose**: Test inc_subtensor (increment values)

```python
def test_inc_subtensor_slice():
    """Test inc_subtensor operation (increment)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.subtensor import inc_subtensor

    x = pt.vector('x', dtype='float32')
    y = inc_subtensor(x[2:5], np.array([10, 20, 30], dtype='float32'))

    x_val = np.arange(10, dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val.copy()
    expected[2:5] += [10, 20, 30]
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail with increment handling

---

### Test Category 6: Reduction Operations

**Test File**: `tests/link/onnx/test_math.py`
**Purpose**: Test reduction operations (Sum, Prod, Max, Min, etc.)

#### Test: `test_sum_basic`
**Purpose**: Test sum reduction

```python
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum_basic(axis):
    """Test sum reduction with different axes."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=axis)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.sum(x_val, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify ONNX uses ReduceSum
    node_types = get_onnx_node_types(fn)
    assert 'ReduceSum' in node_types, \
        f"Expected 'ReduceSum' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Sum/CAReduce

#### Test: `test_prod`
**Purpose**: Test product reduction

```python
def test_prod():
    """Test product reduction."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = pt.prod(x, axis=1)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.prod(x_val, axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    node_types = get_onnx_node_types(fn)
    assert 'ReduceProd' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Prod

#### Test: `test_max_min_reductions`
**Purpose**: Test max/min reductions

```python
@pytest.mark.parametrize("op,onnx_op", [
    (pt.max, 'ReduceMax'),
    (pt.min, 'ReduceMin'),
])
def test_max_min_reductions(op, onnx_op):
    """Test max and min reductions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = op(x, axis=0)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    if op == pt.max:
        expected = np.max(x_val, axis=0)
    else:
        expected = np.min(x_val, axis=0)

    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Max/Min

#### Test: `test_argmax_argmin`
**Purpose**: Test argmax/argmin operations

```python
@pytest.mark.parametrize("op,onnx_op", [
    (pt.argmax, 'ArgMax'),
    (pt.argmin, 'ArgMin'),
])
def test_argmax_argmin(op, onnx_op):
    """Test argmax and argmin operations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = op(x, axis=1)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    if op == pt.argmax:
        expected = np.argmax(x_val, axis=1)
    else:
        expected = np.argmin(x_val, axis=1)

    np.testing.assert_array_equal(result, expected)

    # Verify output dtype is int64
    assert result.dtype == np.int64, \
        f"Expected dtype int64, got {result.dtype}"

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Argmax/Argmin

#### Test: `test_logical_reductions`
**Purpose**: Test All/Any reductions

```python
@pytest.mark.parametrize("op,np_op,onnx_op", [
    (pt.all, np.all, 'ReduceMin'),
    (pt.any, np.any, 'ReduceMax'),
])
def test_logical_reductions(op, np_op, onnx_op):
    """Test All and Any logical reductions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='bool')
    y = op(x, axis=1)

    x_val = np.random.rand(3, 4) > 0.5  # Random boolean array

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val, axis=1)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    # All/Any map to ReduceMin/ReduceMax for boolean types
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` for All/Any

#### Test: `test_multiple_axes_reduction`
**Purpose**: Test reduction over multiple axes

```python
def test_multiple_axes_reduction():
    """Test reduction over multiple axes."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor3('x', dtype='float32')
    y = pt.sum(x, axis=[0, 2])  # Sum over first and last axes

    x_val = np.random.randn(2, 3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.sum(x_val, axis=(0, 2))
    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

**Expected Failure Mode**: May fail with multi-axis handling

---

### Test Category 7: Allocation Operations

**Test File**: `tests/link/onnx/test_tensor_basic.py`
**Purpose**: Test tensor allocation operations

#### Test: `test_alloc_scalar`
**Purpose**: Test Alloc broadcasting scalar to shape

```python
def test_alloc_scalar():
    """Test Alloc broadcasting scalar to shape."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Broadcast scalar 5.0 to shape (3, 4)
    x = pt.alloc(5.0, 3, 4)

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.full((3, 4), 5.0, dtype='float64')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Expand or ConstantOfShape
    node_types = get_onnx_node_types(fn)
    assert any(op in node_types for op in ['Expand', 'ConstantOfShape']), \
        f"Expected 'Expand' or 'ConstantOfShape' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Alloc

#### Test: `test_alloc_with_scalar_input`
**Purpose**: Test Alloc with scalar input variable

```python
def test_alloc_with_scalar_input():
    """Test Alloc with scalar input variable."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    a = pt.scalar('a', dtype='float32')
    x = pt.alloc(a, 2, 3)

    a_val = np.array(7.0, dtype='float32')

    fn, result = compare_onnx_and_py([a], x, [a_val])

    expected = np.full((2, 3), 7.0, dtype='float32')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail with dynamic value allocation

#### Test: `test_alloc_empty`
**Purpose**: Test AllocEmpty (uninitialized allocation)

```python
def test_alloc_empty():
    """Test AllocEmpty creates array with correct shape and dtype."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.AllocEmpty('float32')(3, 4)

    # Custom assertion: only check shape and dtype, not values
    def assert_shape_dtype(a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"

    fn, result = compare_onnx_and_py([], x, [], assert_fn=assert_shape_dtype)

    assert result.shape == (3, 4)
    assert result.dtype == np.float32
```

**Expected Failure Mode**: `NotImplementedError` for AllocEmpty

#### Test: `test_make_vector`
**Purpose**: Test MakeVector creating vector from scalars

```python
def test_make_vector():
    """Test MakeVector creates vector from scalars."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    a = pt.scalar('a', dtype='float32')
    b = pt.scalar('b', dtype='float32')
    c = pt.scalar('c', dtype='float32')

    x = pt.make_vector(a, b, c)

    a_val = np.array(1.0, dtype='float32')
    b_val = np.array(2.0, dtype='float32')
    c_val = np.array(3.0, dtype='float32')

    fn, result = compare_onnx_and_py([a, b, c], x, [a_val, b_val, c_val])

    expected = np.array([1.0, 2.0, 3.0], dtype='float32')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Concat or similar
    node_types = get_onnx_node_types(fn)
    # May use Concat, Reshape, or custom pattern
```

**Expected Failure Mode**: `NotImplementedError` for MakeVector

#### Test: `test_arange_basic`
**Purpose**: Test ARange with constant parameters

```python
def test_arange_basic():
    """Test ARange with constant parameters."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # ARange requires constant inputs in ONNX
    x = pt.arange(0, 10, 2, dtype='int64')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.arange(0, 10, 2, dtype='int64')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses Range
    node_types = get_onnx_node_types(fn)
    assert 'Range' in node_types, \
        f"Expected 'Range' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for ARange

#### Test: `test_arange_negative_step`
**Purpose**: Test ARange with negative step (descending)

```python
def test_arange_negative_step():
    """Test ARange with negative step (descending)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.arange(10, 0, -2, dtype='int64')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.arange(10, 0, -2, dtype='int64')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail with negative step

#### Test: `test_arange_empty`
**Purpose**: Test ARange with empty range

```python
def test_arange_empty():
    """Test ARange with empty range."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Empty range: stop < start with positive step
    x = pt.arange(10, 5, 1, dtype='int64')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.arange(10, 5, 1, dtype='int64')
    assert result.shape == (0,), f"Expected empty array, got shape {result.shape}"
```

**Expected Failure Mode**: May fail with empty range handling

#### Test: `test_eye_basic`
**Purpose**: Test Eye creating identity matrix

```python
def test_eye_basic():
    """Test Eye creates identity matrix."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.eye(4, dtype='float32')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.eye(4, dtype='float32')
    np.testing.assert_array_equal(result, expected)

    # Verify ONNX uses EyeLike or custom pattern
    node_types = get_onnx_node_types(fn)
    # May use various patterns depending on implementation
```

**Expected Failure Mode**: `NotImplementedError` for Eye

#### Test: `test_eye_non_square`
**Purpose**: Test Eye with non-square matrix

```python
def test_eye_non_square():
    """Test Eye with non-square matrix."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # 3 rows, 5 columns
    x = pt.eye(3, 5, dtype='float32')

    fn, result = compare_onnx_and_py([], x, [])

    expected = np.eye(3, 5, dtype='float32')
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 5)
```

**Expected Failure Mode**: May fail with non-square handling

---

### Test Category 8: Integration Tests

**Test File**: `tests/link/onnx/test_integration.py`
**Purpose**: Test combined operations in realistic scenarios

#### Test: `test_mean_variance`
**Purpose**: Test computing mean and variance (uses multiple ops)

```python
def test_mean_variance():
    """Test computing mean and variance using reductions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

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
```

**Expected Failure Mode**: May fail if reductions not implemented

#### Test: `test_normalize_rows`
**Purpose**: Test normalizing matrix rows (reshape + reductions)

```python
def test_normalize_rows():
    """Test normalizing matrix rows using reshape and reductions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    # Normalize each row: x / sum(x, axis=1, keepdims=True)
    row_sums = pt.sum(x, axis=1, keepdims=True)
    normalized = x / row_sums

    x_val = np.random.rand(5, 10).astype('float32') + 0.1  # Avoid zeros

    fn, result = compare_onnx_and_py([x], normalized, [x_val])

    # Verify each row sums to 1
    row_sums_result = np.sum(result, axis=1)
    np.testing.assert_allclose(row_sums_result, np.ones(5), rtol=1e-5)
```

**Expected Failure Mode**: May fail with keepdims handling

#### Test: `test_reshape_and_slice`
**Purpose**: Test combined reshape and slicing

```python
def test_reshape_and_slice():
    """Test combined reshape and slicing operations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    # Reshape to 3x4, then take middle 2 rows
    reshaped = x.reshape((3, 4))
    sliced = reshaped[1:3, :]

    x_val = np.arange(12, dtype='float32')

    fn, result = compare_onnx_and_py([x], sliced, [x_val])

    expected = np.arange(12).reshape(3, 4)[1:3, :].astype('float32')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: May fail if either reshape or slicing fails

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
