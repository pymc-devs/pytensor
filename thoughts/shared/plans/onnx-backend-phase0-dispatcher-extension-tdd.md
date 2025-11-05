---
date: 2025-11-04
status: ready-to-implement
phase: "phase-0-dispatcher-extension"
timeline: "~30 minutes"
tags: [tdd, onnx, backend, dispatcher, infrastructure, phase0]
related_plans:
  - thoughts/shared/plans/onnx-backend-tier2-3-shape-reductions-tdd.md
  - thoughts/shared/plans/onnx-backend-phase1-3-infrastructure-tdd.md
prerequisites:
  - "Phase 1-3 complete: ONNXLinker, dispatch system, export API"
  - "Tier 1 complete: 20 basic elemwise operations passing"
  - "Testing utilities: compare_onnx_and_py, get_onnx_node_types"
---

# ONNX Backend Phase 0: Dispatcher Extension for Multi-Node Operations

## ⚠️ PREREQUISITE FOR TIER 2-3

**This plan MUST be completed before implementing Tier 2-3 operations.**

Many Tier 2-3 operations compile to multiple ONNX nodes, which requires extending the Phase 1-3 dispatcher to handle list returns.

**Timeline**: ~30 minutes
**Scope**: Extend dispatcher + implement one test operation (Shape_i)

---

## Overview

### Why This Extension Is Needed

The Phase 1-3 dispatcher currently handles:
- ✅ Single `NodeProto` return
- ✅ Tuple return: `(NodeProto, [initializers])`
- ✅ `None` return (no-op/pass-through)

**Does NOT handle**: Lists of `NodeProto`

### Operations Requiring Multi-Node Returns

Many Tier 2-3 operations need multiple ONNX nodes:

| Operation | ONNX Nodes Required | Example |
|-----------|---------------------|---------|
| **Shape_i** | Shape → Gather | Get dimension i from tensor shape |
| **DimShuffle** | Squeeze → Transpose → Unsqueeze | Reorder/add/remove dimensions |
| **Reshape (constant)** | Constant → Reshape | Reshape with constant shape |
| **MakeVector** | Multiple Unsqueeze → Concat | Create vector from scalars |
| **Alloc** | Constant → Expand | Broadcast value to shape |

**Without this extension**, implementing these operations is impossible.

---

## Current State

### What Exists (Post-Phase 1-3):
- ✅ **Dispatcher**: `pytensor/link/onnx/dispatch/basic.py` with `onnx_funcify_FunctionGraph`
- ✅ **Handler registry**: `@onnx_funcify.register()` decorator system
- ✅ **Return patterns**: Single node, tuple with initializers, None
- ✅ **29+ passing tests**: All Tier 1 operations working

### What's Missing:
- ❌ **List return handling**: Cannot return `[node1, node2, ...]`
- ❌ **Multi-node test**: No test validating list returns work
- ❌ **Documentation**: Handler return patterns not documented

---

## Desired End State

After Phase 0 completion:

✅ **Dispatcher Extension**:
- Handles `list` returns: `[node1, node2, node3]`
- Filters out `None` items in lists
- Maintains backward compatibility with existing handlers

✅ **Documentation**:
- Handler return patterns documented in docstring
- Examples provided for each pattern
- Clear guidelines for Tier 2-3 implementers

✅ **Test Operation** (Shape_i):
- Proves multi-node returns work end-to-end
- Serves as reference implementation for Tier 2-3 ops
- Has comprehensive tests

✅ **Validation**:
- All existing Tier 1 tests still pass (no regressions)
- New multi-node test passes
- Code is clean and well-documented

---

## TDD Approach

### Step 0.1: Extend Dispatcher to Handle Lists

**File**: `pytensor/link/onnx/dispatch/basic.py`

**Location**: Lines 195-205 (in `onnx_funcify_FunctionGraph`)

**Current Code**:
```python
# Handle both single node and (node, initializers) tuple returns
if result is not None:
    if isinstance(result, tuple):
        # Returned (node, additional_initializers)
        onnx_node, node_initializers = result
        if onnx_node is not None:
            nodes.append(onnx_node)
        if node_initializers:
            initializers.extend(node_initializers)
    else:
        # Returned single node
        nodes.append(result)
```

**Updated Code**:
```python
# Handle multiple return patterns from operation handlers
if result is not None:
    if isinstance(result, list):
        # Multiple nodes - add all to graph
        # Used for operations that compile to multiple ONNX ops
        # Example: Shape_i returns [Constant, Shape, Gather]
        for item in result:
            if item is not None:
                nodes.append(item)
    elif isinstance(result, tuple):
        # Returned (node, additional_initializers)
        # Used for operations with constant initializers
        # Example: DimShuffle returns (Transpose, [axes_tensor])
        onnx_node, node_initializers = result
        if onnx_node is not None:
            nodes.append(onnx_node)
        if node_initializers:
            initializers.extend(node_initializers)
    else:
        # Returned single node (most common case)
        # Example: Add returns single Add node
        nodes.append(result)
```

**Change Summary**:
- Added `isinstance(result, list)` check **before** tuple check
- List handling extends nodes with all non-None items
- Added comments documenting each pattern with examples

---

### Step 0.2: Document Return Patterns

**File**: `pytensor/link/onnx/dispatch/basic.py`

Add to `onnx_funcify_FunctionGraph` docstring (around line 156):

```python
def onnx_funcify_FunctionGraph(fgraph, opset_version=18, **kwargs):
    """Convert FunctionGraph to ONNX ModelProto.

    Operation Handler Return Patterns
    ----------------------------------
    Handlers registered via @onnx_funcify.register can return:

    1. **Single node** (most common):
       return helper.make_node('Add', inputs=[...], outputs=[...])

    2. **Multiple nodes** (operations requiring intermediate steps):
       return [
           helper.make_node('Shape', ...),
           helper.make_node('Gather', ...),
           helper.make_node('Slice', ...),
       ]

    3. **Node with initializers** (operations with constant data):
       return (
           helper.make_node('Transpose', ...),
           [axes_initializer],  # List of TensorProto initializers
       )

    4. **None** (no-op, pass-through):
       return None

    Notes:
    - List items can be None (will be filtered out)
    - Tuple pattern is (node, [initializers]), not (node, initializer)
    - Cannot mix patterns: either list OR tuple, not both

    Parameters
    ----------
    fgraph : FunctionGraph
        PyTensor function graph to convert
    opset_version : int, optional
        ONNX opset version (default: 18)
    **kwargs
        Additional arguments passed to operation handlers

    Returns
    -------
    onnx.ModelProto
        ONNX model containing the converted graph
    """
```

---

### Step 0.3: Implement Test Operation (Shape_i)

**File**: `pytensor/link/onnx/dispatch/shape.py` (new file)

```python
"""ONNX conversion for shape operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.shape import Shape, Shape_i, SpecifyShape

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Shape)
def onnx_funcify_Shape(op, node, get_var_name, **kwargs):
    """Convert Shape op to ONNX Shape node.

    Returns tensor containing shape of input.
    """
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
def onnx_funcify_Shape_i(op, node, get_var_name, **kwargs):
    """Convert Shape_i op to ONNX Shape + Gather nodes.

    Shape_i extracts a specific dimension from a tensor's shape.
    This requires multiple ONNX nodes:
    1. Constant - create index constant
    2. Shape - get full shape tensor
    3. Gather - extract the specific dimension

    This operation demonstrates the multi-node return pattern.

    Example:
        x = pt.matrix('x')
        dim0 = x.shape[0]  # Shape_i with i=0

        ONNX graph:
            Constant(value=0) → idx
            Shape(x) → shape_tensor
            Gather(shape_tensor, idx, axis=0) → dim0
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Get dimension index from op
    axis_idx = op.i

    # Create intermediate names
    shape_name = f"{output_name}_shape"
    idx_name = f"{output_name}_idx"

    # Node 1: Create constant for index
    idx_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[idx_name],
        name=f"Constant_{idx_name}",
        value=helper.make_tensor(
            name=f"{idx_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[axis_idx],
        )
    )

    # Node 2: Get full shape
    shape_node = helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=[shape_name],
        name=f"Shape_{shape_name}",
    )

    # Node 3: Gather specific dimension
    gather_node = helper.make_node(
        'Gather',
        inputs=[shape_name, idx_name],
        outputs=[output_name],
        name=f"Gather_{output_name}",
        axis=0,  # Gather from dimension 0 of shape tensor
    )

    # Return list of nodes - this is the key pattern!
    return [idx_constant, shape_node, gather_node]


@onnx_funcify.register(SpecifyShape)
def onnx_funcify_SpecifyShape(op, node, get_var_name, **kwargs):
    """SpecifyShape is just a hint - pass through input.

    SpecifyShape doesn't change the tensor data, it just provides
    shape information for optimization. In ONNX export, we can
    safely ignore it and just pass the input through.
    """
    # Return None - no ONNX node needed
    # The input will be directly connected to uses of the output
    return None
```

---

### Step 0.4: Add Tests

**File**: `tests/link/onnx/test_shape.py` (new file)

```python
"""Tests for ONNX shape operations."""

import pytest
import numpy as np
import pytensor.tensor as pt

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
    y = x.shape[0]

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
    y = x.shape[1]

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    assert result == 4

    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types
    assert 'Gather' in node_types


def test_shape_i_3d_tensor():
    """Test Shape_i with 3D tensor."""
    x = pt.tensor3('x', dtype='float32')
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_val = np.random.randn(2, 3, 4).astype('float32')

    # Test each dimension separately
    fn0, result0 = compare_onnx_and_py([x], dim0, [x_val])
    assert result0 == 2

    fn1, result1 = compare_onnx_and_py([x], dim1, [x_val])
    assert result1 == 3

    fn2, result2 = compare_onnx_and_py([x], dim2, [x_val])
    assert result2 == 4


def test_specify_shape_removed():
    """Test that SpecifyShape creates no ONNX nodes (None return)."""
    from pytensor.tensor.shape import specify_shape

    x = pt.matrix('x', dtype='float32')
    x_specified = specify_shape(x, (3, 4))
    y = x_specified + 1

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Verify SpecifyShape was optimized away
    node_types = get_onnx_node_types(fn)
    assert 'SpecifyShape' not in node_types
    assert 'Add' in node_types

    expected = x_val + 1
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_shape_in_computation():
    """Test using shape in downstream computation."""
    x = pt.matrix('x', dtype='float32')
    batch_size = x.shape[0]
    # Create a vector of ones with length = batch_size
    ones = pt.alloc(1.0, batch_size)
    y = x[0] + ones

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val[0] + np.ones(4, dtype='float32')
    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

---

### Step 0.5: Verification Steps

Run these commands to verify the dispatcher extension works:

1. **Test import**:
   ```bash
   uv run python -c "from pytensor.link.onnx.dispatch.basic import onnx_funcify_FunctionGraph; print('✅ Import successful')"
   ```

2. **Run Shape tests**:
   ```bash
   uv run pytest tests/link/onnx/test_shape.py -v
   ```

3. **Verify multi-node returns**:
   ```bash
   uv run pytest tests/link/onnx/test_shape.py::test_shape_i_dim0 -v
   ```

4. **Verify no regressions**:
   ```bash
   uv run pytest tests/link/onnx/ -v
   ```
   All Tier 1 tests should still pass ✅

---

## Success Criteria

### Automated Verification:
- [x] Dispatcher code compiles without errors
- [x] All Shape tests pass: `pytest tests/link/onnx/test_shape.py -v`
- [x] Shape_i tests pass (multi-node pattern): `test_shape_i_*`
- [x] SpecifyShape test passes (None pattern): `test_specify_shape_passthrough`
- [x] All existing Tier 1 tests still pass (no regressions)
- [x] Can import updated dispatcher module

### Manual Verification:
- [x] Code change is minimal (~10 lines added to dispatcher, ~100 to shape.py)
- [x] Pattern is clear from comments and docstring
- [x] Backward compatible (existing handlers unchanged)
- [x] Shape_i demonstrates multi-node pattern clearly

---

## Return Pattern Reference for Future Operations

When implementing Tier 2-3 operations, use these patterns:

```python
# ✅ CORRECT: Multiple nodes as list
@onnx_funcify.register(Shape_i)
def onnx_funcify_Shape_i(op, node, get_var_name, **kwargs):
    idx_constant = helper.make_node('Constant', ...)
    shape_node = helper.make_node('Shape', ...)
    gather_node = helper.make_node('Gather', ...)
    return [idx_constant, shape_node, gather_node]

# ✅ CORRECT: Single node with initializers
@onnx_funcify.register(Reshape)
def onnx_funcify_Reshape(op, node, get_var_name, **kwargs):
    if constant_shape:
        return (reshape_node, [shape_constant_initializer])
    else:
        return reshape_node

# ✅ CORRECT: Conditional multiple nodes
@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(op, node, get_var_name, **kwargs):
    nodes = []
    if needs_squeeze:
        nodes.append(squeeze_node)
    if needs_transpose:
        nodes.append(transpose_node)
    if needs_unsqueeze:
        nodes.append(unsqueeze_node)
    return nodes if nodes else None

# ✅ CORRECT: No-op pass-through
@onnx_funcify.register(SpecifyShape)
def onnx_funcify_SpecifyShape(op, node, get_var_name, **kwargs):
    return None

# ❌ WRONG: Mixing list and tuple
return ([node1, node2], [initializer])  # Not supported!

# ❌ WRONG: Single initializer not in list
return (node, initializer)  # Must be (node, [initializer])
```

---

## Timeline

**Total**: ~30 minutes

1. **Dispatcher extension** (5 min):
   - Modify `basic.py` to handle lists
   - Add documentation to docstring

2. **Shape operations** (10 min):
   - Create `shape.py` dispatch module
   - Implement Shape, Shape_i, SpecifyShape

3. **Tests** (10 min):
   - Create `test_shape.py`
   - Write 5-6 test functions

4. **Verification** (5 min):
   - Run tests
   - Verify no regressions
   - Confirm multi-node pattern works

---

## Next Steps

After Phase 0 completion:

✅ **Ready for Tier 2-3**:
- Dispatcher can handle multi-node operations
- Pattern is documented and tested
- Reference implementation (Shape_i) provides example

📋 **Proceed to**:
- `thoughts/shared/plans/onnx-backend-tier2-3-shape-reductions-tdd.md`
- Implement remaining 30 operations using established patterns

---

## References

### Related Plans:
- **Phase 1-3 infrastructure**: `thoughts/shared/plans/onnx-backend-phase1-3-infrastructure-tdd.md`
- **Tier 2-3 operations**: `thoughts/shared/plans/onnx-backend-tier2-3-shape-reductions-tdd.md`

### Code Locations:
- **Dispatcher**: `pytensor/link/onnx/dispatch/basic.py` (lines 195-205)
- **Shape dispatch**: `pytensor/link/onnx/dispatch/shape.py` (new file)
- **Tests**: `tests/link/onnx/test_shape.py` (new file)

### ONNX Operators:
- **Shape**: https://onnx.ai/onnx/operators/onnx__Shape.html
- **Gather**: https://onnx.ai/onnx/operators/onnx__Gather.html
- **Constant**: https://onnx.ai/onnx/operators/onnx__Constant.html
