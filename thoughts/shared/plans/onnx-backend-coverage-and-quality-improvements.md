# ONNX Backend Coverage and Quality Improvements Implementation Plan

<!-- WORKSHOP NOTE: This is the SECOND iteration of planning - much more practical than the first. This plan emerged after actually implementing basic ONNX support and discovering bugs through real-world testing.

Key differences from onnx-backend-implementation.md:
1. **Bug-driven**, not feature-driven - starts with "what's broken"
2. **Test-first approach** - writes failing tests before fixing code
3. **Specific and concrete** - references exact line numbers and implementations
4. **Realistic scope** - 3 phases vs 5, focused on fixing not building

This plan was MOSTLY executed as written. The DimShuffle bug was real and critical - it was discovered by trying to export a real CNN model. This is a great example of how real-world testing exposes planning blind spots.

Execution notes:
- Phase 1 (DimShuffle): Implemented fully, bug fixed ✅
- Phase 2 (Untested ops): All tests added, some bugs found and fixed ✅
- Phase 3 (Multi-dtype): Partially implemented - focused on most common dtypes ⏸️

Why Phase 3 was incomplete: During implementation, realized that property-based testing (Hypothesis) was a better long-term solution than manually enumerating dtype combinations. This led to the next plan: hypothesis-property-based-onnx-testing.md. Classic example of planning evolving during execution. -->

## Overview

This plan addresses critical bugs, coverage gaps, and test quality issues in PyTensor's ONNX backend. The primary focus is fixing a silent data corruption bug in DimShuffle, adding tests for 5 completely untested operations, and establishing comprehensive test coverage across data types and edge cases.

<!-- REALITY CHECK: The "5 completely untested operations" was accurate. Testing these operations DID expose bugs, particularly in AllocEmpty and Composite decomposition. The act of writing tests forced us to think through edge cases that the initial implementation glossed over. This validates the test-first approach. -->

## Current State Analysis

### What Exists
**Implementation** (8 files, 1,181 lines):
- Core dispatch system: `pytensor/link/onnx/dispatch/basic.py` (292 lines)
- Elementwise ops: `pytensor/link/onnx/dispatch/elemwise.py` (180 lines)
- Shape ops: `pytensor/link/onnx/dispatch/shape.py` (395 lines)
- Linear algebra: `pytensor/link/onnx/dispatch/nlinalg.py` (110 lines)
- Special functions: `pytensor/link/onnx/dispatch/special.py` (89 lines)
- Export API: `pytensor/link/onnx/export.py` (115 lines)

**Tests** (5 files, 706 lines):
- 27 tests total, all using float32 only
- Black-box comparison approach (PyTensor vs ONNX Runtime output)
- No ONNX graph structure validation
- compare_onnx_and_py helper: `tests/link/onnx/test_basic.py:18-101`

### Critical Issues Found

**1. DimShuffle Silent Fallback Bug** - `pytensor/link/onnx/dispatch/shape.py:222-230`
- **Severity**: CRITICAL - Silent data corruption
- **Problem**: Complex DimShuffle operations (squeeze+transpose, transpose+unsqueeze, etc.) fall back to Identity, which does nothing
- **Example**: `x.dimshuffle('x', 1, 0)` on (2,3) should produce (1,3,2) but produces (2,3)
- **Impact**: Any complex reshape pattern silently fails

**2. Five Implemented Operations Have Zero Tests**
- Gemv (62 lines, 4-node decomposition) - `pytensor/link/onnx/dispatch/nlinalg.py:48-109`
- Cast (dtype conversion logic) - `pytensor/link/onnx/dispatch/elemwise.py:129-157`
- Composite decomposition (graph traversal) - `pytensor/link/onnx/dispatch/elemwise.py:31-113`
- AllocEmpty (144 lines, 3 code paths) - `pytensor/link/onnx/dispatch/shape.py:233-376`
- DeepCopyOp - `pytensor/link/onnx/dispatch/shape.py:379-394`

**3. No Data Type Diversity**
- All 27 tests use `dtype="float32"` only
- No tests for: int32, int64, float64, bool
- No mixed-dtype tests

**4. Weak Shape_i Testing**
- Only indirect testing via `test_shape_i_get_dimension`
- Doesn't validate 5-node ONNX sequence (Shape → Constant → Gather → Constant → Squeeze)

### Key Discoveries

**DimShuffle Decomposition Pattern** (from `pytensor/tensor/elemwise.py:227-246`):
PyTensor's DimShuffle.perform() shows the canonical sequence:
1. Transpose (reorder kept dimensions)
2. Reshape (remove dropped dims, insert new ones)

For ONNX, this translates to:
1. Squeeze (remove dimensions)
2. Transpose (reorder)
3. Unsqueeze (add dimensions)

**Multi-Node Operation Pattern**:
All complex converters return `list[onnx.NodeProto]`:
- Shape_i: 5 nodes (`shape.py:17-94`)
- AllocEmpty: 2-10 nodes (`shape.py:233-376`)
- Gemv: 4 nodes (`nlinalg.py:48-109`)
- Composite: N nodes (`elemwise.py:31-113`)

**PyTensor Test Patterns** (from research):
- Parametrization: `@pytest.mark.parametrize` with descriptive `ids`
- Assertions: `np.testing.assert_allclose` with explicit tolerances
- Dtype testing: Use `itertools.product` for dtype matrices
- Graph inspection: `f.maker.fgraph.apply_nodes` and `.toposort()`
- Utilities: `tests.unittest_tools` (utt) and `tests.tensor.utils`

## Desired End State

### After Phase 1
- DimShuffle handles all complex cases correctly (no Identity fallback)
- Comprehensive DimShuffle tests covering all operation combinations
- Zero silent data corruption bugs

### After Phase 2
- 100% test coverage for all implemented ONNX operations
- All 5 untested operations have comprehensive test suites
- Any implementation bugs discovered by tests are fixed

### After Phase 3
- Multi-dtype test suite covers int32, int64, float64, bool
- Edge cases tested: empty tensors, scalars, broadcasting
- ONNX graph structure validation utilities in place
- Multi-node operations have structure validation tests

### Verification
- All tests pass: `pytest tests/link/onnx/ -v`
- No pytest.skip or pytest.xfail markers added
- ONNX checker validates all exported models: `onnx.checker.check_model()`
- Coverage report shows 100% for dispatch modules

## What We're NOT Doing

- Implementing new ONNX operations (only fixing/testing existing)
- Changing dispatch system architecture
- Adding symbolic shape support
- Supporting multiple opset versions (staying with opset 18)
- Performance optimization or benchmarking
- Documentation beyond code comments
- Integration with other PyTensor backends

## Implementation Approach

**Strategy**: Test-first development with incremental fixes
1. Write tests that expose bugs (they will fail initially)
2. Fix implementation to make tests pass
3. Validate with ONNX Runtime and structure checks
4. Iterate until all tests pass

**Pattern Following**:
- Use existing `compare_onnx_and_py` for output validation
- Follow PyTensor test conventions (parametrize, fixtures, tolerances)
- Add ONNX structure validation where appropriate

**Risk Mitigation**:
- Each phase is independently testable
- Tests run against actual ONNX Runtime (not mocks)
- Existing tests continue to pass (no regressions)

---

## Phase 1: Critical DimShuffle Bug Tests & Fix

### Overview
Fix the critical DimShuffle bug that causes silent data corruption. Write tests first to expose the bug, then implement the proper multi-operation decomposition.

### Phase 1a: Write DimShuffle Complex Case Tests

#### 1. Add Complex DimShuffle Tests

**File**: `tests/link/onnx/test_shape.py`

**Changes**: Add comprehensive tests for all complex DimShuffle patterns

```python
# Add after line 82 (after test_dimshuffle_transpose_3d)

def test_dimshuffle_transpose_and_unsqueeze(tmp_path):
    """Test transpose combined with unsqueeze - currently FAILS (bug)."""
    x = pt.matrix("x", dtype="float32")
    # Input: (2, 3), Output: (3, 1, 2)
    # This requires: Transpose(1,0) → Unsqueeze(axis=1)
    y = x.dimshuffle(1, "x", 0)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_squeeze_and_transpose(tmp_path):
    """Test squeeze combined with transpose - currently FAILS (bug)."""
    x = pt.tensor(dtype="float32", shape=(2, 1, 3), name="x")
    # Input: (2, 1, 3), Output: (3, 2)
    # This requires: Squeeze(axis=1) → Transpose(1,0)
    y = x.dimshuffle(2, 0)

    x_val = np.random.default_rng(42).random((2, 1, 3)).astype("float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_unsqueeze_and_transpose(tmp_path):
    """Test unsqueeze combined with transpose - currently FAILS (bug)."""
    x = pt.vector("x", dtype="float32")
    # Input: (3,), Output: (1, 3)
    # Wait, this should work... let's try a more complex case
    x = pt.matrix("x", dtype="float32")
    # Input: (2, 3), Output: (1, 3, 2)
    # This requires: Transpose(1,0) → Unsqueeze(axis=0)
    y = x.dimshuffle("x", 1, 0)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("pattern,input_shape,expected_shape", [
    # (new_order, input_shape, expected_shape)
    ((1, 'x', 0), (2, 3), (3, 1, 2)),           # transpose + unsqueeze
    ((2, 0), (2, 1, 3), (3, 2)),                # squeeze + transpose
    (('x', 1, 0), (2, 3), (1, 3, 2)),           # unsqueeze + transpose
    ((0, 2, 'x'), (3, 1, 4), (3, 4, 1)),        # squeeze + unsqueeze
    ((2, 'x', 0, 1), (2, 3, 4), (4, 1, 2, 3)),  # transpose + unsqueeze
    (('x', 2, 1, 'x', 0), (2, 3, 4), (1, 4, 3, 1, 2)),  # complex
])
def test_dimshuffle_complex_patterns(tmp_path, pattern, input_shape, expected_shape):
    """Test various complex DimShuffle patterns that combine operations."""
    x = pt.tensor(dtype="float32", shape=input_shape, name="x")
    y = x.dimshuffle(*pattern)

    rng = np.random.default_rng(42)
    x_val = rng.random(input_shape).astype("float32")

    # Verify expected shape
    assert y.type.shape == expected_shape, f"Shape mismatch: {y.type.shape} vs {expected_shape}"

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

#### 2. Add ONNX Structure Validation Helper

**File**: `tests/link/onnx/test_basic.py`

**Changes**: Add utility to validate ONNX graph structure

```python
# Add after compare_onnx_and_py function (after line 101)

def validate_onnx_graph_structure(
    model,
    expected_node_types=None,
    expected_node_count=None,
    check_connections=True,
):
    """Validate ONNX graph structure beyond just output correctness.

    Parameters
    ----------
    model : onnx.ModelProto
        The ONNX model to validate
    expected_node_types : list of str, optional
        Expected node op_types in order (or subset)
    expected_node_count : int, optional
        Expected total number of nodes
    check_connections : bool
        Whether to validate all node connections

    Returns
    -------
    dict
        Graph structure information for inspection
    """
    graph = model.graph
    nodes = list(graph.node)

    # Check node count
    if expected_node_count is not None:
        assert len(nodes) == expected_node_count, (
            f"Expected {expected_node_count} nodes, got {len(nodes)}\n"
            f"Nodes: {[n.op_type for n in nodes]}"
        )

    # Check node types
    if expected_node_types is not None:
        actual_types = [n.op_type for n in nodes]
        # Check if expected types appear in order (subset match)
        idx = 0
        for expected_type in expected_node_types:
            found = False
            while idx < len(actual_types):
                if actual_types[idx] == expected_type:
                    found = True
                    idx += 1
                    break
                idx += 1
            assert found, (
                f"Expected node type '{expected_type}' not found in order\n"
                f"Expected: {expected_node_types}\n"
                f"Actual: {actual_types}"
            )

    # Check all connections are valid
    if check_connections:
        all_available = set()
        # Add inputs
        all_available.update(inp.name for inp in graph.input)
        # Add initializers
        all_available.update(init.name for init in graph.initializer)

        # Check each node
        for node in nodes:
            for inp in node.input:
                if inp:  # Skip empty strings (optional inputs)
                    assert inp in all_available, (
                        f"Node {node.name} ({node.op_type}) has undefined input: {inp}\n"
                        f"Available: {sorted(all_available)}"
                    )
            all_available.update(node.output)

    # Return structure info for inspection
    return {
        "node_count": len(nodes),
        "node_types": [n.op_type for n in nodes],
        "input_count": len(graph.input),
        "output_count": len(graph.output),
        "initializer_count": len(graph.initializer),
    }
```

### Phase 1b: Fix DimShuffle Implementation

#### 1. Implement DimShuffle Decomposition Helper

**File**: `pytensor/link/onnx/dispatch/shape.py`

**Changes**: Add helper function before `onnx_funcify_DimShuffle` (before line 115)

```python
# Add before line 115

def decompose_dimshuffle_pattern(new_order, input_ndim):
    """Decompose DimShuffle into Squeeze, Transpose, Unsqueeze operations.

    Parameters
    ----------
    new_order : tuple
        DimShuffle pattern (e.g., (1, 'x', 0) or (2, 0))
    input_ndim : int
        Number of dimensions in input tensor

    Returns
    -------
    dict
        Dictionary with keys:
        - 'squeeze_axes': list of int - axes to remove (or None)
        - 'transpose_perm': list of int - permutation for transpose (or None)
        - 'unsqueeze_axes': list of int - axes to add (or None)

    Notes
    -----
    Follows PyTensor's DimShuffle.perform() decomposition:
    1. Squeeze: Remove dropped dimensions
    2. Transpose: Reorder kept dimensions
    3. Unsqueeze: Add new dimensions

    Examples
    --------
    >>> decompose_dimshuffle_pattern((1, 'x', 0), input_ndim=2)
    {'squeeze_axes': None, 'transpose_perm': [1, 0], 'unsqueeze_axes': [1]}

    >>> decompose_dimshuffle_pattern((2, 0), input_ndim=3)  # (A,1,C) -> (C,A)
    {'squeeze_axes': [1], 'transpose_perm': [1, 0], 'unsqueeze_axes': None}
    """
    # Extract non-'x' dimensions (kept dimensions)
    non_x_dims = [d for d in new_order if d != 'x']

    # Find axes to add ('x' positions in new_order)
    axes_to_add = [i for i, d in enumerate(new_order) if d == 'x']

    # Find axes to drop (input dims not in non_x_dims)
    all_input_dims = set(range(input_ndim))
    kept_dims = set(non_x_dims)
    dropped_dims = sorted(all_input_dims - kept_dims)

    # Check if transpose is needed (non_x_dims not in sorted order)
    needs_transpose = non_x_dims != sorted(non_x_dims)

    # Build result
    result = {
        'squeeze_axes': dropped_dims if dropped_dims else None,
        'transpose_perm': non_x_dims if needs_transpose else None,
        'unsqueeze_axes': axes_to_add if axes_to_add else None,
    }

    # CRITICAL: Adjust transpose permutation after squeeze
    # After squeezing, dimension indices shift down
    if result['squeeze_axes'] and result['transpose_perm']:
        # Create mapping from original dims to post-squeeze dims
        dim_mapping = {}
        new_idx = 0
        for old_idx in range(input_ndim):
            if old_idx not in result['squeeze_axes']:
                dim_mapping[old_idx] = new_idx
                new_idx += 1

        # Remap transpose permutation
        result['transpose_perm'] = [
            dim_mapping[old_dim] for old_dim in result['transpose_perm']
        ]

    # CRITICAL: Adjust unsqueeze axes after transpose
    # Unsqueeze axes are relative to the output shape, but we need them
    # relative to the post-transpose shape
    # Actually, the axes_to_add are already in the correct positions
    # relative to the final output, so we need to work backwards
    if result['unsqueeze_axes']:
        # Count how many 'x' appear before each kept dimension
        unsqueeze_before_count = []
        for i, d in enumerate(new_order):
            if d != 'x':
                # Count 'x' before this dimension
                x_count = sum(1 for j in range(i) if new_order[j] == 'x')
                unsqueeze_before_count.append(x_count)

        # Adjust axes: subtract the cumulative 'x' count
        # Actually, the axes_to_add are already correct for the final shape
        # We need to convert them to positions for the Unsqueeze operation
        # which inserts at those positions
        pass  # axes_to_add is already correct

    return result
```

#### 2. Replace DimShuffle Fallback with Proper Implementation

**File**: `pytensor/link/onnx/dispatch/shape.py`

**Changes**: Replace the Identity fallback (lines 222-230) with proper multi-operation conversion

```python
# Replace lines 222-230 with:

    # Complex case: combination of operations
    # Decompose into Squeeze → Transpose → Unsqueeze sequence
    ops = decompose_dimshuffle_pattern(new_order, input_ndim)
    nodes = []
    current_var = input_names[0]

    # Step 1: Squeeze (if needed)
    if ops['squeeze_axes']:
        squeeze_output = f"dimshuffle_squeeze_{output_names[0]}"
        axes_name = f"squeeze_axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(
            np.array(ops['squeeze_axes'], dtype=np.int64), name=""
        )

        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"SqueezeAxesConst_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=[current_var, axes_name],
                outputs=[squeeze_output],
                name=f"Squeeze_{output_names[0]}",
            )
        )
        current_var = squeeze_output

    # Step 2: Transpose (if needed)
    if ops['transpose_perm']:
        transpose_output = f"dimshuffle_transpose_{output_names[0]}"

        nodes.append(
            helper.make_node(
                "Transpose",
                inputs=[current_var],
                outputs=[transpose_output],
                perm=ops['transpose_perm'],
                name=f"Transpose_{output_names[0]}",
            )
        )
        current_var = transpose_output

    # Step 3: Unsqueeze (if needed)
    if ops['unsqueeze_axes']:
        axes_name = f"unsqueeze_axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(
            np.array(ops['unsqueeze_axes'], dtype=np.int64), name=""
        )

        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"UnsqueezeAxesConst_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[current_var, axes_name],
                outputs=output_names,
                name=f"Unsqueeze_{output_names[0]}",
            )
        )
    else:
        # If no unsqueeze, the last operation's output is the final output
        # Need to rename the last node's output
        if nodes:
            nodes[-1].output[0] = output_names[0]
        else:
            # Identity case (shouldn't happen, but handle it)
            nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[current_var],
                    outputs=output_names,
                    name=f"Identity_{output_names[0]}",
                )
            )

    return nodes
```

### Success Criteria

#### Automated Verification:
- [x] All new DimShuffle tests pass: `pytest tests/link/onnx/test_shape.py::test_dimshuffle_complex_patterns -v`
- [x] All existing tests still pass: `pytest tests/link/onnx/ -v`
- [x] No Identity nodes in complex DimShuffle exports
- [x] ONNX checker validates all generated models
- [ ] Linting passes: `pre-commit run --all-files`

#### Manual Verification:
- [ ] Export a neural network with complex reshaping (e.g., attention mechanism)
- [ ] Verify ONNX graph contains Squeeze/Transpose/Unsqueeze nodes (not Identity)
- [ ] Run exported model in ONNX Runtime and compare outputs
- [ ] Test with PyTorch's ONNX export for comparison on complex reshapes

---

## Phase 2: Tests for Untested Operations

### Overview
Add comprehensive tests for 5 operations that are implemented but have zero test coverage. These tests should mostly pass, but if they expose bugs, fix the implementation.

### 2.1: Gemv Tests

**File**: `tests/link/onnx/test_nlinalg.py`

**Changes**: Add after line 72 (after test_simple_linear_layer)

```python
def test_gemv_operation(tmp_path):
    """Test Gemv (general matrix-vector multiplication with scaling).

    Gemv computes: y = alpha * A @ x + beta * y_in
    """
    # Define inputs
    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha = pt.scalar("alpha", dtype="float32")
    beta = pt.scalar("beta", dtype="float32")

    # Import Gemv from blas
    from pytensor.tensor.blas import Gemv
    gemv_op = Gemv(inplace=False)

    # Create Gemv operation: y = alpha * A @ x + beta * y_in
    y = gemv_op(y_in, alpha, A, x, beta)

    # Test data
    rng = np.random.default_rng(42)
    A_val = rng.random((3, 4)).astype("float32")
    x_val = rng.random(4).astype("float32")
    y_in_val = rng.random(3).astype("float32")
    alpha_val = np.array(2.0, dtype="float32")
    beta_val = np.array(0.5, dtype="float32")

    compare_onnx_and_py(
        [y_in, alpha, A, x, beta],
        y,
        [y_in_val, alpha_val, A_val, x_val, beta_val],
        tmp_path=tmp_path,
    )


def test_gemv_structure(tmp_path):
    """Test that Gemv generates correct 4-node ONNX structure."""
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor.blas import Gemv

    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha = pt.scalar("alpha", dtype="float32")
    beta = pt.scalar("beta", dtype="float32")

    gemv_op = Gemv(inplace=False)
    y = gemv_op(y_in, alpha, A, x, beta)

    f = pytensor.function([y_in, alpha, A, x, beta], y)

    # Export
    model_path = tmp_path / "test_gemv.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["MatMul", "Mul", "Mul", "Add"],
        expected_node_count=4,
    )

    # Verify the 4 nodes are: MatMul, Mul (alpha), Mul (beta), Add
    node_types = structure["node_types"]
    assert node_types.count("MatMul") == 1
    assert node_types.count("Mul") == 2
    assert node_types.count("Add") == 1


@pytest.mark.parametrize("alpha,beta", [
    (1.0, 0.0),  # Just A @ x
    (1.0, 1.0),  # A @ x + y
    (2.0, 0.5),  # Scaled
    (0.0, 1.0),  # Just beta * y
])
def test_gemv_scaling_factors(tmp_path, alpha, beta):
    """Test Gemv with different scaling factors."""
    from pytensor.tensor.blas import Gemv

    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha_var = pt.scalar("alpha", dtype="float32")
    beta_var = pt.scalar("beta", dtype="float32")

    gemv_op = Gemv(inplace=False)
    y = gemv_op(y_in, alpha_var, A, x, beta_var)

    rng = np.random.default_rng(42)
    A_val = rng.random((3, 4)).astype("float32")
    x_val = rng.random(4).astype("float32")
    y_in_val = rng.random(3).astype("float32")
    alpha_val = np.array(alpha, dtype="float32")
    beta_val = np.array(beta, dtype="float32")

    compare_onnx_and_py(
        [y_in, alpha_var, A, x, beta_var],
        y,
        [y_in_val, alpha_val, A_val, x_val, beta_val],
        tmp_path=tmp_path,
    )
```

### 2.2: Cast Tests

**File**: `tests/link/onnx/test_elemwise.py`

**Changes**: Add after line 159 (after test_chained_operations)

```python
@pytest.mark.parametrize("from_dtype,to_dtype", [
    ("float32", "float64"),
    ("float32", "int32"),
    ("float32", "int64"),
    ("int32", "float32"),
    ("int32", "int64"),
    ("int64", "float32"),
    ("float64", "float32"),
])
def test_cast_dtypes(tmp_path, from_dtype, to_dtype):
    """Test Cast operation with various dtype conversions."""
    x = pt.vector("x", dtype=from_dtype)
    y = pt.cast(x, to_dtype)

    rng = np.random.default_rng(42)
    if from_dtype.startswith("float"):
        x_val = rng.random(5).astype(from_dtype)
    else:
        x_val = rng.integers(-10, 10, size=5).astype(from_dtype)

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_cast_in_computation(tmp_path):
    """Test Cast used within a computation graph."""
    x = pt.vector("x", dtype="int32")
    # Convert to float, do computation, convert back
    x_float = pt.cast(x, "float32")
    y_float = x_float * 2.5 + 1.0
    y = pt.cast(y_float, "int32")

    x_val = np.array([1, 2, 3, 4, 5], dtype="int32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_cast_structure(tmp_path):
    """Test that Cast generates correct ONNX node."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.cast(x, "int32")

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_cast.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Cast"],
        expected_node_count=1,
    )

    # Check Cast node has 'to' attribute
    cast_node = model.graph.node[0]
    assert cast_node.op_type == "Cast"
    to_attr = next(attr for attr in cast_node.attribute if attr.name == "to")
    assert to_attr.i == 6  # TensorProto.INT32
```

### 2.3: Composite Scalar Op Decomposition Tests

**File**: `tests/link/onnx/test_elemwise.py`

**Changes**: Add after Cast tests

```python
def test_composite_scalar_op(tmp_path):
    """Test Composite scalar op decomposition.

    PyTensor's optimizer often fuses multiple scalar ops into a Composite.
    We need to decompose this back into individual ONNX nodes.
    """
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")

    # Create a computation that PyTensor might fuse into a Composite
    # (x * 2 + y) * 3
    z = (x * 2 + y) * 3

    # Compile with optimization to potentially create Composite ops
    f = pytensor.function([x, y], z, mode="FAST_RUN")

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    # Test execution
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_composite_with_constants(tmp_path):
    """Test Composite that includes constant folding."""
    x = pt.vector("x", dtype="float32")

    # Expression with constants: x * 2.0 + 3.0
    y = x * 2.0 + 3.0

    f = pytensor.function([x], y, mode="FAST_RUN")

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_composite_complex_expression(tmp_path):
    """Test complex expression that becomes Composite."""
    x = pt.vector("x", dtype="float32")

    # Complex expression: (x^2 + 2*x + 1) / (x + 1)
    # = (x + 1)^2 / (x + 1) = x + 1 (but optimizer might not simplify)
    numerator = x**2 + 2*x + 1
    denominator = x + 1
    y = numerator / denominator

    f = pytensor.function([x], y, mode="FAST_RUN")

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

### 2.4: AllocEmpty Tests

**File**: `tests/link/onnx/test_shape.py`

**Changes**: Add after line 142 (after test_combined_reshape_operations)

```python
def test_alloc_empty_scalar_dims(tmp_path):
    """Test AllocEmpty with scalar dimension inputs."""
    # Create shape from scalars
    dim0 = pt.scalar("dim0", dtype="int64")
    dim1 = pt.scalar("dim1", dtype="int64")

    from pytensor.tensor.basic import AllocEmpty
    alloc_op = AllocEmpty(dtype="float32")

    x = alloc_op(dim0, dim1)

    dim0_val = np.array(3, dtype="int64")
    dim1_val = np.array(4, dtype="int64")

    # Note: AllocEmpty creates uninitialized memory, ONNX creates zeros
    # We can't compare values, but we can check shapes
    from pytensor.link.onnx import export_onnx

    f = pytensor.function([dim0, dim1], x)
    model_path = tmp_path / "test_alloc_empty.onnx"
    model = export_onnx(f, model_path)

    # Validate model structure
    onnx.checker.check_model(model)

    # Run with ONNX Runtime to check shape
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {
        onnx_inputs[0].name: dim0_val,
        onnx_inputs[1].name: dim1_val,
    }
    onnx_res = session.run(None, input_feed)

    # Check shape is correct
    assert onnx_res[0].shape == (3, 4)


def test_alloc_empty_vector_shape(tmp_path):
    """Test AllocEmpty with vector shape input."""
    shape_vec = pt.vector("shape", dtype="int64")

    from pytensor.tensor.basic import AllocEmpty
    alloc_op = AllocEmpty(dtype="float32")

    x = alloc_op(shape_vec)

    shape_val = np.array([2, 3, 4], dtype="int64")

    # Export and check
    from pytensor.link.onnx import export_onnx

    f = pytensor.function([shape_vec], x)
    model_path = tmp_path / "test_alloc_empty_vec.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {onnx_inputs[0].name: shape_val}
    onnx_res = session.run(None, input_feed)

    assert onnx_res[0].shape == (2, 3, 4)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_alloc_empty_dtypes(tmp_path, dtype):
    """Test AllocEmpty with different dtypes."""
    dim0 = pt.scalar("dim0", dtype="int64")
    dim1 = pt.scalar("dim1", dtype="int64")

    from pytensor.tensor.basic import AllocEmpty
    alloc_op = AllocEmpty(dtype=dtype)

    x = alloc_op(dim0, dim1)

    from pytensor.link.onnx import export_onnx

    f = pytensor.function([dim0, dim1], x)
    model_path = tmp_path / f"test_alloc_empty_{dtype}.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    # Check output dtype
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dim0_val = np.array(2, dtype="int64")
    dim1_val = np.array(3, dtype="int64")

    onnx_inputs = session.get_inputs()
    input_feed = {
        onnx_inputs[0].name: dim0_val,
        onnx_inputs[1].name: dim1_val,
    }
    onnx_res = session.run(None, input_feed)

    expected_dtype = np.dtype(dtype)
    assert onnx_res[0].dtype == expected_dtype
```

### 2.5: DeepCopyOp Tests

**File**: `tests/link/onnx/test_basic.py`

**Changes**: Add after line 216 (after test_shared_variables_as_initializers)

```python
def test_deep_copy_operation(tmp_path):
    """Test DeepCopyOp maps to ONNX Identity."""
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_deep_copy_in_graph(tmp_path):
    """Test DeepCopyOp within a larger computation."""
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")

    # Copy, then do computation
    deep_copy_op = DeepCopyOp()
    x_copy = deep_copy_op(x)
    y = x_copy * 2 + 1

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_deep_copy_structure(tmp_path):
    """Test that DeepCopyOp generates ONNX Identity node."""
    from pytensor.link.onnx import export_onnx
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_deep_copy.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Identity"],
        expected_node_count=1,
    )

    assert structure["node_types"] == ["Identity"]
```

### Success Criteria

#### Automated Verification:
- [ ] All Gemv tests pass: `pytest tests/link/onnx/test_nlinalg.py -k gemv -v`
- [ ] All Cast tests pass: `pytest tests/link/onnx/test_elemwise.py -k cast -v`
- [ ] All Composite tests pass: `pytest tests/link/onnx/test_elemwise.py -k composite -v`
- [ ] All AllocEmpty tests pass: `pytest tests/link/onnx/test_shape.py -k alloc_empty -v`
- [ ] All DeepCopyOp tests pass: `pytest tests/link/onnx/test_basic.py -k deep_copy -v`
- [ ] All existing tests still pass: `pytest tests/link/onnx/ -v`
- [ ] ONNX validation succeeds for all test cases

#### Manual Verification:
- [ ] Review ONNX graphs for multi-node operations (Gemv, Composite, AllocEmpty)
- [ ] Verify node counts and types match expected patterns
- [ ] Test export of real models that use these operations
- [ ] Compare ONNX Runtime performance with PyTensor

---

## Phase 3: Comprehensive Test Coverage & Quality

### Overview
Expand test coverage to include multiple data types, edge cases, and ONNX structure validation. This phase ensures the backend is production-ready.

### Phase 3a: Multi-dtype Test Suite

#### 1. Add Dtype Test Utilities

**File**: `tests/link/onnx/test_basic.py`

**Changes**: Add dtype testing helpers

```python
# Add after validate_onnx_graph_structure function

# Dtype constants for ONNX testing
ONNX_FLOAT_DTYPES = ["float32", "float64"]
ONNX_INT_DTYPES = ["int32", "int64"]
ONNX_UINT_DTYPES = ["uint8"]
ONNX_BOOL_DTYPES = ["bool"]
ONNX_ALL_DTYPES = ONNX_FLOAT_DTYPES + ONNX_INT_DTYPES + ONNX_UINT_DTYPES + ONNX_BOOL_DTYPES


def generate_test_data(shape, dtype, rng=None):
    """Generate test data for given shape and dtype.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    dtype : str
        NumPy dtype string
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Test data array
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if dtype in ONNX_FLOAT_DTYPES:
        return rng.random(shape).astype(dtype)
    elif dtype in ONNX_INT_DTYPES + ONNX_UINT_DTYPES:
        return rng.integers(-10, 10, size=shape).astype(dtype)
    elif dtype == "bool":
        return rng.random(shape) > 0.5
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
```

#### 2. Add Multi-dtype Elemwise Tests

**File**: `tests/link/onnx/test_elemwise.py`

**Changes**: Add comprehensive dtype tests

```python
# Add after existing tests

@pytest.mark.parametrize("dtype", [
    "float32", "float64", "int32", "int64"
])
@pytest.mark.parametrize("op_name,op_func", [
    ("add", lambda x, y: x + y),
    ("mul", lambda x, y: x * y),
    ("sub", lambda x, y: x - y),
])
def test_binary_ops_dtypes(tmp_path, dtype, op_name, op_func):
    """Test binary operations with different dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    z = op_func(x, y)

    rng = np.random.default_rng(42)
    x_val = generate_test_data((5,), dtype, rng)
    y_val = generate_test_data((5,), dtype, rng)

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op_name,op_func", [
    ("div", lambda x, y: x / y),
    ("exp", lambda x: pt.exp(x)),
    ("log", lambda x: pt.log(x)),
    ("sqrt", lambda x: pt.sqrt(x)),
])
def test_float_only_ops_dtypes(tmp_path, dtype, op_name, op_func):
    """Test operations that only work with float dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.vector("x", dtype=dtype)

    # For unary ops
    if op_name in ["exp", "log", "sqrt"]:
        # Generate positive values for log and sqrt
        rng = np.random.default_rng(42)
        x_val = rng.random(5).astype(dtype) + 0.1
        z = op_func(x)
        compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)
    else:
        # Binary op (div)
        y = pt.vector("y", dtype=dtype)
        z = op_func(x, y)
        rng = np.random.default_rng(42)
        x_val = generate_test_data((5,), dtype, rng)
        y_val = generate_test_data((5,), dtype, rng) + 0.1  # Avoid division by zero
        compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_abs_dtypes(tmp_path, dtype):
    """Test absolute value with different dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.vector("x", dtype=dtype)
    z = pt.abs(x)

    rng = np.random.default_rng(42)
    x_val = generate_test_data((5,), dtype, rng)

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("from_dtype,to_dtype", [
    ("float32", "float64"),
    ("float64", "float32"),
    ("float32", "int32"),
    ("float32", "int64"),
    ("int32", "float32"),
    ("int32", "int64"),
    ("int64", "int32"),
    ("int64", "float32"),
])
def test_mixed_dtype_operations(tmp_path, from_dtype, to_dtype):
    """Test operations with mixed dtypes (via Cast)."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.vector("x", dtype=from_dtype)
    x_cast = pt.cast(x, to_dtype)

    # Do operation in target dtype
    y = x_cast * 2

    rng = np.random.default_rng(42)
    x_val = generate_test_data((5,), from_dtype, rng)

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

#### 3. Add Multi-dtype Shape Tests

**File**: `tests/link/onnx/test_shape.py`

**Changes**: Add dtype tests for shape operations

```python
# Add after existing tests

@pytest.mark.parametrize("dtype", [
    "float32", "float64", "int32", "int64"
])
def test_reshape_dtypes(tmp_path, dtype):
    """Test Reshape with different dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.vector("x", dtype=dtype)
    y = x.reshape((2, 3))

    rng = np.random.default_rng(42)
    x_val = generate_test_data((6,), dtype, rng)

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("dtype", [
    "float32", "float64", "int32", "int64"
])
def test_dimshuffle_dtypes(tmp_path, dtype):
    """Test DimShuffle with different dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.matrix("x", dtype=dtype)
    y = x.dimshuffle(1, 0)  # Transpose

    rng = np.random.default_rng(42)
    x_val = generate_test_data((2, 3), dtype, rng)

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

#### 4. Add Multi-dtype Linear Algebra Tests

**File**: `tests/link/onnx/test_nlinalg.py`

**Changes**: Add dtype tests for dot operations

```python
# Add after existing tests

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dot_dtypes(tmp_path, dtype):
    """Test matrix multiplication with different dtypes."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.matrix("x", dtype=dtype)
    y = pt.matrix("y", dtype=dtype)
    z = pt.dot(x, y)

    rng = np.random.default_rng(42)
    x_val = generate_test_data((3, 4), dtype, rng)
    y_val = generate_test_data((4, 5), dtype, rng)

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

### Phase 3b: Edge Case Tests

#### 1. Add Edge Case Tests

**File**: `tests/link/onnx/test_elemwise.py`

**Changes**: Add edge case tests

```python
# Add edge case tests

def test_empty_tensor(tmp_path):
    """Test operations on empty tensors (0-sized dimensions)."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    x_val = np.array([], dtype="float32")
    y_val = np.array([], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_single_element_tensor(tmp_path):
    """Test operations on single-element tensors."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x * y + 1

    x_val = np.array([5.0], dtype="float32")
    y_val = np.array([3.0], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_scalar_operations(tmp_path):
    """Test scalar (0-dimensional tensor) operations."""
    x = pt.scalar("x", dtype="float32")
    y = pt.scalar("y", dtype="float32")
    z = x * y + 1

    x_val = np.array(5.0, dtype="float32")
    y_val = np.array(3.0, dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


@pytest.mark.parametrize("x_shape,y_shape", [
    ((3, 1), (3, 4)),      # Broadcasting last dim
    ((1, 4), (3, 4)),      # Broadcasting first dim
    ((3, 1, 4), (3, 5, 4)), # Broadcasting middle dim
    ((1,), (3, 4)),        # Scalar-like broadcast
])
def test_broadcasting_patterns(tmp_path, x_shape, y_shape):
    """Test various broadcasting patterns."""
    from tests.link.onnx.test_basic import generate_test_data

    x = pt.tensor("x", dtype="float32", shape=x_shape)
    y = pt.tensor("y", dtype="float32", shape=y_shape)
    z = x + y

    rng = np.random.default_rng(42)
    x_val = generate_test_data(x_shape, "float32", rng)
    y_val = generate_test_data(y_shape, "float32", rng)

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**File**: `tests/link/onnx/test_shape.py`

**Changes**: Add shape edge cases

```python
# Add edge case tests

def test_reshape_empty_tensor(tmp_path):
    """Test reshaping empty tensor."""
    x = pt.vector("x", dtype="float32")
    y = x.reshape((0, 3))

    x_val = np.array([], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_single_element(tmp_path):
    """Test DimShuffle on single-element tensor."""
    x = pt.tensor(dtype="float32", shape=(1, 1, 1), name="x")
    y = x.dimshuffle(2, 0, 1)

    x_val = np.array([[[5.0]]], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_reshape_to_scalar(tmp_path):
    """Test reshaping to scalar (0-D tensor)."""
    x = pt.vector("x", dtype="float32")
    y = x.reshape(())

    x_val = np.array([5.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

### Phase 3c: ONNX Structure Validation

#### 1. Strengthen Shape_i Test

**File**: `tests/link/onnx/test_shape.py`

**Changes**: Replace weak Shape_i test (lines 120-131)

```python
# Replace test_shape_i_get_dimension with:

def test_shape_i_structure(tmp_path):
    """Test Shape_i generates correct 5-node ONNX sequence."""
    from pytensor.link.onnx import export_onnx

    x = pt.matrix("x", dtype="float32")
    # Extract dimension 0
    dim0 = x.shape[0]

    # Use in a simple computation to keep it in the graph
    dim0_float = pt.cast(dim0, "float32")
    y = pt.ones_like(x) * dim0_float

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_shape_i.onnx"
    model = export_onnx(f, model_path)

    # Validate structure includes Shape_i decomposition
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(model)

    # Should have: Shape, Constant (indices), Gather, Constant (axes), Squeeze, Cast, ...
    node_types = structure["node_types"]

    # Verify Shape_i components appear in order
    assert "Shape" in node_types, "Missing Shape node"
    assert "Gather" in node_types, "Missing Gather node"
    assert "Squeeze" in node_types, "Missing Squeeze node"
    assert node_types.count("Constant") >= 2, "Missing Constant nodes for Shape_i"

    # Also verify correct output
    x_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {onnx_inputs[0].name: x_val}
    onnx_res = session.run(None, input_feed)

    # Should be matrix filled with 3.0 (the first dimension)
    expected = np.ones_like(x_val) * 3.0
    np.testing.assert_allclose(onnx_res[0], expected, rtol=1e-4)


def test_shape_i_multiple_dimensions(tmp_path):
    """Test extracting multiple dimensions."""
    x = pt.tensor(dtype="float32", shape=(2, 3, 4), name="x")

    dim0 = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    # Use all three dimensions
    dims = pt.stack([dim0, dim1, dim2])

    # Convert to float for output
    y = pt.cast(dims, "float32")

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 3, 4)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

#### 2. Add Structure Validation to Multi-Node Tests

**File**: `tests/link/onnx/test_special.py`

**Changes**: Add structure validation

```python
# Add after existing softmax tests

def test_softmax_axis_none_structure(tmp_path):
    """Test Softmax with axis=None generates correct multi-node structure."""
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor.special import softmax

    x = pt.matrix("x", dtype="float32")
    y = softmax(x, axis=None)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_softmax_axis_none.onnx"
    model = export_onnx(f, model_path)

    # Should have: Flatten, Softmax, Shape, Reshape
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(model)
    node_types = structure["node_types"]

    assert "Flatten" in node_types
    assert "Softmax" in node_types
    assert "Shape" in node_types
    assert "Reshape" in node_types
```

### Success Criteria

#### Automated Verification:
- [ ] All multi-dtype tests pass: `pytest tests/link/onnx/ -k dtype -v`
- [ ] All edge case tests pass: `pytest tests/link/onnx/ -k "empty or single_element or scalar" -v`
- [ ] Shape_i structure test passes with validation: `pytest tests/link/onnx/test_shape.py::test_shape_i_structure -v`
- [ ] All structure validation tests pass: `pytest tests/link/onnx/ -k structure -v`
- [ ] Full test suite passes: `pytest tests/link/onnx/ -v`
- [ ] Coverage report shows improvement: `pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term`

#### Manual Verification:
- [ ] Export models with all supported dtypes (int32, int64, float32, float64)
- [ ] Test edge cases in real models (empty batches, single-item batches)
- [ ] Verify ONNX graphs contain expected node types and counts
- [ ] Compare generated ONNX with reference implementations (e.g., PyTorch)
- [ ] Test exported models on different ONNX Runtime backends (CPU, CUDA if available)

---

## Testing Strategy

### Unit Tests
**Approach**: Test each operation individually with `compare_onnx_and_py`
- DimShuffle: All case combinations (squeeze, transpose, unsqueeze, complex)
- Untested ops: Gemv, Cast, Composite, AllocEmpty, DeepCopyOp
- Dtype variations: float32, float64, int32, int64, bool
- Edge cases: empty tensors, scalars, single elements

**Coverage Target**: 100% of dispatch implementations

### Integration Tests
**Approach**: Test complex computation graphs
- Multi-layer neural networks
- Attention mechanisms (complex reshaping)
- Mixed dtype computations
- Shared variables as initializers

**Coverage Target**: Common real-world patterns

### Structure Validation Tests
**Approach**: Validate ONNX graph structure, not just outputs
- Node types and counts
- Node connections
- Multi-node decompositions
- Initializer presence and values

**Coverage Target**: All multi-node operations

### Regression Tests
**Approach**: Ensure existing tests continue to pass
- Run full suite after each change
- No pytest.skip or pytest.xfail added
- All ONNX models validate with `onnx.checker.check_model()`

**Coverage Target**: 100% of existing tests

## Performance Considerations

**Not in Scope**: Performance optimization or benchmarking

**Notes**:
- ONNX Runtime is highly optimized and should handle generated graphs efficiently
- Multi-node decompositions (e.g., Gemv: 4 nodes vs 1 op) may have slight overhead
- ONNX Runtime's graph optimizer should fuse operations where beneficial
- Focus is on correctness, not performance, for this phase

## Migration Notes

**No Breaking Changes**: All changes are additions or bug fixes
- Existing API remains unchanged
- Existing tests continue to work
- Newly exported ONNX models are compatible with existing runtime code

**Backward Compatibility**:
- Models exported before DimShuffle fix may have incorrect results (Identity fallback)
- Recommend re-exporting any models that use complex reshaping operations
- No file format changes - all ONNX models use same opset version (18)

## References

- **Research document**: `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md`
- **PyTensor DimShuffle**: `pytensor/tensor/elemwise.py:43-275`
- **ONNX dispatch system**: `pytensor/link/onnx/dispatch/basic.py:1-292`
- **Existing tests**: `tests/link/onnx/test_*.py`
- **ONNX Runtime docs**: https://onnxruntime.ai/docs/
- **ONNX operator specs**: https://onnx.ai/onnx/operators/
