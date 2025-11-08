# Phase 3: Shape Operations Property-Based Tests TDD Implementation Plan

## Overview

Create individual property-based tests for 8 shape operations using strategies from the `SHAPE_OPERATIONS` registry. Unlike elemwise operations, shape operations have diverse behaviors requiring separate test functions for each operation.

## Current State Analysis

### Current Testing Landscape:
- Testing framework: pytest with Hypothesis (configured in tests/link/onnx/conftest.py)
- Test utilities: `compare_onnx_and_py()` and `get_onnx_node_types()` at tests/link/onnx/test_basic.py
- Registry: `SHAPE_OPERATIONS` exists in tests/link/onnx/strategies.py:156-241
- Property test pattern: Individual tests per operation (recommended in research doc)

### Current Shape Tests:
- 10 manual tests in tests/link/onnx/test_shape.py
- Test coverage: shape, shape_i, specify_shape, concatenate, stack, split
- Missing from tests: reshape, transpose, dimshuffle operations
- Manual tests are well-written, will augment with property tests

### Shape Operations Characteristics:
- **Heterogeneous behavior**: Each operation has unique validation requirements
- **Shape transformations**: Output shapes differ significantly from inputs
- **Multi-output operations**: Split returns multiple outputs
- **Pass-through operations**: SpecifyShape generates no ONNX nodes

## Desired End State

A comprehensive property-based test suite with:
- **8 individual property test functions** (one per shape operation)
- **Retained manual tests** for specific edge cases
- **80+ test scenarios** (8 operations × 10 examples minimum)
- **Clear validation** for each operation's unique behavior

### Key Discoveries:
- Research decision #2 (line 384-414): Shape operations need individual tests due to unique validation
- Existing SHAPE_OPERATIONS registry has strategies ready (strategies.py:159-241)
- Shape operations have complex outputs (shapes, tuples, multiple values)
- Some operations (SpecifyShape) are pass-through and need different validation

## What We're NOT Testing/Implementing

- Not testing reshape with -1 (inferred dimension) yet
- Not testing dynamic shapes (non-constant shape inputs)
- Not testing all dimshuffle permutations (focus on common patterns)
- Not modifying ONNX backend implementation (only tests)
- Not testing shape operations with non-float32 dtypes yet

## TDD Approach

### Test Design Philosophy:
- Each operation gets its own property test (clear isolation)
- Test failures clearly indicate which specific operation failed
- Validate both numerical correctness and shape transformations
- Use existing strategies from SHAPE_OPERATIONS registry

---

## Phase 1: Test Design & Implementation

### Overview
Write individual property-based tests for each shape operation using the SHAPE_OPERATIONS registry.

### Test Categories:

#### 1. Shape Inspection Operations

##### Test: `test_shape_operation_correctness`
**Purpose**: Property test for Shape operation (get tensor shape)
**Test Data**: Random tensors with various shapes
**Expected Behavior**: Returns correct shape as int64 array
**Assertions**: Shape correctness, ONNX node type

```python
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
    op_config = SHAPE_OPERATIONS['shape']

    # Generate test tensor
    test_data = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate result
    expected_shape = np.array(test_data.shape, dtype='int64')
    np.testing.assert_array_equal(result, expected_shape)

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types, \
        f"Expected 'Shape' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError (array comparison)
- Expected message: Arrays not equal (shape mismatch)
- Points to: Shape operation implementation

##### Test: `test_shape_i_operation_correctness`
**Purpose**: Property test for Shape_i operation (get specific dimension)
**Test Data**: Random tensors with dimension index
**Expected Behavior**: Returns correct dimension value
**Assertions**: Dimension value correctness, multi-node ONNX pattern

```python
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
    op_config = SHAPE_OPERATIONS['shape_i']

    # Generate test data (tensor and valid dimension index)
    test_data = data.draw(op_config['strategy'])
    x_val, dim_index = test_data

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, dim_index)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Validate result
    expected_dim = x_val.shape[dim_index]
    assert result == expected_dim, \
        f"Expected dimension {dim_index} to be {expected_dim}, got {result}"

    # Verify ONNX node pattern (multi-node return)
    node_types = get_onnx_node_types(fn)
    assert 'Shape' in node_types, "Expected 'Shape' node"
    assert 'Gather' in node_types, "Expected 'Gather' node"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Dimension value mismatch
- Points to: Shape_i implementation

##### Test: `test_specify_shape_passthrough_correctness`
**Purpose**: Property test verifying SpecifyShape creates no ONNX nodes
**Test Data**: Random tensors
**Expected Behavior**: Pass-through, no ONNX nodes generated
**Assertions**: No SpecifyShape nodes, computation continues correctly

```python
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
    x_val = np.random.randn(*shape).astype('float32')

    # Build graph with SpecifyShape in the middle
    x = pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim)
    x_specified = specify_shape(x, x_val.shape)
    y = x_specified * 2.0  # Some computation after SpecifyShape

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Validate numerical correctness
    expected = x_val * 2.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify SpecifyShape doesn't appear in ONNX
    node_types = get_onnx_node_types(fn)
    assert 'SpecifyShape' not in node_types, \
        "SpecifyShape should not appear in ONNX graph (it's a pass-through)"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Numerical mismatch OR SpecifyShape appears in graph
- Points to: SpecifyShape dispatcher or pass-through logic

#### 2. Reshape Operations

##### Test: `test_reshape_operation_correctness`
**Purpose**: Property test for Reshape operation
**Test Data**: Tensors with compatible reshape targets
**Expected Behavior**: Correct reshaping with same total elements
**Assertions**: Shape transformation, numerical correctness

```python
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
    op_config = SHAPE_OPERATIONS['reshape']

    # Generate tensor and compatible reshape target
    test_data = data.draw(op_config['strategy'])
    x_val, new_shape = test_data

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, new_shape)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Validate shape transformation
    expected = x_val.reshape(new_shape)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == new_shape, \
        f"Expected shape {new_shape}, got {result.shape}"

    # Verify total elements preserved
    assert result.size == x_val.size, \
        f"Element count changed: {x_val.size} -> {result.size}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Reshape' in node_types, \
        f"Expected 'Reshape' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Shape mismatch or array not equal
- Points to: Reshape operation implementation

##### Test: `test_transpose_operation_correctness`
**Purpose**: Property test for Transpose operation (matrix transpose)
**Test Data**: 2D matrices
**Expected Behavior**: Correct transposition (axes swapped)
**Assertions**: Shape swap, element correctness

```python
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
    op_config = SHAPE_OPERATIONS['transpose']

    # Generate 2D matrix
    test_data = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate transposition
    expected = test_data.T
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (test_data.shape[1], test_data.shape[0]), \
        f"Expected shape {test_data.T.shape}, got {result.shape}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Transpose' in node_types, \
        f"Expected 'Transpose' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal or shape mismatch
- Points to: Transpose/DimShuffle implementation

##### Test: `test_dimshuffle_add_dim_correctness`
**Purpose**: Property test for DimShuffle adding dimension
**Test Data**: Vectors
**Expected Behavior**: Adds dimension at specified position
**Assertions**: Shape change, ONNX node type

```python
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
    op_config = SHAPE_OPERATIONS['dimshuffle_add_dim']

    # Generate vector
    test_data = data.draw(op_config['strategy'])

    # Build graph (adds dimension at position 0)
    graph_inputs, graph_output = op_config['build_graph'](test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate dimension addition
    expected = test_data[np.newaxis, :]  # Add dimension at position 0
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (1, test_data.shape[0]), \
        f"Expected shape (1, {test_data.shape[0]}), got {result.shape}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Unsqueeze' in node_types, \
        f"Expected 'Unsqueeze' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Shape mismatch
- Points to: DimShuffle Unsqueeze implementation

##### Test: `test_dimshuffle_squeeze_correctness`
**Purpose**: Property test for DimShuffle removing dimension
**Test Data**: Tensors with singleton dimension
**Expected Behavior**: Removes singleton dimension
**Assertions**: Shape reduction, numerical correctness

```python
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
    op_config = SHAPE_OPERATIONS['dimshuffle_squeeze']

    # Generate tensor with singleton dimension
    test_data = data.draw(op_config['strategy'])

    # Build graph (removes dimension at position 1)
    graph_inputs, graph_output = op_config['build_graph'](test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Validate dimension removal
    expected = test_data.squeeze(axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.ndim == test_data.ndim - 1, \
        f"Expected {test_data.ndim - 1} dimensions, got {result.ndim}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Squeeze' in node_types, \
        f"Expected 'Squeeze' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Dimension count or shape mismatch
- Points to: DimShuffle Squeeze implementation

#### 3. Join/Split Operations

##### Test: `test_concatenate_operation_correctness`
**Purpose**: Property test for concatenate operation
**Test Data**: Two tensors with compatible shapes
**Expected Behavior**: Correct concatenation along specified axis
**Assertions**: Shape, concatenation correctness

```python
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
    op_config = SHAPE_OPERATIONS['concatenate']

    # Generate two compatible tensors and axis
    test_data = data.draw(op_config['strategy'])
    a_val, b_val, axis = test_data

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](a_val, b_val, axis)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [a_val, b_val])

    # Validate concatenation
    expected = np.concatenate([a_val, b_val], axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify shape along concatenation axis
    expected_shape = list(a_val.shape)
    expected_shape[axis] = a_val.shape[axis] + b_val.shape[axis]
    assert result.shape == tuple(expected_shape), \
        f"Expected shape {tuple(expected_shape)}, got {result.shape}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'Concat' in node_types, \
        f"Expected 'Concat' node, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal or shape mismatch
- Points to: Join/Concatenate implementation

##### Test: `test_stack_operation_correctness`
**Purpose**: Property test for stack operation
**Test Data**: Two tensors with same shape
**Expected Behavior**: Correct stacking (adds new dimension)
**Assertions**: Shape expansion, element positioning

```python
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
    op_config = SHAPE_OPERATIONS['stack']

    # Generate two tensors with same shape
    test_data = data.draw(op_config['strategy'])
    a_val, b_val = test_data

    # Build graph (stack along axis 0)
    graph_inputs, graph_output = op_config['build_graph'](a_val, b_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [a_val, b_val])

    # Validate stacking
    expected = np.stack([a_val, b_val], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify shape (added dimension)
    assert result.ndim == a_val.ndim + 1, \
        f"Expected {a_val.ndim + 1} dimensions, got {result.ndim}"
    assert result.shape[0] == 2, \
        f"Expected size 2 along axis 0, got {result.shape[0]}"

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    assert 'Concat' in node_types or 'Unsqueeze' in node_types, \
        f"Expected 'Concat' or 'Unsqueeze' nodes, got {node_types}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal or dimension mismatch
- Points to: Stack/Join implementation

### Test Implementation Steps:

1. **Modify existing test file**: `tests/link/onnx/test_shape.py`

2. **Add imports at top of file**:
   ```python
   from hypothesis import given, strategies as st, settings
   from hypothesis.extra.numpy import array_shapes
   from functools import partial
   from tests.link.onnx.strategies import SHAPE_OPERATIONS
   ```

3. **Add property test section**:
   ```python
   # ============================================================================
   # PROPERTY-BASED TESTS (Primary Coverage)
   # ============================================================================
   ```

4. **Implement each property test** as specified above

5. **Keep existing manual tests** below property tests for reference and edge cases

### Success Criteria:

#### Automated Verification:
- [ ] All test functions created with proper structure
- [ ] Tests use SHAPE_OPERATIONS registry correctly
- [ ] Tests are discoverable: `uv run pytest --collect-only tests/link/onnx/test_shape.py`
- [ ] Test code follows project conventions: `make lint-tests`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertion messages are diagnostic
- [ ] Shape validation is thorough

---

## Phase 2: Test Failure Verification

### Overview
Run the property tests and verify they work correctly or expose any implementation issues.

### Verification Steps:

1. **Run the test suite**:
   ```bash
   uv run pytest tests/link/onnx/test_shape.py -k "correctness" -v
   ```

2. **For each test, verify**:
   - Test runs without collection errors
   - Test either passes or fails with clear message
   - Failure messages show which shape operation failed
   - Shape mismatches are clearly reported

3. **Document outcomes**:
   - Which operations pass all property tests
   - Which operations have issues
   - Any edge cases discovered by Hypothesis

### Expected Outcomes:

**Scenario 1: All tests pass**
- Shape operations are well-implemented
- Property tests validate existing functionality
- Proceed to Phase 4 (refactoring)

**Scenario 2: Some tests fail**
- Specific shape operations have bugs
- Hypothesis shows minimal failing examples
- Document issues for Phase 3

**Scenario 3: Test infrastructure issues**
- Registry access problems
- Strategy issues
- Fix in strategies.py

### Expected Test Behavior:

- **test_shape_operation_correctness**: Should pass (Shape is basic)
- **test_shape_i_operation_correctness**: Should pass (already tested manually)
- **test_specify_shape_passthrough_correctness**: Should pass (pass-through)
- **test_reshape_operation_correctness**: May reveal edge cases
- **test_transpose_operation_correctness**: Should pass (matrix transpose simple)
- **test_dimshuffle_add_dim_correctness**: Should pass (Unsqueeze)
- **test_dimshuffle_squeeze_correctness**: Should pass (Squeeze)
- **test_concatenate_operation_correctness**: Should pass (already tested)
- **test_stack_operation_correctness**: Should pass (already tested)

### Success Criteria:

#### Automated Verification:
- [ ] All tests run without collection errors
- [ ] Tests complete execution (10 examples each)
- [ ] No import or strategy errors

#### Manual Verification:
- [ ] Test failures (if any) are informative
- [ ] Can identify operation and input causing failure
- [ ] Hypothesis shrinking provides minimal examples
- [ ] No confusing error messages

### Adjustment Phase:

If tests don't run properly:
- [ ] Fix registry key names
- [ ] Fix strategy access
- [ ] Adjust shape validation logic
- [ ] Improve error messages

---

## Phase 3: Implementation / Bug Fixes (If Needed)

### Overview
Fix any implementation bugs revealed by property tests. Skip this phase if all tests pass.

### Implementation Strategy:

**Only proceed if Phase 2 revealed bugs**

**Order of fixes:**
1. Simple shape operations (shape, shape_i)
2. Reshape and transpose
3. DimShuffle operations
4. Join/split operations

### Implementation Steps:

#### Fix 1: Shape/Reshape Edge Cases

**Symptom**: Reshape fails with certain shape combinations
**Location**: pytensor/link/onnx/dispatch/shape.py

**Debugging Approach**:
1. Hypothesis shows minimal failing example
2. Check shape compatibility validation
3. Verify ONNX Reshape node generation
4. Test fix with property test

#### Fix 2: DimShuffle Issues

**Symptom**: Unsqueeze/Squeeze fails or wrong dimensions
**Location**: pytensor/link/onnx/dispatch/shape.py:122

**Debugging Approach**:
1. Check dimension index handling
2. Verify ONNX axes parameter
3. Test with minimal example
4. Validate with property test

**Not providing specific fixes** - depends on what tests reveal

### Success Criteria:

#### Automated Verification:
- [ ] All property tests pass: `uv run pytest tests/link/onnx/test_shape.py -k "correctness" -v`
- [ ] No regressions in existing tests
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Fixes are minimal and targeted
- [ ] Code comments explain any edge cases
- [ ] No workarounds, proper solutions only

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor test code for clarity and organization.

### Refactoring Targets:

1. **Test Organization**:
   - Group tests by category (inspection, reshape, join/split)
   - Add section comments
   - Order tests logically

2. **Remove Redundant Tests**:
   - Identify manual tests covered by property tests
   - Keep unique edge case tests
   - Document retention rationale

3. **Documentation**:
   - Add module docstring explaining test strategy
   - Document which operations are tested
   - Explain property vs manual test split

### Refactoring Steps:

1. **Ensure all tests pass**: `uv run pytest tests/link/onnx/test_shape.py -v`

2. **Reorganize file**:
   ```python
   """
   Tests for ONNX shape operations.

   Test Strategy:
   - Property-based tests provide primary coverage (80+ scenarios)
   - Individual property test per operation (8 operations)
   - Manual tests retained for specific edge cases

   Operations: shape, shape_i, specify_shape, reshape, transpose,
               dimshuffle (unsqueeze/squeeze), concatenate, stack
   """

   # ============================================================================
   # PROPERTY-BASED TESTS - Shape Inspection
   # ============================================================================

   def test_shape_operation_correctness(...):
       ...

   def test_shape_i_operation_correctness(...):
       ...

   # ============================================================================
   # PROPERTY-BASED TESTS - Reshape Operations
   # ============================================================================

   def test_reshape_operation_correctness(...):
       ...

   # ============================================================================
   # PROPERTY-BASED TESTS - Join/Split Operations
   # ============================================================================

   def test_concatenate_operation_correctness(...):
       ...

   # ============================================================================
   # MANUAL EDGE CASE TESTS
   # ============================================================================

   def test_split_unequal(...):  # Kept: specific split pattern
       ...
   ```

3. **Consider consolidating manual tests**:
   - test_shape_basic → Covered by property test (can remove)
   - test_shape_i_dim0/dim1 → Covered by property test (can remove)
   - test_concatenate_axis0/axis1 → Covered by property test (can remove)
   - Keep test_split_equal/unequal → Split not in SHAPE_OPERATIONS yet

4. **Add helpful comments**:
   - Explain why certain manual tests are kept
   - Document any operation-specific quirks
   - Note ONNX limitations if any

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] Test count reduced appropriately
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Code is more organized and readable
- [ ] Clear distinction between property and manual tests
- [ ] No important coverage lost

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] 8 shape operations covered by individual property tests
- [ ] 80+ test scenarios (8 ops × 10 examples minimum)
- [ ] Shape transformations validated
- [ ] ONNX node types verified
- [ ] Edge cases covered by retained manual tests

### Test Organization:
- Individual property tests: One per operation (clear isolation)
- Manual tests: Specific edge cases (split with unequal sizes, etc.)
- Test utilities: compare_onnx_and_py, get_onnx_node_types

### Running Tests:

```bash
# Run all shape tests
uv run pytest tests/link/onnx/test_shape.py -v

# Run only property tests
uv run pytest tests/link/onnx/test_shape.py -k "correctness" -v

# Run specific operation test
uv run pytest tests/link/onnx/test_shape.py::test_reshape_operation_correctness -v

# Run with Hypothesis verbose output
uv run pytest tests/link/onnx/test_shape.py -k "correctness" -v --hypothesis-show-statistics
```

## Performance Considerations

- Property tests generate small tensors (max 10 elements per dimension)
- Shape operations are fast (metadata operations mostly)
- Full suite should complete in seconds
- No performance concerns expected

## Migration Notes

### Tests to Keep:
- test_split_equal, test_split_unequal (Split not in SHAPE_OPERATIONS yet)
- Any unique regression tests

### Tests to Consider Removing:
- test_shape_basic (covered by property test)
- test_shape_i_dim0/dim1/3d_tensor (covered by property test)
- test_specify_shape_passthrough (covered by property test)
- test_concatenate_axis0/axis1 (covered by property test)
- test_stack_axis0 (covered by property test)

## References

- Original research: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md:384-414`
- SHAPE_OPERATIONS registry: `tests/link/onnx/strategies.py:156-241`
- Test utilities: `tests/link/onnx/test_basic.py:30`
- Shape dispatchers: `pytensor/link/onnx/dispatch/shape.py`
- Existing shape tests: `tests/link/onnx/test_shape.py`
