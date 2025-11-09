# Phase 5: Argmax Property Test TDD Implementation Plan

## Overview

Create an individual property-based test for the Argmax operation, separating it from general reduction operations. Argmax has unique behavior (returns indices rather than values) requiring its own test.

## Current State Analysis

### Current Testing Landscape:
- Testing framework: pytest with Hypothesis
- Test utilities: `compare_onnx_and_py()` and `get_onnx_node_types()` at tests/link/onnx/test_basic.py
- Registry: `REDUCTION_OPERATIONS` in tests/link/onnx/strategies.py includes argmax (line 285)
- Current test: `test_reduction_operations_correctness` in test_math.py:23 covers argmax with other reductions

### Current Argmax Test Coverage:
- **Included in reduction operations property test** (test_math.py:23-49)
- **Manual test** `test_argmax_argmin` in test_math.py:133-153
- Test scenarios: Currently bundled with 6 other reduction operations
- Good coverage, but argmax has unique characteristics warranting separate test

### Argmax Operation Characteristics:
- **Returns indices, not values** (unlike other reductions that return aggregated values)
- **Requires axis parameter** (cannot reduce over all axes like sum)
- **Output dtype is int64** (not float like input)
- **Used differently** than value-based reductions

## Desired End State

A focused property-based test for Argmax:
- **One dedicated property test function** for argmax
- **Retained in reduction test** for consistency (already passing)
- **Additional test for argmin** if needed
- **10+ test scenarios** (argmax × 10 examples)
- **Clear validation** of index correctness

### Key Discoveries:
- Research recommendation (line 508-516): Create separate test for argmax
- Argmax already in REDUCTION_OPERATIONS registry (strategies.py:285-292)
- Strategy uses `tensor_with_axis_strategy(allow_none=False)` (requires explicit axis)
- Manual test covers both argmax and argmin (test_math.py:133-153)

## What We're NOT Testing/Implementing

- Not testing argmax without axis (not meaningful for ONNX)
- Not testing keepdims variations (simple behavior)
- Not testing argmin separately (can be combined with argmax)
- Not modifying ONNX backend implementation (only tests)

## TDD Approach

### Test Design Philosophy:
- Dedicated test highlights argmax's unique behavior (returns indices)
- Test clearly validates index correctness (not just numerical values)
- Assertion messages distinguish between index and value errors
- Can remain in reduction operations test too (consistency check)

---

## Phase 1: Test Design & Implementation

### Overview
Write a dedicated property-based test for argmax (and optionally argmin) operations.

### Test Categories:

#### 1. Argmax Operation Test

##### Test: `test_argmax_operation_correctness`
**Purpose**: Property test specifically for argmax operation
**Test Data**: Tensors with explicit axis for reduction
**Expected Behavior**: Correct indices of maximum values
**Assertions**: Index correctness, int64 dtype, ONNX node type

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_argmax_operation_correctness(data):
    """
    Property test: Argmax operation returns correct indices.

    This test verifies:
    - Argmax returns indices of maximum values along axis
    - Output dtype is int64 (indices, not values)
    - ONNX output matches Python reference
    - Correct ONNX node type (ArgMax)
    - Works with various tensor shapes and axes

    Note: Argmax requires explicit axis (cannot reduce over all axes)
    """
    op_config = REDUCTION_OPERATIONS['argmax']

    # Generate test data (tensor and axis)
    test_data = data.draw(op_config['strategy'])
    x_val, axis = test_data

    # Verify axis is not None (argmax requires explicit axis)
    assert axis is not None, \
        "Argmax requires explicit axis parameter"

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, axis)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Verify output dtype is int64 (indices, not values)
    assert result.dtype == np.int64, \
        f"Argmax should return int64 indices, got {result.dtype}"

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    assert 'ArgMax' in node_types, \
        f"Expected 'ArgMax' node, got {node_types}"

    # Additional validation: verify indices are within valid range
    assert np.all(result >= 0), \
        "Indices should be non-negative"
    assert np.all(result < x_val.shape[axis]), \
        f"Indices should be less than dimension size {x_val.shape[axis]}"

    # Verify correctness: check that result points to maximum values
    # For each index in result, verify it points to the max value
    expected_result = np.argmax(x_val, axis=axis)
    np.testing.assert_array_equal(result, expected_result)
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal (indices mismatch) OR dtype mismatch
- Points to: Argmax implementation

##### Test: `test_argmin_operation_correctness`
**Purpose**: Property test specifically for argmin operation
**Test Data**: Tensors with explicit axis
**Expected Behavior**: Correct indices of minimum values
**Assertions**: Index correctness, int64 dtype

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_argmin_operation_correctness(data):
    """
    Property test: Argmin operation returns correct indices.

    This test verifies:
    - Argmin returns indices of minimum values along axis
    - Output dtype is int64 (indices, not values)
    - ONNX output matches Python reference
    - Correct ONNX node pattern (Neg + ArgMax or ArgMin)

    Note: Argmin may be implemented as argmax of negated input
    """
    op_config = REDUCTION_OPERATIONS['argmin']

    # Generate test data (tensor and axis)
    test_data = data.draw(op_config['strategy'])
    x_val, axis = test_data

    # Verify axis is not None
    assert axis is not None, \
        "Argmin requires explicit axis parameter"

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, axis)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val])

    # Verify output dtype is int64
    assert result.dtype == np.int64, \
        f"Argmin should return int64 indices, got {result.dtype}"

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    # Argmin may be implemented as -argmax(-x)
    assert 'ArgMax' in node_types or 'ArgMin' in node_types, \
        f"Expected 'ArgMax' or 'ArgMin' node, got {node_types}"

    # Additional validation: verify indices are within valid range
    assert np.all(result >= 0), \
        "Indices should be non-negative"
    assert np.all(result < x_val.shape[axis]), \
        f"Indices should be less than dimension size {x_val.shape[axis]}"

    # Verify correctness
    expected_result = np.argmin(x_val, axis=axis)
    np.testing.assert_array_equal(result, expected_result)
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal (indices mismatch)
- Points to: Argmin implementation

#### 2. Argmax with Keepdims (Optional)

##### Test: `test_argmax_keepdims_correctness`
**Purpose**: Property test for argmax with keepdims parameter
**Test Data**: Tensors with axis and keepdims=True
**Expected Behavior**: Output shape preserves reduced dimension (size 1)
**Assertions**: Shape correctness, index correctness

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_argmax_keepdims_correctness(data):
    """
    Property test: Argmax with keepdims preserves dimension.

    This test verifies:
    - Argmax with keepdims=True preserves reduced dimension
    - Output shape has size 1 along reduced axis
    - Indices still correct
    - ONNX output matches Python reference
    """
    # Generate test data
    shape = data.draw(array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=10))
    x_val = data.draw(arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)
    ))
    axis = data.draw(st.integers(0, len(shape) - 1))

    # Build graph with keepdims=True
    x = pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim)
    y = pt.argmax(x, axis=axis, keepdims=True)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Verify shape with keepdims
    expected_shape = list(x_val.shape)
    expected_shape[axis] = 1
    assert result.shape == tuple(expected_shape), \
        f"Expected shape {tuple(expected_shape)}, got {result.shape}"

    # Verify correctness (squeeze to compare with numpy)
    expected_result = np.argmax(x_val, axis=axis, keepdims=True)
    np.testing.assert_array_equal(result, expected_result)
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Shape mismatch or arrays not equal
- Points to: Argmax keepdims implementation

### Test Implementation Steps:

1. **Add to existing test file**: `tests/link/onnx/test_math.py`

2. **Add imports** (if not already present):
   ```python
   from hypothesis import given, strategies as st, settings
   from hypothesis.extra.numpy import arrays, array_shapes
   from tests.link.onnx.strategies import REDUCTION_OPERATIONS
   ```

3. **Add new property tests** after existing `test_reduction_operations_correctness`

4. **Add section comment**:
   ```python
   # ============================================================================
   # PROPERTY-BASED TESTS - Argmax/Argmin (Separate from Value Reductions)
   # ============================================================================
   ```

5. **Implement the argmax and argmin property tests** as specified above

6. **Keep existing manual tests** for reference and specific patterns

### Success Criteria:

#### Automated Verification:
- [ ] Test functions created with proper structure
- [ ] Tests use REDUCTION_OPERATIONS registry correctly
- [ ] Tests are discoverable: `uv run pytest --collect-only tests/link/onnx/test_math.py`
- [ ] Test code follows project conventions: `make lint-tests`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertions validate index correctness (not just values)
- [ ] Docstrings explain why argmax is tested separately

---

## Phase 2: Test Failure Verification

### Overview
Run the new argmax property tests and verify they work correctly.

### Verification Steps:

1. **Run the test suite**:
   ```bash
   uv run pytest tests/link/onnx/test_math.py::test_argmax_operation_correctness -v
   uv run pytest tests/link/onnx/test_math.py::test_argmin_operation_correctness -v
   ```

2. **For each test, verify**:
   - Test runs without collection errors
   - Test either passes or fails with clear message
   - Failure messages distinguish index vs value errors
   - Hypothesis shows minimal failing examples

3. **Document outcomes**:
   - Whether argmax/argmin pass
   - Any edge cases discovered
   - Comparison with existing reduction test results

### Expected Outcomes:

**Scenario 1: All tests pass**
- Argmax/argmin are well-implemented
- Property tests validate existing functionality
- Proceed to Phase 4 (refactoring)

**Scenario 2: Tests fail**
- Argmax/argmin have bugs
- Hypothesis shows minimal failing examples
- Document issues for Phase 3

**Scenario 3: Tests redundant with existing**
- New tests don't provide additional value
- Consider keeping only one approach
- Document decision

### Expected Test Behavior:

- **test_argmax_operation_correctness**: Should pass (already tested in reduction operations)
- **test_argmin_operation_correctness**: Should pass (already tested manually)
- **test_argmax_keepdims_correctness** (if implemented): May reveal keepdims issues

### Success Criteria:

#### Automated Verification:
- [ ] All tests run without collection errors
- [ ] Tests complete execution (10 examples each)
- [ ] No import or strategy errors

#### Manual Verification:
- [ ] Test failures (if any) are informative
- [ ] Can identify axis causing failure
- [ ] Hypothesis shrinking provides minimal examples
- [ ] Index errors are clearly distinguished from value errors

### Adjustment Phase:

If tests don't run properly:
- [ ] Fix registry access
- [ ] Fix strategy usage (axis handling)
- [ ] Adjust assertions
- [ ] Improve error messages

---

## Phase 3: Implementation / Bug Fixes (If Needed)

### Overview
Fix any implementation bugs revealed by property tests. Skip this phase if all tests pass.

### Implementation Strategy:

**Only proceed if Phase 2 revealed bugs**

Given that argmax is already tested in reduction operations test, bugs are unlikely. If found:

**Order of fixes:**
1. Argmax axis handling
2. Argmin implementation (may use -argmax(-x))
3. Keepdims behavior

### Implementation Steps:

#### Fix 1: Argmax Axis Issues

**Symptom**: Argmax returns wrong indices for certain axes
**Location**: pytensor/link/onnx/dispatch/math.py:94

**Debugging Approach**:
1. Hypothesis shows minimal failing example (tensor and axis)
2. Check ONNX ArgMax node generation
3. Verify axis parameter passed correctly
4. Test fix with property test

#### Fix 2: Argmin Implementation

**Symptom**: Argmin returns wrong indices
**Location**: pytensor/link/onnx/dispatch/math.py (if separate implementation)

**Debugging Approach**:
1. Check if argmin uses -argmax(-x) pattern
2. Verify negation doesn't affect index computation
3. Test with minimal example
4. Validate with property test

**Not providing specific fixes** - bugs are unlikely given existing tests pass

### Success Criteria:

#### Automated Verification:
- [ ] All property tests pass: `uv run pytest tests/link/onnx/test_math.py -k "argm" -v`
- [ ] No regressions in reduction operations test
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
   - Group argmax tests together
   - Add section comment explaining separation from reductions
   - Organize by complexity (basic, then keepdims)

2. **Evaluate Redundancy**:
   - Determine if argmax in reduction test is still needed
   - Consider keeping both (consistency check + focused test)
   - Document rationale

3. **Documentation**:
   - Add comments explaining why argmax tested separately
   - Document unique characteristics (indices vs values)
   - Update module docstring

### Refactoring Steps:

1. **Ensure all tests pass**: `uv run pytest tests/link/onnx/test_math.py -v`

2. **Organize argmax tests**:
   ```python
   # ============================================================================
   # PROPERTY-BASED TESTS - Reductions (Value-Based)
   # ============================================================================

   @given(...)
   def test_reduction_operations_correctness(...):
       """
       Property test for value-based reductions.
       Note: Argmax/argmin also tested here for consistency with other reductions.
       """
       ...

   # ============================================================================
   # PROPERTY-BASED TESTS - Argmax/Argmin (Index-Based Reductions)
   # ============================================================================

   @given(...)
   def test_argmax_operation_correctness(...):
       """
       Dedicated property test for argmax.

       Argmax tested separately because:
       - Returns indices (int64), not values (float32)
       - Has unique validation requirements (index bounds)
       - Different failure modes than value reductions
       """
       ...
   ```

3. **Update module docstring**:
   ```python
   """
   Tests for ONNX math operations (reductions).

   Test Strategy:
   - Property-based tests for value reductions (sum, prod, max, min)
   - Separate property tests for index reductions (argmax, argmin)
   - Manual tests for edge cases (keepdims, multiple axes, etc.)

   Coverage: 8 reduction operations + argmax/argmin
   """
   ```

4. **Decide on redundancy**:
   - **Option A**: Keep argmax in both tests (consistency + focused validation)
   - **Option B**: Remove argmax from reduction test (avoid duplication)
   - **Recommendation**: Keep in both - small overhead, provides consistency check

   **Rationale for keeping argmax in both tests**:
   - **Consistency check**: If argmax passes in reduction test but fails in dedicated test (or vice versa), it indicates a test infrastructure issue
   - **Different validation**: Reduction test validates argmax behaves like other reductions; dedicated test validates index-specific behavior
   - **Low cost**: 10 extra examples is negligible overhead (~1 second)
   - **Documentation**: Having both tests clearly signals that argmax has dual nature (reduction + index operation)
   - **Regression protection**: If someone accidentally breaks index handling, both tests catch it

5. **Consider consolidating manual tests**:
   - test_argmax_argmin → Covered by property tests (can remove)
   - Keep if it tests unique patterns not in property test

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] No test failures introduced
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Code is more organized and readable
- [ ] Clear explanation for separate argmax tests
- [ ] No important coverage lost
- [ ] Decision on redundancy documented

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Argmax tested separately from value reductions
- [ ] 10+ test scenarios for argmax
- [ ] Optional: 10+ scenarios for argmin
- [ ] Optional: Keepdims variations tested
- [ ] Index correctness validated (not just values)
- [ ] Dtype correctness validated (int64 output)

### Test Organization:
- Dedicated property test: test_argmax_operation_correctness
- Optional dedicated test: test_argmin_operation_correctness
- Optional keepdims test: test_argmax_keepdims_correctness
- Existing coverage: argmax in test_reduction_operations_correctness (for consistency)
- Manual tests: test_argmax_argmin (may be redundant)

### Running Tests:

```bash
# Run all math tests
uv run pytest tests/link/onnx/test_math.py -v

# Run only argmax/argmin property tests
uv run pytest tests/link/onnx/test_math.py -k "argm" -v

# Run specific test
uv run pytest tests/link/onnx/test_math.py::test_argmax_operation_correctness -v

# Run with Hypothesis verbose output
uv run pytest tests/link/onnx/test_math.py::test_argmax_operation_correctness -v --hypothesis-show-statistics
```

## Performance Considerations

- Argmax property tests generate small tensors (same as reduction tests)
- Argmax is fast (single pass through data)
- Full suite should complete in seconds
- No performance concerns

## Migration Notes

### Tests to Keep:
- test_reduction_operations_correctness (includes argmax for consistency)
- New test_argmax_operation_correctness (dedicated validation)
- test_argmax_argmin (if it tests patterns not in property tests)

### Tests to Consider Removing:
- test_argmax_argmin → Covered by property tests (can remove if redundant)

### Decision Points:
1. **Keep argmax in reduction test?**
   - Recommendation: Yes (consistency check)
2. **Test argmin separately?**
   - Recommendation: Yes (similar to argmax, worth dedicated test)
3. **Test keepdims?**
   - Recommendation: Optional (can add if needed)

## References

- Original research: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md:508-516`
- REDUCTION_OPERATIONS registry: `tests/link/onnx/strategies.py:285-302`
- Test utilities: `tests/link/onnx/test_basic.py:30`
- Argmax dispatcher: `pytensor/link/onnx/dispatch/math.py:94`
- Existing reduction tests: `tests/link/onnx/test_math.py:23-49`
- Existing argmax manual test: `tests/link/onnx/test_math.py:133-153`
