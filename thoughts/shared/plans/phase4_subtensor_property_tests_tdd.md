# Phase 4: Subtensor Operations Property-Based Tests TDD Implementation Plan

## Overview

Create individual property-based tests for 4 subtensor operations (slicing and indexing) using strategies from the `SUBTENSOR_OPERATIONS` and `INCSUBTENSOR_OPERATIONS` registries. Subtensor operations have complex constraints and edge cases requiring careful test design.

## Current State Analysis

### Current Testing Landscape:
- Testing framework: pytest with Hypothesis
- Test utilities: `compare_onnx_and_py()` and `get_onnx_node_types()` at tests/link/onnx/test_basic.py
- Registries: `SUBTENSOR_OPERATIONS` and `INCSUBTENSOR_OPERATIONS` in tests/link/onnx/strategies.py:345-407
- Test pattern: Individual tests per operation (due to complexity)

### Current Subtensor Tests:
- 14 tests in tests/link/onnx/test_subtensor.py across 3 test classes
- **TestSubtensorBasic** (9 tests): Basic slicing patterns (1D, 2D, 3D, with step)
- **TestSubtensorNegativeIndices** (2 tests, SKIPPED): Negative indices not implemented
- **TestAdvancedSubtensor** (2 tests): Integer array indexing
- **TestIncSubtensor** (2 tests): set_subtensor and inc_subtensor

### Known Limitations:
- **Negative indices NOT supported** (research doc lines 666-670, test_subtensor.py:115-137)
- Documentation at pytensor/link/onnx/dispatch/subtensor.py:122-127 confirms limitation
- Research design decision #3: Don't test negative indices in property tests

### Subtensor Operations Characteristics:
- **Complex constraints**: Slice bounds, valid indices, shape compatibility
- **Multiple patterns**: Basic slicing, advanced indexing, set/inc operations
- **Edge cases**: Empty slices, out-of-bounds (should error), step values
- **Multi-input operations**: set_subtensor and inc_subtensor take values to insert

## Desired End State

A comprehensive property-based test suite with:
- **4 individual property test functions** (one per operation type)
- **Retained manual tests** for specific patterns and edge cases
- **40+ test scenarios** (4 operations × 10 examples minimum)
- **Clear validation** for slicing correctness and index handling

### Key Discoveries:
- Research design decision #3 (lines 666-670): Exclude negative indices from property tests
- Existing strategies in strategies.py:348-386 are basic patterns
- Manual tests cover good variety (1D, 2D, 3D, with step)
- Advanced indexing uses integer arrays (AdvancedSubtensor1, AdvancedSubtensor)

## What We're NOT Testing/Implementing

- **Not testing negative indices** (known limitation, documented in subtensor.py:122-127)
- Not testing out-of-bounds access (should error, not normal behavior)
- Not testing all possible slicing patterns (focus on common ones)
- Not testing dynamic bounds (runtime-determined slice indices)
- Not modifying ONNX backend implementation (only tests)

## TDD Approach

### Test Design Philosophy:
- Each operation type gets its own property test
- Property tests generate valid slices/indices only
- Test failures clearly indicate which slicing pattern failed
- Validate both numerical correctness and shape transformations
- Explicitly exclude unsupported features (negative indices)

---

## Phase 1: Test Design & Implementation

### Overview
Write individual property-based tests for each subtensor operation using existing registries and strategies.

### Test Categories:

#### 1. Basic Slicing Operations

##### Test: `test_subtensor_basic_slicing_correctness`
**Purpose**: Property test for basic Subtensor operation (slicing)
**Test Data**: Tensors with valid slice patterns
**Expected Behavior**: Correct slicing results
**Assertions**: Numerical correctness, shape validation

```python
@given(
    op_name=st.sampled_from(['slice_basic', 'slice_multidim', 'slice_with_step']),
    data=st.data(),
)
@settings(max_examples=20, deadline=None)  # Higher count for slicing edge cases
def test_subtensor_basic_slicing_correctness(op_name, data):
    """
    Property test: Basic subtensor slicing operations produce correct results.

    This test verifies:
    - Basic slicing (x[2:5]) works correctly
    - Multi-dimensional slicing (x[1:3, 2:4]) works correctly
    - Slicing with step (x[::2], x[1:8:2]) works correctly
    - ONNX output matches Python reference
    - Correct ONNX node type (Slice)

    Operations tested: slice_basic, slice_multidim, slice_with_step
    Total: 3 patterns × 10 examples = 30 test scenarios

    Note: This test does NOT cover negative indices (not yet supported in ONNX backend)
    """
    op_config = SUBTENSOR_OPERATIONS[op_name]

    # Generate test data (tensor with valid size for slicing)
    test_data = data.draw(op_config['strategy'])

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](test_data)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [test_data])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"{op_name}: Expected one of {expected_ops}, got {node_types}"

    # Additional validation: verify result shape is reasonable
    assert result.ndim <= test_data.ndim, \
        f"Result should not have more dimensions than input"
    assert result.size <= test_data.size, \
        f"Slice result should not be larger than input"
```

**Expected Failure Mode**:
- Error type: AssertionError from array comparison
- Expected message: Arrays not equal
- Points to: Subtensor/Slice implementation

#### 2. Advanced Indexing Operations

##### Test: `test_advanced_subtensor_indexing_correctness`
**Purpose**: Property test for AdvancedSubtensor (integer array indexing)
**Test Data**: Tensors with integer index arrays
**Expected Behavior**: Correct indexed selection
**Assertions**: Numerical correctness, Gather node

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_advanced_subtensor_indexing_correctness(data):
    """
    Property test: Advanced subtensor indexing produces correct results.

    This test verifies:
    - Integer array indexing (x[indices]) works correctly
    - Selected elements match Python reference
    - ONNX output matches PyTensor
    - Correct ONNX node type (Gather)

    Note: Uses advanced_index_strategy to generate valid indices
          (all indices are non-negative and within bounds)
    """
    op_config = SUBTENSOR_OPERATIONS['advanced_index']

    # Generate test data (tensor and valid integer indices)
    test_data = data.draw(op_config['strategy'])
    x_val, indices_val = test_data

    # Verify indices are valid (strategy constraint)
    assert np.all(indices_val >= 0), \
        "Indices should be non-negative (negative indices not supported)"
    assert np.all(indices_val < x_val.shape[0]), \
        "Indices should be within bounds"

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, indices_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, indices_val])

    # Verify ONNX node type
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"Expected one of {expected_ops}, got {node_types}"

    # Validate result shape
    expected_shape = (indices_val.shape[0],) + x_val.shape[1:]
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal or shape mismatch
- Points to: AdvancedSubtensor/Gather implementation

#### 3. Set Subtensor Operations

##### Test: `test_set_subtensor_operation_correctness`
**Purpose**: Property test for set_subtensor (x[2:5] = values)
**Test Data**: Tensors with slice and replacement values
**Expected Behavior**: Correct value replacement
**Assertions**: Numerical correctness, ScatterElements node

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_set_subtensor_operation_correctness(data):
    """
    Property test: set_subtensor correctly replaces slice with values.

    This test verifies:
    - set_subtensor replaces slice with provided values
    - Other elements remain unchanged
    - ONNX output matches PyTensor
    - Correct ONNX node types (ScatterElements/ScatterND)

    Note: Uses set_subtensor_strategy to generate compatible shapes
    """
    op_config = INCSUBTENSOR_OPERATIONS['set_subtensor']

    # Generate test data (tensor and replacement values)
    test_data = data.draw(op_config['strategy'])
    x_val, values_val = test_data

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, values_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, values_val])

    # Verify ONNX node types
    node_types = get_onnx_node_types(fn)
    expected_ops = op_config['expected_onnx_ops']
    assert any(op in node_types for op in expected_ops), \
        f"Expected one of {expected_ops}, got {node_types}"

    # Use Hypothesis assume() to filter edge case where new values equal old
    # This avoids false failures when values_val happens to equal x_val[2:5]
    from hypothesis import assume
    assume(not np.array_equal(values_val, x_val[2:5]))

    # Validate that slice was modified
    # (This assertion is now guaranteed to be meaningful)
    assert not np.array_equal(result[2:5], x_val[2:5]), \
        "Slice should have been modified"

    # Validate that values were set correctly
    np.testing.assert_array_equal(result[2:5], values_val)

    # Validate that other elements unchanged
    np.testing.assert_array_equal(result[:2], x_val[:2])
    np.testing.assert_array_equal(result[5:], x_val[5:])
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal (slice not set correctly)
- Points to: IncSubtensor/ScatterElements implementation

##### Test: `test_inc_subtensor_operation_correctness`
**Purpose**: Property test for inc_subtensor (x[2:5] += values)
**Test Data**: Tensors with slice and increment values
**Expected Behavior**: Correct value increment
**Assertions**: Numerical correctness, Add + ScatterElements nodes

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_inc_subtensor_operation_correctness(data):
    """
    Property test: inc_subtensor correctly increments slice values.

    This test verifies:
    - inc_subtensor adds values to existing slice
    - Other elements remain unchanged
    - ONNX output matches PyTensor
    - Correct ONNX node types (Gather, Add, ScatterElements)

    Note: inc_subtensor is more complex than set_subtensor
          (requires gather, add, then scatter)
    """
    op_config = INCSUBTENSOR_OPERATIONS['inc_subtensor']

    # Generate test data (tensor and increment values)
    test_data = data.draw(op_config['strategy'])
    x_val, values_val = test_data

    # Build graph
    graph_inputs, graph_output = op_config['build_graph'](x_val, values_val)

    # Compare ONNX vs PyTensor
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, [x_val, values_val])

    # Verify ONNX node types (should include Gather, Add, ScatterElements)
    node_types = get_onnx_node_types(fn)
    # Note: inc_subtensor requires multiple operations
    assert 'Gather' in node_types or 'Slice' in node_types, \
        "Expected gather/slice operation"
    assert 'Add' in node_types, \
        "Expected Add operation (for increment)"
    assert 'ScatterElements' in node_types or 'ScatterND' in node_types, \
        "Expected scatter operation"

    # Use Hypothesis assume() to filter edge case where increment values are zero
    # This avoids false failures when values_val is all zeros
    from hypothesis import assume
    assume(not np.allclose(values_val, 0))

    # Validate that slice was modified
    # (This assertion is now guaranteed to be meaningful)
    assert not np.array_equal(result[2:5], x_val[2:5]), \
        "Slice should have been modified"

    # Validate that values were incremented correctly
    expected_slice = x_val[2:5] + values_val
    np.testing.assert_allclose(result[2:5], expected_slice, rtol=1e-5)

    # Validate that other elements unchanged
    np.testing.assert_array_equal(result[:2], x_val[:2])
    np.testing.assert_array_equal(result[5:], x_val[5:])
```

**Expected Failure Mode**:
- Error type: AssertionError
- Expected message: Arrays not equal (increment not applied correctly)
- Points to: IncSubtensor increment implementation

### Test Implementation Steps:

1. **Modify existing test file**: `tests/link/onnx/test_subtensor.py`

2. **Add imports at top of file**:
   ```python
   from hypothesis import given, strategies as st, settings
   from functools import partial
   from tests.link.onnx.strategies import SUBTENSOR_OPERATIONS, INCSUBTENSOR_OPERATIONS
   ```

3. **Add property test section before existing classes**:
   ```python
   # ============================================================================
   # PROPERTY-BASED TESTS (Primary Coverage)
   # ============================================================================
   ```

4. **Implement each property test** as specified above

5. **Keep existing manual test classes** for specific patterns and edge cases

### Success Criteria:

#### Automated Verification:
- [x] All test functions created with proper structure
- [x] Tests use registries correctly
- [x] Tests are discoverable: `uv run pytest --collect-only tests/link/onnx/test_subtensor.py`
- [ ] Test code follows project conventions: `make lint-tests`

#### Manual Verification:
- [x] Each test has clear, informative docstring
- [x] Test names clearly describe what they test
- [x] Negative indices explicitly excluded (documented in comments)
- [x] Assertion messages are diagnostic

---

## Phase 2: Test Failure Verification

### Overview
Run the property tests and verify they work correctly or expose any implementation issues.

### Verification Steps:

1. **Run the test suite**:
   ```bash
   uv run pytest tests/link/onnx/test_subtensor.py -k "correctness" -v
   ```

2. **For each test, verify**:
   - Test runs without collection errors
   - Test either passes or fails with clear message
   - Failure messages show which slice pattern failed
   - Hypothesis shows minimal failing example

3. **Document outcomes**:
   - Which slicing patterns pass
   - Which patterns have issues
   - Any edge cases discovered

### Expected Outcomes:

**Scenario 1: All tests pass**
- Subtensor operations are well-implemented
- Property tests validate existing functionality
- Proceed to Phase 4 (refactoring)

**Scenario 2: Some tests fail**
- Specific slicing patterns have bugs
- Hypothesis shows minimal failing examples
- Document issues for Phase 3

**Scenario 3: Test infrastructure issues**
- Registry or strategy problems
- Fix in strategies.py

### Expected Test Behavior:

- **test_subtensor_basic_slicing_correctness**: Should pass (slicing is basic)
- **test_advanced_subtensor_indexing_correctness**: Should pass (already tested manually)
- **test_set_subtensor_operation_correctness**: May reveal edge cases
- **test_inc_subtensor_operation_correctness**: More complex, may reveal issues

### Success Criteria:

#### Automated Verification:
- [x] All tests run without collection errors
- [x] Tests complete execution (10 examples each)
- [x] No import or strategy errors

#### Manual Verification:
- [x] Test failures (if any) are informative
- [x] Can identify slice pattern causing failure
- [x] Hypothesis shrinking provides minimal examples
- [x] No confusing error messages

**Result**: All tests pass! No bugs found in the ONNX subtensor implementation.

### Adjustment Phase:

If tests don't run properly:
- [ ] Fix registry access
- [ ] Fix strategy usage
- [ ] Adjust slice validation
- [ ] Improve error messages

---

## Phase 3: Implementation / Bug Fixes (If Needed)

### Overview
Fix any implementation bugs revealed by property tests. Skip this phase if all tests pass.

### Implementation Strategy:

**Only proceed if Phase 2 revealed bugs**

**Order of fixes:**
1. Basic slicing issues (most fundamental)
2. Advanced indexing bugs
3. Set subtensor problems
4. Inc subtensor issues (most complex)

### Implementation Steps:

#### Fix 1: Basic Slicing Edge Cases

**Symptom**: Slicing fails with certain patterns
**Location**: pytensor/link/onnx/dispatch/subtensor.py:12

**Debugging Approach**:
1. Hypothesis shows minimal failing slice
2. Check slice bounds calculation
3. Verify ONNX Slice node generation
4. Test fix with property test

#### Fix 2: Advanced Indexing Issues

**Symptom**: Integer array indexing produces wrong results
**Location**: pytensor/link/onnx/dispatch/subtensor.py:191

**Debugging Approach**:
1. Check index array handling
2. Verify ONNX Gather operation
3. Test with minimal example
4. Validate with property test

#### Fix 3: Set/Inc Subtensor Problems

**Symptom**: Values not set/incremented correctly
**Location**: pytensor/link/onnx/dispatch/subtensor.py:235

**Debugging Approach**:
1. Check ScatterElements generation
2. Verify index calculation for scatter
3. For inc_subtensor, check gather-add-scatter pipeline
4. Test with minimal example

**Not providing specific fixes** - depends on what tests reveal

### Success Criteria:

#### Automated Verification:
- [ ] All property tests pass: `uv run pytest tests/link/onnx/test_subtensor.py -k "correctness" -v`
- [ ] No regressions in existing tests
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Fixes are minimal and targeted
- [ ] Code comments explain edge cases
- [ ] No workarounds, proper solutions only

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor test code for clarity and organization.

### Refactoring Targets:

1. **Test Organization**:
   - Group property tests at top
   - Keep manual test classes below
   - Add section comments

2. **Consolidate Manual Tests**:
   - Identify tests covered by property tests
   - Keep unique edge case tests
   - Document retention rationale

3. **Documentation**:
   - Add module docstring explaining test strategy
   - Document negative index limitation
   - Explain property vs manual test split

### Refactoring Steps:

1. **Ensure all tests pass**: `uv run pytest tests/link/onnx/test_subtensor.py -v`

2. **Reorganize file**:
   ```python
   """
   Tests for ONNX subtensor (slicing and indexing) operations.

   Test Strategy:
   - Property-based tests provide primary coverage (40+ scenarios)
   - Individual property test per operation type (4 operations)
   - Manual tests retained for specific patterns and edge cases

   Operations: Subtensor (slicing), AdvancedSubtensor (integer indexing),
               set_subtensor, inc_subtensor

   Known Limitations:
   - Negative indices NOT supported (limitation documented in subtensor.py:122-127)
   - Property tests explicitly exclude negative indices
   - Manual tests for negative indices are skipped (will be enabled when supported)
   """

   # ============================================================================
   # PROPERTY-BASED TESTS (Primary Coverage)
   # ============================================================================

   @given(...)
   def test_subtensor_basic_slicing_correctness(...):
       """
       Property test for basic slicing.
       Note: Does NOT test negative indices (not yet supported).
       """
       ...

   # ============================================================================
   # MANUAL EDGE CASE TESTS
   # ============================================================================

   class TestSubtensorBasic:
       """Test specific slicing patterns."""
       # Keep a few representative tests
       ...

   class TestSubtensorNegativeIndices:
       """Tests for negative indices (currently skipped)."""
       # Keep these skipped tests as documentation of known limitation
       ...
   ```

3. **Consider consolidating TestSubtensorBasic**:
   - test_slice_1d_basic → Covered by property test (can remove)
   - test_slice_1d_with_step → Covered by property test (can remove)
   - test_slice_2d_basic → Covered by property test (can remove)
   - Keep test_slice_3d → Good example of 3D slicing
   - Keep TestSubtensorNegativeIndices → Documents known limitation

4. **Add helpful comments**:
   - Explain why negative index tests are skipped
   - Reference limitation documentation
   - Note when feature might be implemented

5. **Add @pytest.mark.xfail tests for negative indices** (optional but recommended):
   ```python
   @pytest.mark.xfail(reason="Negative indices not yet supported in ONNX backend - see subtensor.py:122-127")
   def test_slice_negative_indices_future():
       """
       Test for negative indices - currently expected to fail.

       This test documents the expected behavior once negative indices
       are implemented. Remove @pytest.mark.xfail when feature is ready.

       See: pytensor/link/onnx/dispatch/subtensor.py:122-127 for limitation docs
       GitHub Issue: [link to issue tracking negative index support]
       """
       x_val = np.array([1, 2, 3, 4, 5], dtype='float32')
       x = pt.tensor('x', dtype='float32', shape=(None,))
       y = x[-2:]  # Should return [4, 5]

       fn, result = compare_onnx_and_py([x], y, [x_val])
       np.testing.assert_array_equal(result, np.array([4, 5], dtype='float32'))
   ```

   Benefits of xfail tests:
   - Documents expected behavior for future implementation
   - Provides ready-made test when feature is implemented
   - Tracks known limitations in test suite
   - Can link to GitHub issues for tracking

### Success Criteria:

#### Automated Verification:
- [x] All tests still pass (16 passed, 2 skipped)
- [x] Test count appropriate (kept all manual tests for documentation)
- [ ] Linting passes: `make lint` (no Makefile in project)

#### Manual Verification:
- [x] Code is more organized and readable
- [x] Limitation clearly documented in test class docstrings
- [x] No important coverage lost (all manual tests retained)

---

## Testing Strategy Summary

### Test Coverage Goals:
- [x] 4 subtensor operations covered by property tests
- [x] 50 test scenarios (3 slice ops × 20 examples + advanced indexing × 10 + set/inc × 10 each)
- [x] Basic slicing patterns validated
- [x] Advanced indexing tested
- [x] Set/inc subtensor operations verified
- [x] Negative indices explicitly excluded (documented limitation)

### Test Organization:
- Property tests: Primary coverage for supported operations
- Manual tests: Specific patterns, edge cases, and documentation of limitations
- Test utilities: compare_onnx_and_py, get_onnx_node_types

### Running Tests:

```bash
# Run all subtensor tests
uv run pytest tests/link/onnx/test_subtensor.py -v

# Run only property tests
uv run pytest tests/link/onnx/test_subtensor.py -k "correctness" -v

# Run specific operation test
uv run pytest tests/link/onnx/test_subtensor.py::test_set_subtensor_operation_correctness -v

# Run manual test classes
uv run pytest tests/link/onnx/test_subtensor.py::TestSubtensorBasic -v

# Run with Hypothesis verbose output
uv run pytest tests/link/onnx/test_subtensor.py -k "correctness" -v --hypothesis-show-statistics
```

## Performance Considerations

- Property tests generate small tensors (10-20 elements typical)
- Slicing operations are fast
- Set/inc subtensor slightly slower (multiple ONNX nodes)
- Full suite should complete in seconds

## Migration Notes

### Tests to Keep:
- test_slice_3d (good example of 3D slicing)
- TestSubtensorNegativeIndices (documents known limitation)
- TestIncSubtensor (documents expected ONNX node patterns)

### Tests to Consider Removing:
- test_slice_1d_basic (covered by property test)
- test_slice_1d_from_start (covered by property test)
- test_slice_1d_to_end (covered by property test)
- test_slice_1d_with_step (covered by property test)
- test_slice_1d_with_step_range (covered by property test)
- test_slice_2d_basic (covered by property test)
- test_slice_2d_one_axis (covered by property test)
- test_integer_array_indexing (covered by property test)
- test_integer_array_indexing_2d (covered by property test)

## References

- Original research: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md:416-453`
- Research design decision #3: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md:666-676`
- SUBTENSOR_OPERATIONS registry: `tests/link/onnx/strategies.py:348-386`
- INCSUBTENSOR_OPERATIONS registry: `tests/link/onnx/strategies.py:393-407`
- Test utilities: `tests/link/onnx/test_basic.py:30`
- Subtensor dispatchers: `pytensor/link/onnx/dispatch/subtensor.py`
- Negative index limitation: `pytensor/link/onnx/dispatch/subtensor.py:122-127`
- Existing subtensor tests: `tests/link/onnx/test_subtensor.py`

---

## Post-Implementation Analysis

**Date**: 2025-11-11 (Same day as implementation)
**Analyzed by**: clsandoval
**Implementation Period**: 2025-11-11 (single session implementation)
**Status**: Implementation completed successfully, not yet committed

### What Worked As Planned

- **Phase 1: Test Design & Implementation** - All 4 property-based tests were created exactly as specified in the plan (tests/link/onnx/test_subtensor.py:35-225)
- **Phase 2: Test Verification** - All tests passed on first attempt after registry fix, with no ONNX backend bugs found
- **Phase 4: Refactoring** - Documentation and test organization completed as planned
- **Test Coverage** - Achieved 50 test scenarios (exceeding the 40+ goal): 60 basic slicing + 10 advanced indexing + 10 set_subtensor + 10 inc_subtensor
- **Module Documentation** - Comprehensive docstrings added to all test classes as planned (test_subtensor.py:1-15, 237-245, 343-351, 379-383, 421-429)

### Divergences from Plan

#### Tests

**Issue 1: Registry Design Mismatch**
- **Planned**: Plan assumed registries would work directly with test code as written (lines 103-124)
- **Actual**: Initial test failures revealed registries expected PyTensor variables but received numpy arrays
- **Files**: `tests/link/onnx/strategies.py:460-518`
- **Root Cause**: The plan code examples showed `build_graph` being called with `test_data`, but didn't account for the fact that strategies generate numpy arrays while `build_graph` functions needed to create PyTensor symbolic variables
- **Why**: The SUBTENSOR and INCSUBTENSOR registries were inconsistent with the ELEMWISE_OPERATIONS registry pattern (which properly wraps numpy→PyTensor conversion)

**Fix Applied**:
```python
# Before (strategies.py:461-462)
"build_graph": lambda x: ([x], x[2:5]),

# After (strategies.py:461-463)
"build_graph": lambda x_val: (
    lambda x: ([x], x[2:5])
)(pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim)),
```

**Issue 2: Graph Inputs Pattern**
- **Planned**: Plan showed `graph_inputs, graph_output = op_config['build_graph'](test_data)` (line 109)
- **Actual**: Had to adjust for operations with multiple inputs:
  - Basic slicing: Single numpy array input `x_val`
  - Advanced indexing: Tuple input `(x_val, indices_val)`
  - Set/Inc subtensor: Tuple input `(x_val, values_val)`
- **Files**: `tests/link/onnx/test_subtensor.py:59, 98-99, 142, 191`
- **Why**: The plan's code examples didn't show the tuple unpacking needed for multi-input operations

**Issue 3: Advanced Indexing Shape Validation**
- **Planned**: `expected_shape = (indices_val.shape[0],) + x_val.shape[1:]` (line 181)
- **Actual**: `expected_shape = (indices_val.shape[0],)` (test_subtensor.py:120-122)
- **Why**: The strategy generates 1D tensors, so there are no additional dimensions. The plan assumed 2D+ tensors.

#### Implementation

**Issue 1: Registry Pattern Inconsistency**
- **Planned**: Assumed existing SUBTENSOR_OPERATIONS registry would work as-is
- **Actual**: Had to refactor all 4 operations in SUBTENSOR_OPERATIONS and both in INCSUBTENSOR_OPERATIONS
- **Files**: `tests/link/onnx/strategies.py:460-518`
- **Commits**: Not yet committed (working changes)
- **Why**: The registries were created before the ELEMWISE_OPERATIONS pattern was established, leading to inconsistency

**Issue 2: Import Requirements**
- **Planned**: Listed imports as `from functools import partial` (line 330)
- **Actual**: Didn't need `partial`, but needed `assume` from Hypothesis (test_subtensor.py:19)
- **Why**: Plan included unnecessary import from copying pattern from other tests; needed `assume()` for filtering edge cases

**Issue 3: Test Data Variable Naming**
- **Planned**: Used `test_data = data.draw(...)` throughout examples (lines 106, 159, 217, 280)
- **Actual**: Used `x_val = data.draw(...)` for single inputs and tuple unpacking for multi-input cases
- **Why**: More descriptive variable names improve readability and match the registry parameter names

#### Additional Changes

- **No additional files needed** - Implementation stayed within the two files mentioned in plan
- **No unexpected dependencies** - All required tools (Hypothesis, pytest) were already in place
- **Registry graph_inputs return value** - Changed from returning single variable to returning list of variables consistently (strategies.py:499, 518, 530)

### Bugs and Fixes Encountered

#### Bug: AttributeError - numpy.ndarray has no attribute 'owner'
- **Symptom**: `AttributeError: 'numpy.ndarray' object has no attribute 'owner'` when running property tests
- **Root Cause**: Registry `build_graph` functions were receiving numpy arrays but treating them as PyTensor variables directly
- **Fix**: Wrapped registry `build_graph` lambdas to convert numpy arrays to PyTensor symbolic variables
- **Commit**: Not yet committed
- **Plan Gap**: Plan should have included verification that registry patterns matched established ELEMWISE pattern before proceeding with test implementation

#### Bug: TypeError - x must be the result of a subtensor operation
- **Symptom**: `TypeError: x must be the result of a subtensor operation` in set/inc_subtensor tests
- **Root Cause**: PyTensor's `set_subtensor` and `inc_subtensor` require the first argument to be a sliced view (result of subtensor operation), but registries were passing constants
- **Fix**: Changed registry to create proper symbolic graph with `x[2:5]` where `x` is a symbolic variable
- **Commit**: Not yet committed
- **Plan Gap**: Plan didn't research how `set_subtensor`/`inc_subtensor` validate their inputs; should have checked PyTensor source

### Success Criteria Gaps

#### Automated Checks
- [x] All test functions created with proper structure - **PASSED**
- [x] Tests use registries correctly - **PASSED** (after registry fix)
- [x] Tests discoverable - **PASSED** (18 tests collected)
- [ ] Test code follows project conventions - **NOT RUN** (no Makefile or working ruff config in project)

#### Manual Verification
- [x] Clear, informative docstrings - **PASSED**
- [x] Test names clearly describe what they test - **PASSED**
- [x] Negative indices explicitly excluded - **PASSED**
- [x] Diagnostic assertion messages - **PASSED**

#### Additional Success Metrics (Not in Plan)
- [x] All manual tests still pass (16 passed, 2 skipped)
- [x] Hypothesis generates good variety (verified with --hypothesis-show-statistics)
- [x] No test flakiness (41% invalid for inc_subtensor due to zero-filtering is acceptable)

### Lessons Learned

#### For Future Planning

1. **Verify Registry Patterns Before Writing Tests**
   - **What happened**: Assumed registries followed correct pattern, but they were inconsistent
   - **Next time**: Before writing tests, inspect and verify registry patterns match established conventions (especially ELEMWISE_OPERATIONS pattern)
   - **Action**: Add a Phase 0 step: "Verify registry implementation matches expected pattern"

2. **Include Registry Pattern Examples in Plan**
   - **What happened**: Plan showed registry usage but not registry structure
   - **Next time**: Include a section showing how registries should be structured, with examples from existing working registries
   - **Action**: Add "Registry Pattern Reference" section showing correct lambda wrapping pattern

3. **Test the Test Infrastructure First**
   - **What happened**: Wrote all 4 tests before discovering registry issues
   - **Next time**: Write a single minimal test first to verify infrastructure works, then expand
   - **Action**: Modify TDD phases to include "Phase 1a: Infrastructure Validation with Single Test"

4. **Research API Constraints**
   - **What happened**: Didn't realize `set_subtensor`/`inc_subtensor` validate that first arg is a subtensor result
   - **Next time**: Before planning tests for unfamiliar APIs, read their source or docs for constraints
   - **Action**: Add research step: "Check API validation requirements and constraints"

#### For Test Design

1. **Strategy Output Format Consistency**
   - **Example**: Mix of single values vs tuples from strategies required careful handling
   - **Next time**: Document in plan what format each strategy returns (single value or tuple)
   - **Action**: Add "Strategy Return Types" table in plan

2. **Hypothesis assume() for Edge Cases**
   - **Example**: Used `assume()` to filter zero increments and equal values (not mentioned in plan)
   - **Next time**: Anticipate edge cases where generated values might cause false failures
   - **Action**: Add section "Expected Edge Cases and Filtering" to test design

3. **Shape Validation Assumptions**
   - **Example**: Plan assumed multi-dimensional tensors, but strategies generated 1D
   - **Next time**: Verify strategy output shapes before planning assertions
   - **Action**: Include sample strategy output in plan examples

#### For Implementation

1. **Follow Established Patterns**
   - **Example**: ELEMWISE registry pattern was correct; SUBTENSOR needed to match it
   - **Next time**: When adding to existing infrastructure, find and follow the newest/best pattern
   - **Action**: Add step: "Identify most recent similar implementation to use as template"

2. **Variable Naming for Clarity**
   - **Example**: Using `x_val`, `indices_val` was clearer than generic `test_data`
   - **Next time**: Use descriptive variable names that indicate data type (numpy array vs PyTensor variable)
   - **Action**: Establish naming convention: `*_val` for numpy arrays, plain `x` for PyTensor variables

3. **Incremental Testing**
   - **Example**: Running tests after each test function would have caught registry issue earlier
   - **Next time**: Test after each function implementation, not after all 4 functions
   - **Action**: Add to TDD workflow: "Run test suite after each new test function"

### Recommendations for Next Similar Plan

1. **Add Phase 0: Infrastructure Validation**
   - Verify registries follow correct pattern
   - Write one minimal test to validate infrastructure
   - Document any pattern deviations that need fixing
   - **Why**: Catches infrastructure issues before writing all tests

2. **Include Registry Pattern Documentation**
   - Show correct registry structure with examples
   - Reference existing working registries (e.g., ELEMWISE_OPERATIONS)
   - Explain the numpy→PyTensor wrapping pattern
   - **Why**: Makes implementation faster and reduces errors

3. **Document Strategy Return Types**
   - Create table showing what each strategy returns
   - Note which strategies return tuples vs single values
   - Include shape information for arrays
   - **Why**: Prevents mismatched expectations in test code

4. **Research API Constraints Section**
   - Check PyTensor source for validation requirements
   - Document any constraints on inputs
   - Note any "magic" behavior (like `inc_subtensor` requiring subtensor result)
   - **Why**: Prevents surprises during implementation

5. **Add Expected Edge Cases Section**
   - List edge cases where Hypothesis might generate problematic values
   - Plan where to use `assume()` for filtering
   - Note acceptable invalid example rates (e.g., 41% for zero-filtering)
   - **Why**: Makes testing strategy explicit and avoids confusion

6. **Include Incremental Testing Checkpoints**
   - Add "Run tests" step after each function implementation
   - Don't wait until all tests are written
   - **Why**: Catches issues earlier when they're easier to fix

### Patterns Worth Documenting

- **Registry Lambda Wrapping Pattern**: The two-lambda pattern for converting numpy arrays to PyTensor variables
  ```python
  "build_graph": lambda x_val: (
      lambda x: ([x], x + 1)
  )(pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim))
  ```
  - **Where used**: tests/link/onnx/strategies.py throughout all operation registries
  - **Why valuable**: This pattern is needed for all registries that use property-based testing

- **Hypothesis assume() for Value Filtering**: Using `assume()` to filter out edge cases that would cause false failures
  ```python
  from hypothesis import assume
  assume(not np.allclose(values_val, 0))  # Filter zero increments
  assume(not np.array_equal(values_val, x_val[2:5]))  # Filter equal values
  ```
  - **Where used**: tests/link/onnx/test_subtensor.py:159, 213
  - **Why valuable**: Better than complicated custom strategies for filtering rare edge cases

- **Dual Test Coverage Pattern**: Property tests for broad coverage + manual tests for documentation
  - **Where used**: Throughout test_subtensor.py
  - **Why valuable**: Property tests catch edge cases; manual tests serve as readable examples and explicit regression tests

### Open Questions for Future Work

- Should we consolidate manual tests now that property tests provide broader coverage? (Plan suggested removing some, but decided to keep all for documentation)
- Should we add property tests for negative indices as `@pytest.mark.xfail` to document expected behavior? (Plan suggested this but wasn't implemented)
- Would it be valuable to increase `max_examples` for critical operations? (Currently 10-20, could go higher for more confidence)
- Should we standardize all operation registries to follow the ELEMWISE pattern? (Would require refactoring SHAPE, REDUCTION, ALLOCATION registries)
- Is the 41% invalid rate for `inc_subtensor` acceptable, or should we adjust the strategy to generate fewer zero values?

---

*This post-implementation analysis documents that the implementation was remarkably smooth once the registry pattern issue was identified and fixed. The main lesson is to validate infrastructure patterns before implementing tests. The plan was accurate in its test design and expected outcomes; the only gap was not anticipating the registry pattern inconsistency.*
