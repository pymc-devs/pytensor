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
@settings(max_examples=10, deadline=None)
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

    # Validate that slice was modified
    # (values at indices 2:5 should be different from original)
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

    # Validate that slice was modified
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
- [ ] All test functions created with proper structure
- [ ] Tests use registries correctly
- [ ] Tests are discoverable: `uv run pytest --collect-only tests/link/onnx/test_subtensor.py`
- [ ] Test code follows project conventions: `make lint-tests`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Negative indices explicitly excluded (documented in comments)
- [ ] Assertion messages are diagnostic

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
- [ ] All tests run without collection errors
- [ ] Tests complete execution (10 examples each)
- [ ] No import or strategy errors

#### Manual Verification:
- [ ] Test failures (if any) are informative
- [ ] Can identify slice pattern causing failure
- [ ] Hypothesis shrinking provides minimal examples
- [ ] No confusing error messages

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

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] Test count reduced appropriately
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Code is more organized and readable
- [ ] Limitation clearly documented
- [ ] No important coverage lost

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] 4 subtensor operations covered by property tests
- [ ] 40+ test scenarios (4 ops × 10 examples minimum)
- [ ] Basic slicing patterns validated
- [ ] Advanced indexing tested
- [ ] Set/inc subtensor operations verified
- [ ] Negative indices explicitly excluded (documented limitation)

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
