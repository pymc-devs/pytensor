# ONNX Backend Property-Based Testing - Master Implementation Plan

**Date**: 2025-11-08
**Based on Research**: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md`
**Approach**: Test-Driven Development (TDD)

## Overview

This master plan coordinates the implementation of comprehensive property-based testing for PyTensor's ONNX backend. The goal is to achieve complete test coverage for 41+ ONNX operations through 5 coordinated phases, replacing or augmenting 69 manual tests with 400+ property-based test scenarios.

## Strategic Approach

### Testing Philosophy

**Property-Based Testing Advantages**:
- Automatically tests diverse inputs (Hypothesis generates test cases)
- Catches edge cases developers might miss
- Provides regression protection through shrinking
- Tests operations systematically rather than manually
- One test function can cover multiple operations

**TDD Process**:
1. **Write tests first** - Define expected behavior through tests
2. **Verify tests fail properly** - Ensure tests catch real issues
3. **Implement to pass tests** - Make tests green one at a time
4. **Refactor with confidence** - Tests protect during cleanup

### Operation Categorization

Based on research analysis (research doc lines 324-338), operations are grouped into:

1. **Category-based testing** (homogeneous operations):
   - Elemwise operations (18 ops) - similar validation logic
   - Reduction operations (6 ops) - value-based aggregations
   - Allocation operations (4 ops) - tensor creation

2. **Individual testing** (heterogeneous operations):
   - Shape operations (8 ops) - diverse transformation behaviors
   - Subtensor operations (4 ops) - complex indexing constraints
   - Argmax/argmin (2 ops) - index-based, unique from reductions

## Implementation Phases

### Phase 1: Elemwise Operations Registry
**File**: `thoughts/shared/plans/phase1_elemwise_registry_tdd.md`
**Status**: Plan Complete
**Goal**: Create `ELEMWISE_OPERATIONS` registry and supporting strategies

**Deliverables**:
- `ELEMWISE_OPERATIONS` registry with 18 operation configurations
- Helper strategies: `binary_float32_arrays_strategy()`, `unary_float32_array_strategy()`, etc.
- Constraint-respecting strategies for log, sqrt, pow
- Test file: `tests/link/onnx/test_strategies.py` (new)

**Test Coverage**: Infrastructure validation (registry structure, strategy correctness)

**Dependencies**: None (foundational phase)

**Estimated Effort**: 3-4 hours (registry creation + strategy testing)

---

### Phase 2: Elemwise Property Tests
**File**: `thoughts/shared/plans/phase2_elemwise_property_tests_tdd.md`
**Status**: Plan Complete
**Goal**: Create property-based tests for all 18 elemwise operations

**Deliverables**:
- Main property test: `test_elemwise_operations_correctness()` (13 unconstrained ops)
- Constrained operation tests: `test_log_operation_correctness()`, `test_sqrt_operation_correctness()`, `test_pow_operation_correctness()`, `test_clip_operation_correctness()`
- Updated test file: `tests/link/onnx/test_elemwise.py`
- Cleanup: Remove redundant manual tests

**Test Coverage**: 180+ test scenarios (18 operations × 10 examples minimum)

**Dependencies**: Phase 1 (requires ELEMWISE_OPERATIONS registry)

**Estimated Effort**: 4-5 hours (test implementation + validation + refactoring)

---

### Phase 3: Shape Operations Property Tests
**File**: `thoughts/shared/plans/phase3_shape_property_tests_tdd.md`
**Status**: Plan Complete
**Goal**: Create individual property tests for 8 shape operations

**Deliverables**:
- 8 individual property test functions:
  - `test_shape_operation_correctness()`
  - `test_shape_i_operation_correctness()`
  - `test_specify_shape_passthrough_correctness()`
  - `test_reshape_operation_correctness()`
  - `test_transpose_operation_correctness()`
  - `test_dimshuffle_add_dim_correctness()`
  - `test_dimshuffle_squeeze_correctness()`
  - `test_concatenate_operation_correctness()`
  - `test_stack_operation_correctness()`
- Updated test file: `tests/link/onnx/test_shape.py`
- Cleanup: Remove redundant manual tests

**Test Coverage**: 80+ test scenarios (8 operations × 10 examples)

**Dependencies**: None (SHAPE_OPERATIONS registry already exists)

**Estimated Effort**: 5-6 hours (8 individual tests + validation + refactoring)

---

### Phase 4: Subtensor Operations Property Tests
**File**: `thoughts/shared/plans/phase4_subtensor_property_tests_tdd.md`
**Status**: Plan Complete
**Goal**: Create individual property tests for 4 subtensor operations

**Deliverables**:
- 4 individual property test functions:
  - `test_subtensor_basic_slicing_correctness()` (3 slice patterns)
  - `test_advanced_subtensor_indexing_correctness()`
  - `test_set_subtensor_operation_correctness()`
  - `test_inc_subtensor_operation_correctness()`
- Updated test file: `tests/link/onnx/test_subtensor.py`
- Cleanup: Remove redundant manual tests, document negative index limitation

**Test Coverage**: 40+ test scenarios (4 operations × 10 examples)

**Dependencies**: None (SUBTENSOR_OPERATIONS registry already exists)

**Important Note**: Negative indices NOT supported (research doc design decision #3, lines 666-676)

**Estimated Effort**: 4-5 hours (4 tests + validation + refactoring + documentation)

---

### Phase 5: Argmax Property Test
**File**: `thoughts/shared/plans/phase5_argmax_property_test_tdd.md`
**Status**: Plan Complete
**Goal**: Create dedicated property test for argmax/argmin operations

**Deliverables**:
- 2-3 individual property test functions:
  - `test_argmax_operation_correctness()`
  - `test_argmin_operation_correctness()`
  - (Optional) `test_argmax_keepdims_correctness()`
- Updated test file: `tests/link/onnx/test_math.py`
- Cleanup: Evaluate redundancy with existing reduction test

**Test Coverage**: 20+ test scenarios (2 operations × 10 examples)

**Dependencies**: None (REDUCTION_OPERATIONS registry already has argmax)

**Estimated Effort**: 2-3 hours (simpler phase, builds on existing infrastructure)

---

## Execution Strategy

### Recommended Order

Execute phases in sequence (1 → 2 → 3 → 4 → 5):

**Rationale**:
1. Phase 1 creates foundational registry pattern for Phase 2
2. Phases 3-5 can technically run in parallel (independent)
3. Sequential execution builds confidence and experience

**Alternative Approach** (Parallel Execution):
- Phase 1 → Phase 2 (sequential, dependent)
- Phase 3, 4, 5 in parallel (independent)

### Per-Phase Workflow

Each phase follows the same TDD structure:

#### Stage 1: Test Design & Implementation (30-40% of time)
- Write tests that define expected behavior
- Tests should fail initially (features not implemented yet OR tests more comprehensive)
- Focus on clear, informative test failures

#### Stage 2: Test Failure Verification (10-15% of time)
- Run tests, verify they fail as expected
- Confirm failure messages are diagnostic
- Document failure patterns

#### Stage 3: Implementation / Bug Fixes (30-40% of time)
- Make tests pass one at a time
- Fix any bugs revealed by property tests
- Re-run tests frequently

#### Stage 4: Refactoring & Cleanup (15-20% of time)
- Improve code quality while keeping tests green
- Remove redundant tests
- Add documentation

### Success Metrics

**Per-Phase Metrics**:
- [ ] All property tests pass
- [ ] No regressions in existing tests
- [ ] Code passes linting (`make lint`)
- [ ] Test code is maintainable and clear

**Overall Project Metrics**:
- [ ] 400+ property-based test scenarios
- [ ] 41 operations covered
- [ ] Reduced manual test count (remove redundancy)
- [ ] Comprehensive test documentation

## Coverage Summary

### Before Property-Based Testing
- **Total Operations**: 44+
- **Property-Based Coverage**: 12 operations (27%)
  - Reductions: 6 operations (test_math.py)
  - Allocations: 4 operations (test_tensor_basic.py)
  - Argmax/argmin: 2 operations (test_math.py)
- **Manual Tests**: 69 tests across 13 files
- **Test Scenarios**: ~150 (manual tests)

### After Property-Based Testing (Target)
- **Total Operations**: 44+
- **Property-Based Coverage**: 41 operations (93%)
  - Elemwise: 18 operations (Phase 2)
  - Reductions: 6 operations (existing)
  - Allocations: 4 operations (existing)
  - Shape: 8 operations (Phase 3)
  - Subtensor: 4 operations (Phase 4)
  - Argmax/argmin: 2 operations (Phase 5) [dedicated tests]
- **Manual Tests**: ~30 tests (edge cases only)
- **Test Scenarios**: 400+ (property-based) + ~30 (manual)

### Operations Not Covered
- **Core operations** (3): Constant, DeepCopyOp, FunctionGraph
  - Reason: System-level operations, tested via integration tests

## Key Design Decisions (From Research)

### Decision 1: Constrained Operations (Research lines 654-664)
**Question**: Should all elemwise operations share a single property test?

**Decision**: Operations with special constraints (log, sqrt, pow) have separate tests.

**Rationale**: Allows operation-specific input filtering and clearer failure messages.

### Decision 2: Tolerance Values (Research lines 660-664)
**Question**: What tolerance values for numerically unstable operations?

**Decision**: Use reasonable defaults (rtol=1e-5, atol=1e-8), relax for unstable ops (log, exp, pow).

**Rationale**: Balance accuracy with real-world precision limits. Document non-default tolerances.

### Decision 3: Negative Indices (Research lines 666-676)
**Question**: Should subtensor tests cover negative indices?

**Decision**: No, explicitly exclude negative indices from property tests.

**Rationale**: Current ONNX backend has known limitation (documented at subtensor.py:122-127). Testing unsupported features creates false failures.

### Decision 4: Expected Failures (Research lines 672-676)
**Question**: Should we test unsupported features as "expected to fail"?

**Decision**: No, exclude unsupported features entirely. Document in code comments.

**Rationale**: Property tests should validate working functionality. Clear documentation is preferable to confusing xfail tests.

### Decision 5: Opset Versions (Research lines 679-683)
**Question**: Test multiple ONNX opset versions?

**Decision**: Only test default opset version (18).

**Rationale**: Simplifies test infrastructure. Can extend later if needed.

### Decision 6: Hypothesis Database (Research lines 684-688)
**Question**: Commit `.hypothesis/` directory to version control?

**Decision**: Remain in `.gitignore`.

**Rationale**: Database is local/platform-specific. Reproducibility achieved through deterministic seed.

### Decision 7: Broadcasting (Research lines 690-694)
**Question**: Test broadcasting explicitly?

**Decision**: Yes, create strategies generating compatible but different shapes.

**Rationale**: Broadcasting is critical for elemwise operations and should be validated.

### Decision 8: Graph Structure Validation (Research lines 696-700)
**Question**: Validate graph structure or only numerical correctness?

**Decision**: Validate numerical correctness only.

**Rationale**: Graph structure validation is brittle. ONNX model validation via `onnx.checker.check_model()` ensures structural correctness.

## Testing Infrastructure

### Hypothesis Configuration

**Profiles** (defined in tests/link/onnx/conftest.py:28-68):
- **dev** (default): 10 examples, no deadline, default verbosity
- **ci**: 100 examples, no deadline, suppresses health checks
- **debug**: 10 examples, verbose output, explicit phases

**Usage**:
```bash
# Default (dev profile)
uv run pytest tests/link/onnx/test_elemwise.py -v

# CI profile (more examples)
uv run pytest tests/link/onnx/test_elemwise.py -v --hypothesis-profile=ci

# Debug profile (verbose)
uv run pytest tests/link/onnx/test_elemwise.py -v --hypothesis-profile=debug
```

### Core Test Utilities

**compare_onnx_and_py()** (test_basic.py:30):
- Compiles graph with ONNX and Python backends
- Executes both with same inputs
- Validates ONNX model
- Compares outputs with configurable tolerance
- Returns: `(onnx_function, onnx_result)`

**get_onnx_node_types()** (test_basic.py:107):
- Extracts ONNX node types from compiled function
- Returns: List of ONNX operation names
- Used for validation: `assert 'Add' in get_onnx_node_types(fn)`

### Registry Pattern

**Structure**:
```python
OPERATION_REGISTRY = {
    'operation_name': {
        'build_graph': lambda ...: (inputs, output),  # Builds PyTensor graph
        'strategy': custom_strategy(),                # Hypothesis strategy
        'expected_onnx_ops': ['ONNXOp1', 'ONNXOp2'], # Expected ONNX nodes
        'description': 'Human-readable description'   # Documentation
    }
}
```

**Existing Registries**:
- `ELEMWISE_OPERATIONS` (Phase 1 creates this)
- `REDUCTION_OPERATIONS` (exists, strategies.py:248)
- `ALLOCATION_OPERATIONS` (exists, strategies.py:311)
- `SHAPE_OPERATIONS` (exists, strategies.py:159)
- `SUBTENSOR_OPERATIONS` (exists, strategies.py:348)
- `INCSUBTENSOR_OPERATIONS` (exists, strategies.py:393)

## Common Commands

### Running Tests

```bash
# Run all ONNX tests
uv run pytest tests/link/onnx/ -v

# Run specific phase tests
uv run pytest tests/link/onnx/test_elemwise.py -v        # Phase 2
uv run pytest tests/link/onnx/test_shape.py -v          # Phase 3
uv run pytest tests/link/onnx/test_subtensor.py -v      # Phase 4
uv run pytest tests/link/onnx/test_math.py -k "argm" -v # Phase 5

# Run only property tests
uv run pytest tests/link/onnx/ -k "correctness" -v

# Run with more examples (CI mode)
uv run pytest tests/link/onnx/ --hypothesis-profile=ci -v

# Run with Hypothesis statistics
uv run pytest tests/link/onnx/test_elemwise.py --hypothesis-show-statistics
```

### Code Quality

```bash
# Linting
make lint

# Type checking (if applicable)
make typecheck

# Run tests with coverage
uv run pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term-missing
```

### Debugging

```bash
# Run specific test with verbose output
uv run pytest tests/link/onnx/test_elemwise.py::test_elemwise_operations_correctness -vv

# Show full traceback
uv run pytest tests/link/onnx/test_elemwise.py --tb=long

# Show local variables in traceback
uv run pytest tests/link/onnx/test_elemwise.py --tb=short --showlocals
```

## Risk Management

### Potential Risks

**Risk 1: Property Tests Too Slow**
- **Mitigation**: Use small tensors (max 10 elements per dimension), limit examples
- **Fallback**: Reduce max_examples in CI if needed

**Risk 2: Hypothesis Generates Invalid Inputs**
- **Mitigation**: Use constraint strategies (positive_float32, non_negative_float32)
- **Fallback**: Add filters to strategies

**Risk 3: False Failures Due to Numerical Precision**
- **Mitigation**: Use appropriate tolerances (rtol, atol), document relaxed tolerances
- **Fallback**: Investigate and adjust tolerances per operation

**Risk 4: Property Tests Reveal Many Bugs**
- **Mitigation**: This is actually good! Document bugs, fix systematically
- **Fallback**: Create issues for bugs, fix in separate PRs if needed

**Risk 5: Redundancy with Existing Tests**
- **Mitigation**: Carefully evaluate which manual tests to remove
- **Fallback**: Keep both if removal creates risk, document why

## Timeline Estimate

### Phase-by-Phase (Sequential Execution)

| Phase | Effort | Cumulative | Description |
|-------|--------|------------|-------------|
| Phase 1 | 3-4h | 3-4h | Registry infrastructure |
| Phase 2 | 4-5h | 7-9h | Elemwise property tests |
| Phase 3 | 5-6h | 12-15h | Shape property tests |
| Phase 4 | 4-5h | 16-20h | Subtensor property tests |
| Phase 5 | 2-3h | 18-23h | Argmax property tests |

**Total Estimated Effort**: 18-23 hours (2-3 days of focused work)

### Parallel Execution (Phases 3-5)

| Stage | Effort | Description |
|-------|--------|-------------|
| Phase 1 | 3-4h | Registry infrastructure (sequential) |
| Phase 2 | 4-5h | Elemwise tests (sequential, depends on Phase 1) |
| Phases 3-5 | 5-6h | Shape, Subtensor, Argmax (parallel, independent) |

**Total Estimated Effort**: 12-15 hours (1.5-2 days with parallel execution)

**Recommendation**: Sequential execution for first implementation (builds confidence), parallel for future enhancements.

## Success Criteria

### Phase Completion Criteria
- [ ] All property tests implemented
- [ ] All tests passing
- [ ] No regressions in existing tests
- [ ] Code quality maintained (linting, type checking)
- [ ] Documentation updated
- [ ] Redundant tests removed

### Project Completion Criteria
- [ ] 400+ property-based test scenarios
- [ ] 93% operation coverage (41/44 operations)
- [ ] Comprehensive test documentation
- [ ] Clear test failure messages
- [ ] Maintainable test codebase
- [ ] Property-based testing pattern established for future operations

## Future Enhancements

### Post-Implementation Improvements
1. **Increase examples in CI**: Use `max_examples=100` in CI profile
2. **Add broadcasting tests**: Explicit tests for broadcasting behavior
3. **Test mixed dtypes**: Add float64, int32, etc. tests
4. **Test negative indices**: When ONNX backend supports them
5. **Test dynamic shapes**: When ONNX backend supports them
6. **Add performance benchmarks**: Track test execution time

### New Operations
When new ONNX operations are added:
1. Add to appropriate registry (or create new registry)
2. Create Hypothesis strategy
3. Write property test following established patterns
4. Document in this master plan

## References

### Primary Documents
- **Research Document**: `thoughts/shared/research/2025-11-07_12-08-07_hypothesis-property-based-onnx-testing.md`
- **Phase Plans**: `thoughts/shared/plans/phase[1-5]_*.md`

### Code References
- **Test Utilities**: `tests/link/onnx/test_basic.py`
- **Strategies**: `tests/link/onnx/strategies.py`
- **Hypothesis Config**: `tests/link/onnx/conftest.py:28-68`
- **ONNX Dispatchers**: `pytensor/link/onnx/dispatch/`

### External Resources
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [PyTensor Documentation](https://pytensor.readthedocs.io/)

## Conclusion

This master plan coordinates 5 phases of TDD implementation to achieve comprehensive property-based testing for PyTensor's ONNX backend. Following this plan will:

1. **Improve test coverage**: 27% → 93% property-based coverage
2. **Increase test scenarios**: 150 → 400+ scenarios
3. **Enhance bug detection**: Property tests catch edge cases automatically
4. **Reduce maintenance**: Fewer, more powerful tests
5. **Establish patterns**: Template for future ONNX operations

The phased approach allows for systematic, confidence-building implementation while maintaining code quality and test reliability throughout.

---

**Next Steps**: Begin with Phase 1 (Elemwise Registry) by following `thoughts/shared/plans/phase1_elemwise_registry_tdd.md`.
