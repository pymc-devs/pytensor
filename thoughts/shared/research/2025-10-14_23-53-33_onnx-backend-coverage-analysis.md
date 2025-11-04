---
date: 2025-10-14T23:53:33+0000
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: onnx-backend
repository: pytensor
topic: "ONNX Backend Coverage Gaps, Issues, and Compensatory Test Patterns"
tags: [research, codebase, onnx, testing, coverage, quality]
status: complete
last_updated: 2025-10-14
last_updated_by: Claude
---

# Research: ONNX Backend Coverage Gaps, Issues, and Compensatory Test Patterns

**Date**: 2025-10-14T23:53:33+0000
**Researcher**: Claude
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: onnx-backend
**Repository**: pytensor

## Research Question

What are the coverage gaps, glaring obvious issues, and compensatory test patterns in the current ONNX backend implementation and tests?

## Summary

The ONNX backend implementation is functional but has significant coverage gaps and quality issues:

**Critical Issues Found:**
1. **DimShuffle fallback bug** - Complex cases silently fall back to Identity instead of proper implementation
2. **5 implemented ops lack any tests** - Gemv, Cast, AllocEmpty, DeepCopyOp, Composite decomposition
3. **Weak Shape_i testing** - Only indirectly tested, not validated for ONNX structure
4. **No dtype diversity** - All tests use float32 only
5. **Missing edge case coverage** - No empty tensors, single elements, error paths

**Compensatory Patterns:**
Tests use integration/pattern testing to compensate for lack of granular unit tests on individual operations.

## Detailed Findings

### 1. Implemented Operations (What Exists)

**Elementwise Operations** (`pytensor/link/onnx/dispatch/elemwise.py:1-180`)
- Supported scalar ops: Add, Mul, Sub, TrueDiv, Neg, Exp, Log, Sqrt, Pow, Abs, ScalarMaximum, ScalarMinimum
- Cast operation with dtype mapping
- Composite scalar op decomposition (lines 31-113)

**Shape Operations** (`pytensor/link/onnx/dispatch/shape.py:1-395`)
- Shape_i (extract dimension) - lines 17-94
- Reshape - lines 97-112
- DimShuffle (unsqueeze/squeeze/transpose) - lines 115-230
- AllocEmpty (constant-filled tensor) - lines 233-376
- DeepCopyOp (maps to Identity) - lines 379-394

**Linear Algebra** (`pytensor/link/onnx/dispatch/nlinalg.py:1-110`)
- Dot (general matrix multiplication) - lines 13-29
- Dot22 (optimized 2x2 dot) - lines 32-45
- Gemv (general matrix-vector with alpha/beta) - lines 48-109

**Special Functions** (`pytensor/link/onnx/dispatch/special.py:1-89`)
- Softmax with axis support (including axis=None with flatten/reshape) - lines 12-88

### 2. Test Coverage (What's Tested)

**test_basic.py** (217 lines):
- Basic import and dispatcher registration
- Simple addition export
- Multiple operations chaining
- Unsupported op error handling (SVD)
- Shared variables as initializers

**test_elemwise.py** (160 lines):
- All basic elemwise ops: add, mul, sub, div, neg, exp, log, sqrt, pow, abs
- Different tensor shapes (vector, matrix, 3D)
- Chained operations

**test_shape.py** (143 lines):
- DimShuffle variants: unsqueeze (start/end/multiple), squeeze (first/last), transpose (2D/3D)
- Reshape: vector→matrix, with -1, flatten
- Flatten method
- Shape_i indirectly (in computation)
- Combined reshape operations

**test_nlinalg.py** (73 lines):
- Dot: vector-vector, matrix-vector, matrix-matrix
- Simple linear layer pattern (W @ x + b)

**test_special.py** (113 lines):
- Softmax (basic and axis variations)
- Maximum/Minimum operations
- ReLU via maximum(x, 0) pattern
- Two-layer neural network integration test

### 3. Coverage Gaps (What's Missing)

#### 3.1 Untested Implemented Operations

1. **Gemv** - Fully implemented (lines 48-109 in `nlinalg.py`) but zero tests
   - Complex 4-node decomposition: MatMul + 2 Mul + Add
   - High risk for bugs in node generation

2. **Cast** - Implemented (lines 129-157 in `elemwise.py`) but not explicitly tested
   - Critical for dtype conversions
   - Has dtype mapping logic that could fail

3. **AllocEmpty** - Implemented (lines 233-376 in `shape.py`) but not tested
   - Complex logic with 3 different input cases (144 lines!)
   - Handles scalar/vector/multiple inputs differently

4. **DeepCopyOp** - Implemented (lines 379-394 in `shape.py`) but not tested
   - Simple Identity mapping, low risk but still untested

5. **Composite scalar op decomposition** - Implemented (lines 31-113 in `elemwise.py`) but not explicitly tested
   - Complex graph traversal and node generation
   - Handles constants, intermediate results, final outputs
   - High complexity = high risk

#### 3.2 Missing Edge Cases

- **Empty tensors** - No tests for 0-sized dimensions
- **Single element tensors** - No tests for scalars or (1,) shapes in most ops
- **Very large tensors** - No performance or correctness tests
- **Broadcasting edge cases** - Only basic broadcasting tested
- **Multiple outputs** - No tests for ops that produce multiple outputs
- **Shared intermediate results** - No tests for DAGs with shared nodes
- **Error conditions in shape ops** - No tests for invalid reshape dimensions

#### 3.3 Data Type Coverage

- **Only float32 tested** - All 27 tests use `dtype="float32"`
- **No int32/int64 tests** - Despite implementation support
- **No bool tests** - Despite dtype_map including bool
- **No float64 tests** - Despite implementation support
- **No mixed-dtype tests** - No tests where inputs have different dtypes

#### 3.4 ONNX-Specific Testing Gaps

- **No opset version testing** - Only uses default opset 18
- **No model structure validation** - Only checks outputs match, not node structure
- **No initializer validation** - Only one test checks initializers (test_shared_variables_as_initializers)
- **No symbolic shape testing** - All shapes are concrete values
- **No ONNX checker failure tests** - Only one validation test (in test_shared_variables_as_initializers)

### 4. Glaring Issues

#### 4.1 CRITICAL: DimShuffle Silent Fallback Bug

**Location**: `pytensor/link/onnx/dispatch/shape.py:222-230`

```python
# Complex case: combination of operations
# For now, fall back to identity and let ONNX optimize
# TODO: Handle complex cases with multiple operations
return helper.make_node(
    "Identity",
    inputs=input_names,
    outputs=output_names,
    name=f"Identity_{output_names[0]}",
)
```

**Problem**: DimShuffle operations that combine squeeze/unsqueeze with transpose silently fall back to Identity, which **does nothing**. This will produce incorrect results with no error!

**Example that would fail**:
```python
x.dimshuffle('x', 1, 0)  # Add dim + transpose (2,3) -> (1,3,2)
```
This would export as Identity, returning the original shape instead of (1,3,2).

**Impact**: HIGH - Silent data corruption for complex reshape operations

#### 4.2 HIGH: Shape_i Test Doesn't Validate ONNX Structure

**Location**: `tests/link/onnx/test_shape.py:120-131`

```python
def test_shape_i_get_dimension(tmp_path):
    """Test extracting specific dimensions with shape_i."""
    x = pt.matrix("x", dtype="float32")
    dim0 = x.shape[0]
    dim0_float = pt.cast(dim0, "float32")
    y = x + dim0_float  # Broadcasting scalar with matrix
```

**Problem**: This test doesn't validate that Shape_i generates the correct 5-node ONNX sequence (Shape → Constant → Gather → Constant → Squeeze). It only checks that the final output is correct.

**Why it matters**: The Shape_i implementation is complex (lines 17-94, 78 lines) and could generate incorrect ONNX structure that happens to work in simple cases but fails in complex graphs.

**Impact**: MEDIUM - Could export invalid ONNX that fails in production

#### 4.3 MEDIUM: No Testing of Multi-Node Operations

**Affected operations**:
- Gemv: 4 nodes (MatMul, 2×Mul, Add)
- Shape_i: 5 nodes (Shape, 2×Constant, Gather, Squeeze)
- AllocEmpty: 2-10 nodes depending on inputs
- Softmax(axis=None): 4 nodes (Flatten, Softmax, Shape, Reshape)
- Composite: N nodes for N ops in composite

**Problem**: These operations return lists of ONNX nodes, but no tests verify:
1. Correct number of nodes generated
2. Correct node types
3. Correct intermediate variable names
4. Proper connection between nodes

**Impact**: MEDIUM - Could generate invalid ONNX graphs

#### 4.4 MEDIUM: Gemv Completely Untested

**Location**: `pytensor/link/onnx/dispatch/nlinalg.py:48-109` (62 lines)

**Complexity**:
- 4 separate ONNX nodes
- Input unpacking: `y_in, alpha, A, x, beta = node.inputs`
- 3 intermediate variables
- Returns list of nodes

**Problem**: This is one of the most complex converters (62 lines) with zero test coverage.

**Risk factors**:
- Complex input handling (5 inputs)
- Multi-node generation
- Intermediate variable naming
- Could have node ordering issues

**Impact**: MEDIUM - Likely to have bugs on first use

#### 4.5 LOW: AllocEmpty Untested Despite Complexity

**Location**: `pytensor/link/onnx/dispatch/shape.py:233-376` (144 lines!)

**Complexity**:
- 3 different input cases (single vector, single scalar, multiple scalars)
- Different code paths for each case
- Multiple nodes generated (2-10 nodes)
- Dtype mapping

**Problem**: This is the longest converter implementation (144 lines) with complex branching logic, but it has zero tests.

**Impact**: LOW - Not commonly used, but will break when needed

### 5. Compensatory Test Patterns

These tests are designed to work around limitations or uncertainty in the implementation:

#### 5.1 Indirect Shape_i Testing

**Test**: `test_shape_i_get_dimension` (test_shape.py:120-131)

**Pattern**: Instead of directly testing Shape_i ONNX export, it embeds `x.shape[0]` in a computation and validates the final result.

**Compensating for**: Uncertainty about whether Shape_i generates correct ONNX structure

**Why problematic**: Doesn't validate the ONNX graph structure, only end-to-end behavior

#### 5.2 Pattern-Based Testing

**Tests**:
- `test_simple_linear_layer` (test_nlinalg.py:58-72) - Tests "W @ x + b" pattern
- `test_two_layer_network` (test_special.py:78-112) - Tests complete neural network
- `test_relu_via_maximum` (test_special.py:44-51) - Tests ReLU as maximum(x, 0)

**Pattern**: Tests common usage patterns rather than individual operations

**Compensating for**: Lack of confidence in individual op correctness

**Why used**: Integration tests catch more bugs when unit tests are missing

**Benefit**: Actually useful for validating real-world usage

**Drawback**: Can't pinpoint which operation fails when test breaks

#### 5.3 Combined Operations Testing

**Tests**:
- `test_combined_reshape_operations` (test_shape.py:134-142)
- `test_chained_operations` (test_elemwise.py:151-159)
- `test_export_multiple_ops` (test_basic.py:145-163)

**Pattern**: Tests multiple operations in sequence to verify they compose correctly

**Compensating for**: Uncertainty about whether individual ops generate compatible ONNX

**Why problematic**: When this fails, which operation is broken?

#### 5.4 compare_onnx_and_py Helper Abstraction

**Location**: `tests/link/onnx/test_basic.py:18-101` (84 lines)

**Pattern**: Comprehensive helper that:
- Compiles PyTensor function
- Exports to ONNX
- Validates ONNX model
- Runs with ONNX Runtime
- Compares outputs with tolerance

**Compensating for**: Complexity of ONNX testing workflow

**Benefit**: Makes writing tests much easier

**Drawback**: Abstracts away details that should be tested (e.g., initializers, node structure)

#### 5.5 Parametrized Shape Testing

**Test**: `test_add_different_shapes` (test_elemwise.py:130-148)

**Pattern**: Uses `@pytest.mark.parametrize` to test multiple shapes with one test

**Why used**: Efficiently covers multiple shape scenarios

**Compensating for**: Lack of comprehensive shape testing elsewhere

**Benefit**: Good practice, should be used more widely

## Code References

### Implementation Files
- `pytensor/link/onnx/__init__.py:1-25` - Module exports
- `pytensor/link/onnx/export.py:1-115` - Main export API
- `pytensor/link/onnx/dispatch/__init__.py:1-17` - Dispatch registration
- `pytensor/link/onnx/dispatch/basic.py:1-292` - Core dispatch system and FunctionGraph converter
- `pytensor/link/onnx/dispatch/elemwise.py:1-180` - Elementwise operations
- `pytensor/link/onnx/dispatch/shape.py:1-395` - Shape operations
- `pytensor/link/onnx/dispatch/nlinalg.py:1-110` - Linear algebra operations
- `pytensor/link/onnx/dispatch/special.py:1-89` - Special functions and activations

### Test Files
- `tests/link/onnx/test_basic.py:1-217` - Core functionality and utilities
- `tests/link/onnx/test_elemwise.py:1-160` - Elementwise operation tests
- `tests/link/onnx/test_shape.py:1-143` - Shape operation tests
- `tests/link/onnx/test_nlinalg.py:1-73` - Linear algebra tests
- `tests/link/onnx/test_special.py:1-113` - Special function tests

### Specific Issues
- `pytensor/link/onnx/dispatch/shape.py:222-230` - DimShuffle fallback bug (CRITICAL)
- `pytensor/link/onnx/dispatch/nlinalg.py:48-109` - Gemv untested (HIGH)
- `pytensor/link/onnx/dispatch/shape.py:233-376` - AllocEmpty untested (MEDIUM)
- `pytensor/link/onnx/dispatch/elemwise.py:31-113` - Composite decomposition untested (MEDIUM)
- `pytensor/link/onnx/dispatch/elemwise.py:129-157` - Cast untested (MEDIUM)
- `tests/link/onnx/test_shape.py:120-131` - Weak Shape_i test (HIGH)

## Architecture Insights

### Dispatch System Design

The ONNX backend uses Python's `singledispatch` to register converters for each Op type:

```python
@onnx_funcify.register(OpClass)
def onnx_funcify_OpClass(op, node, var_names, get_var_name, **kwargs):
    # Return onnx.NodeProto or list of onnx.NodeProto
```

**Strengths**:
- Clean separation of concerns (one converter per op)
- Easy to extend (just register new converters)
- Type-safe dispatch

**Weaknesses**:
- No validation that converters return correct types
- Multi-node converters return lists, single-node return single nodes (inconsistent)
- No framework for testing individual converters

### Test Architecture

Tests use a **black-box comparison approach**:
1. Define symbolic computation in PyTensor
2. Compile to both PyTensor and ONNX
3. Run same inputs through both
4. Compare outputs with tolerance

**Strengths**:
- Validates end-to-end correctness
- Catches numerical errors
- Easy to write

**Weaknesses**:
- Doesn't validate ONNX structure
- Can't detect suboptimal ONNX generation
- Hard to debug when it fails (which operation broke?)

### Missing Test Infrastructure

**What would help**:
1. **ONNX graph validator** - Check node types, connections, counts
2. **Converter unit tests** - Test each converter in isolation
3. **Fixture library** - Reusable test data for different dtypes/shapes
4. **ONNX diff tool** - Compare expected vs actual ONNX structure

## Recommendations

### Priority 1: Fix Critical Issues

1. **Fix DimShuffle fallback** - Implement proper handling for complex cases
2. **Add Gemv test** - Before someone uses it and discovers it's broken
3. **Improve Shape_i test** - Validate ONNX structure, not just output
4. **Add Cast test** - Critical for multi-dtype support

### Priority 2: Fill Coverage Gaps

5. **Test all implemented ops** - AllocEmpty, DeepCopyOp, Composite
6. **Add dtype tests** - int32, int64, float64, bool
7. **Add edge case tests** - empty tensors, scalars, error conditions
8. **Test multi-node converters** - Validate graph structure

### Priority 3: Improve Test Quality

9. **Add ONNX structure validation** - Don't just check outputs
10. **Create converter unit tests** - Test each converter independently
11. **Add fixture library** - Standardize test data
12. **Document compensatory patterns** - Make intentional what's accidental

## Open Questions

1. **What's the plan for complex DimShuffle cases?** - Currently broken with TODO comment
2. **Should all tests validate ONNX structure?** - Or just outputs?
3. **What's the target opset version?** - Only 18 tested, should support others?
4. **Are there plans for symbolic shapes?** - All tests use concrete shapes
5. **What's the error handling strategy?** - Only one error test exists
6. **Should Gemv be tested/fixed before release?** - 62 lines of untested code
7. **Why is AllocEmpty so complex?** - 144 lines seems excessive for ConstantOfShape
