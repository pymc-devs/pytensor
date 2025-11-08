---
date: 2025-11-07T12:08:07-06:00
researcher: Claude
git_commit: 0b11ba7026b72d6f8fe53dc2fc5cec3360d6c00d
branch: onnx-backend
repository: clsandoval/pytensor-workshop-demo
topic: "Hypothesis Property-Based Testing for ONNX Backend Operations"
tags: [research, codebase, onnx, hypothesis, property-based-testing, testing]
status: complete
last_updated: 2025-11-08
last_updated_by: Claude
design_decisions_finalized: 2025-11-08
---

# Research: Hypothesis Property-Based Testing for ONNX Backend Operations

**Date**: 2025-11-07T12:08:07-06:00
**Researcher**: Claude
**Git Commit**: 0b11ba7026b72d6f8fe53dc2fc5cec3360d6c00d
**Branch**: onnx-backend
**Repository**: clsandoval/pytensor-workshop-demo

## Research Question

How can we implement hypothesis property-based testing for the ONNX backend with one well-defined test per operation, specifically for ONNX backend operations only?

## Summary

The codebase already has a **partial property-based testing infrastructure** in place for ONNX backend operations. Currently, 2 property-based test functions cover 12 operations (reductions and allocations) using an **operation registry pattern**. To achieve one test per operation, we need to:

1. **Extend the operation registry pattern** from `tests/link/onnx/strategies.py` to cover all 44+ ONNX operations
2. **Create operation-specific test functions** for operations requiring specialized validation (e.g., shape operations, subtensor operations, elemwise operations)
3. **Leverage existing Hypothesis strategies** and create new ones for uncovered operation types
4. **Follow the established testing pattern** using `compare_onnx_and_py()` utility for validation

The current implementation demonstrates that property-based testing successfully caught bugs across multiple operations automatically, making it the preferred approach over manual enumeration.

## Detailed Findings

### Current Hypothesis Infrastructure

#### Existing Property-Based Test Files

**1. Reduction Operations** (`tests/link/onnx/test_math.py`):
- **Single property test function**: `test_reduction_operations_correctness()`
- **Operations covered**: 8 operations (sum, prod, max, min, argmax, argmin, all, any)
- **Test scenarios**: 80 (8 operations × 10 examples per Hypothesis settings)
- **Strategy**: Uses `REDUCTION_OPERATIONS` registry from `strategies.py`
- **Pattern**: Registry-based with `@given(op_name=st.sampled_from(list(REDUCTION_OPERATIONS.keys())))`

**2. Allocation Operations** (`tests/link/onnx/test_tensor_basic.py`):
- **Single property test function**: `test_allocation_operations_correctness()`
- **Operations covered**: 4 operations (alloc, alloc_empty, make_vector, arange)
- **Test scenarios**: 40 (4 operations × 10 examples)
- **Strategy**: Uses `ALLOCATION_OPERATIONS` registry from `strategies.py`
- **Pattern**: Same registry-based approach

**Total current coverage**: 12 operations with property-based tests out of 44+ total ONNX operations (27% coverage)

#### Hypothesis Configuration (`tests/link/onnx/conftest.py:28-68`)

Three profiles available:
- **dev** (default): 10 examples, no deadline, default verbosity
- **ci**: 100 examples, no deadline, suppresses health checks
- **debug**: 10 examples, verbose output, explicit phases

Settings applied module-wide via `settings.register_profile()` and `settings.load_profile()`.

#### Custom Hypothesis Strategies (`tests/link/onnx/strategies.py`)

**Existing Composite Strategies**:
1. `reshape_strategy()` - Generates tensors with compatible reshape dimensions
2. `concatenate_strategy()` - Generates lists of tensors for concatenation
3. `tensor_with_axis_strategy()` - Generates tensors with valid axis for reduction
4. `alloc_strategy()` - Generates value and shape for allocation operations
5. `arange_strategy()` - Generates start, stop, step for range operations
6. `set_subtensor_strategy()` - Generates tensor, slice, and values for IncSubtensor
7. `advanced_index_strategy()` - Generates tensor and integer array indices

**Strategy Patterns Used**:
- `st.data()` - Interactive data drawing
- `st.sampled_from()` - Sample from collections
- `st.integers()`, `st.floats()` - Numeric generation with constraints
- `st.lists()` - List generation with min/max size
- `st.one_of()` - Choice between strategies
- `arrays()` from `hypothesis.extra.numpy` - NumPy array generation
- `array_shapes()` from `hypothesis.extra.numpy` - Shape tuple generation

#### Operation Registry Pattern

**Structure** (from `strategies.py`):
```python
OPERATION_REGISTRY = {
    'operation_name': {
        'build_graph': lambda ...: (inputs, output),
        'strategy': custom_strategy(),
        'expected_onnx_ops': ['ONNXOp1', 'ONNXOp2'],
        'description': 'Human-readable description'
    }
}
```

**Current Registries**:
1. `REDUCTION_OPERATIONS` - 8 reduction operations
2. `ALLOCATION_OPERATIONS` - 4 allocation operations
3. `SHAPE_OPERATIONS` - Shape operations (registry exists but not yet used in property tests)
4. `SUBTENSOR_OPERATIONS` - Subtensor operations (registry exists but not yet used in property tests)
5. `INCSUBTENSOR_OPERATIONS` - IncSubtensor operations (registry exists but not yet used in property tests)

### ONNX Backend Operations Inventory

#### Complete List of 44+ Implemented Operations

**1. Core Operations (3)**:
- Constant (pytensor/link/onnx/dispatch/basic.py:305)
- DeepCopyOp (pytensor/link/onnx/dispatch/basic.py:313)
- FunctionGraph (pytensor/link/onnx/dispatch/basic.py:126)

**2. Element-wise Scalar Operations (18)** via `pytensor/link/onnx/dispatch/elemwise.py`:
- Add, Mul, Sub, TrueDiv, IntDiv, Neg, Abs, Exp, Log, Sqrt, Pow, Floor, Ceil, RoundHalfToEven, RoundHalfAwayFromZero, Maximum, Minimum, Clip
- **Dispatcher**: Single `@onnx_funcify.register(Elemwise)` at line 34
- **Mapping**: `SCALAR_OP_TO_ONNX` dictionary at lines 10-31

**3. Reduction Operations (6)** via `pytensor/link/onnx/dispatch/math.py`:
- ReduceSum (Add), ReduceProd (Mul), ReduceMax (Maximum), ReduceMin (Minimum), ReduceMin (AND), ReduceMax (OR)
- **Dispatcher**: `@onnx_funcify.register(CAReduce)` at line 25
- **Mapping**: `REDUCE_OP_MAP` dictionary

**4. Argmax Operations (1)**:
- Argmax (pytensor/link/onnx/dispatch/math.py:94)

**5. Shape Operations (8)** via `pytensor/link/onnx/dispatch/shape.py`:
- Shape (line 20), Shape_i (line 39), SpecifyShape (line 105), DimShuffle (line 122), Reshape (line 206), Join (line 264), Split (line 304)

**6. Tensor Creation Operations (4)** via `pytensor/link/onnx/dispatch/tensor_basic.py`:
- Alloc (line 11), AllocEmpty (line 134), MakeVector (line 254), ARange (line 343)

**7. Indexing/Subtensor Operations (4)** via `pytensor/link/onnx/dispatch/subtensor.py`:
- Subtensor (line 12), AdvancedSubtensor1 (line 162), AdvancedSubtensor (line 191), IncSubtensor (line 235)

### Testing Architecture

#### Core Test Utilities (`tests/link/onnx/test_basic.py`)

**1. `compare_onnx_and_py(graph_inputs, graph_outputs, test_inputs, **kwargs)`** (line ~50):
- Compiles graph with both ONNX linker and Python backend
- Executes both with same test inputs
- Validates ONNX model via `onnx.checker.check_model()`
- Compares results using `np.testing.assert_allclose()`
- Returns: `(onnx_function, onnx_result)`

**Key parameters**:
- `rtol` (default 1e-5): Relative tolerance for floating-point comparison
- `atol` (default 1e-8): Absolute tolerance
- Can be overridden per test

**2. `get_onnx_node_types(fn)`** (line ~140):
- Extracts ONNX node types from compiled function
- Returns: Set of ONNX operation names (e.g., {'Add', 'Mul'})
- Used for validation: `assert 'Add' in get_onnx_node_types(fn)`

#### Compilation Modes

**ONNX Mode** (`test_basic.py`):
```python
onnx_linker = ONNXLinker(opset_version=18)
onnx_mode = Mode(linker=onnx_linker, optimizer=None)
```
- No graph optimizations - exports as-is
- Opset version 18 (ONNX standard)

**Python Mode** (`test_basic.py`):
```python
py_mode = Mode(linker='py', optimizer=None)
```
- Reference implementation for comparison

### Current Test Coverage

#### Existing Test Files (13 total, 69 tests)

**Property-Based Tests (2 files)**:
1. `tests/link/onnx/test_math.py` - 10 tests (80 property test scenarios)
2. `tests/link/onnx/test_tensor_basic.py` - 7 tests (40 property test scenarios)

**Manual/Parametrized Tests (8 files)**:
1. `tests/link/onnx/test_elemwise.py` - 14 tests for elemwise operations
2. `tests/link/onnx/test_shape.py` - 10 tests for shape operations
3. `tests/link/onnx/test_subtensor.py` - 14 tests (3 test classes) for subtensor operations
4. `tests/link/onnx/test_linker.py` - 3 tests for linker system
5. `tests/link/onnx/test_export.py` - 3 tests for export API
6. `tests/link/onnx/test_dispatch_basic.py` - 3 tests for dispatch system
7. `tests/link/onnx/test_imports.py` - 3 tests for import structure
8. `tests/link/onnx/conftest.py` - Fixtures and configuration

**Test Pattern Distribution**:
- **Property-based**: 2 files (27% of operations)
- **Class-based**: 1 file (subtensor operations)
- **Standard pytest functions**: 8 files

#### Operations Without Property-Based Tests

**Missing from Property-Based Testing**:
1. **Element-wise operations** (18 ops) - Currently tested with 14 manual tests in `test_elemwise.py`
2. **Shape operations** (8 ops) - Currently tested with 10 manual tests in `test_shape.py`
   - Has registry in `strategies.py` but no property test function yet
3. **Subtensor operations** (4 ops) - Currently tested with 14 manual tests in `test_subtensor.py`
   - Has registries in `strategies.py` but no property test functions yet
4. **Core operations** (3 ops) - Tested via system-level tests
5. **Argmax** (1 op) - Included in `REDUCTION_OPERATIONS` registry

### ONNX Backend Architecture

#### Dispatcher System

**Core Components**:
1. **`onnx_funcify`** (`pytensor/link/onnx/dispatch/basic.py:60`) - Main dispatcher for op conversion
2. **`onnx_typify`** (`pytensor/link/onnx/dispatch/basic.py:28`) - Type conversion dispatcher

**Registration Pattern**:
```python
@onnx_funcify.register(PyTensorOpClass)
def onnx_funcify_OpName(op, node, get_var_name, **kwargs):
    # Convert PyTensor op to ONNX node(s)
    return onnx_node  # or [nodes] or (node, initializers) or None
```

**Return Patterns**:
1. **Single node**: Most common - append directly
2. **List of nodes**: Multi-step operations (e.g., Shape_i → Constant + Shape + Gather)
3. **Tuple (node, initializers)**: Operations with constant data (e.g., Subtensor)
4. **None**: Pass-through operations (e.g., SpecifyShape)

#### Dispatcher Files

- `pytensor/link/onnx/dispatch/basic.py` - Core infrastructure, Constant, DeepCopyOp, FunctionGraph
- `pytensor/link/onnx/dispatch/elemwise.py` - 18 elemwise operations via mapping table
- `pytensor/link/onnx/dispatch/math.py` - Reduction and argmax operations
- `pytensor/link/onnx/dispatch/shape.py` - Shape manipulation operations
- `pytensor/link/onnx/dispatch/tensor_basic.py` - Tensor creation operations
- `pytensor/link/onnx/dispatch/subtensor.py` - Indexing/slicing operations

### Implementation Strategy for Complete Property-Based Coverage

#### Pattern 1: One Test Per Operation (Most Granular)

Create individual test functions for each operation:

```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_add_correctness(data):
    """Property test for Add operation."""
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x + y

    x_val = data.draw(arrays(np.float32, (5,)))
    y_val = data.draw(arrays(np.float32, (5,)))

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = x_val + y_val
    np.testing.assert_allclose(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Add' in node_types
```

**Advantages**:
- Clear isolation - each operation has its own test
- Easy to identify failures - test name directly indicates which operation failed
- Specialized strategies per operation
- Can set operation-specific tolerances and validation

**Disadvantages**:
- More test functions to maintain (44+ functions)
- Some code duplication
- Longer test file

#### Pattern 2: One Test Per Operation Category (Current Approach)

Group related operations into registries, one property test per category:

```python
# In strategies.py
ELEMWISE_OPERATIONS = {
    'add': {
        'build_graph': lambda: ...,
        'strategy': ...,
        'expected_onnx_ops': ['Add'],
        'description': 'Addition'
    },
    # ... 17 more elemwise operations
}

# In test_elemwise.py
@given(
    op_name=st.sampled_from(list(ELEMWISE_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_elemwise_operations_correctness(op_name, data):
    op_config = ELEMWISE_OPERATIONS[op_name]
    test_data = data.draw(op_config['strategy'])
    graph_inputs, graph_output = op_config['build_graph'](*test_data)
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, test_data)
    # Common validation logic
```

**Advantages**:
- Less duplication - validation logic shared
- Scalable - easy to add new operations to registry
- Consistent testing patterns across operation categories
- Fewer test functions

**Disadvantages**:
- Test failure indicates category, requires looking at Hypothesis example to see specific operation
- Harder to set operation-specific settings
- All operations in category share same strategy constraints

#### Pattern 3: Hybrid Approach (Recommended)

**Category-based for homogeneous operations**:
- Elemwise operations (18 ops) → `test_elemwise_operations_correctness()`
- Reduction operations (6 ops) → Already implemented in `test_math.py`
- Allocation operations (4 ops) → Already implemented in `test_tensor_basic.py`

**Individual tests for heterogeneous operations**:
- Shape operations (8 ops) → 8 individual test functions
- Subtensor operations (4 ops) → 4 individual test functions
- Argmax (1 op) → Individual test function

**Rationale**:
- Elemwise operations share nearly identical validation logic (element-wise comparison)
- Shape operations have diverse behaviors (transpose, reshape, split, join, etc.)
- Subtensor operations have complex edge cases (negative indices, advanced indexing, etc.)
- Hybrid approach balances maintainability with specificity

### Recommended Operation-Specific Implementations

#### 1. Elemwise Operations (18 ops) - Category Test

**File**: `tests/link/onnx/test_elemwise.py`

**Strategy** (new registry in `strategies.py`):
```python
ELEMWISE_OPERATIONS = {
    'add': {
        'build_graph': lambda x, y: ([x, y], x + y),
        'strategy': two_float32_vectors_strategy(),
        'expected_onnx_ops': ['Add'],
    },
    'mul': {
        'build_graph': lambda x, y: ([x, y], x * y),
        'strategy': two_float32_vectors_strategy(),
        'expected_onnx_ops': ['Mul'],
    },
    # ... 16 more operations
}
```

**Test function**:
```python
@given(
    op_name=st.sampled_from(list(ELEMWISE_OPERATIONS.keys())),
    data=st.data(),
)
@settings(max_examples=10, deadline=None)
def test_elemwise_operations_correctness(op_name, data):
    """Property test for all elemwise operations."""
    op_config = ELEMWISE_OPERATIONS[op_name]
    test_data = data.draw(op_config['strategy'])

    graph_inputs, graph_output = op_config['build_graph'](*test_data)
    fn, result = compare_onnx_and_py(graph_inputs, graph_output, test_data)

    # Validate ONNX node types
    node_types = get_onnx_node_types(fn)
    for expected_op in op_config['expected_onnx_ops']:
        assert expected_op in node_types, f"Expected {expected_op} in {node_types}"
```

#### 2. Shape Operations (8 ops) - Individual Tests

**File**: `tests/link/onnx/test_shape.py`

**Example for Reshape**:
```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_reshape_correctness(data):
    """Property test for Reshape operation."""
    test_data = data.draw(reshape_strategy())
    x_val, new_shape = test_data

    x = pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim)
    y = x.reshape(new_shape)

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val.reshape(new_shape)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Reshape' in node_types
```

**Rationale**: Each shape operation has unique validation requirements:
- `Shape` → compare shape tuple
- `Reshape` → validate shape transformation
- `DimShuffle` → validate axis permutation
- `Join` → validate concatenation
- `Split` → validate split results

#### 3. Subtensor Operations (4 ops) - Individual Tests

**File**: `tests/link/onnx/test_subtensor.py`

**Example for Subtensor**:
```python
@given(data=st.data())
@settings(max_examples=10, deadline=None)
def test_subtensor_basic_slicing_correctness(data):
    """Property test for Subtensor with basic slicing."""
    # Generate tensor and valid slice
    tensor_strategy = arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10)
    )
    x_val = data.draw(tensor_strategy)

    # Generate valid slice for this tensor
    slice_obj = data.draw(generate_valid_slice(x_val.shape))

    x = pt.tensor('x', dtype='float32', shape=(None,) * x_val.ndim)
    y = x[slice_obj]

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = x_val[slice_obj]
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Slice' in node_types
```

**Rationale**: Subtensor operations have complex constraints:
- `Subtensor` → slice validation, negative indices, step handling
- `AdvancedSubtensor1` → integer array indexing, bounds checking
- `AdvancedSubtensor` → multi-dimensional advanced indexing
- `IncSubtensor` → set vs increment mode, value broadcasting

### Implementation Steps

#### Phase 1: Extend Registries (Strategies)

**File**: `tests/link/onnx/strategies.py`

1. **Create `ELEMWISE_OPERATIONS` registry** for 18 elemwise operations
2. **Add helper strategies**:
   - `two_float32_vectors_strategy()` - For binary ops
   - `single_float32_vector_strategy()` - For unary ops
   - `float32_vector_and_scalar_strategy()` - For mixed ops (e.g., Pow)

3. **Expand existing registries**:
   - Add missing operations to `SHAPE_OPERATIONS` (DimShuffle, Reshape, Join, Split)
   - Add missing operations to `SUBTENSOR_OPERATIONS`

#### Phase 2: Create Category-Based Property Tests

**File**: `tests/link/onnx/test_elemwise.py`

1. **Replace existing manual tests** with single property test function
2. **Use `ELEMWISE_OPERATIONS` registry** with `@given(op_name=st.sampled_from(...))`
3. **Common validation**: ONNX node type checking, numerical correctness

**Result**: 18 elemwise operations → 1 property test function (180 test scenarios)

#### Phase 3: Create Individual Property Tests for Shape Operations

**File**: `tests/link/onnx/test_shape.py`

Create 8 property test functions:
1. `test_shape_correctness()` - Shape operation
2. `test_shape_i_correctness()` - Shape_i operation
3. `test_specify_shape_correctness()` - SpecifyShape operation
4. `test_dimshuffle_correctness()` - DimShuffle operation
5. `test_reshape_correctness()` - Reshape operation
6. `test_join_correctness()` - Join operation
7. `test_split_correctness()` - Split operation
8. Keep existing manual tests for edge cases

**Result**: 8 shape operations → 8 property test functions (80 test scenarios)

#### Phase 4: Create Individual Property Tests for Subtensor Operations

**File**: `tests/link/onnx/test_subtensor.py`

Create 4 property test functions:
1. `test_subtensor_correctness()` - Basic slicing
2. `test_advanced_subtensor1_correctness()` - 1D integer array indexing
3. `test_advanced_subtensor_correctness()` - Multi-dimensional integer array indexing
4. `test_inc_subtensor_correctness()` - In-place subtensor modification

**Result**: 4 subtensor operations → 4 property test functions (40 test scenarios)

#### Phase 5: Add Argmax Individual Property Test

**File**: `tests/link/onnx/test_math.py`

1. **Create `test_argmax_correctness()`** - Separate from reduction operations
2. **Use `tensor_with_axis_strategy()`** for test data generation
3. **Validate both axis and keepdims variations**

**Result**: 1 argmax operation → 1 property test function (10 test scenarios)

### Coverage Summary After Implementation

| Operation Category | Operations | Pattern | Test Functions | Scenarios |
|-------------------|------------|---------|----------------|-----------|
| Elemwise          | 18         | Category| 1              | 180       |
| Reductions        | 6          | Category| 1 (existing)   | 60        |
| Allocations       | 4          | Category| 1 (existing)   | 40        |
| Shape             | 8          | Individual| 8            | 80        |
| Subtensor         | 4          | Individual| 4            | 40        |
| Argmax            | 1          | Individual| 1            | 10        |
| **Total**         | **41**     | —       | **16**         | **410**   |

**Core operations (Constant, DeepCopyOp, FunctionGraph)** tested via system-level tests - not suitable for property-based testing.

### Code References

**Key Files for Implementation**:

**Strategies and Registries**:
- `tests/link/onnx/strategies.py` - All Hypothesis strategies and operation registries

**Test Files**:
- `tests/link/onnx/test_math.py` - Reduction and argmax tests
- `tests/link/onnx/test_tensor_basic.py` - Allocation tests
- `tests/link/onnx/test_elemwise.py` - Elemwise tests
- `tests/link/onnx/test_shape.py` - Shape operation tests
- `tests/link/onnx/test_subtensor.py` - Subtensor operation tests

**Test Utilities**:
- `tests/link/onnx/test_basic.py:50` - `compare_onnx_and_py()` function
- `tests/link/onnx/test_basic.py:140` - `get_onnx_node_types()` function
- `tests/link/onnx/conftest.py:28-68` - Hypothesis profile configuration

**ONNX Backend Implementation**:
- `pytensor/link/onnx/dispatch/basic.py:60` - `onnx_funcify` dispatcher
- `pytensor/link/onnx/dispatch/elemwise.py:34` - Elemwise dispatcher
- `pytensor/link/onnx/dispatch/math.py:25` - CAReduce dispatcher
- `pytensor/link/onnx/dispatch/shape.py` - Shape operation dispatchers
- `pytensor/link/onnx/dispatch/tensor_basic.py` - Tensor creation dispatchers
- `pytensor/link/onnx/dispatch/subtensor.py` - Subtensor dispatchers

## Architecture Insights

### Property-Based Testing Success Factors

**1. Operation Registry Pattern**:
The registry pattern (`REDUCTION_OPERATIONS`, `ALLOCATION_OPERATIONS`, etc.) enables:
- Declarative operation specification
- Centralized strategy management
- Easy addition of new operations
- Consistent testing patterns

**2. Composite Strategies**:
Custom `@st.composite` strategies like `tensor_with_axis_strategy()` encapsulate:
- Validity constraints (e.g., axis within tensor dimensions)
- Inter-parameter relationships (e.g., shape compatibility for reshape)
- Complex data generation logic

**3. Validation Utilities**:
The `compare_onnx_and_py()` utility provides:
- Dual compilation (ONNX + Python reference)
- Automatic result comparison with configurable tolerances
- ONNX model validation via `onnx.checker.check_model()`
- Consistent error reporting

**4. Hypothesis Configuration**:
Three profiles (dev, ci, debug) enable:
- Fast local development (10 examples)
- Thorough CI testing (100 examples)
- Debugging with verbose output and explicit phases

### Dispatcher Architecture Insights

**1. Singledispatch Pattern**:
- No inheritance hierarchy - purely registration-based
- `@onnx_funcify.register(OpClass)` decorator for each PyTensor op type
- Enables modular, extensible dispatch system

**2. Return Pattern Polymorphism**:
Handlers return different structures based on operation complexity:
- Single node: Simple 1:1 mappings (e.g., Add → Add)
- List: Multi-step conversions (e.g., Shape_i → Constant + Shape + Gather)
- Tuple: Node + initializers (e.g., Subtensor with slice constants)
- None: Pass-through (e.g., SpecifyShape)

**3. Variable Naming**:
- `get_var_name()` closure maintains PyTensor Variable → ONNX name mapping
- Ensures uniqueness via counter: `"{base_name}_{counter}"`
- Passed to all handlers via kwargs

**4. Constant Handling**:
- Constants converted to ONNX initializers, not nodes
- Special case: scalar int constants auto-upcast to float32 to prevent type mismatches

## Historical Context (from thoughts/)

### Planning Documents

**1. Main Implementation Plan** (`thoughts/shared/plans/onnx-backend-tier2-3-shape-reductions-tdd.md`):
- Documents shift from manual tests to property-based testing
- Contains strategy examples for reductions and allocations
- Emphasizes property-based testing as "the way forward"

**2. Bug Fix Documentation** (`thoughts/shared/plans/onnx-backend-bugfixes-2025-01-04.md`):
- Notes that property-based tests **automatically caught issues across multiple operations**
- Validates the approach: "This is the power of property-based testing—one fix, many operations benefit."

**3. Quality Improvements Plan** (`thoughts/shared/plans/onnx-backend-coverage-and-quality-improvements.md`):
- Explains decision to use property-based testing instead of manual dtype enumeration
- "Rather than manually enumerating dtypes, we can use Hypothesis to generate diverse test cases."

**4. Deleted Planning Document** (mentioned in `thoughts/shared/research/2025-11-04_05-44-21_dev-environment-onnx-backend-setup.md`):
- Reference to deleted file: `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`
- Likely contained initial planning for property-based testing approach
- Now superseded by actual implementation

### Evolution of Testing Approach

**Phase 1**: Manual tests with explicit examples (test_elemwise.py, test_shape.py, test_subtensor.py)

**Phase 2**: Introduction of property-based testing for reductions (test_math.py)

**Phase 3**: Extension to allocations (test_tensor_basic.py)

**Current State**: Hybrid approach with 27% property-based coverage

**Future Direction**: Full property-based coverage for all ONNX operations as documented in this research

## Related Research

- `thoughts/shared/research/2025-11-04_05-44-21_dev-environment-onnx-backend-setup.md` - Development environment setup and historical context

## Design Decisions

The following questions were resolved on 2025-11-08:

1. **Should all elemwise operations share a single property test, or should operations with special constraints have separate tests?**

   **Decision**: Operations with special constraints (e.g., Pow with negative bases, Sqrt with negative values, Log with non-positive values) should have separate tests.

   **Rationale**: This allows for operation-specific input filtering, specialized error handling, and clearer test failure messages when constraints are violated.

2. **What tolerance values (`rtol`, `atol`) should be used for operations with known numerical instability?**

   **Decision**: Use reasonable tolerance values based on operation characteristics. Default values (`rtol=1e-5`, `atol=1e-8`) are acceptable for most operations. For numerically unstable operations (e.g., Exp, Log, Pow), consider slightly relaxed tolerances (e.g., `rtol=1e-4`).

   **Rationale**: Tolerances should balance numerical accuracy with real-world precision limits. Document any non-default tolerances in test docstrings.

3. **Should subtensor tests cover negative indices and dynamic bounds?**

   **Decision**: No, these should not be tested in property-based tests.

   **Rationale**: Current ONNX backend has known limitations with negative indices (see `subtensor.py:122-127`). Testing unsupported features would create false failures.

4. **Should we test unsupported features as "expected to fail" tests to document limitations?**

   **Decision**: Exclude unsupported features from property tests entirely. Document limitations in code comments and docstrings instead.

   **Rationale**: Property-based tests should validate working functionality. Unsupported features should be documented in implementation files and tracked as future enhancements. Using `pytest.mark.xfail` for property tests can be confusing and makes test results harder to interpret. Clear documentation is preferable.

5. **How should we handle operations that require specific ONNX opset versions?**

   **Decision**: Only test the default opset version (18).

   **Rationale**: Simplifies test infrastructure. If opset version becomes configurable in the future, tests can be extended.

6. **Should the Hypothesis example database (`.hypothesis/` directory) be committed to version control?**

   **Decision**: Remain in `.gitignore`.

   **Rationale**: The example database is local and may contain platform-specific artifacts. Test reproducibility is achieved through Hypothesis's deterministic seed, not through committing the database.

7. **What's the best strategy for operations with broadcasting?**

   **Decision**: Test broadcasting behavior explicitly with dedicated strategies that generate compatible shapes.

   **Rationale**: Broadcasting is a critical feature of elemwise operations and should be validated explicitly. Create strategies that generate pairs of arrays with compatible but different shapes (e.g., `(5, 1)` and `(1, 3)` → broadcast to `(5, 3)`).

8. **Should property tests validate graph structure or only validate numerical correctness?**

   **Decision**: Validate numerical correctness only.

   **Rationale**: The primary goal is to ensure correct computation results. Graph structure validation (e.g., counting ONNX nodes) is brittle and may break with legitimate optimizations. ONNX model validation via `onnx.checker.check_model()` already ensures structural correctness.

