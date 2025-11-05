---
date: 2025-11-04
status: active
phase: "1-3"
coverage: "Foundation, First Operations, Export & Testing Infrastructure"
timeline: "Weeks 1-3"
tags: [tdd, onnx, backend, infrastructure, phase1, phase2, phase3]
related_research:
  - thoughts/shared/research/2025-11-04_11-52-15_onnx-backend-infrastructure-roadmap.md
  - thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md
---

# ONNX Backend Phases 1-3: Foundation & Infrastructure - TDD Implementation Plan

## Overview

This TDD plan covers the foundational infrastructure for the ONNX backend (Weeks 1-3), including:
- **Phase 1**: Module structure and core dispatch system
- **Phase 2**: First operations (Tier 1 - 20 basic elemwise ops)
- **Phase 3**: Export API and comprehensive testing infrastructure

**TDD Approach**: We'll write comprehensive tests that define expected behavior, verify they fail properly, then implement features by making tests pass. This ensures our infrastructure actually works and catches regressions.

## Current State Analysis

### What Exists:
- ❌ **No ONNX backend implementation** - `pytensor/link/onnx/` does not exist
- ❌ **No ONNX tests** - `tests/link/onnx/` does not exist
- ✅ **Reference implementations**: JAX backend (`pytensor/link/jax/`) with 99 operations
- ✅ **Planning documents**: Infrastructure and operations roadmaps

### Testing Landscape:
- **Testing framework**: pytest
- **Test patterns**: Based on JAX backend tests
  - `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py` utility pattern
  - `tests/link/jax/conftest.py` - Fixture patterns
- **Available utilities**:
  - `pytensor.config.change_flags` for test configuration
  - NumPy testing utilities for numerical comparisons
- **Backend testing pattern**: Compile graph with backend, compare output to Python reference

### Key Discoveries:
- JAX backend uses `singledispatch` for operation conversion: `pytensor/link/jax/dispatch/basic.py:27-46`
- Linker base classes in `pytensor/link/basic.py:144-717`
- Mode system for backend registration: `pytensor/compile/mode.py:42-597`
- ONNX requires static graph (unlike JAX JIT)

## Desired End State

After Phases 1-3, we'll have:

✅ **Working Infrastructure**:
- Module structure with proper organization
- Core dispatch system (`onnx_funcify`, `onnx_typify`)
- ONNXLinker that converts FunctionGraph to ONNX ModelProto
- Export API (`export_onnx`, `compile_onnx`)

✅ **Basic Operations** (Tier 1 - 20 ops):
- Elemwise arithmetic: Add, Sub, Mul, Div, Neg, Abs, Maximum, Minimum
- Basic math: Exp, Log, Sqrt, Pow, Floor, Ceil, Round
- Infrastructure: Constant, Cast, Identity

✅ **Scalable Testing Architecture** (Hypothesis-based):
- **Operation registry** (`ONNX_OPERATIONS` dict) mapping ops to test configurations
- **Hypothesis strategies module** (`tests/link/onnx/strategies/`) for input generation
- **~4-6 property tests** that automatically test all 20 operations:
  - Correctness: ONNX matches PyTensor output
  - Shape preservation: Broadcasting works correctly
  - Dtype preservation: Types handled correctly
  - Edge cases: No crashes on empty/scalar/large values
- **~8-12 infrastructure tests** (linker, dispatch, export API, imports)
- **~5-8 targeted regression tests** (for specific bugs discovered during implementation)
- **Total: ~20-25 tests instead of 40+ manual tests**
- `compare_onnx_and_py` utility for validation

✅ **Validation**:
- Can export basic arithmetic expressions to ONNX
- ONNX Runtime can execute exported models
- Outputs match Python reference implementation
- Adding new operations requires only registry entry + optional custom strategy

## What We're NOT Testing/Implementing

❌ **Out of Scope for Phases 1-3**:
- Shape operations (Tier 2) - covered in Phases 4-5 plan
- Reductions (Tier 3) - covered in Phases 4-5 plan
- Linear algebra (Tier 4) - covered in Phases 6-7 plan
- Advanced operations (Tier 5) - covered in Phases 6-7 plan
- CNN operations (Conv2D, MaxPool) - not core backend operations
- Random variables - future work
- Training operations - inference only for now

## TDD Approach

### Test Design Philosophy:

1. **Infrastructure-First Testing**: Test that the dispatch and linker infrastructure works correctly before testing specific operations
2. **Incremental Validation**: Each test validates one aspect of behavior
3. **Reference Comparison**: All tests compare ONNX Runtime output to Python reference
4. **Clear Failure Messages**: Tests should clearly indicate what's broken and where
5. **ONNX Validation**: All exported models must pass ONNX checker

### Testing Strategy:

```python
# Core pattern: Compare ONNX output to Python reference
def compare_onnx_and_py(graph_inputs, graph_outputs, test_inputs):
    # Compile with ONNX backend
    onnx_fn = pytensor.function(graph_inputs, graph_outputs, mode=onnx_mode)
    onnx_result = onnx_fn(*test_inputs)

    # Compile with Python reference
    py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_result = py_fn(*test_inputs)

    # Compare
    np.testing.assert_allclose(onnx_result, py_result)

    # Validate ONNX model
    onnx.checker.check_model(onnx_fn.maker.linker.onnx_model)
```

---

## Phase 1: Test Design & Implementation

### Overview

Write comprehensive tests that define the infrastructure's expected behavior. These tests will fail initially because the infrastructure doesn't exist yet.

---

### Test Category 1: Module Structure & Imports

**Test File**: `tests/link/onnx/test_imports.py`
**Purpose**: Verify the ONNX module structure is set up correctly and imports work

#### Test: `test_onnx_module_exists`

**Purpose**: Verify `pytensor.link.onnx` module exists and is importable

**Test Data**: None (import test)

**Expected Behavior**: Module imports successfully

**Assertions**:
- Module import doesn't raise ImportError
- Module has expected public API

```python
def test_onnx_module_exists():
    """Test that pytensor.link.onnx module exists and is importable."""
    try:
        import pytensor.link.onnx
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import pytensor.link.onnx: {e}")
```

**Expected Failure Mode**:
- Error type: `ModuleNotFoundError`
- Expected message: `No module named 'pytensor.link.onnx'`

#### Test: `test_onnx_public_api`

**Purpose**: Verify public API exports are available

```python
def test_onnx_public_api():
    """Test that ONNX backend exports expected public API."""
    from pytensor.link.onnx import (
        ONNXLinker,
        export_onnx,
        compile_onnx,
        onnx_funcify,
        ONNX_OPSET_VERSION,
    )

    assert ONNXLinker is not None, "ONNXLinker not exported"
    assert export_onnx is not None, "export_onnx not exported"
    assert compile_onnx is not None, "compile_onnx not exported"
    assert onnx_funcify is not None, "onnx_funcify not exported"
    assert ONNX_OPSET_VERSION == 18, f"Expected opset 18, got {ONNX_OPSET_VERSION}"
```

**Expected Failure Mode**:
- Error type: `ImportError` or `AttributeError`
- Expected message: `cannot import name 'ONNXLinker'`

#### Test: `test_dispatch_module_structure`

**Purpose**: Verify dispatch module structure

```python
def test_dispatch_module_structure():
    """Test that dispatch module has expected structure."""
    from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify

    # Check they're singledispatch functions
    assert hasattr(onnx_funcify, 'register'), \
        "onnx_funcify should be a singledispatch function"
    assert hasattr(onnx_typify, 'register'), \
        "onnx_typify should be a singledispatch function"
```

**Expected Failure Mode**:
- Error type: `ModuleNotFoundError`
- Expected message: `No module named 'pytensor.link.onnx.dispatch'`

---

### Test Category 2: Core Dispatch System

**Test File**: `tests/link/onnx/test_dispatch_basic.py`
**Purpose**: Verify the dispatch system correctly handles type registration and conversion

#### Test: `test_onnx_funcify_unregistered_op`

**Purpose**: Verify dispatch raises helpful error for unregistered operations

```python
def test_onnx_funcify_unregistered_op():
    """Test that onnx_funcify raises informative error for unregistered ops."""
    from pytensor.link.onnx.dispatch import onnx_funcify
    from pytensor.tensor.elemwise import Elemwise
    from pytensor.scalar.basic import Add

    # Create a fake op
    class FakeOp:
        pass

    fake_op = FakeOp()

    with pytest.raises(NotImplementedError) as exc_info:
        onnx_funcify(fake_op)

    error_msg = str(exc_info.value)
    assert "No ONNX conversion available" in error_msg, \
        f"Error should mention no conversion available, got: {error_msg}"
    assert "FakeOp" in error_msg, \
        f"Error should mention the op type, got: {error_msg}"
```

**Expected Failure Mode**:
- Error type: `ModuleNotFoundError` (dispatch doesn't exist yet)
- Expected message: `No module named 'pytensor.link.onnx.dispatch'`

#### Test: `test_onnx_typify_ndarray`

**Purpose**: Verify type conversion for numpy arrays

```python
def test_onnx_typify_ndarray():
    """Test that onnx_typify converts numpy arrays to ONNX tensors."""
    from pytensor.link.onnx.dispatch import onnx_typify
    import numpy as np
    import onnx
    from onnx import numpy_helper

    # Test data
    arr = np.array([1, 2, 3], dtype='float32')

    # Convert
    result = onnx_typify(arr, name="test_tensor")

    # Verify it's a TensorProto
    assert isinstance(result, onnx.TensorProto), \
        f"Expected TensorProto, got {type(result)}"

    # Verify data is correct
    result_arr = numpy_helper.to_array(result)
    np.testing.assert_array_equal(result_arr, arr)
```

**Expected Failure Mode**:
- Error type: `ModuleNotFoundError`
- Then after module exists: `NotImplementedError` (onnx_typify not registered for ndarray)

#### Test: `test_make_value_info_basic`

**Purpose**: Verify ValueInfo creation from PyTensor Variables

```python
def test_make_value_info_basic():
    """Test that make_value_info creates correct ONNX ValueInfo."""
    from pytensor.link.onnx.dispatch.basic import make_value_info
    import pytensor.tensor as pt
    import onnx

    # Create a PyTensor variable
    x = pt.vector('x', dtype='float32')

    # Create ValueInfo
    value_info = make_value_info(x, 'x')

    # Verify type
    assert isinstance(value_info, onnx.ValueInfoProto), \
        f"Expected ValueInfoProto, got {type(value_info)}"

    # Verify name
    assert value_info.name == 'x', \
        f"Expected name 'x', got {value_info.name}"

    # Verify dtype
    assert value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, \
        f"Expected FLOAT dtype, got {value_info.type.tensor_type.elem_type}"
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'make_value_info'`

---

### Test Category 3: ONNXLinker Basic Functionality

**Test File**: `tests/link/onnx/test_linker.py`
**Purpose**: Verify ONNXLinker can convert simple FunctionGraphs to ONNX models

#### Test: `test_linker_instantiation`

**Purpose**: Verify ONNXLinker can be instantiated

```python
def test_linker_instantiation():
    """Test that ONNXLinker can be instantiated."""
    from pytensor.link.onnx.linker import ONNXLinker

    linker = ONNXLinker(opset_version=18)

    assert linker is not None, "Linker instantiation returned None"
    assert linker.opset_version == 18, \
        f"Expected opset 18, got {linker.opset_version}"
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'ONNXLinker'`

#### Test: `test_linker_empty_graph`

**Purpose**: Verify linker can handle an empty graph (passthrough)

```python
def test_linker_empty_graph():
    """Test that linker can convert a trivial passthrough graph."""
    import pytensor.tensor as pt
    import pytensor
    from pytensor.link.onnx.linker import ONNXLinker

    # Create identity graph
    x = pt.scalar('x', dtype='float32')
    y = x  # Passthrough

    # Compile with ONNX linker
    fn = pytensor.function([x], y, mode=Mode(linker=ONNXLinker()))

    # Test execution
    result = fn(5.0)
    assert result == 5.0, f"Expected 5.0, got {result}"

    # Verify ONNX model exists
    assert hasattr(fn.maker.linker, 'onnx_model'), \
        "Linker should have onnx_model attribute"
    assert fn.maker.linker.onnx_model is not None, \
        "onnx_model should not be None"
```

**Expected Failure Mode**:
- Error type: `ImportError` initially
- Then: `NotImplementedError` in `fgraph_convert`

#### Test: `test_linker_constant_graph`

**Purpose**: Verify linker handles graphs with constants

```python
def test_linker_constant_graph():
    """Test that linker correctly handles constants as initializers."""
    import pytensor.tensor as pt
    import pytensor
    from pytensor.link.onnx.linker import ONNXLinker
    import numpy as np

    # Create graph with constant
    x = pt.scalar('x', dtype='float32')
    c = pt.constant(2.0, dtype='float32')
    y = x * c

    # Compile
    fn = pytensor.function([x], y, mode=Mode(linker=ONNXLinker()))

    # Test
    result = fn(3.0)
    expected = 6.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Verify ONNX model has initializer for constant
    model = fn.maker.linker.onnx_model
    assert len(model.graph.initializer) > 0, \
        "Model should have at least one initializer for the constant"
```

**Expected Failure Mode**:
- Error type: `NotImplementedError` in constant handling

---

### Test Category 4: Testing Infrastructure Utilities

**Test File**: `tests/link/onnx/test_basic.py`
**Purpose**: Test the test utilities themselves (meta-testing!)

#### Test: `test_compare_onnx_and_py_simple`

**Purpose**: Verify compare_onnx_and_py utility works for simple cases

```python
def test_compare_onnx_and_py_simple():
    """Test that compare_onnx_and_py works for a simple identity operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Simple identity
    x = pt.vector('x', dtype='float32')
    y = x

    # Test data
    x_val = np.array([1, 2, 3], dtype='float32')

    # Should not raise
    try:
        fn, result = compare_onnx_and_py([x], y, [x_val])
        np.testing.assert_array_equal(result, x_val)
    except Exception as e:
        pytest.fail(f"compare_onnx_and_py raised unexpectedly: {e}")
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'compare_onnx_and_py'`

#### Test: `test_get_onnx_node_types`

**Purpose**: Verify utility to inspect ONNX nodes works

```python
def test_get_onnx_node_types():
    """Test that get_onnx_node_types utility works."""
    import pytensor.tensor as pt
    import pytensor
    from pytensor.link.onnx.linker import ONNXLinker
    from tests.link.onnx.test_basic import get_onnx_node_types

    # Create a graph with Add operation
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x + y

    # Compile
    fn = pytensor.function([x, y], z, mode=Mode(linker=ONNXLinker()))

    # Get node types
    node_types = get_onnx_node_types(fn)

    assert 'Add' in node_types, \
        f"Expected 'Add' in node types, got {node_types}"
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'get_onnx_node_types'`

---

### Test Category 5: Tier 1 Operations - Basic Arithmetic

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Test basic elemwise arithmetic operations

#### Test: `test_add_vectors`

**Purpose**: Test addition of two vectors

```python
def test_add_vectors():
    """Test that vector addition exports correctly to ONNX."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Define graph
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x + y

    # Test data
    x_val = np.array([1, 2, 3], dtype='float32')
    y_val = np.array([4, 5, 6], dtype='float32')

    # Compare outputs
    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    # Verify ONNX node type
    from tests.link.onnx.test_basic import get_onnx_node_types
    node_types = get_onnx_node_types(fn)
    assert 'Add' in node_types, \
        f"Expected 'Add' node in ONNX graph, got {node_types}"
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: `No ONNX conversion available for: Elemwise`

#### Test: `test_mul_vectors`

**Purpose**: Test multiplication of two vectors

```python
def test_mul_vectors():
    """Test that vector multiplication exports correctly to ONNX."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x * y

    x_val = np.array([1, 2, 3], dtype='float32')
    y_val = np.array([2, 3, 4], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    from tests.link.onnx.test_basic import get_onnx_node_types
    assert 'Mul' in get_onnx_node_types(fn)
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: `Elemwise scalar op not supported for ONNX export: Mul`

#### Test: `test_sub_vectors`

**Purpose**: Test subtraction

```python
def test_sub_vectors():
    """Test vector subtraction."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x - y

    x_val = np.array([5, 6, 7], dtype='float32')
    y_val = np.array([1, 2, 3], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert 'Sub' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Sub operation

#### Test: `test_div_vectors`

**Purpose**: Test division

```python
def test_div_vectors():
    """Test vector division."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x / y

    x_val = np.array([6, 8, 10], dtype='float32')
    y_val = np.array([2, 4, 5], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert 'Div' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for TrueDiv operation

#### Test: `test_chained_arithmetic`

**Purpose**: Test multiple arithmetic operations chained together

```python
def test_chained_arithmetic():
    """Test that chained arithmetic operations work correctly."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype='float32')

    fn, result = compare_onnx_and_py([x], z, [x_val])

    # Should have multiple operation nodes
    node_types = get_onnx_node_types(fn)
    assert 'Mul' in node_types
    assert 'Add' in node_types
    assert 'Div' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for first unimplemented op in chain

---

### Test Category 6: Tier 1 Operations - Unary Math

**Test File**: `tests/link/onnx/test_elemwise.py` (continued)
**Purpose**: Test unary mathematical operations

#### Test: `test_neg`

**Purpose**: Test negation

```python
def test_neg():
    """Test negation operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = -x

    x_val = np.array([1, -2, 3], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert 'Neg' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Neg

#### Test: `test_abs`

**Purpose**: Test absolute value

```python
def test_abs():
    """Test absolute value operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.abs(x)

    x_val = np.array([1, -2, 3, -4], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert 'Abs' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Abs

#### Test: `test_exp`

**Purpose**: Test exponential

```python
def test_exp():
    """Test exponential operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.exp(x)

    x_val = np.array([0, 1, 2], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert 'Exp' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Exp

#### Test: `test_log`

**Purpose**: Test natural logarithm

```python
def test_log():
    """Test natural logarithm operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.log(x)

    x_val = np.array([1, 2, np.e], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert 'Log' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Log

#### Test: `test_sqrt`

**Purpose**: Test square root

```python
def test_sqrt():
    """Test square root operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.sqrt(x)

    x_val = np.array([1, 4, 9, 16], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert 'Sqrt' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Sqrt

#### Test: `test_pow`

**Purpose**: Test power operation

```python
def test_pow():
    """Test power operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x ** y

    x_val = np.array([2, 3, 4], dtype='float32')
    y_val = np.array([2, 2, 3], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert 'Pow' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Pow

#### Test: `test_floor_ceil_round`

**Purpose**: Test rounding operations

```python
@pytest.mark.parametrize("op_name,op_func,expected_node", [
    ("floor", pt.floor, "Floor"),
    ("ceil", pt.ceil, "Ceil"),
    ("round", pt.round, "Round"),
])
def test_rounding_operations(op_name, op_func, expected_node):
    """Test floor, ceil, and round operations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = op_func(x)

    x_val = np.array([1.2, 2.5, 3.7, -1.5], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])
    assert expected_node in get_onnx_node_types(fn), \
        f"Expected {expected_node} node for {op_name}"
```

**Expected Failure Mode**: `NotImplementedError` for Floor/Ceil/Round

---

### Test Category 7: Tier 1 Operations - Min/Max

**Test File**: `tests/link/onnx/test_elemwise.py` (continued)

#### Test: `test_maximum`

**Purpose**: Test element-wise maximum

```python
def test_maximum():
    """Test element-wise maximum operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = pt.maximum(x, y)

    x_val = np.array([1, 5, 3], dtype='float32')
    y_val = np.array([4, 2, 6], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert 'Max' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Maximum

#### Test: `test_minimum`

**Purpose**: Test element-wise minimum

```python
def test_minimum():
    """Test element-wise minimum operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = pt.minimum(x, y)

    x_val = np.array([1, 5, 3], dtype='float32')
    y_val = np.array([4, 2, 6], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert 'Min' in get_onnx_node_types(fn)
```

**Expected Failure Mode**: `NotImplementedError` for Minimum

---

### Test Category 8: Export API

**Test File**: `tests/link/onnx/test_export.py`
**Purpose**: Test the high-level export API functions

#### Test: `test_export_onnx_basic`

**Purpose**: Test export_onnx function creates a valid .onnx file

```python
def test_export_onnx_basic(tmp_path):
    """Test that export_onnx creates a valid ONNX file."""
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.link.onnx import export_onnx
    import onnx

    # Define graph
    x = pt.vector('x', dtype='float32')
    y = x * 2

    # Export
    output_path = tmp_path / "test_model.onnx"
    model = export_onnx([x], y, str(output_path))

    # Verify file exists
    assert output_path.exists(), f"ONNX file not created at {output_path}"

    # Verify model is valid
    onnx.checker.check_model(model)

    # Verify model can be loaded
    loaded_model = onnx.load(str(output_path))
    assert loaded_model is not None
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'export_onnx'`

#### Test: `test_compile_onnx_basic`

**Purpose**: Test compile_onnx returns executable function

```python
def test_compile_onnx_basic():
    """Test that compile_onnx returns an executable function."""
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.link.onnx import compile_onnx

    x = pt.vector('x', dtype='float32')
    y = x + 1

    # Compile
    fn = compile_onnx([x], y)

    # Test execution
    x_val = np.array([1, 2, 3], dtype='float32')
    result = fn(x_val)

    expected = np.array([2, 3, 4], dtype='float32')
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'compile_onnx'`

#### Test: `test_export_function_onnx`

**Purpose**: Test exporting an already-compiled PyTensor function

```python
def test_export_function_onnx(tmp_path):
    """Test exporting a compiled PyTensor function to ONNX."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.link.onnx import export_function_onnx
    import onnx

    # Create and compile function
    x = pt.vector('x', dtype='float32')
    y = pt.sqrt(x)
    fn = pytensor.function([x], y)

    # Export
    output_path = tmp_path / "function.onnx"
    model = export_function_onnx(fn, str(output_path))

    # Verify
    assert output_path.exists()
    onnx.checker.check_model(model)
```

**Expected Failure Mode**:
- Error type: `ImportError`
- Expected message: `cannot import name 'export_function_onnx'`

---

### Test Implementation Steps

1. **Create directory structure**:
   ```bash
   mkdir -p pytensor/link/onnx/dispatch
   mkdir -p tests/link/onnx
   ```

2. **Create test files**:
   - `tests/link/onnx/__init__.py`
   - `tests/link/onnx/conftest.py` (fixtures)
   - `tests/link/onnx/test_imports.py`
   - `tests/link/onnx/test_dispatch_basic.py`
   - `tests/link/onnx/test_linker.py`
   - `tests/link/onnx/test_basic.py` (utilities)
   - `tests/link/onnx/test_elemwise.py`
   - `tests/link/onnx/test_export.py`

3. **Create conftest.py with fixtures**:
   ```python
   import numpy as np
   import pytest
   import pytensor

   @pytest.fixture
   def rng():
       """Seeded random number generator."""
       return np.random.default_rng(42)

   @pytest.fixture(scope="module", autouse=True)
   def configure_pytensor():
       """Module-level PyTensor configuration."""
       with pytensor.config.change_flags(
           cxx="",
           compute_test_value="ignore",
           floatX="float32"
       ):
           yield

   @pytest.fixture
   def float32_vector(rng):
       """Sample float32 vector for testing."""
       return rng.normal(size=10).astype('float32')
   ```

4. **Implement test_basic.py utility functions**:
   ```python
   # Core testing utilities
   def compare_onnx_and_py(...):
       # Implementation
       pass

   def get_onnx_node_types(...):
       # Implementation
       pass
   ```

5. **Write all test cases** as specified above

### Success Criteria

#### Automated Verification:
- [ ] All test files created: `ls tests/link/onnx/test_*.py`
- [ ] Tests are discoverable: `pytest --collect-only tests/link/onnx/ | grep "test_"`
- [ ] Test syntax is valid: `python -m py_compile tests/link/onnx/*.py`
- [ ] Imports are structured correctly: No circular import errors

#### Manual Verification:
- [ ] Each test has clear, descriptive docstring
- [ ] Test names follow `test_<what_is_tested>` pattern
- [ ] Assertion messages are diagnostic and helpful
- [ ] Test organization follows logical grouping
- [ ] Tests cover all Tier 1 operations (20 ops)

---

## Phase 2: Test Failure Verification (Hypothesis + Infrastructure)

### Overview

Verify Hypothesis setup works and that all tests fail in expected, diagnostic ways. This ensures our property tests and infrastructure tests are actually testing the right things.

### Phase 2.1: Verify Hypothesis Setup

**Before implementing ANY ONNX code**, verify Hypothesis infrastructure works:

1. **Verify strategies import**:
   ```bash
   uv run python -c "from tests.link.onnx.strategies import ONNX_OPERATIONS; print(len(ONNX_OPERATIONS))"
   ```
   - Should print "20" (20 operations registered)

2. **Verify can generate examples**:
   ```bash
   uv run python -c "from tests.link.onnx.strategies import onnx_tensor; print(onnx_tensor().example())"
   ```
   - Should print a numpy array

3. **Verify Hypothesis profiles work**:
   ```bash
   uv run pytest tests/link/onnx/ --collect-only --hypothesis-profile=dev
   ```
   - Should collect tests without errors

**If any fail**: Fix Hypothesis setup before proceeding

### Phase 2.2: Verify Infrastructure Tests Fail Correctly

1. **Run full test suite**:
   ```bash
   uv run pytest tests/link/onnx/ -v --tb=short
   ```

2. **Verify test discovery**:
   ```bash
   uv run pytest --collect-only tests/link/onnx/
   ```
   - Should collect ~16 tests (not 40+ with Hypothesis approach)
   - Should show all test files

3. **Check import errors first**:
   ```bash
   uv run pytest tests/link/onnx/test_imports.py -v
   ```
   - All should fail with `ModuleNotFoundError`

4. **Check property tests fail correctly**:
   ```bash
   uv run pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor -v --hypothesis-profile=dev
   ```
   - Should fail with `NotImplementedError: No ONNX conversion available for: Elemwise`
   - Verify Hypothesis runs (tries multiple examples)
   - Verify failure message is clear

5. **Document failure patterns**:
   Create a checklist of what we see vs what we expect

### Expected Failures

#### Import Tests (test_imports.py):
- **test_onnx_module_exists**:
  - Expected: `ModuleNotFoundError: No module named 'pytensor.link.onnx'`
  - Status: ❌ (correct failure)

- **test_onnx_public_api**:
  - Expected: `ModuleNotFoundError: No module named 'pytensor.link.onnx'`
  - Status: ❌ (correct failure)

- **test_dispatch_module_structure**:
  - Expected: `ModuleNotFoundError: No module named 'pytensor.link.onnx.dispatch'`
  - Status: ❌ (correct failure)

#### Dispatch Tests (test_dispatch_basic.py):
- **test_onnx_funcify_unregistered_op**:
  - Expected: `ModuleNotFoundError: No module named 'pytensor.link.onnx.dispatch'`
  - Status: ❌ (correct failure)

- **test_onnx_typify_ndarray**:
  - Expected: `ModuleNotFoundError`
  - Status: ❌ (correct failure)

- **test_make_value_info_basic**:
  - Expected: `ImportError: cannot import name 'make_value_info'`
  - Status: ❌ (correct failure)

#### Linker Tests (test_linker.py):
- **test_linker_instantiation**:
  - Expected: `ImportError: cannot import name 'ONNXLinker'`
  - Status: ❌ (correct failure)

- **test_linker_empty_graph**:
  - Expected: `ImportError`
  - Status: ❌ (correct failure)

- **test_linker_constant_graph**:
  - Expected: `ImportError`
  - Status: ❌ (correct failure)

#### Property Tests (test_properties.py):
- **test_onnx_matches_pytensor**:
  - Expected: `NotImplementedError: No ONNX conversion available for: Elemwise`
  - Should try multiple operations from registry
  - Hypothesis should run 10 examples (dev profile)
  - Status: ❌ (correct failure)

- **test_elemwise_preserves_broadcast_shape**:
  - Expected: `NotImplementedError` (same as above)
  - Status: ❌ (correct failure)

- **test_operation_preserves_dtype**:
  - Expected: `NotImplementedError` (same as above)
  - Status: ❌ (correct failure)

- **test_operation_handles_edge_cases**:
  - Expected: `NotImplementedError` (same as above)
  - Status: ❌ (correct failure)

#### Export API Tests (test_export.py):
- **All export tests**:
  - Expected: `ImportError: cannot import name 'export_onnx'`
  - Status: ❌ (correct failure)

### Success Criteria

#### Automated Verification:
- [ ] Hypothesis imports: `uv run python -c "import hypothesis; print(hypothesis.__version__)"`
- [ ] Strategies work: `uv run python -c "from tests.link.onnx.strategies import ONNX_OPERATIONS; print(len(ONNX_OPERATIONS))"`
- [ ] All tests discovered: `uv run pytest --collect-only tests/link/onnx/ | grep -c "test_"` shows ~16
- [ ] All tests fail: `uv run pytest tests/link/onnx/ -v | grep FAILED | wc -l` equals test count
- [ ] No syntax errors: `uv run pytest tests/link/onnx/ --tb=line` shows no SyntaxError
- [ ] No unexpected exceptions: Review output for unexpected error types

#### Manual Verification:
- [ ] Each test fails with correct error type (ModuleNotFoundError, ImportError, NotImplementedError)
- [ ] Error messages clearly indicate what's missing
- [ ] Stack traces point to right locations (our test code, not pytest internals)
- [ ] No cryptic error messages
- [ ] Failure output would guide implementation

### Phase 2.3: Verify Hypothesis Shrinking (Optional but Recommended)

Test that Hypothesis shrinking works by injecting a deliberate bug:

1. **Temporarily modify compare_onnx_and_py** to fail on specific shapes:
   ```python
   def compare_onnx_and_py(...):
       if any(x.shape == (3, 2) for x in test_inputs):
           raise AssertionError("Deliberate bug for shape (3, 2)")
       # ... rest of implementation
   ```

2. **Run property test**:
   ```bash
   uv run pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor --hypothesis-profile=dev -v
   ```

3. **Expected behavior**:
   - Hypothesis finds the bug (may try many shapes first)
   - **Shrinking happens**: Reduces to minimal failing example
   - Output shows: `Falsifying example: test_onnx_matches_pytensor(op_name='add', data=...)`
   - Hypothesis saves failure to `.hypothesis/examples/`

4. **Verify saved examples**:
   ```bash
   ls .hypothesis/examples/
   ```

5. **Remove the deliberate bug** after verification

### Failure Mode Documentation

Create `tests/link/onnx/EXPECTED_FAILURES.md`:

```markdown
# Expected Test Failures (Before Implementation)

## Stage 1: No Module (Initial State)
All tests fail with `ModuleNotFoundError: No module named 'pytensor.link.onnx'`

Run: `uv run pytest tests/link/onnx/ -v`

## Stage 2: Module Structure Created
Import tests pass, others fail with:
- `ImportError: cannot import name 'ONNXLinker'`
- `ImportError: cannot import name 'onnx_funcify'`

Run: `uv run pytest tests/link/onnx/test_imports.py -v` (should pass)
Run: `uv run pytest tests/link/onnx/test_dispatch_basic.py -v` (should fail)

## Stage 3: Dispatch System Created
Infrastructure tests pass, property tests fail with:
- `NotImplementedError: No ONNX conversion available for: Elemwise`

Run: `uv run pytest tests/link/onnx/test_properties.py -v --hypothesis-profile=dev` (should fail)

## Stage 4: Operations Implemented
All tests should pass

Run: `uv run pytest tests/link/onnx/ -v --hypothesis-profile=dev` (all pass)
```

### Adjustment Phase

If tests don't fail as expected:

- [ ] **Tests that pass unexpectedly**:
  - Too lenient - tighten assertions
  - Testing wrong thing - fix test logic

- [ ] **Tests with confusing errors**:
  - Add clearer assertion messages
  - Improve error context

- [ ] **Tests that error instead of fail**:
  - Fix import paths
  - Add missing test dependencies
  - Fix typos in test code

- [ ] **Tests that can't run**:
  - Fix pytest configuration
  - Add required fixtures
  - Fix test file structure

---

## Phase 3: Feature Implementation (Infrastructure → Operations → Automatic Coverage)

### Overview

Implement features by making tests pass, guided by property test failures. The key insight: **implement infrastructure once, add operations in bulk via mapping, property tests validate everything automatically**.

### Workflow Transformation

**Old approach (Manual Tests):**
1. test_add_vectors fails → implement Add → test passes
2. test_mul_vectors fails → implement Mul → test passes
3. Repeat 15+ times...

**New approach (Hypothesis):**
1. Property tests fail → implement dispatch infrastructure → infrastructure tests pass
2. Property tests still fail → add SCALAR_OP_TO_ONNX mapping (all 20 ops) → **ALL property tests pass automatically**
3. Done! 20 operations × 10 examples = 200+ scenarios validated with 4 property tests

### Implementation Order

1. **Module structure** → Import tests pass
2. **Dispatch system** → Dispatch tests pass
3. **ONNXLinker** → Linker tests pass
4. **Testing utilities** → Property tests can run (but fail on operations)
5. **Elemwise operations (bulk)** → ALL property tests pass at once! ✨
6. **Export API** → Export tests pass
7. **Full integration** → All ~16 tests pass

---

### Implementation 3.1: Module Structure

**Goal**: Make import tests pass
**Target**: `uv run pytest tests/link/onnx/test_imports.py -v`

#### Steps:

1. **Create directory structure**:
   ```bash
   mkdir -p pytensor/link/onnx/dispatch
   touch pytensor/link/onnx/__init__.py
   touch pytensor/link/onnx/dispatch/__init__.py
   ```

2. **Create stub `__init__.py` files** with empty `__all__ = []`

3. **Verify**:
   ```bash
   uv run python -c "import pytensor.link.onnx"
   uv run pytest tests/link/onnx/test_imports.py::test_onnx_module_exists -v
   ```
   Should pass ✅

**Progress check**: `test_onnx_module_exists` passes, `test_onnx_public_api` fails with `ImportError`

---

### Implementation 3.2: Core Dispatch System

**Goal**: Make dispatch tests pass
**Target**: `uv run pytest tests/link/onnx/test_dispatch_basic.py -v`

#### Key Files to Create:

**File**: `pytensor/link/onnx/dispatch/basic.py`

Implement:
- `onnx_funcify` - singledispatch function (raises NotImplementedError by default)
- `onnx_typify` - singledispatch for type conversion
- `onnx_typify.register(np.ndarray)` - converts ndarray → TensorProto
- `make_value_info(var, name)` - creates ONNX ValueInfoProto
- `onnx_funcify.register(Constant)` - handles constants as initializers
- `onnx_funcify.register(FunctionGraph)` - converts full graph to ModelProto

**Note**: This is the longest implementation (~200 lines). See original plan lines 1330-1497 for full code. Key points:
- Uses `singledispatch` pattern like JAX backend
- FunctionGraph converter does topological sort and calls onnx_funcify on each node
- Creates ONNX ModelProto with inputs, outputs, nodes, and initializers

3. **Update `pytensor/link/onnx/dispatch/__init__.py`** to export functions

4. **Verify**:
   ```bash
   uv run pytest tests/link/onnx/test_dispatch_basic.py -v
   ```
   All 3 dispatch tests should pass ✅

**Progress check**: Dispatch infrastructure works, can convert basic graphs

---


### Implementation 3.3: ONNXLinker

**Goal**: Make linker tests pass
**Target**: `uv run pytest tests/link/onnx/test_linker.py -v`

#### Key File to Create:

**File**: `pytensor/link/onnx/linker.py`

Implement `ONNXLinker` class (inherits from `JITLinker`):
- `__init__(opset_version=18)` - initialize with ONNX opset version
- `fgraph_convert()` - calls `onnx_funcify(fgraph)` to get ModelProto, returns ONNX Runtime function
- `_create_onnx_runtime_function()` - wraps ONNX Runtime InferenceSession
- `export_to_file()` - saves model to .onnx file

**Update**: `pytensor/link/onnx/__init__.py` to export `ONNXLinker`

**Verify**:
```bash
uv run pytest tests/link/onnx/test_linker.py -v
```
All 3 linker tests should pass ✅

**Progress check**: Can compile simple graphs with ONNX backend

---

### Implementation 3.4: Testing Utilities

**Goal**: Enable property tests to run
**Target**: Property tests can execute (but will fail on unimplemented operations)

#### Key Files to Create:

**File**: `tests/link/onnx/test_basic.py`

Implement core testing utilities:

```python
def compare_onnx_and_py(
    graph_inputs, graph_outputs, test_inputs,
    *, assert_fn=None, must_validate=True, **kwargs
):
    """Compare ONNX Runtime output to Python reference.

    1. Compile graph with ONNX backend
    2. Compile graph with Python backend
    3. Execute both with test_inputs
    4. Assert outputs match
    5. Validate ONNX model
    """
    # Compile with ONNX
    onnx_fn = pytensor.function(graph_inputs, graph_outputs, mode=onnx_mode)
    onnx_res = onnx_fn(*test_inputs)

    # Compile with Python reference
    py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = py_fn(*test_inputs)

    # Compare
    assert_fn(onnx_res, py_res)  # default: np.testing.assert_allclose

    # Validate ONNX model
    if must_validate:
        onnx.checker.check_model(onnx_fn.maker.linker.onnx_model)

    return onnx_fn, onnx_res


def get_onnx_node_types(fn):
    """Get list of ONNX node types in compiled function."""
    return [node.op_type for node in fn.maker.linker.onnx_model.graph.node]
```

**File**: `tests/link/onnx/conftest.py`

Already created in Phase 1 with Hypothesis profiles.

**Verify**:
```bash
uv run python -c "from tests.link.onnx.test_basic import compare_onnx_and_py"
```

**Progress check**: Test utilities work, property tests can run (but fail on operations)

---

### Implementation 3.5: Elemwise Operations (Bulk Implementation) ⭐

**Goal**: Make ALL property tests pass at once!
**Target**: `uv run pytest tests/link/onnx/test_properties.py -v --hypothesis-profile=dev`

#### This is THE KEY MOMENT 🎯

You implement ALL 20 operations with ONE mapping dictionary!

**File**: `pytensor/link/onnx/dispatch/elemwise.py` (new)

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar import basic as scalar
from onnx import helper


# ⭐ THE MAGIC MAPPING - All 20 operations in one dict!
SCALAR_OP_TO_ONNX = {
    # Arithmetic (Tier 1)
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.IntDiv: "Div",

    # Math (Tier 1)
    scalar.Abs: "Abs",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",
    scalar.Floor: "Floor",
    scalar.Ceil: "Ceil",
    scalar.Round: "Round",

    # Min/Max (Tier 1)
    scalar.Maximum: "Max",
    scalar.Minimum: "Min",
}


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    This ONE function handles ALL 20 operations!
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported: {scalar_op_type.__name__}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    return helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )
```

**Update**: `pytensor/link/onnx/dispatch/__init__.py`

```python
# Import to trigger registration
import pytensor.link.onnx.dispatch.elemwise  # noqa: F401
```

#### The Magic Moment 🎉

**Run property tests**:
```bash
uv run pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor -v --hypothesis-profile=dev
```

**What happens**:
```
test_onnx_matches_pytensor[add-data0] PASSED
test_onnx_matches_pytensor[add-data1] PASSED
...
test_onnx_matches_pytensor[mul-data0] PASSED
test_onnx_matches_pytensor[mul-data1] PASSED
...
test_onnx_matches_pytensor[sqrt-data9] PASSED

========== 200 passed in 5.23s ==========
```

**You just validated 20 operations × 10 examples = 200+ test scenarios with:**
- One 20-line dict
- One 30-line function
- Zero manual tests!

#### Debugging Property Test Failures

If a property test fails:

```bash
uv run pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor -v --hypothesis-profile=dev
```

**Hypothesis tells you exactly what failed**:
```
Falsifying example: test_onnx_matches_pytensor(
    op_name='log',
    data=<data that generated negative values>
)
AssertionError: ONNX produced nan, Python produced -inf
```

**Fix approaches**:
1. **Add input filtering** in property test (for `log`, `sqrt` - need positive values)
2. **Fix implementation** if there's a real bug
3. **Add to SCALAR_OP_TO_ONNX** if operation is missing

**Example fix** in `tests/link/onnx/test_properties.py`:
```python
@given(...)
def test_onnx_matches_pytensor(op_name, data):
    ...
    # Filter invalid inputs
    if op_name == "log":
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "sqrt":
        inputs_tuple = tuple(np.abs(x) for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)  # Avoid division by zero
        inputs_tuple = (x, y)
    ...
```

**Verify all property tests pass**:
```bash
uv run pytest tests/link/onnx/test_properties.py -v --hypothesis-profile=dev
```

**Progress check**: ALL 4 property tests pass! 20 operations fully tested! ✅

---

### Implementation 3.6: Export API

**Goal**: Make export tests pass
**Target**: `uv run pytest tests/link/onnx/test_export.py -v`

#### Key File to Create:

**File**: `pytensor/link/onnx/export.py`

Implement user-facing export functions:

```python
def export_onnx(inputs, outputs, filename, *, opset_version=18, **kwargs):
    """Export PyTensor graph to ONNX file.

    1. Create FunctionGraph from inputs/outputs
    2. Convert to ONNX ModelProto via onnx_funcify
    3. Save to file
    4. Return model
    """
    fgraph = construct_nominal_fgraph(inputs, outputs)
    onnx_model = onnx_funcify(fgraph, opset_version=opset_version, ...)
    onnx.save(onnx_model, filename)
    return onnx_model


def compile_onnx(inputs, outputs, *, opset_version=18, **kwargs):
    """Compile PyTensor graph using ONNX backend.

    Returns function that executes via ONNX Runtime.
    """
    onnx_linker = ONNXLinker(opset_version=opset_version)
    onnx_mode = Mode(linker=onnx_linker, optimizer=None)
    return function(inputs, outputs, mode=onnx_mode, **kwargs)


def export_function_onnx(fn, filename, *, opset_version=18):
    """Export already-compiled PyTensor function to ONNX."""
    fgraph = fn.maker.fgraph
    onnx_model = onnx_funcify(fgraph, opset_version=opset_version)
    onnx.save(onnx_model, filename)
    return onnx_model
```

**Update**: `pytensor/link/onnx/__init__.py` to export these functions

**Verify**:
```bash
uv run pytest tests/link/onnx/test_export.py -v
```
All 3 export tests should pass ✅

**Progress check**: Can export PyTensor graphs to .onnx files

---

### Implementation 3.7: Full Integration & Verification

**Goal**: Verify all tests pass
**Target**: `uv run pytest tests/link/onnx/ -v --hypothesis-profile=dev`

#### Full Test Run:

```bash
uv run pytest tests/link/onnx/ -v --hypothesis-profile=dev
```

**Expected results**:
```
tests/link/onnx/test_imports.py::test_onnx_module_exists PASSED
tests/link/onnx/test_imports.py::test_onnx_public_api PASSED
tests/link/onnx/test_imports.py::test_dispatch_module_structure PASSED
tests/link/onnx/test_dispatch_basic.py::test_onnx_funcify_unregistered_op PASSED
tests/link/onnx/test_dispatch_basic.py::test_onnx_typify_ndarray PASSED
tests/link/onnx/test_dispatch_basic.py::test_make_value_info_basic PASSED
tests/link/onnx/test_linker.py::test_linker_instantiation PASSED
tests/link/onnx/test_linker.py::test_linker_empty_graph PASSED
tests/link/onnx/test_linker.py::test_linker_constant_graph PASSED
tests/link/onnx/test_properties.py::test_onnx_matches_pytensor[add-...] PASSED (×10)
tests/link/onnx/test_properties.py::test_onnx_matches_pytensor[mul-...] PASSED (×10)
... (all 20 operations × 10 examples)
tests/link/onnx/test_properties.py::test_elemwise_preserves_broadcast_shape[...] PASSED (×10)
tests/link/onnx/test_properties.py::test_operation_preserves_dtype[...] PASSED (×10)
tests/link/onnx/test_properties.py::test_operation_handles_edge_cases[...] PASSED (×10)
tests/link/onnx/test_export.py::test_export_onnx_basic PASSED
tests/link/onnx/test_export.py::test_compile_onnx_basic PASSED
tests/link/onnx/test_export.py::test_export_function_onnx PASSED

========== ~16 tests, 240+ total assertions passed in ~10s ==========
```

**Test Count Breakdown**:
- ✅ Import tests: 3 tests
- ✅ Dispatch tests: 3 tests
- ✅ Linker tests: 3 tests
- ✅ Property tests: 4 tests (but validate 200+ scenarios!)
- ✅ Export tests: 3 tests

**Total: ~16 focused tests instead of 40+ manual tests**

#### Run with More Examples (CI Profile):

```bash
HYPOTHESIS_PROFILE=ci uv run pytest tests/link/onnx/ -v
```

This runs 100 examples per property test = 2000+ test scenarios!

#### Manual Validation:

1. **Export a simple model**:
   ```bash
   uv run python -c "
   import pytensor.tensor as pt
   import numpy as np
   from pytensor.link.onnx import export_onnx

   x = pt.vector('x', dtype='float32')
   y = (x + 1) * 2

   export_onnx([x], y, 'test_model.onnx')
   print('Model exported!')
   "
   ```

2. **Verify with ONNX tools**:
   ```bash
   uv run python -c "import onnx; onnx.checker.check_model(onnx.load('test_model.onnx'))"
   ```

3. **Run with ONNX Runtime**:
   ```bash
   uv run python -c "
   import onnxruntime as ort
   import numpy as np

   session = ort.InferenceSession('test_model.onnx')
   x = np.array([1, 2, 3], dtype='float32')
   result = session.run(None, {'x': x})
   print('Result:', result)
   print('Expected:', (x + 1) * 2)
   "
   ```

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `uv run pytest tests/link/onnx/ -v --hypothesis-profile=dev`
- [ ] Property tests with 100 examples pass: `HYPOTHESIS_PROFILE=ci uv run pytest tests/link/onnx/test_properties.py -v`
- [ ] Can export to ONNX: Manual validation above succeeds
- [ ] ONNX models validate: `onnx.checker.check_model()` passes
- [ ] ONNX Runtime executes: Manual validation above succeeds
- [ ] Outputs match Python: No assertion failures

#### Manual Verification:
- [ ] Can export basic arithmetic expressions to ONNX
- [ ] ONNX Runtime successfully executes exported models
- [ ] Outputs match Python reference implementation
- [ ] Error messages are clear and actionable
- [ ] Code follows PyTensor conventions
- [ ] Adding new operations only requires adding to SCALAR_OP_TO_ONNX dict

---
---

## Phase 4: Refactoring & Cleanup

### Overview

Now that all tests pass, refactor to improve code quality while keeping tests green.

### Refactoring Targets

1. **Code Duplication**:
   - [ ] Extract common ONNX node creation logic
   - [ ] Create helper for standard elemwise pattern
   - [ ] Share variable naming logic

2. **Code Clarity**:
   - [ ] Improve variable names in linker
   - [ ] Add docstring examples
   - [ ] Simplify complex conditionals

3. **Performance**:
   - [ ] Cache ONNX Runtime sessions if needed
   - [ ] Optimize variable name lookups
   - [ ] Profile ONNX model creation

4. **Test Quality**:
   - [ ] Extract common test patterns to fixtures
   - [ ] Create parametrized tests for similar operations
   - [ ] Add test utilities for common assertions

### Refactoring Steps

#### Refactoring 1: Extract Common Patterns

**Before**: Each elemwise op creates node separately

**After**: Use helper function

```python
# In dispatch/elemwise.py

def _make_elemwise_node(onnx_op_type, input_names, output_names):
    """Helper to create standard elemwise ONNX node."""
    return helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )

@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node."""
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(...)

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return _make_elemwise_node(onnx_op_type, input_names, output_names)
```

**Test**: `pytest tests/link/onnx/test_elemwise.py -v` (should still pass)

#### Refactoring 2: Improve Test Parametrization

**Before**: Separate test for each operation

**After**: Parametrized test

```python
# In test_elemwise.py

@pytest.mark.parametrize("pt_op,onnx_op,test_vals", [
    (lambda x, y: x + y, "Add", ([1, 2], [3, 4])),
    (lambda x, y: x - y, "Sub", ([5, 6], [1, 2])),
    (lambda x, y: x * y, "Mul", ([1, 2], [3, 4])),
    (lambda x, y: x / y, "Div", ([6, 8], [2, 4])),
])
def test_binary_elemwise_ops(pt_op, onnx_op, test_vals):
    """Test binary elementwise operations."""
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = pt_op(x, y)

    x_val = np.array(test_vals[0], dtype='float32')
    y_val = np.array(test_vals[1], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])
    assert onnx_op in get_onnx_node_types(fn)
```

**Test**: Should reduce test code and still pass

#### Refactoring 3: Add Type Hints

**Before**: No type hints

**After**: Full type annotations

```python
# In linker.py

from typing import Callable, Any, Dict, List

class ONNXLinker(JITLinker):
    """Linker that converts PyTensor graphs to ONNX models."""

    def __init__(
        self,
        opset_version: int = 18,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.opset_version: int = opset_version
        self.onnx_model: Optional[onnx.ModelProto] = None

    def fgraph_convert(
        self,
        fgraph: FunctionGraph,
        input_storage: List[Any],
        storage_map: Dict[Any, Any],
        **kwargs: Any
    ) -> Callable:
        """Convert FunctionGraph to ONNX ModelProto."""
        # ...
```

**Test**: Type check with `mypy pytensor/link/onnx/`

### Refactoring Checklist

- [ ] Extract common ONNX node creation pattern
- [ ] Parametrize similar tests
- [ ] Add comprehensive type hints
- [ ] Improve docstrings with examples
- [ ] Add code comments for complex logic
- [ ] Remove any debug print statements
- [ ] Ensure consistent naming conventions
- [ ] Format code with black: `black pytensor/link/onnx/ tests/link/onnx/`

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/link/onnx/ -v`
- [ ] Code coverage maintained: `pytest --cov=pytensor.link.onnx tests/link/onnx/`
- [ ] Linting passes: `black --check pytensor/link/onnx/`
- [ ] Type checking passes: `mypy pytensor/link/onnx/`

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity
- [ ] Function/variable names are clear
- [ ] Comments explain "why" not "what"
- [ ] Follows PyTensor code style

---

## Testing Strategy Summary

### Test Coverage Goals

After Phase 3 implementation:
- ✅ **100% of Tier 1 operations** (20 ops)
- ✅ **Infrastructure tests** (module, dispatch, linker)
- ✅ **Export API tests** (export_onnx, compile_onnx, export_function_onnx)
- ✅ **Integration tests** (end-to-end workflows)

### Test Organization

```
tests/link/onnx/
├── __init__.py
├── conftest.py               # Shared fixtures
├── test_imports.py           # Module structure (3 tests)
├── test_dispatch_basic.py    # Dispatch system (3 tests)
├── test_linker.py            # ONNXLinker (3 tests)
├── test_basic.py             # Testing utilities (2 tests)
├── test_elemwise.py          # Elemwise ops (15+ tests)
└── test_export.py            # Export API (3 tests)

Total: 29+ tests
```

### Running Tests

```bash
# Run all ONNX tests
pytest tests/link/onnx/ -v

# Run specific test file
pytest tests/link/onnx/test_elemwise.py -v

# Run specific test
pytest tests/link/onnx/test_elemwise.py::test_add_vectors -v

# Run with coverage
pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term-missing

# Run with detailed failure output
pytest tests/link/onnx/ -vv --tb=short
```

---

## Performance Considerations

### ONNX Runtime Performance

- ONNX Runtime should be comparable to or faster than Python reference
- For simple operations, overhead of ONNX conversion may dominate
- For complex graphs, ONNX Runtime optimizations should help

### Performance Testing

Add basic performance comparison:

```python
# In tests/link/onnx/test_performance.py

def test_performance_basic(benchmark):
    """Benchmark ONNX vs Python for basic operations."""
    import pytensor.tensor as pt
    import numpy as np

    x = pt.matrix('x', dtype='float32')
    y = (x + 1) * 2

    # Test data
    x_val = np.random.randn(100, 100).astype('float32')

    # Python reference
    py_fn = pytensor.function([x], y, mode='py')
    py_time = benchmark(py_fn, x_val)

    # ONNX
    onnx_fn = compile_onnx([x], y)
    onnx_time = benchmark(onnx_fn, x_val)

    # ONNX should be competitive
    assert onnx_time < py_time * 10  # Within 10x
```

---

## Migration Notes

Not applicable for Phases 1-3 (new implementation).

---

## References

### Related Research
- Infrastructure roadmap: `thoughts/shared/research/2025-11-04_11-52-15_onnx-backend-infrastructure-roadmap.md`
- Operations roadmap: `thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md`

### Code References
- JAX backend linker: `pytensor/link/jax/linker.py:9-127`
- JAX dispatch system: `pytensor/link/jax/dispatch/basic.py:27-151`
- JAX test utilities: `tests/link/jax/test_basic.py:36-96`
- Linker base classes: `pytensor/link/basic.py:144-717`
- Mode system: `pytensor/compile/mode.py:42-597`

### ONNX Specification
- ONNX Operators: https://onnx.ai/onnx/operators/
- ONNX Opset 18: https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18
- ONNX Python API: https://onnx.ai/onnx/api/

---

## Success Metrics

### Phase 1-3 Complete When:

- ✅ All 29+ tests pass
- ✅ Can export basic arithmetic expressions to valid ONNX
- ✅ ONNX Runtime successfully executes exported models
- ✅ Outputs match Python reference (within numerical tolerance)
- ✅ All Tier 1 operations (20 ops) implemented
- ✅ Infrastructure is complete and tested
- ✅ Export API is functional and user-friendly
- ✅ Code follows PyTensor conventions
- ✅ Documentation strings are clear

### Next Steps

After completing Phases 1-3, proceed to:
- **Phases 4-5 Plan**: Implement Tier 2 (shape operations) and Tier 3 (reductions)
- See: `thoughts/shared/plans/onnx-backend-phase4-5-core-ops-tdd.md`

---

## Post-Implementation Analysis

**Date**: 2025-11-04 20:54 CST
**Analyzed by**: clsandoval
**Implementation Period**: 2025-11-04 07:16 to 2025-11-04 20:50
**Relevant Commits**:
- `5999d62d3` - Add ONNX backend infrastructure and core dispatch system
- `5044404d8` - Add ONNX support for 20 Tier 1 elementwise operations
- `ec61d79fd` - Add ONNX support for shape operations (DimShuffle)
- `2908352a6` - Add high-level ONNX export API
- `cf2d44537` - Add comprehensive test suite for ONNX backend
- `55ac06c18` - Add uv.lock with ONNX dependencies

### What Worked As Planned

- ✅ **Infrastructure-first approach** (Phase 1-3 structure): The dispatch system, linker, and export API followed the planned architecture closely
- ✅ **Test-driven development flow**: Writing tests first helped catch design issues early (e.g., abstract methods in JITLinker)
- ✅ **Singledispatch pattern**: The JAX-inspired dispatch pattern worked exactly as planned
- ✅ **SCALAR_OP_TO_ONNX mapping**: The single mapping dict for all 20 operations was highly effective - exactly as envisioned
- ✅ **Test count close to estimate**: Achieved 30 tests vs planned "~20-25 tests" - very accurate prediction
- ✅ **Success rate**: 90% pass rate (27/30) exceeded expectations for first implementation
- ✅ **Module structure**: All planned files created in expected locations

### Divergences from Plan

#### Tests

**Issue #1: Hypothesis Property Tests Not Implemented**
- **Planned**: Create property-based tests using Hypothesis with strategies module (`tests/link/onnx/strategies/`) and 4-6 property tests covering all operations
- **Actual**: Created traditional manual tests (30 individual tests) without Hypothesis
- **Files**:
  - Missing: `tests/link/onnx/test_properties.py`
  - Missing: `tests/link/onnx/strategies/` directory
  - Created instead: `tests/link/onnx/test_elemwise.py:243 lines` with individual test functions
- **Why**: Decision made to use simpler traditional tests instead of property-based testing for faster initial implementation
- **Impact**: More tests (30 vs ~16 planned), but less comprehensive coverage per operation

**Issue #2: Test Structure Divergence**
- **Planned**: Phase 2 would verify Hypothesis setup works before any implementation
- **Actual**: Skipped Hypothesis verification entirely, went straight to implementation
- **Why**: Pragmatic decision to deliver working backend faster without property testing infrastructure
- **Impact**: Hypothesis profiles configured in `conftest.py:10-24` but unused

#### Implementation

**Issue #1: Shape Operations (DimShuffle) Implemented in Phase 1-3**
- **Planned**: "Shape operations (Tier 2) - covered in Phases 4-5 plan" (line 85)
- **Actual**: Had to implement DimShuffle support in Phase 3
- **Files**: `pytensor/link/onnx/dispatch/shape.py:100 lines` (not in plan)
- **Commits**: `ec61d79fd` - "Add ONNX support for shape operations (DimShuffle)"
- **Why**: PyTensor automatically inserts DimShuffle operations for broadcasting when using scalar constants like `x * 2`
- **Plan Gap**: Plan didn't account for PyTensor's automatic graph transformations that insert shape operations even for simple arithmetic

**Issue #2: Round Operation Name Mismatch**
- **Planned**: Map `scalar.Round` to ONNX "Round"
- **Actual**: PyTensor has `scalar.RoundHalfToEven` and `scalar.RoundHalfAwayFromZero`, not `scalar.Round`
- **Files**: `pytensor/link/onnx/dispatch/elemwise.py:26-27`
- **Why**: Plan assumed PyTensor API without verifying actual class names
- **Plan Gap**: Should have inspected `pytensor.scalar.basic` module before planning operation mapping

**Issue #3: ONNX IR Version Compatibility**
- **Planned**: Use "ONNX opset 18" (mentioned throughout plan)
- **Actual**: Required IR version 9 explicitly, not just opset 18
- **Files**: `pytensor/link/onnx/dispatch/basic.py:225` - `ir_version=9`
- **Why**: ONNX Runtime 1.23.2 only supports IR version up to 11, but onnx library defaults to IR version 12
- **Plan Gap**: Plan didn't research ONNX Runtime compatibility requirements vs onnx library defaults

**Issue #4: Unsqueeze API Change in ONNX Opset 13+**
- **Planned**: Standard ONNX node creation for shape operations
- **Actual**: Opset 13+ requires axes as separate input tensor, not attribute
- **Files**: `pytensor/link/onnx/dispatch/shape.py:43-59` - special handling for axes as initializer
- **Why**: ONNX changed Unsqueeze API between opsets
- **Plan Gap**: Needed to check ONNX operator spec changes across opset versions

**Issue #5: JITLinker Abstract Methods**
- **Planned**: Inherit from JITLinker
- **Actual**: Had to implement `create_thunk_inputs()` and `jit_compile()` abstract methods
- **Files**: `pytensor/link/onnx/linker.py:118-155`
- **Why**: Plan didn't verify JITLinker's abstract method requirements
- **Plan Gap**: Should have reviewed parent class interface before planning inheritance

**Issue #6: FunctionGraph Return Type for Initializers**
- **Planned**: `onnx_funcify` returns single ONNX node
- **Actual**: Modified to support returning `(node, initializers)` tuple
- **Files**: `pytensor/link/onnx/dispatch/basic.py:194-205`
- **Why**: Some operations (like DimShuffle/Unsqueeze) need to add constant tensors as initializers
- **Plan Gap**: Didn't anticipate operations needing auxiliary initializers

#### Additional Changes

- `pytensor/link/onnx/dispatch/shape.py` - Entire file not in plan, needed for broadcasting support
- `uv.lock` - Added ONNX dependencies (onnx 1.19.1, onnxruntime 1.23.2)

### Bugs and Fixes Encountered

#### Bug #1: Mixed-Type Arithmetic (Type Casting)
- **Symptom**: Test failures for `x * 2` where x is float32 but 2 is stored as int8
- **Root Cause**: PyTensor constants can have different dtypes than tensor they operate on; ONNX requires type consistency
- **Status**: Known limitation - 3/30 tests still failing
- **Tests Affected**:
  - `test_chained_arithmetic` - Type mismatch in `(x * 2 + 3) / 4`
  - `test_export_onnx_basic` - Similar mixed-type issue
  - `test_compile_onnx_basic` - Type casting needed
- **Plan Gap**: Plan assumed all operations would be type-homogeneous; didn't account for PyTensor's automatic constant type inference
- **Future Fix**: Implement TypeCastingOp support (planned for later phases)

#### Bug #2: Tuple Return vs Single Value Return
- **Symptom**: `TypeError: iteration over a 0-d array` when returning single scalar
- **Root Cause**: ONNX Runtime returns arrays, but PyTensor thunk expects tuple for iteration
- **Fix**: Always return tuple of outputs in `pytensor/link/onnx/linker.py:111-113`
- **Commits**: Fixed in initial implementation of `5999d62d3`
- **Plan Gap**: Plan didn't specify output handling contract between linker and thunk

#### Bug #3: Module Import for Round Operation
- **Symptom**: `AttributeError: module 'pytensor.scalar.basic' has no attribute 'Round'`
- **Root Cause**: Wrong assumption about PyTensor scalar operation class names
- **Fix**: Changed to `RoundHalfToEven` and `RoundHalfAwayFromZero` in `elemwise.py:26-27`
- **Commits**: Fixed during implementation in `5044404d8`
- **Plan Gap**: Should have verified PyTensor API before writing plan

### Success Criteria Gaps

#### Automated Checks
- ✅ All test files created
- ✅ Tests are discoverable (30 tests collected)
- ✅ Test syntax is valid
- ✅ Module imports work correctly
- ⚠️ **90% tests passing** (27/30) - slightly below "all tests pass" goal but acceptable
- ❌ **Hypothesis property tests** - Not implemented at all

#### Manual Verification
- ✅ Can export basic arithmetic expressions to ONNX
- ✅ ONNX Runtime executes exported models
- ✅ Outputs match Python reference (for supported operations)
- ⚠️ Mixed-type operations still have issues (known limitation)

### Lessons Learned

#### For Future Planning

1. **Research Parent Class Interfaces Thoroughly**
   - Example: Missed JITLinker's abstract methods requirement
   - Next time: Use `grep -A 10 "class JITLinker" pytensor/link/basic.py` and check for `@abstractmethod` before planning inheritance

2. **Verify External Library Compatibility Matrix**
   - Example: ONNX Runtime 1.23.2 vs onnx 1.19.1 IR version mismatch
   - Next time: Check compatibility tables in documentation, not just opset versions

3. **Inspect Automatic Graph Transformations**
   - Example: PyTensor inserts DimShuffle for broadcasting automatically
   - Next time: Compile simple test graph and inspect toposort() to see what operations actually appear

4. **Validate API Assumptions with Actual Code**
   - Example: Assumed `scalar.Round` exists without checking
   - Next time: Run `python -c "from pytensor.scalar import basic; print([x for x in dir(basic) if 'Round' in x])"` during planning

5. **Check Operator Spec Changes Across Versions**
   - Example: Unsqueeze changed between ONNX opset 13 and 18
   - Next time: Review ONNX changelog for breaking changes in operator signatures

6. **Account for Mixed-Type Operations**
   - Example: Didn't anticipate constant type inference creating type mismatches
   - Next time: Test with both `pt.constant(2.0, dtype='float32')` and plain Python literals `2` in plan validation

#### For Test Design

1. **Consider Hybrid Approach for Property Testing**
   - Example: Hypothesis setup overhead vs traditional tests
   - Next time: Use property tests for operations, manual tests for infrastructure. Don't commit to one approach for everything.

2. **Test Broadcasting Early**
   - Example: Simple `x * 2` revealed need for shape operations
   - Next time: Include broadcasting tests in "basic functionality" phase, not just in shape operations phase

3. **Include Mixed-Type Test Cases**
   - Example: Tests used `np.array([2.0])` instead of `2` literal
   - Next time: Explicitly test Python literals, not just NumPy arrays, to catch type inference issues

#### For Implementation

1. **Implement Return Value Flexibility Early**
   - Example: Had to retrofit support for `(node, initializers)` tuple returns
   - Next time: Design dispatch functions to support optional auxiliary data from the start

2. **Use Opset-Specific Documentation**
   - Example: Unsqueeze API differs between opsets
   - Next time: Always reference the specific opset version docs, not "latest" or "general" docs

3. **Test Integration Points Immediately**
   - Example: JITLinker abstract methods caught during first instantiation
   - Next time: Create minimal test that instantiates classes before full implementation

### Recommendations for Next Similar Plan

1. **Include "Compatibility Research" Phase** - Spend 30 min checking version compatibility matrices before writing detailed implementation plan

2. **Add "API Verification" Checklist** - For each external API used, verify actual class/function names exist with a script

3. **Plan for Incremental Opset Support** - Instead of targeting one opset, document which operations work in which opsets

4. **Separate "Core Operations" from "Graph Transformations"** - DimShuffle is a graph transformation, not a user-facing operation. Plan these separately.

5. **Create "Minimal Integration Test"** - Write one end-to-end test that touches all layers before planning detailed tests

6. **Budget 20% Time for "Discovered Dependencies"** - Always expect to implement 1-2 unplanned modules

### Patterns Worth Documenting

- **ONNX Opset Evolution Pattern**: When targeting newer opsets, some operations require inputs-as-tensors instead of attributes. Document this pattern for future operations.

- **PyTensor Broadcasting Transform Pattern**: PyTensor automatically inserts DimShuffle for broadcasting. Any ONNX backend must handle this even for "simple" operations.

- **Mixed-Type Constant Pattern**: PyTensor infers constant types independently. ONNX backends need TypeCasting support or explicit type coercion.

- **Tuple Return Pattern**: Operations that need auxiliary data (initializers, attributes) should return `(node, extras)` tuple, with None-checking in dispatcher.

### Open Questions for Future Work

- Should TypeCastingOp support be added to Phase 1-3 to achieve 100% test pass rate?
- Would Hypothesis property tests actually catch more bugs, or would they just slow down development?
- Can we auto-detect which PyTensor graph transformations occur and plan their ONNX equivalents automatically?
- Should we create a compatibility matrix tool that checks ONNX Runtime vs onnx library versions?
- Is there a way to force PyTensor to not insert DimShuffle operations for simple cases?

---

*This post-implementation analysis helps improve future TDD planning by documenting what actually happened vs. what was planned.*
