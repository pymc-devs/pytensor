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

✅ **Comprehensive Testing**:
- `compare_onnx_and_py` utility for validation
- Test fixtures and utilities
- 20+ passing tests for Tier 1 operations

✅ **Validation**:
- Can export basic arithmetic expressions to ONNX
- ONNX Runtime can execute exported models
- Outputs match Python reference implementation

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

## Phase 2: Test Failure Verification

### Overview

Run all tests and verify they fail in expected, diagnostic ways. This ensures our tests are actually testing the right things and will catch regressions.

### Verification Steps

1. **Run full test suite**:
   ```bash
   pytest tests/link/onnx/ -v --tb=short
   ```

2. **Verify test discovery**:
   ```bash
   pytest --collect-only tests/link/onnx/
   ```
   - Should collect 40+ tests
   - Should show all test files

3. **Check import errors first**:
   ```bash
   pytest tests/link/onnx/test_imports.py -v
   ```
   - All should fail with `ModuleNotFoundError`

4. **Document failure patterns**:
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

#### Elemwise Tests (test_elemwise.py):
- **All arithmetic tests** (test_add_vectors, test_mul_vectors, etc.):
  - Expected: `ModuleNotFoundError` initially
  - After infrastructure: `NotImplementedError: No ONNX conversion available for: Elemwise`
  - Status: ❌ (correct failure progression)

#### Export API Tests (test_export.py):
- **All export tests**:
  - Expected: `ImportError: cannot import name 'export_onnx'`
  - Status: ❌ (correct failure)

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `pytest --collect-only tests/link/onnx/ | grep -c "test_"` shows 40+
- [ ] All tests fail: `pytest tests/link/onnx/ -v | grep FAILED | wc -l` equals test count
- [ ] No syntax errors: `pytest tests/link/onnx/ --tb=line` shows no SyntaxError
- [ ] No unexpected exceptions: Review output for unexpected error types

#### Manual Verification:
- [ ] Each test fails with correct error type (ModuleNotFoundError, ImportError, NotImplementedError)
- [ ] Error messages clearly indicate what's missing
- [ ] Stack traces point to right locations (our test code, not pytest internals)
- [ ] No cryptic error messages
- [ ] Failure output would guide implementation

### Failure Mode Documentation

Create `tests/link/onnx/EXPECTED_FAILURES.md`:

```markdown
# Expected Test Failures (Before Implementation)

## Phase 1: No Module (Initial State)
All tests fail with `ModuleNotFoundError: No module named 'pytensor.link.onnx'`

## Phase 2: Module Structure Created
Import tests pass, others fail with:
- `ImportError: cannot import name 'ONNXLinker'`
- `ImportError: cannot import name 'onnx_funcify'`

## Phase 3: Dispatch System Created
Infrastructure tests pass, operation tests fail with:
- `NotImplementedError: No ONNX conversion available for: Elemwise`
- `NotImplementedError: Elemwise scalar op not supported: Add`

## Phase 4: Operations Implemented
All tests should pass
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

## Phase 3: Feature Implementation (Red → Green)

### Overview

Implement features by making tests pass, one group at a time. Work like you're debugging - let test failures guide you.

### Implementation Order

1. Module structure (make import tests pass)
2. Dispatch system (make dispatch tests pass)
3. ONNXLinker basic (make linker tests pass)
4. Testing utilities (make test_basic tests pass)
5. Tier 1 operations (make elemwise tests pass)
6. Export API (make export tests pass)

---

### Implementation 1: Module Structure

**Target Tests**: `tests/link/onnx/test_imports.py`
**Current Failures**: `ModuleNotFoundError: No module named 'pytensor.link.onnx'`

#### Changes Required

**Step 1.1**: Create directory structure

```bash
mkdir -p pytensor/link/onnx/dispatch
touch pytensor/link/onnx/__init__.py
touch pytensor/link/onnx/dispatch/__init__.py
```

**Step 1.2**: Create stub files

**File**: `pytensor/link/onnx/__init__.py`
```python
"""ONNX backend for PyTensor."""

# Placeholder exports - will implement later
__all__ = []
```

**File**: `pytensor/link/onnx/dispatch/__init__.py`
```python
"""ONNX dispatch system."""

# Placeholder - will implement later
__all__ = []
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_imports.py::test_onnx_module_exists -v`
2. Should now pass (module exists)
3. Run: `pytest tests/link/onnx/test_imports.py::test_onnx_public_api -v`
4. Should fail with `ImportError: cannot import name 'ONNXLinker'`
5. This is progress - we've moved from ModuleNotFoundError to ImportError

#### Success Criteria

##### Automated Verification:
- [ ] Module imports: `python -c "import pytensor.link.onnx"`
- [ ] test_onnx_module_exists passes: `pytest tests/link/onnx/test_imports.py::test_onnx_module_exists -v`
- [ ] Directory structure exists: `ls pytensor/link/onnx/dispatch/`

##### Manual Verification:
- [ ] Clean directory structure
- [ ] __init__.py files present
- [ ] No circular imports

---

### Implementation 2: Core Dispatch System

**Target Tests**: `tests/link/onnx/test_dispatch_basic.py`, part of `test_imports.py`
**Current Failures**: `ImportError: cannot import name 'onnx_funcify'`

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/basic.py`

```python
"""Core ONNX dispatch system."""

from functools import singledispatch
from typing import Dict
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install onnx"
    ) from e

from pytensor.graph.basic import Variable, Constant
from pytensor.graph.fg import FunctionGraph

# Target ONNX opset version
ONNX_OPSET_VERSION = 18


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX node(s).

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation to convert
    node : Apply, optional
        The Apply node containing the op
    **kwargs
        Additional conversion parameters

    Returns
    -------
    onnx.NodeProto or List[onnx.NodeProto]
        ONNX node(s) representing the operation

    Raises
    ------
    NotImplementedError
        If no converter is registered for this Op type
    """
    raise NotImplementedError(
        f"No ONNX conversion available for: {type(op).__name__}\n"
        f"Op: {op}\n"
        f"This operation is not yet supported for ONNX export.\n\n"
        f"To add support, register a converter:\n"
        f"  @onnx_funcify.register({type(op).__name__})\n"
        f"  def onnx_funcify_{type(op).__name__}(op, node, **kwargs):\n"
        f"      # Return onnx.NodeProto\n"
    )


@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert Python/NumPy data to ONNX-compatible types.

    Parameters
    ----------
    data : Any
        Data to convert
    dtype : str, optional
        Target dtype

    Returns
    -------
    onnx.TensorProto or data
        ONNX tensor or original data
    """
    if dtype is None:
        return data
    else:
        return np.array(data, dtype=dtype)


@onnx_typify.register(np.ndarray)
def onnx_typify_ndarray(data, dtype=None, name="", **kwargs):
    """Convert numpy array to ONNX TensorProto."""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


def make_value_info(var: Variable, name: str) -> onnx.ValueInfoProto:
    """Create ONNX ValueInfoProto from PyTensor Variable.

    Parameters
    ----------
    var : Variable
        PyTensor variable
    name : str
        Name for the ONNX value

    Returns
    -------
    onnx.ValueInfoProto
        ONNX value info with type and shape
    """
    # Map PyTensor dtype to ONNX dtype
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
        "bool": TensorProto.BOOL,
        "complex64": TensorProto.COMPLEX64,
        "complex128": TensorProto.COMPLEX128,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (handle symbolic dimensions)
    if hasattr(var.type, 'shape'):
        shape = []
        for i, dim in enumerate(var.type.shape):
            if dim is None or (isinstance(dim, int) and dim < 0):
                # Dynamic dimension
                shape.append(f"dim_{i}")
            else:
                shape.append(int(dim))
    else:
        shape = None

    # Create tensor type
    tensor_type = helper.make_tensor_type_proto(
        elem_type=onnx_dtype, shape=shape
    )

    return helper.make_value_info(name, tensor_type)


@onnx_funcify.register(Constant)
def onnx_funcify_Constant(op, node, **kwargs):
    """Constants are handled as initializers, not nodes."""
    return None


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    node=None,
    opset_version: int = ONNX_OPSET_VERSION,
    model_name: str = "pytensor_model",
    **kwargs,
) -> onnx.ModelProto:
    """Convert FunctionGraph to ONNX ModelProto.

    Parameters
    ----------
    fgraph : FunctionGraph
        The graph to convert
    opset_version : int
        ONNX opset version
    model_name : str
        Model name

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    from typing import List

    # Track nodes and initializers
    onnx_nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []

    # Variable naming
    var_names: Dict[Variable, str] = {}
    name_counter = 0

    def get_var_name(var: Variable) -> str:
        """Get or create unique name for variable."""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, 'name') and var.name:
                base_name = var.name
                if base_name in var_names.values():
                    base_name = f"{base_name}_{name_counter}"
                    name_counter += 1
                var_names[var] = base_name
            else:
                var_names[var] = f"var_{name_counter}"
                name_counter += 1
        return var_names[var]

    # Convert constants to initializers
    for node in fgraph.apply_nodes:
        for inp in node.inputs:
            if isinstance(inp, Constant):
                name = get_var_name(inp)
                if name not in [init.name for init in initializers]:
                    tensor = numpy_helper.from_array(
                        np.asarray(inp.data), name=name
                    )
                    initializers.append(tensor)

    # Convert ops in topological order
    for node in fgraph.toposort():
        onnx_node_or_nodes = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            opset_version=opset_version,
            **kwargs,
        )

        if onnx_node_or_nodes is not None:
            if isinstance(onnx_node_or_nodes, list):
                onnx_nodes.extend(onnx_node_or_nodes)
            else:
                onnx_nodes.append(onnx_node_or_nodes)

    # Create inputs (non-constant only)
    input_protos = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(make_value_info(inp, name))

    # Create outputs
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(make_value_info(out, name))

    # Create graph
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=f"{model_name}_graph",
        inputs=input_protos,
        outputs=output_protos,
        initializer=initializers,
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name="PyTensor",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Validate
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}") from e

    return model
```

**File**: `pytensor/link/onnx/dispatch/__init__.py`

```python
"""ONNX dispatch system."""

from pytensor.link.onnx.dispatch.basic import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

__all__ = [
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_dispatch_basic.py::test_onnx_funcify_unregistered_op -v`
2. Should now pass (dispatch raises NotImplementedError correctly)
3. Run: `pytest tests/link/onnx/test_dispatch_basic.py::test_onnx_typify_ndarray -v`
4. Should pass (typify converts numpy arrays)
5. Run: `pytest tests/link/onnx/test_dispatch_basic.py::test_make_value_info_basic -v`
6. Should pass (make_value_info creates ValueInfo)

#### Success Criteria

##### Automated Verification:
- [ ] Dispatch tests pass: `pytest tests/link/onnx/test_dispatch_basic.py -v`
- [ ] Can import dispatch: `python -c "from pytensor.link.onnx.dispatch import onnx_funcify"`
- [ ] singledispatch works: Test unregistered op raises NotImplementedError

##### Manual Verification:
- [ ] Error messages are helpful
- [ ] Type mappings are correct
- [ ] Variable naming works correctly

---

### Implementation 3: ONNXLinker

**Target Tests**: `tests/link/onnx/test_linker.py`
**Current Failures**: `ImportError: cannot import name 'ONNXLinker'`

#### Changes Required

**File**: `pytensor/link/onnx/linker.py`

```python
"""ONNX Linker for PyTensor."""

from pytensor.link.basic import JITLinker
from pytensor.link.onnx.dispatch import onnx_funcify

try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "ONNX backend requires 'onnx' and 'onnxruntime'. "
        "Install with: pip install onnx onnxruntime"
    ) from e


class ONNXLinker(JITLinker):
    """Linker that converts PyTensor graphs to ONNX models.

    Parameters
    ----------
    opset_version : int, optional
        ONNX opset version to target (default: 18)
    """

    def __init__(self, opset_version=18, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opset_version = opset_version
        self.onnx_model = None

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        """Convert FunctionGraph to ONNX ModelProto.

        Parameters
        ----------
        fgraph : FunctionGraph
            Graph to convert
        input_storage : list
            Input storage
        storage_map : dict
            Storage map

        Returns
        -------
        callable
            Function that executes via ONNX Runtime
        """
        # Convert graph to ONNX
        self.onnx_model = onnx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            opset_version=self.opset_version,
            **kwargs
        )

        # Return ONNX Runtime executor
        return self._create_onnx_runtime_function(self.onnx_model)

    def _create_onnx_runtime_function(self, onnx_model):
        """Create ONNX Runtime inference session.

        Parameters
        ----------
        onnx_model : onnx.ModelProto
            ONNX model

        Returns
        -------
        callable
            Function that runs inference
        """
        # Serialize model
        model_bytes = onnx_model.SerializeToString()

        # Create session
        session = ort.InferenceSession(model_bytes)

        def onnx_runtime_fn(*inputs):
            """Execute ONNX model via ONNX Runtime."""
            # Map inputs to ONNX names
            input_names = [inp.name for inp in session.get_inputs()]
            input_dict = {name: inp for name, inp in zip(input_names, inputs)}

            # Run inference
            output_names = [out.name for out in session.get_outputs()]
            outputs = session.run(output_names, input_dict)

            return outputs if len(outputs) > 1 else outputs[0]

        return onnx_runtime_fn

    def jit_compile(self, fn):
        """No-op for ONNX (already compiled as static graph)."""
        return fn

    def create_thunk_inputs(self, storage_map):
        """Standard input preparation."""
        return [storage_map[n] for n in self.fgraph.inputs]

    def export_to_file(self, filename):
        """Export ONNX model to file.

        Parameters
        ----------
        filename : str
            Path to save model
        """
        if self.onnx_model is None:
            raise RuntimeError("No ONNX model has been generated yet")

        onnx.save(self.onnx_model, filename)
```

**File**: `pytensor/link/onnx/__init__.py` (update)

```python
"""ONNX backend for PyTensor."""

from pytensor.link.onnx.linker import ONNXLinker
from pytensor.link.onnx.dispatch import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

__all__ = [
    "ONNXLinker",
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_linker.py::test_linker_instantiation -v`
2. Should pass (linker can be created)
3. Run: `pytest tests/link/onnx/test_linker.py::test_linker_empty_graph -v`
4. May fail with NotImplementedError for Identity op
5. Need to implement Identity first, then re-test

#### Success Criteria

##### Automated Verification:
- [ ] Linker instantiates: `pytest tests/link/onnx/test_linker.py::test_linker_instantiation -v`
- [ ] Can import: `python -c "from pytensor.link.onnx import ONNXLinker"`
- [ ] Inherits from JITLinker correctly

##### Manual Verification:
- [ ] Linker follows PyTensor linker patterns
- [ ] ONNX Runtime integration works
- [ ] Model export method exists

---

### Implementation 4: Testing Utilities

**Target Tests**: `tests/link/onnx/test_basic.py`
**Current Failures**: `ImportError: cannot import name 'compare_onnx_and_py'`

#### Changes Required

**File**: `tests/link/onnx/test_basic.py`

```python
"""Core testing utilities for ONNX backend."""

import numpy as np
import pytest
from functools import partial

# Import ONNX and skip tests if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.onnx.linker import ONNXLinker
from pytensor.graph import RewriteDatabaseQuery


# Configure ONNX mode for testing
optimizer = RewriteDatabaseQuery(include=["onnx"], exclude=["cxx_only", "BlasOpt"])
onnx_mode = Mode(linker=ONNXLinker(), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_onnx_and_py(
    graph_inputs,
    graph_outputs,
    test_inputs,
    *,
    assert_fn=None,
    must_validate=True,
    onnx_mode=onnx_mode,
    py_mode=py_mode,
    opset_version=None,
):
    """Compare ONNX Runtime output to Python reference.

    Parameters
    ----------
    graph_inputs : list of Variable
        Symbolic input variables
    graph_outputs : Variable or list of Variable
        Symbolic output variables
    test_inputs : list
        Concrete test values
    assert_fn : callable, optional
        Custom assertion function
    must_validate : bool, optional
        Whether ONNX model must pass validation
    onnx_mode : Mode, optional
        ONNX compilation mode
    py_mode : Mode, optional
        Python reference mode
    opset_version : int, optional
        ONNX opset version

    Returns
    -------
    onnx_fn : Function
        Compiled ONNX function
    onnx_res : array or list
        ONNX results

    Raises
    ------
    AssertionError
        If outputs don't match
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-6)

    # Validate inputs are root variables
    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables (no owner)")

    # Compile with ONNX backend
    pytensor_onnx_fn = pytensor.function(graph_inputs, graph_outputs, mode=onnx_mode)

    # Execute with ONNX Runtime
    onnx_res = pytensor_onnx_fn(*test_inputs)

    # Validate ONNX model if required
    if must_validate:
        onnx_model = pytensor_onnx_fn.maker.linker.onnx_model
        try:
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            pytest.fail(f"ONNX model validation failed: {e}")

    # Compile with Python backend (reference)
    pytensor_py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    # Compare results
    if isinstance(graph_outputs, (list, tuple)):
        assert len(onnx_res) == len(py_res), "Output count mismatch"
        for i, (o, p) in enumerate(zip(onnx_res, py_res, strict=True)):
            try:
                assert_fn(o, p)
            except AssertionError as e:
                raise AssertionError(f"Output {i} mismatch: {e}") from e
    else:
        assert_fn(onnx_res, py_res)

    return pytensor_onnx_fn, onnx_res


def get_onnx_node_types(fn):
    """Get list of ONNX node types in compiled function.

    Parameters
    ----------
    fn : Function
        Compiled PyTensor function with ONNX backend

    Returns
    -------
    list of str
        ONNX operator types
    """
    onnx_model = fn.maker.linker.onnx_model
    return [node.op_type for node in onnx_model.graph.node]


def get_onnx_node_by_type(fn, op_type):
    """Get ONNX node by operator type.

    Parameters
    ----------
    fn : Function
        Compiled function
    op_type : str
        ONNX operator type

    Returns
    -------
    onnx.NodeProto or None
        First matching node
    """
    onnx_model = fn.maker.linker.onnx_model
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            return node
    return None


# Module-level fixtures
@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    """Configure PyTensor for ONNX testing."""
    with pytensor.config.change_flags(cxx="", compute_test_value="ignore"):
        yield


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)
```

**File**: `tests/link/onnx/conftest.py`

```python
"""Shared pytest fixtures for ONNX backend tests."""

import numpy as np
import pytest
import pytensor


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def float32_data(rng):
    """Common float32 test data."""
    return rng.normal(size=(3, 4)).astype('float32')


@pytest.fixture
def matrix_pair(rng):
    """Pair of compatible matrices for operations like dot."""
    A = rng.normal(size=(3, 4)).astype('float32')
    B = rng.normal(size=(4, 5)).astype('float32')
    return A, B


@pytest.fixture(scope="module", autouse=True)
def configure_pytensor():
    """Module-level PyTensor configuration."""
    with pytensor.config.change_flags(
        cxx="",
        compute_test_value="ignore",
        floatX="float32"
    ):
        yield
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_basic.py -v`
2. Utilities should work (but dependent tests will still fail)
3. Can now use compare_onnx_and_py in other tests

#### Success Criteria

##### Automated Verification:
- [ ] Utilities importable: `python -c "from tests.link.onnx.test_basic import compare_onnx_and_py"`
- [ ] Fixtures work: `pytest tests/link/onnx/conftest.py --collect-only`

##### Manual Verification:
- [ ] compare_onnx_and_py follows JAX pattern
- [ ] Error messages are clear
- [ ] Fixtures are useful

---

### Implementation 5: Tier 1 Operations - Elemwise

**Target Tests**: `tests/link/onnx/test_elemwise.py`
**Current Failures**: `NotImplementedError: No ONNX conversion for: Elemwise`

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/elemwise.py`

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise, DimShuffle
from pytensor.scalar import basic as scalar

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    # Arithmetic (Tier 1)
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.IntDiv: "Div",  # Map to Div with type casting

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

    Elemwise ops perform element-wise operations on tensors.
    They map directly to ONNX ops like Add, Mul, etc.
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}\n"
            f"Supported scalar ops: {', '.join(op.__name__ for op in SCALAR_OP_TO_ONNX.keys())}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    onnx_node = helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )

    return onnx_node
```

**File**: `pytensor/link/onnx/dispatch/__init__.py` (update)

```python
"""ONNX dispatch system."""

from pytensor.link.onnx.dispatch.basic import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

# Import dispatch modules to trigger registration
import pytensor.link.onnx.dispatch.elemwise  # noqa: F401

__all__ = [
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_elemwise.py::test_add_vectors -v`
2. Should now pass (Add is implemented)
3. Run each elemwise test one at a time
4. All Tier 1 elemwise tests should pass

#### Success Criteria

##### Automated Verification:
- [ ] All Tier 1 elemwise tests pass: `pytest tests/link/onnx/test_elemwise.py -v -k "test_add or test_mul or test_sub or test_div or test_neg or test_abs or test_exp or test_log or test_sqrt or test_pow or test_floor or test_ceil or test_round or test_maximum or test_minimum"`
- [ ] Chained operations work: `pytest tests/link/onnx/test_elemwise.py::test_chained_arithmetic -v`

##### Manual Verification:
- [ ] ONNX nodes are correct types
- [ ] Broadcasting works correctly
- [ ] Output values match Python reference

---

### Implementation 6: Export API

**Target Tests**: `tests/link/onnx/test_export.py`
**Current Failures**: `ImportError: cannot import name 'export_onnx'`

#### Changes Required

**File**: `pytensor/link/onnx/export.py`

```python
"""User-facing API for ONNX export."""

from pathlib import Path
from typing import Iterable, Union
import onnx

from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.compile.function import function
from pytensor.link.onnx.linker import ONNXLinker
from pytensor.link.onnx.dispatch import onnx_funcify


def export_onnx(
    inputs: Iterable[Variable],
    outputs: Union[Variable, Iterable[Variable]],
    filename: Union[str, Path],
    *,
    opset_version: int = 18,
    model_name: str = "pytensor_model",
    doc_string: str = "",
    optimize: bool = True,
) -> onnx.ModelProto:
    """Export a PyTensor computation graph to ONNX format.

    Parameters
    ----------
    inputs : list of Variable
        Input variables
    outputs : Variable or list of Variable
        Output variables
    filename : str or Path
        Path to save ONNX model
    opset_version : int, optional
        ONNX opset version (default: 18)
    model_name : str, optional
        Model name (default: "pytensor_model")
    doc_string : str, optional
        Documentation string
    optimize : bool, optional
        Apply optimizations (default: True)

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model
    """
    # Validate inputs
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of Variables")

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    # Create FunctionGraph
    from pytensor.compile.builders import construct_nominal_fgraph

    fgraph = construct_nominal_fgraph(inputs, outputs)

    # Apply optimizations if requested
    if optimize:
        # Basic optimizations only (no CXX-specific)
        from pytensor.graph.rewriting.basic import GraphRewriter
        from pytensor.tensor.rewriting.basic import register_canonicalize

        optimizer = GraphRewriter()
        fgraph = optimizer.rewrite(fgraph)

    # Convert to ONNX
    onnx_model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name=model_name,
    )

    # Add doc string
    if doc_string:
        onnx_model.doc_string = doc_string

    # Save to file
    onnx.save(onnx_model, str(filename))

    print(f"ONNX model exported to: {filename}")
    print(f"  Opset version: {opset_version}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    return onnx_model


def export_function_onnx(
    fn,
    filename: Union[str, Path],
    *,
    opset_version: int = 18,
) -> onnx.ModelProto:
    """Export a compiled PyTensor function to ONNX.

    Parameters
    ----------
    fn : pytensor.compile.function_module.Function
        Compiled PyTensor function
    filename : str or Path
        Path to save model
    opset_version : int, optional
        ONNX opset version (default: 18)

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model
    """
    # Extract FunctionGraph
    fgraph = fn.maker.fgraph

    # Get inputs and outputs
    inputs = fgraph.inputs
    outputs = fgraph.outputs

    # Convert to ONNX
    onnx_model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name="pytensor_function",
    )

    # Save
    onnx.save(onnx_model, str(filename))

    return onnx_model


def compile_onnx(
    inputs: Iterable[Variable],
    outputs: Union[Variable, Iterable[Variable]],
    *,
    opset_version: int = 18,
    **kwargs
):
    """Compile a PyTensor graph using ONNX backend.

    This returns a function that executes via ONNX Runtime.

    Parameters
    ----------
    inputs : list of Variable
        Input variables
    outputs : Variable or list of Variable
        Output variables
    opset_version : int, optional
        ONNX opset version (default: 18)
    **kwargs
        Additional arguments passed to pytensor.function()

    Returns
    -------
    Function
        Compiled function that executes via ONNX Runtime
    """
    from pytensor.compile.mode import Mode

    # Use ONNX linker
    onnx_linker = ONNXLinker(opset_version=opset_version)
    onnx_mode = Mode(linker=onnx_linker, optimizer=None)

    return function(inputs, outputs, mode=onnx_mode, **kwargs)
```

**File**: `pytensor/link/onnx/__init__.py` (final update)

```python
"""ONNX backend for PyTensor."""

from pytensor.link.onnx.linker import ONNXLinker
from pytensor.link.onnx.export import (
    export_onnx,
    export_function_onnx,
    compile_onnx,
)
from pytensor.link.onnx.dispatch import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

__all__ = [
    "ONNXLinker",
    "export_onnx",
    "export_function_onnx",
    "compile_onnx",
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

#### Debugging Approach

1. Run: `pytest tests/link/onnx/test_export.py::test_export_onnx_basic -v`
2. Should pass (can export to file)
3. Run: `pytest tests/link/onnx/test_export.py::test_compile_onnx_basic -v`
4. Should pass (can compile and execute)
5. Run all export tests

#### Success Criteria

##### Automated Verification:
- [ ] All export tests pass: `pytest tests/link/onnx/test_export.py -v`
- [ ] Can import export functions: `python -c "from pytensor.link.onnx import export_onnx, compile_onnx"`
- [ ] Exported files are valid: ONNX checker validates them

##### Manual Verification:
- [ ] Export API is user-friendly
- [ ] Error messages are helpful
- [ ] Documentation strings are clear

---

### Complete Feature Implementation

#### Final Integration Test

Run full test suite to ensure everything works together:

```bash
pytest tests/link/onnx/ -v
```

#### Expected Results

All tests should pass:
- ✅ Import tests (3 tests)
- ✅ Dispatch tests (3 tests)
- ✅ Linker tests (3 tests)
- ✅ Testing utility tests (2 tests)
- ✅ Elemwise tests (15+ tests for all Tier 1 ops)
- ✅ Export API tests (3 tests)

**Total**: 29+ passing tests

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `pytest tests/link/onnx/ -v | grep "passed"`
- [ ] No regressions: `pytest` (full suite) shows no new failures
- [ ] Linting passes: `make lint` or `black pytensor/link/onnx/ tests/link/onnx/`
- [ ] ONNX models validate: All exported models pass `onnx.checker.check_model`

#### Manual Verification:
- [ ] Can export basic arithmetic expressions
- [ ] ONNX Runtime executes exported models correctly
- [ ] Outputs match Python reference implementation
- [ ] Error messages are clear and actionable
- [ ] Code follows PyTensor conventions

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
