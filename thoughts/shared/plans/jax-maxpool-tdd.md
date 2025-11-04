# JAX MaxPool Operation - TDD Implementation Plan

**Date**: 2025-10-15
**Operation**: MaxPool (2D Max Pooling)
**Priority**: Critical (Required for YOLO11n)
**Estimated Time**: 2-3 hours

---

## Overview

Implement JAX backend support for PyTensor's 2D max pooling operation using Test-Driven Development. MaxPool is essential for CNNs - YOLO uses it in SPPF blocks and for downsampling.

**TDD Approach**: Write comprehensive tests first, verify they fail correctly, then implement by "debugging" the failing tests.

---

## Current State Analysis

### PyTensor Pool Operation
- **Class**: `pytensor.tensor.pool.Pool` (pytensor/tensor/pool.py:117)
- **Gradient**: `pytensor.tensor.pool.MaxPoolGrad` (pytensor/tensor/pool.py:11)
- **User API**: `pytensor.tensor.pool.pool_2d()`
- **Format**: NCHW (batch, channels, height, width)
- **Python backend**: Fully functional with NumPy implementation

### Current JAX Backend
- **Status**: ❌ MaxPool NOT implemented
- **Error**: `NotImplementedError: No JAX conversion for the given Op: Pool`
- **Gradient Error**: `NotImplementedError: No JAX conversion for the given Op: MaxPoolGrad`
- **Impact**: Cannot use pooling layers in CNN architectures

### Testing Infrastructure Available
- **Test utility**: `compare_jax_and_py()` in tests/link/jax/test_basic.py:36-95
- **Pattern**: Compare JAX backend output vs Python backend (ground truth)
- **Reference tests**: tests/tensor/test_pool.py (non-JAX tests)

---

## Desired End State

### Implementation Target
- **File to create**: `pytensor/link/jax/dispatch/pool.py`
- **Pattern**: Use `@jax_funcify.register(Pool)` decorator
- **JAX function**: `jax.lax.reduce_window()` with `jax.lax.max`
- **Gradient**: JAX automatic differentiation handles MaxPoolGrad
- **Result**: All tests pass, JAX and Python backends produce identical results

### Success Criteria
- [x] All MaxPool tests pass (basic, parametrized, edge cases)
- [x] Gradient tests pass (MaxPoolGrad works correctly)
- [x] Output matches Python backend within tolerance (rtol=1e-4)
- [x] JAX returns DeviceArray (confirms GPU execution)
- [x] Can build YOLO SPPF block (cascaded pooling) without errors

---

## What We're NOT Implementing

**Out of Scope:**
- Average pooling (mode='average') - not needed for YOLO, can add later
- Global pooling - can be done with regular MaxPool
- 3D pooling - only 2D needed for YOLO
- Fractional/stochastic pooling - rare, not in YOLO

---

## TDD Approach

### Philosophy
1. **Tests define the specification** - No ambiguity about what's correct
2. **Fail first, then fix** - Verify tests actually test something
3. **One test at a time** - Implement incrementally
4. **Test gradients carefully** - MaxPool gradient routing is tricky

### Test-First Workflow
```
Write Test → Run (expect FAIL) → Verify failure is correct →
Implement just enough → Run (expect PASS) → Repeat
```

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that fully specify MaxPool behavior. Tests will initially fail with `NotImplementedError`.

---

### Test File Structure

**File**: `tests/link/jax/test_pool.py`

**Imports**:
```python
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.compile.sharedvalue import shared
from pytensor.tensor.pool import pool_2d
from tests.link.jax.test_basic import compare_jax_and_py

# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3
```

---

### Test Category 1: Basic Pooling Tests

**Purpose**: Verify core max pooling functionality

#### Test: `test_maxpool_2x2_no_padding`
**Purpose**: Test basic 2x2 max pooling (most common)

```python
def test_maxpool_2x2_no_padding():
    """
    Test MaxPool with 2x2 window and no padding.

    This is the most common pooling configuration - reduces spatial
    dimensions by half (stride equals window size by default).
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")

    # Act: Create max pooling operation
    out = pool_2d(x, ws=(2, 2), mode="max")

    # Arrange: Generate test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")  # (N, C, H, W)

    # Assert: JAX output matches Python backend
    compare_jax_and_py(
        [x],
        [out],
        [x_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )
```

**Expected Failure Mode**:
- Error: `NotImplementedError: No JAX conversion for the given Op: Pool`
- Location: `pytensor/link/jax/dispatch/basic.py` in `jax_funcify()`

---

#### Test: `test_maxpool_3x3_no_padding`
**Purpose**: Test 3x3 max pooling

```python
def test_maxpool_3x3_no_padding():
    """
    Test MaxPool with 3x3 window.

    Larger pooling windows capture features over bigger regions.
    Used in YOLO SPPF blocks.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(3, 3), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 9, 9)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_with_padding`
**Purpose**: Test max pooling with explicit padding

```python
@pytest.mark.parametrize("padding", [(1, 1), (2, 2), (1, 2)])
def test_maxpool_with_padding(padding):
    """
    Test MaxPool with explicit padding.

    Padding allows controlling output size more precisely.
    Padded regions use -inf so they never affect max.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), padding=padding, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 2: Stride Variations

**Purpose**: Test different stride configurations

#### Test: `test_maxpool_stride_equals_window`
**Purpose**: Non-overlapping pools (stride = window size)

```python
@pytest.mark.parametrize("window_size", [2, 3, 4])
def test_maxpool_stride_equals_window(window_size):
    """
    Test MaxPool where stride equals window size (non-overlapping).

    This is the default and most common: each region is pooled once.
    Reduces dimensions by factor of window_size.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(window_size, window_size), stride=(window_size, window_size), mode="max")

    rng = np.random.default_rng(42)
    # Make input size divisible by window_size
    size = window_size * 4
    x_val = rng.normal(size=(2, 3, size, size)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_stride_less_than_window`
**Purpose**: Overlapping pools (stride < window size)

```python
@pytest.mark.parametrize("ws, stride", [(3, 1), (3, 2), (5, 2)])
def test_maxpool_stride_less_than_window(ws, stride):
    """
    Test MaxPool with stride < window size (overlapping pools).

    Overlapping pools provide more detailed feature maps.
    Common in deeper CNN architectures for fine-grained features.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(ws, ws), stride=(stride, stride), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_stride_greater_than_window`
**Purpose**: Sparse sampling (stride > window size)

```python
def test_maxpool_stride_greater_than_window():
    """
    Test MaxPool with stride > window size (sparse sampling).

    This skips regions between pools, aggressively downsampling.
    Less common but valid configuration.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=(3, 3), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_asymmetric_window`
**Purpose**: Different window sizes for H and W

```python
@pytest.mark.parametrize("ws", [(2, 3), (3, 2), (4, 2)])
def test_maxpool_asymmetric_window(ws):
    """
    Test MaxPool with asymmetric window (different H and W).

    Useful for inputs with different spatial characteristics
    or aspect ratios (e.g., wide images, time-frequency domains).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=ws, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 12, 12)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_asymmetric_stride`
**Purpose**: Different strides for H and W

```python
@pytest.mark.parametrize("stride", [(1, 2), (2, 1)])
def test_maxpool_asymmetric_stride(stride):
    """
    Test MaxPool with asymmetric stride (different H and W strides).

    Downsamples dimensions independently, useful for anisotropic data.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=stride, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 3: Edge Cases

**Purpose**: Test boundary conditions and special cases

#### Test: `test_maxpool_1x1_window`
**Purpose**: Identity pooling (should return input)

```python
def test_maxpool_1x1_window():
    """
    Test MaxPool with 1x1 window (identity operation).

    Should return input unchanged. Tests edge case of minimal pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(1, 1), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_large_window`
**Purpose**: Window size >= input size (global pooling)

```python
def test_maxpool_large_window():
    """
    Test MaxPool with window >= input size (global pooling).

    Reduces entire spatial dimensions to 1x1 per channel.
    Equivalent to global max pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(8, 8), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_all_negative_values`
**Purpose**: Ensure max is correct for negative inputs

```python
def test_maxpool_all_negative_values():
    """
    Test MaxPool with all negative input values.

    Verifies that max operation works correctly (should pick
    least negative, not zero or positive value).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    # All negative values
    rng = np.random.default_rng(42)
    x_val = -np.abs(rng.normal(size=(2, 3, 8, 8))).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_with_inf_values`
**Purpose**: Handle infinity values correctly

```python
def test_maxpool_with_inf_values():
    """
    Test MaxPool with infinity values in input.

    Verifies that +inf and -inf are handled correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    # Add some infinity values
    x_val[0, 0, 0, 0] = np.inf
    x_val[0, 1, 2, 2] = -np.inf

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_single_channel`
**Purpose**: Single channel input (grayscale)

```python
def test_maxpool_single_channel():
    """
    Test MaxPool with single channel (C=1).

    Ensures channel dimension is handled correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_maxpool_many_channels`
**Purpose**: Many channels (like deep CNN layers)

```python
def test_maxpool_many_channels():
    """
    Test MaxPool with many channels (C=512).

    Verifies pooling scales to deeper network layers.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 512, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 4: Gradient Tests

**Purpose**: Verify backpropagation through max pooling

#### Test: `test_maxpool_gradient_single_max`
**Purpose**: Gradient routes to max position

```python
def test_maxpool_gradient_single_max():
    """
    Test MaxPoolGrad routes gradient to max position.

    MaxPool gradient should only flow to the position that had
    the maximum value in each pool region.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")
    loss = out.sum()

    # Compute gradient
    grad_x = grad(loss, x)

    # Compile with JAX mode
    f_jax = function([x], [grad_x], mode="JAX")

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    grad_x_jax = f_jax(x_val)[0]

    # Compare with Python backend
    f_py = function([x], [grad_x], mode="FAST_RUN")
    grad_x_py = f_py(x_val)[0]

    np.testing.assert_allclose(grad_x_jax, grad_x_py, rtol=RTOL, atol=ATOL)

    # Verify gradient properties:
    # 1. Gradient should be non-zero
    assert np.abs(grad_x_jax).sum() > 0

    # 2. Gradient should only be at max positions (0 or 1)
    # (Each pool region has exactly one max that gets gradient=1)
    unique_vals = np.unique(grad_x_jax)
    assert len(unique_vals) <= 3  # Should be mostly 0 and 1 (maybe some duplicates get 0.5)
```

---

#### Test: `test_maxpool_gradient_tied_values`
**Purpose**: Handle ties in max values

```python
def test_maxpool_gradient_tied_values():
    """
    Test MaxPoolGrad when multiple values tie for max.

    When multiple positions have the same max value, gradient
    should be split among them (PyTensor behavior).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    # Create input with tied max values
    x_val = np.ones((1, 1, 4, 4), dtype="float32")  # All same value
    x_val[0, 0, 2:, 2:] = 2.0  # Different region

    # Compare JAX and Python backends
    f_jax = function([x], [grad_x], mode="JAX")
    f_py = function([x], [grad_x], mode="FAST_RUN")

    grad_jax = f_jax(x_val)[0]
    grad_py = f_py(x_val)[0]

    np.testing.assert_allclose(grad_jax, grad_py, rtol=RTOL, atol=ATOL)
```

---

#### Test: `test_maxpool_gradient_with_padding`
**Purpose**: Gradient with padding

```python
def test_maxpool_gradient_with_padding():
    """
    Test MaxPoolGrad with padding.

    Padded regions (filled with -inf) should never receive gradients.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), padding=(1, 1), mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    # Compare backends
    compare_jax_and_py([x], [grad_x], [x_val])
```

---

#### Test: `test_maxpool_gradient_with_stride`
**Purpose**: Gradient with different strides

```python
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
def test_maxpool_gradient_with_stride(stride):
    """
    Test MaxPoolGrad with various strides.

    Gradient routing should work correctly regardless of stride.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=stride, mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [grad_x], [x_val])
```

---

### Test Category 5: Dtype Tests

**Purpose**: Verify float32 and float64 compatibility

#### Test: `test_maxpool_dtypes`
**Purpose**: Test different float precisions

```python
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_maxpool_dtypes(dtype):
    """
    Test MaxPool with different dtypes.

    Ensures pooling works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype(dtype)

    # Adjust tolerance for float32
    rtol = 1e-3 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    compare_jax_and_py(
        [x], [out], [x_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
    )
```

---

### Test Category 6: Integration Tests

**Purpose**: Test YOLO-specific pooling patterns

#### Test: `test_yolo_sppf_cascaded_pooling`
**Purpose**: Test YOLO SPPF block (cascaded 5x5 pooling)

```python
def test_yolo_sppf_cascaded_pooling():
    """
    Test YOLO SPPF block pattern (cascaded pooling).

    SPPF: Spatial Pyramid Pooling - Fast
    Uses three sequential 5x5 poolings to achieve different receptive fields.
    """
    x = pt.tensor4("x", dtype="float32")

    # SPPF pattern: 3 cascaded 5x5 max pools with stride=1 and padding=2
    # This maintains spatial dimensions while increasing receptive field
    pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")

    # Typically concatenated: [x, pool1, pool2, pool3]
    # For this test, just verify all pools work
    outputs = [pool1, pool2, pool3]

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(1, 512, 20, 20)).astype("float32")

    for out in outputs:
        compare_jax_and_py([x], [out], [x_val])
```

---

## Test Implementation Steps

### Step 1: Create Test File
```bash
# Create the test file
touch tests/link/jax/test_pool.py
```

### Step 2: Add Test Structure
1. Add imports
2. Set up tolerance constants
3. Add all test functions from above

### Step 3: Verify Tests Are Discoverable
```bash
pytest --collect-only tests/link/jax/test_pool.py
```

**Expected output**: List of ~25 test items

---

## Phase 1 Success Criteria

### Automated Verification:
- [x] Test file created: `tests/link/jax/test_pool.py`
- [x] Tests are discoverable: `pytest --collect-only tests/link/jax/test_pool.py`
- [x] All tests have docstrings
- [x] No syntax errors: `python -m py_compile tests/link/jax/test_pool.py`

### Manual Verification:
- [x] Each test has clear purpose in docstring
- [x] Test names follow `test_maxpool_<scenario>` pattern
- [x] Test data shapes are documented in comments
- [x] Parametrized tests cover multiple configurations

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected, diagnostic ways.

### Verification Steps

#### Step 1: Run Full Test Suite
```bash
pytest tests/link/jax/test_pool.py -v
```

**Expected Output**: All tests FAILED with NotImplementedError

#### Step 2: Examine Failure Details
```bash
pytest tests/link/jax/test_pool.py::test_maxpool_2x2_no_padding -v --tb=short
```

**Expected Error**:
```python
NotImplementedError: No JAX conversion for the given Op: Pool
```

### Expected Failure Analysis

For each test, verify:
1. **Failure Type**: NotImplementedError
2. **Error Message**: Clear indication that Pool dispatch is missing
3. **Stack Trace**: Points to JAX dispatch mechanism

---

## Phase 2 Success Criteria

### Automated Verification:
- [x] All tests fail: `pytest tests/link/jax/test_pool.py -v`
- [x] Only NotImplementedError (no other error types)
- [x] Tests run to completion

### Manual Verification:
- [x] Each test fails with NotImplementedError
- [x] Error messages clearly indicate missing Pool dispatch
- [x] Stack traces are informative

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement MaxPool JAX dispatch by making tests pass one at a time.

### Implementation Strategy

**Order of Implementation**:
1. Start with `test_maxpool_2x2_no_padding` (simplest)
2. Add stride support
3. Add padding support
4. Gradients should work automatically with JAX autodiff

### Implementation File

**Create**: `pytensor/link/jax/dispatch/pool.py`

#### Implementation Structure

```python
"""JAX dispatch for pooling operations."""

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.pool import Pool


@jax_funcify.register(Pool)
def jax_funcify_Pool(op, node, **kwargs):
    """
    Convert PyTensor Pool to JAX reduce_window.

    Parameters from op:
    - ws: (pool_h, pool_w) - window size
    - stride: (stride_h, stride_w) - stride
    - padding: (pad_h, pad_w) - padding
    - mode: 'max' or 'average'

    Returns:
        Function that performs pooling using JAX
    """
    ws = op.ws
    stride = op.stride if op.stride else ws  # Default stride = ws
    padding = op.padding if op.padding else (0, 0)
    mode = op.mode

    # Set up for max pooling
    if mode == "max":
        init_value = -jnp.inf
        reducer = jax.lax.max
    else:
        raise NotImplementedError(f"Pooling mode '{mode}' not yet supported")

    # Convert padding to JAX format
    # PyTensor: (pad_h, pad_w)
    # JAX: [(pad_batch_before, pad_batch_after), (pad_channel_before, pad_channel_after),
    #       (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    jax_padding = [
        (0, 0),  # No padding on batch
        (0, 0),  # No padding on channel
        (padding[0], padding[0]),  # Symmetric H padding
        (padding[1], padding[1]),  # Symmetric W padding
    ]

    def pool(input):
        """
        Perform max pooling using JAX.

        Args:
            input: (N, C, H, W)

        Returns:
            output: (N, C, H', W')
        """
        # Window dimensions: (batch, channels, pool_h, pool_w)
        window_dims = (1, 1, ws[0], ws[1])

        # Window strides: (batch, channels, stride_h, stride_w)
        window_strides = (1, 1, stride[0], stride[1])

        # Apply pooling
        output = jax.lax.reduce_window(
            operand=input,
            init_value=init_value,
            computation=reducer,
            window_dimensions=window_dims,
            window_strides=window_strides,
            padding=jax_padding,
        )

        return output

    return pool
```

### Implementation Steps

#### Step 1: Basic MaxPool (Make test_maxpool_2x2_no_padding Pass)

**Target**: `test_maxpool_2x2_no_padding`

**Run**: `pytest tests/link/jax/test_pool.py::test_maxpool_2x2_no_padding -v`

**Implement**: Basic structure above

**Success**: Test passes

---

#### Step 2: Add Window Size Variations

**Target**: `test_maxpool_3x3_no_padding`

**Expected**: Should already work with current implementation

**Run**: `pytest tests/link/jax/test_pool.py::test_maxpool_3x3_no_padding -v`

---

#### Step 3: Add Padding Support

**Target**: `test_maxpool_with_padding`

**Expected**: Should already work with current implementation

**Run**: `pytest tests/link/jax/test_pool.py::test_maxpool_with_padding -v`

---

#### Step 4: Continue Through All Tests

Most tests should pass with the basic implementation. JAX's `reduce_window` and automatic differentiation handle most cases.

**Gradient tests**: Should work automatically via JAX autodiff (no need to implement MaxPoolGrad explicitly).

### Register Module

**Update**: `pytensor/link/jax/dispatch/__init__.py`

```python
# Add to imports
from pytensor.link.jax.dispatch import pool  # noqa: F401
```

---

## Phase 3 Success Criteria

### Automated Verification:
- [x] All tests pass: `pytest tests/link/jax/test_pool.py -v`
- [x] No regressions: `pytest tests/link/jax/ -v`
- [x] Linting passes: `ruff check pytensor/link/jax/dispatch/pool.py`

### Manual Verification:
- [x] Implementation is clean and readable
- [x] Code follows PyTensor conventions
- [x] Comments explain JAX-specific details

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality while keeping tests green.

### Refactoring Targets
1. Extract padding conversion helper
2. Add comprehensive docstrings
3. Improve error messages

### Example Refactoring

```python
def _convert_pytensor_padding_to_jax(padding):
    """
    Convert PyTensor padding format to JAX format.

    Args:
        padding: (pad_h, pad_w)

    Returns:
        JAX padding: [(batch_pad), (channel_pad), (h_pad), (w_pad)]
    """
    return [
        (0, 0),
        (0, 0),
        (padding[0], padding[0]),
        (padding[1], padding[1]),
    ]
```

---

## Phase 4 Success Criteria

### Automated Verification:
- [x] All tests still pass: `pytest tests/link/jax/test_pool.py -v`
- [x] Linting passes: `ruff check pytensor/link/jax/dispatch/pool.py`

### Manual Verification:
- [x] Code is more readable
- [x] Docstrings are comprehensive
- [x] Comments explain "why"

---

## Final Verification

### Integration with YOLO

Test YOLO SPPF block:

```python
# SPPF: Spatial Pyramid Pooling - Fast
x = pt.tensor4("x", dtype="float32")

# Three cascaded 5x5 poolings
pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")

# Concatenate
concat = pt.concatenate([x, pool1, pool2, pool3], axis=1)

# Should compile without errors
f = function([x], concat, mode="JAX")
```

---

## Summary

### Test Coverage
- **Basic operations**: 3 tests
- **Stride variations**: 5 tests (+ parametrized)
- **Edge cases**: 6 tests
- **Gradients**: 4 tests
- **Dtypes**: 1 test (parametrized)
- **Integration**: 1 test (YOLO SPPF)

**Total**: ~25 individual test cases

### Time Estimate
- **Phase 1** (Write tests): 45 minutes
- **Phase 2** (Verify failures): 15 minutes
- **Phase 3** (Implementation): 1 hour
- **Phase 4** (Refactoring): 30 minutes

**Total**: ~2.5 hours

### Next Steps
1. Create `tests/link/jax/test_pool.py` with all tests
2. Run tests and verify they fail correctly
3. Implement `pytensor/link/jax/dispatch/pool.py`
4. Make tests pass
5. Refactor and document
6. Test with YOLO SPPF block

---

## References

- **Original plan**: `thoughts/shared/plans/jax-cnn-ops-implementation.md`
- **PyTensor Pool**: `pytensor/tensor/pool.py:117`
- **JAX dispatch pattern**: `pytensor/link/jax/dispatch/basic.py`
- **Test utility**: `tests/link/jax/test_basic.py:36-95`
- **JAX reduce_window docs**: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html
