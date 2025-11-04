# JAX Conv2D Operation - TDD Implementation Plan

**Date**: 2025-10-15
**Operation**: Conv2D (2D Convolution)
**Priority**: Critical (Required for YOLO11n)
**Estimated Time**: 3-4 hours

---

## Overview

Implement JAX backend support for PyTensor's 2D convolution operation using Test-Driven Development. Conv2D is the most critical CNN operation - every YOLO layer uses it.

**TDD Approach**: Write comprehensive tests first, verify they fail correctly, then implement by "debugging" the failing tests.

---

## Current State Analysis

### PyTensor Conv2D Operation
- **Class**: `pytensor.tensor.conv.abstract_conv.BaseAbstractConv` (pytensor/tensor/conv/abstract_conv.py:2059)
- **User API**: `pytensor.tensor.conv.abstract_conv.conv2d()` (line 3514)
- **Format**: NCHW (batch, channels, height, width)
- **Python backend**: Fully functional with NumPy implementation

### Current JAX Backend
- **Status**: ❌ Conv2D NOT implemented
- **Error**: `NotImplementedError: No JAX conversion for the given Op: BaseAbstractConv`
- **Impact**: Cannot use JAX backend for any CNN architectures

### Testing Infrastructure Available
- **Test utility**: `compare_jax_and_py()` in tests/link/jax/test_basic.py:36-95
- **Pattern**: Compare JAX backend output vs Python backend (ground truth)
- **Existing example**: tests/link/jax/signal/test_conv.py (1D convolution, 18 lines)

---

## Desired End State

### Implementation Target
- **File to create**: `pytensor/link/jax/dispatch/conv.py`
- **Pattern**: Use `@jax_funcify.register(BaseAbstractConv)` decorator
- **JAX function**: `jax.lax.conv_general_dilated()`
- **Result**: All tests pass, JAX and Python backends produce identical results

### Success Criteria
- [ ] All Conv2D tests pass (basic, parametrized, edge cases)
- [ ] Gradient tests pass (backpropagation works)
- [ ] Output matches Python backend within tolerance (rtol=1e-4)
- [ ] JAX returns DeviceArray (confirms GPU execution)
- [ ] Can build YOLO ConvBNSiLU block without errors

---

## What We're NOT Implementing

**Out of Scope:**
- 3D convolution (Conv3D) - only 2D needed for YOLO
- Transposed convolution (ConvTranspose) - YOLO uses upsampling instead
- Locally connected layers (unshared=True) - rare, not in YOLO
- Training-mode optimizations - inference correctness first

---

## TDD Approach

### Philosophy
1. **Tests define the specification** - No ambiguity about what's correct
2. **Fail first, then fix** - Verify tests actually test something
3. **One test at a time** - Implement incrementally
4. **Refactor fearlessly** - Tests protect you

### Test-First Workflow
```
Write Test → Run (expect FAIL) → Verify failure is correct →
Implement just enough → Run (expect PASS) → Repeat
```

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that fully specify Conv2D behavior. Tests will initially fail with `NotImplementedError`.

---

### Test File Structure

**File**: `tests/link/jax/test_conv.py`

**Imports**:
```python
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.conv.abstract_conv import conv2d
from tests.link.jax.test_basic import compare_jax_and_py

# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3
```

---

### Test Category 1: Basic Convolution Tests

**Purpose**: Verify core convolution functionality with standard configurations

#### Test: `test_conv2d_valid_padding`
**Purpose**: Test basic convolution with no padding (valid mode)

```python
def test_conv2d_valid_padding():
    """
    Test Conv2D with valid padding (no padding).

    This is the most basic convolution - output is smaller than input.
    Expected output size: (batch, out_channels, H-kH+1, W-kW+1)
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    # Act: Create convolution operation
    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    # Arrange: Generate test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")  # (N, C_in, H, W)
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")  # (C_out, C_in, kH, kW)

    # Assert: JAX output matches Python backend
    compare_jax_and_py(
        [x, filters],
        [out],
        [x_val, filters_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )
```

**Expected Failure Mode**:
- Error: `NotImplementedError: No JAX conversion for the given Op: BaseAbstractConv`
- Location: `pytensor/link/jax/dispatch/basic.py` in `jax_funcify()`

---

#### Test: `test_conv2d_same_padding`
**Purpose**: Test convolution with same padding (output size = input size)

```python
def test_conv2d_same_padding():
    """
    Test Conv2D with same padding.

    Same padding ensures output spatial dimensions equal input dimensions
    (with stride=1). This is common in ResNet and modern architectures.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="same", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

**Expected Failure**: Same as above (NotImplementedError)

---

#### Test: `test_conv2d_explicit_padding`
**Purpose**: Test explicit padding values as tuple

```python
@pytest.mark.parametrize("padding", [(1, 1), (2, 2), (1, 2)])
def test_conv2d_explicit_padding(padding):
    """
    Test Conv2D with explicit padding tuple.

    Padding can be specified as (pad_h, pad_w) to add specific padding.
    This is common when fine control over output size is needed.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode=padding, filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

**Note**: Parametrized to test multiple padding configurations

---

### Test Category 2: Filter Flip Tests

**Purpose**: Verify correct handling of convolution vs cross-correlation

#### Test: `test_conv2d_filter_flip_true_vs_false`
**Purpose**: Compare filter_flip=True (convolution) vs False (cross-correlation)

```python
def test_conv2d_filter_flip_true_vs_false():
    """
    Test filter_flip parameter behavior.

    filter_flip=True: True convolution (flip kernel 180 degrees)
    filter_flip=False: Cross-correlation (no flip)

    Results should be different for non-symmetric kernels.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    # Both modes
    out_flip = conv2d(x, filters, border_mode="valid", filter_flip=True)
    out_no_flip = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    # Non-symmetric kernel to see difference
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Test both
    compare_jax_and_py([x, filters], [out_flip], [x_val, filters_val])
    compare_jax_and_py([x, filters], [out_no_flip], [x_val, filters_val])
```

---

### Test Category 3: Stride Tests

**Purpose**: Verify strided convolution (downsampling)

#### Test: `test_conv2d_stride_2x2`
**Purpose**: Test 2x2 stride (common for downsampling)

```python
def test_conv2d_stride_2x2():
    """
    Test Conv2D with stride=(2, 2).

    Strided convolution reduces spatial dimensions by the stride factor.
    This is commonly used instead of pooling in modern architectures.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, subsample=(2, 2), border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

#### Test: `test_conv2d_stride_asymmetric`
**Purpose**: Test different strides for height and width

```python
@pytest.mark.parametrize("stride", [(2, 1), (1, 2), (3, 2)])
def test_conv2d_stride_asymmetric(stride):
    """
    Test Conv2D with asymmetric strides.

    Different strides for H and W dimensions are occasionally used
    when input has different aspect ratios or anisotropic features.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, subsample=stride, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

### Test Category 4: Dilation Tests

**Purpose**: Verify dilated (atrous) convolution

#### Test: `test_conv2d_dilation_2x2`
**Purpose**: Test dilated convolution with dilation factor 2

```python
def test_conv2d_dilation_2x2():
    """
    Test Conv2D with dilation=(2, 2) (atrous convolution).

    Dilation inserts gaps between kernel elements, expanding receptive
    field without increasing parameters. Used in DeepLab, etc.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(
        x, filters,
        border_mode="valid",
        filter_flip=False,
        filter_dilation=(2, 2)
    )

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

### Test Category 5: Kernel Size Variations

**Purpose**: Test different kernel sizes

#### Test: `test_conv2d_kernel_sizes`
**Purpose**: Test various kernel sizes (1x1, 5x5, 7x7)

```python
@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_conv2d_kernel_sizes(kernel_size):
    """
    Test Conv2D with various kernel sizes.

    - 1x1: Pointwise convolution (channel mixing)
    - 3x3: Most common (VGG, ResNet)
    - 5x5, 7x7: Larger receptive field (older architectures)
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, kernel_size, kernel_size)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

### Test Category 6: Edge Cases

**Purpose**: Test boundary conditions and special cases

#### Test: `test_conv2d_single_channel`
**Purpose**: Grayscale input (1 channel)

```python
def test_conv2d_single_channel():
    """
    Test Conv2D with single input channel (grayscale).

    Ensures broadcasting and indexing work correctly for C=1.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")  # C=1
    filters_val = rng.normal(size=(16, 1, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

#### Test: `test_conv2d_single_batch`
**Purpose**: Batch size of 1

```python
def test_conv2d_single_batch():
    """
    Test Conv2D with batch size 1 (inference mode).

    Common during inference when processing single images.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(1, 3, 8, 8)).astype("float32")  # N=1
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

#### Test: `test_conv2d_large_batch`
**Purpose**: Larger batch sizes

```python
@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_conv2d_large_batch(batch_size):
    """
    Test Conv2D with larger batch sizes.

    Verifies batching works correctly and efficiently.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(batch_size, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

#### Test: `test_conv2d_grouped`
**Purpose**: Grouped convolution (depthwise when groups=channels)

```python
@pytest.mark.parametrize("num_groups", [2, 4])
def test_conv2d_grouped(num_groups):
    """
    Test grouped convolution.

    Grouped conv splits channels into groups, reducing parameters.
    When num_groups == in_channels, it's depthwise convolution.
    Used in MobileNet, ShuffleNet, etc.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    in_channels = 8
    out_channels = 16

    out = conv2d(
        x, filters,
        border_mode="valid",
        filter_flip=False,
        num_groups=num_groups
    )

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, in_channels, 8, 8)).astype("float32")
    # Grouped: out_channels must be divisible by num_groups
    filters_val = rng.normal(size=(out_channels, in_channels // num_groups, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])
```

---

### Test Category 7: Gradient Tests

**Purpose**: Verify backpropagation works correctly

#### Test: `test_conv2d_gradient_wrt_input`
**Purpose**: Test gradient computation w.r.t. input

```python
def test_conv2d_gradient_wrt_input():
    """
    Test Conv2D gradient with respect to input.

    Verifies that JAX's automatic differentiation produces correct
    gradients for the input tensor during backpropagation.
    """
    from pytensor import function, grad
    from pytensor.compile.sharedvalue import shared

    x = pt.tensor4("x", dtype="float32")
    filters_val = np.random.randn(16, 3, 3, 3).astype("float32")
    filters = shared(filters_val, name="filters")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)
    loss = out.sum()

    grad_x = grad(loss, x)

    # Compile with JAX mode
    f = function([x], [loss, grad_x], mode="JAX")

    x_val = np.random.randn(2, 3, 8, 8).astype("float32")
    loss_val, grad_x_val = f(x_val)

    # Verify gradient is not zero (should have meaningful values)
    assert np.abs(grad_x_val).sum() > 0, "Gradient should not be zero"
    assert grad_x_val.shape == x_val.shape, "Gradient shape should match input"

    # Compare with Python backend
    f_py = function([x], [loss, grad_x], mode="FAST_RUN")
    loss_py, grad_x_py = f_py(x_val)

    np.testing.assert_allclose(grad_x_val, grad_x_py, rtol=RTOL, atol=ATOL)
```

**Expected Failure**: NotImplementedError initially, then gradient should work automatically with JAX

---

#### Test: `test_conv2d_gradient_wrt_filters`
**Purpose**: Test gradient computation w.r.t. filters (weight updates)

```python
def test_conv2d_gradient_wrt_filters():
    """
    Test Conv2D gradient with respect to filters.

    This is critical for training - verifies that filter gradients are
    computed correctly for weight updates during backpropagation.
    """
    from pytensor import function, grad
    from pytensor.compile.sharedvalue import shared

    x_val = np.random.randn(2, 3, 8, 8).astype("float32")
    x = shared(x_val, name="x")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)
    loss = out.sum()

    grad_filters = grad(loss, filters)

    f = function([filters], [loss, grad_filters], mode="JAX")

    filters_val = np.random.randn(16, 3, 3, 3).astype("float32")
    loss_val, grad_filters_val = f(filters_val)

    assert np.abs(grad_filters_val).sum() > 0
    assert grad_filters_val.shape == filters_val.shape

    # Compare with Python backend
    f_py = function([filters], [loss, grad_filters], mode="FAST_RUN")
    loss_py, grad_filters_py = f_py(filters_val)

    np.testing.assert_allclose(grad_filters_val, grad_filters_py, rtol=RTOL, atol=ATOL)
```

---

### Test Category 8: Dtype Tests

**Purpose**: Verify float32 and float64 compatibility

#### Test: `test_conv2d_dtypes`
**Purpose**: Test different float precisions

```python
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_conv2d_dtypes(dtype):
    """
    Test Conv2D with different dtypes.

    Ensures convolution works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    filters = pt.tensor4("filters", dtype=dtype)

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype(dtype)
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype(dtype)

    # Adjust tolerance for float32
    rtol = 1e-3 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    compare_jax_and_py(
        [x, filters], [out], [x_val, filters_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
    )
```

---

## Test Implementation Steps

### Step 1: Create Test File
```bash
# Create the test file
touch tests/link/jax/test_conv.py
```

### Step 2: Add Test Structure
1. Add imports
2. Set up tolerance constants
3. Add all test functions from above

### Step 3: Verify Tests Are Discoverable
```bash
pytest --collect-only tests/link/jax/test_conv.py
```

**Expected output**: List of ~18 test items

---

## Phase 1 Success Criteria

### Automated Verification:
- [ ] Test file created: `tests/link/jax/test_conv.py`
- [ ] Tests are discoverable: `pytest --collect-only tests/link/jax/test_conv.py`
- [ ] All tests have docstrings: Check manually
- [ ] No syntax errors: `python -m py_compile tests/link/jax/test_conv.py`

### Manual Verification:
- [ ] Each test has clear purpose in docstring
- [ ] Test names follow `test_conv2d_<scenario>` pattern
- [ ] Test data shapes are documented in comments
- [ ] Parametrized tests cover multiple configurations
- [ ] Code is readable and follows project style

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected, diagnostic ways.

### Verification Steps

#### Step 1: Run Full Test Suite
```bash
pytest tests/link/jax/test_conv.py -v
```

**Expected Output**:
```
tests/link/jax/test_conv.py::test_conv2d_valid_padding FAILED
tests/link/jax/test_conv.py::test_conv2d_same_padding FAILED
tests/link/jax/test_conv.py::test_conv2d_explicit_padding[padding0] FAILED
...
======================== 18 failed in X.XXs ========================
```

#### Step 2: Examine Failure Details
```bash
pytest tests/link/jax/test_conv.py::test_conv2d_valid_padding -v --tb=short
```

**Expected Error**:
```python
NotImplementedError: No JAX conversion for the given Op: <BaseAbstractConv at 0x...>
```

**Stack trace should point to**:
- `pytensor/link/jax/dispatch/basic.py` in `jax_funcify()`
- Shows that JAX dispatch is attempted but not found

### Expected Failure Analysis

#### For Each Test, Verify:

1. **Failure Type**: NotImplementedError (not AttributeError, ImportError, etc.)
2. **Error Message**: Clear indication that Conv2D dispatch is missing
3. **Stack Trace**: Points to JAX dispatch mechanism
4. **No False Passes**: Confirm no test passes (would indicate test is broken)

### Failure Documentation

Create checklist:

- [ ] `test_conv2d_valid_padding`: NotImplementedError ✓
- [ ] `test_conv2d_same_padding`: NotImplementedError ✓
- [ ] `test_conv2d_explicit_padding`: NotImplementedError ✓ (all variants)
- [ ] `test_conv2d_filter_flip_true_vs_false`: NotImplementedError ✓
- [ ] `test_conv2d_stride_2x2`: NotImplementedError ✓
- [ ] `test_conv2d_stride_asymmetric`: NotImplementedError ✓ (all variants)
- [ ] `test_conv2d_dilation_2x2`: NotImplementedError ✓
- [ ] `test_conv2d_kernel_sizes`: NotImplementedError ✓ (all variants)
- [ ] `test_conv2d_single_channel`: NotImplementedError ✓
- [ ] `test_conv2d_single_batch`: NotImplementedError ✓
- [ ] `test_conv2d_large_batch`: NotImplementedError ✓ (all variants)
- [ ] `test_conv2d_grouped`: NotImplementedError ✓ (all variants)
- [ ] `test_conv2d_gradient_wrt_input`: NotImplementedError ✓
- [ ] `test_conv2d_gradient_wrt_filters`: NotImplementedError ✓
- [ ] `test_conv2d_dtypes`: NotImplementedError ✓ (both dtypes)

### Adjustment Phase

If tests don't fail correctly:

**Problem**: Test passes unexpectedly
- **Cause**: Test is too lenient or doesn't actually use Conv2D
- **Fix**: Verify `conv2d()` is actually called in the test

**Problem**: Wrong error type (AttributeError, ImportError)
- **Cause**: Missing import or wrong function call
- **Fix**: Check imports and function signatures

**Problem**: Cryptic error message
- **Cause**: Test setup issue
- **Fix**: Add better assertions and error messages

---

## Phase 2 Success Criteria

### Automated Verification:
- [ ] All tests fail (none pass): `pytest tests/link/jax/test_conv.py -v | grep FAILED | wc -l` → 18+
- [ ] No unexpected errors: No ImportError, AttributeError (only NotImplementedError)
- [ ] Tests run to completion: No crashes or hangs

### Manual Verification:
- [ ] Each test fails with NotImplementedError
- [ ] Error messages clearly indicate missing Conv2D dispatch
- [ ] Stack traces are informative
- [ ] Failure output would help during implementation

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement Conv2D JAX dispatch by making tests pass one at a time. Work like debugging - let test failures guide implementation.

### Implementation Strategy

**Order of Implementation**:
1. Start with `test_conv2d_valid_padding` (simplest case)
2. Then `test_conv2d_same_padding` (add padding logic)
3. Then `test_conv2d_explicit_padding` (generalize padding)
4. Continue in order of complexity

### Implementation File

**Create**: `pytensor/link/jax/dispatch/conv.py`

#### Implementation Structure

```python
"""JAX dispatch for convolution operations."""

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.conv.abstract_conv import BaseAbstractConv


@jax_funcify.register(BaseAbstractConv)
def jax_funcify_BaseAbstractConv(op, node, **kwargs):
    """
    Convert PyTensor Conv2D to JAX conv_general_dilated.

    Parameters from op:
    - subsample: (stride_h, stride_w)
    - border_mode: 'valid', 'same', 'half', 'full', or tuple
    - filter_dilation: (dilation_h, dilation_w)
    - filter_flip: bool (True for convolution, False for cross-correlation)
    - num_groups: int (for grouped/depthwise convolution)

    Returns:
        Function that performs convolution using JAX
    """
    # TODO: Extract op attributes
    # TODO: Convert border_mode to JAX padding format
    # TODO: Set dimension numbers (NCHW format)
    # TODO: Return inner function

    raise NotImplementedError("Conv2D JAX dispatch not yet implemented")
```

### Implementation Steps

#### Step 1: Basic Valid Padding (Make test_conv2d_valid_padding Pass)

**Target**: `test_conv2d_valid_padding`

**Current Failure**: NotImplementedError

**Implementation**:

```python
@jax_funcify.register(BaseAbstractConv)
def jax_funcify_BaseAbstractConv(op, node, **kwargs):
    """Convert PyTensor Conv2D to JAX conv_general_dilated."""

    # Extract op attributes
    subsample = op.subsample  # (stride_h, stride_w)
    border_mode = op.border_mode
    filter_dilation = getattr(op, 'filter_dilation', (1, 1))
    num_groups = getattr(op, 'num_groups', 1)
    filter_flip = op.filter_flip

    # Convert border_mode to JAX padding
    if border_mode == 'valid':
        padding = 'VALID'
    else:
        raise NotImplementedError(f"border_mode={border_mode} not yet supported")

    # Dimension numbers: PyTensor uses NCHW format
    dimension_numbers = ('NCHW', 'OIHW', 'NCHW')

    def conv2d(input, filters):
        """
        Perform convolution using JAX.

        Args:
            input: (N, C_in, H, W)
            filters: (C_out, C_in, kH, kW)

        Returns:
            output: (N, C_out, H', W')
        """
        # Handle filter flip
        if filter_flip:
            # Flip kernel spatially for true convolution
            filters = jnp.flip(filters, axis=(-2, -1))

        # Call JAX convolution
        output = jax.lax.conv_general_dilated(
            lhs=input,
            rhs=filters,
            window_strides=subsample,
            padding=padding,
            lhs_dilation=(1, 1),
            rhs_dilation=filter_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=num_groups,
        )

        return output

    return conv2d
```

**Debugging Approach**:
1. Run: `pytest tests/link/jax/test_conv.py::test_conv2d_valid_padding -v`
2. If error, read message carefully
3. Fix the specific issue
4. Re-run test
5. Repeat until test passes

**Success Criteria**:
- [ ] Test passes: `pytest tests/link/jax/test_conv.py::test_conv2d_valid_padding -v`
- [ ] No new linting errors: `ruff check pytensor/link/jax/dispatch/conv.py`

---

#### Step 2: Same Padding (Make test_conv2d_same_padding Pass)

**Target**: `test_conv2d_same_padding`

**Expected Issue**: border_mode='same' raises NotImplementedError in current code

**Add to implementation**:

```python
# In jax_funcify_BaseAbstractConv, update padding logic:

if border_mode == 'valid':
    padding = 'VALID'
elif border_mode == 'same' or border_mode == 'half':
    padding = 'SAME'
else:
    raise NotImplementedError(f"border_mode={border_mode} not yet supported")
```

**Run**: `pytest tests/link/jax/test_conv.py::test_conv2d_same_padding -v`

**Success Criteria**:
- [ ] Test passes
- [ ] Previous test still passes (no regression)

---

#### Step 3: Explicit Padding (Make test_conv2d_explicit_padding Pass)

**Target**: `test_conv2d_explicit_padding`

**Expected Issue**: Tuple padding not handled

**Add to implementation**:

```python
# Update padding logic:

if border_mode == 'valid':
    padding = 'VALID'
elif border_mode == 'same' or border_mode == 'half':
    padding = 'SAME'
elif isinstance(border_mode, (tuple, list)):
    # Explicit padding: (pad_h, pad_w)
    # JAX expects: [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    if len(border_mode) == 2:
        padding = [(border_mode[0], border_mode[0]), (border_mode[1], border_mode[1])]
    else:
        raise ValueError(f"Invalid border_mode tuple: {border_mode}")
else:
    raise ValueError(f"Unsupported border_mode: {border_mode}")
```

**Run**: `pytest tests/link/jax/test_conv.py::test_conv2d_explicit_padding -v`

**Success Criteria**:
- [ ] All padding tests pass
- [ ] All previous tests still pass

---

#### Step 4: Continue Through Remaining Tests

**Process**:
1. Run next failing test
2. Read error message
3. Implement missing feature
4. Re-run test
5. Verify no regressions: `pytest tests/link/jax/test_conv.py -v`

**Expected Order**:
1. ✓ Valid padding (done)
2. ✓ Same padding (done)
3. ✓ Explicit padding (done)
4. Filter flip tests → Should already work
5. Stride tests → Should already work
6. Dilation tests → Should already work
7. Kernel size tests → Should already work
8. Edge case tests → Should already work
9. Grouped tests → Should already work
10. Gradient tests → Should work automatically with JAX autodiff
11. Dtype tests → Should already work

### Register Module

**Update**: `pytensor/link/jax/dispatch/__init__.py`

Add import so dispatch is registered:

```python
# Add to imports
from pytensor.link.jax.dispatch import conv  # noqa: F401
```

---

## Phase 3 Success Criteria

### Automated Verification:
- [ ] All tests pass: `pytest tests/link/jax/test_conv.py -v`
- [ ] No regressions: `pytest tests/link/jax/ -v` (all JAX tests)
- [ ] Linting passes: `ruff check pytensor/link/jax/dispatch/conv.py`
- [ ] Type checking passes: `mypy pytensor/link/jax/dispatch/conv.py`

### Manual Verification:
- [ ] Implementation is clean and readable
- [ ] Code follows PyTensor conventions
- [ ] Comments explain JAX-specific details
- [ ] No obvious performance issues

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality while keeping tests green.

### Refactoring Targets

#### 1. Code Organization
- Extract padding logic to helper function
- Add clear section comments
- Group related logic

#### 2. Documentation
- Add comprehensive docstring to main function
- Document parameter mappings
- Add examples in comments

#### 3. Error Messages
- Improve error messages for unsupported modes
- Add helpful suggestions

### Refactoring Steps

#### Before Each Change:
```bash
# Ensure tests pass
pytest tests/link/jax/test_conv.py -v
```

#### After Each Change:
```bash
# Verify tests still pass
pytest tests/link/jax/test_conv.py -v

# If pass, commit
git add pytensor/link/jax/dispatch/conv.py
git commit -m "refactor: improve conv.py [specific change]"

# If fail, revert and reconsider
git restore pytensor/link/jax/dispatch/conv.py
```

### Example Refactorings

#### Extract Padding Helper:

```python
def _convert_border_mode_to_jax_padding(border_mode):
    """
    Convert PyTensor border_mode to JAX padding format.

    Args:
        border_mode: 'valid', 'same', 'half', or tuple

    Returns:
        JAX padding: 'VALID', 'SAME', or list of tuples
    """
    if border_mode == 'valid':
        return 'VALID'
    elif border_mode == 'same' or border_mode == 'half':
        return 'SAME'
    elif isinstance(border_mode, (tuple, list)):
        if len(border_mode) == 2:
            return [(border_mode[0], border_mode[0]), (border_mode[1], border_mode[1])]
        else:
            raise ValueError(f"Invalid border_mode tuple: {border_mode}")
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")


@jax_funcify.register(BaseAbstractConv)
def jax_funcify_BaseAbstractConv(op, node, **kwargs):
    """Convert PyTensor Conv2D to JAX conv_general_dilated."""

    # Extract and convert parameters
    subsample = op.subsample
    padding = _convert_border_mode_to_jax_padding(op.border_mode)
    filter_dilation = getattr(op, 'filter_dilation', (1, 1))
    num_groups = getattr(op, 'num_groups', 1)
    filter_flip = op.filter_flip
    dimension_numbers = ('NCHW', 'OIHW', 'NCHW')

    def conv2d(input, filters):
        # ... rest of implementation
```

**Run tests**: `pytest tests/link/jax/test_conv.py -v`

#### Improve Docstrings:

Add detailed docstring to main function with examples, parameter explanations, etc.

**Run tests**: Verify still pass

---

## Phase 4 Success Criteria

### Automated Verification:
- [ ] All tests still pass: `pytest tests/link/jax/test_conv.py -v`
- [ ] No regressions: `pytest tests/link/jax/ -v`
- [ ] Linting passes: `ruff check pytensor/link/jax/dispatch/conv.py`
- [ ] Type hints added: `mypy pytensor/link/jax/dispatch/conv.py`

### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] Helper functions have clear single responsibilities
- [ ] Docstrings are comprehensive
- [ ] Comments explain "why" not "what"
- [ ] No unnecessary complexity

---

## Final Verification

### Integration with YOLO

Test that Conv2D works in YOLO ConvBNSiLU block:

```python
# In separate test file or manual testing
import pytensor.tensor as pt
from pytensor.tensor.conv.abstract_conv import conv2d
from pytensor.tensor.batchnorm import batch_normalization
from pytensor.tensor.nnet import sigmoid

def test_yolo_conv_bn_silu_block():
    """Test YOLO ConvBNSiLU block with JAX backend."""

    # ConvBNSiLU: Conv → BatchNorm → SiLU activation
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")
    gamma = pt.vector("gamma")
    beta = pt.vector("beta")
    mean = pt.vector("mean")
    var = pt.vector("var")

    # Conv
    conv_out = conv2d(x, filters, border_mode="same", filter_flip=False)

    # BatchNorm
    bn_out = batch_normalization(conv_out, gamma, beta, mean, var)

    # SiLU (x * sigmoid(x))
    silu_out = bn_out * sigmoid(bn_out)

    # Should compile without errors
    from pytensor import function
    f = function([x, filters, gamma, beta, mean, var], silu_out, mode="JAX")

    # Run
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(1, 3, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")
    gamma_val = np.ones(16, dtype="float32")
    beta_val = np.zeros(16, dtype="float32")
    mean_val = np.zeros(16, dtype="float32")
    var_val = np.ones(16, dtype="float32")

    result = f(x_val, filters_val, gamma_val, beta_val, mean_val, var_val)

    assert result.shape == (1, 16, 32, 32)
    print("✓ YOLO ConvBNSiLU block works with JAX!")
```

---

## Summary

### Test Coverage
- **Basic operations**: 3 tests (valid, same, explicit padding)
- **Filter flip**: 1 test
- **Stride variations**: 2 tests (+ parametrized)
- **Dilation**: 1 test
- **Kernel sizes**: 1 test (parametrized for 4 sizes)
- **Edge cases**: 3 tests (+ parametrized)
- **Grouped conv**: 1 test (parametrized)
- **Gradients**: 2 tests (input and filter gradients)
- **Dtypes**: 1 test (parametrized)

**Total**: ~18-20 individual test cases (accounting for parametrization)

### Time Estimate
- **Phase 1** (Write tests): 1 hour
- **Phase 2** (Verify failures): 30 minutes
- **Phase 3** (Implementation): 1.5-2 hours
- **Phase 4** (Refactoring): 30 minutes

**Total**: ~3.5-4 hours

### Next Steps
1. Create `tests/link/jax/test_conv.py` with all tests
2. Run tests and verify they fail correctly
3. Implement `pytensor/link/jax/dispatch/conv.py`
4. Make tests pass one by one
5. Refactor and document
6. Test with YOLO ConvBNSiLU block

---

## References

- **Original plan**: `thoughts/shared/plans/jax-cnn-ops-implementation.md`
- **PyTensor Conv2D**: `pytensor/tensor/conv/abstract_conv.py:2059`
- **JAX dispatch pattern**: `pytensor/link/jax/dispatch/basic.py`
- **Test utility**: `tests/link/jax/test_basic.py:36-95`
- **JAX conv docs**: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html
