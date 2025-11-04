# JAX Resize Operation - TDD Implementation Plan

**Date**: 2025-10-15
**Operation**: Resize (Spatial Upsampling/Downsampling)
**Priority**: Critical (Required for YOLO11n FPN)
**Estimated Time**: 1.5-2 hours

---

## Overview

Implement JAX backend support for PyTensor's resize operation using Test-Driven Development. Resize is essential for YOLO's Feature Pyramid Network (FPN) - upsamples feature maps before concatenation.

**TDD Approach**: Write comprehensive tests first, verify they fail correctly, then implement by "debugging" the failing tests.

---

## Current State Analysis

### PyTensor Resize Operation
- **Class**: `pytensor.tensor.resize.Resize` (pytensor/tensor/resize.py:31)
- **User API**: `pytensor.tensor.resize.resize()`
- **Format**: NCHW (batch, channels, height, width)
- **Methods**: 'nearest' (nearest neighbor), 'linear' (bilinear interpolation)
- **Python backend**: Uses NumPy indexing (nearest) or scipy.ndimage.zoom (linear)

### Current JAX Backend
- **Status**: ❌ Resize NOT implemented
- **Error**: `NotImplementedError: No JAX conversion for the given Op: Resize`
- **Impact**: Cannot use upsampling in FPN architectures

### Testing Infrastructure Available
- **Test utility**: `compare_jax_and_py()` in tests/link/jax/test_basic.py:36-95
- **Pattern**: Compare JAX backend output vs Python backend (ground truth)
- **Reference tests**: tests/tensor/test_resize.py (non-JAX tests)

---

## Desired End State

### Implementation Target
- **File to create**: `pytensor/link/jax/dispatch/resize.py`
- **Pattern**: Use `@jax_funcify.register(Resize)` decorator
- **JAX function**: `jax.image.resize()` (handles both nearest and bilinear)
- **Result**: All tests pass, JAX and Python backends produce identical results

### Success Criteria
- [ ] All Resize tests pass (nearest and bilinear modes)
- [ ] Gradient tests pass (backpropagation works)
- [ ] Output matches Python backend within tolerance (rtol=1e-4)
- [ ] JAX returns DeviceArray (confirms GPU execution)
- [ ] Can build YOLO FPN upsampling path without errors

---

## What We're NOT Implementing

**Out of Scope:**
- Bicubic interpolation - JAX supports it, but not in PyTensor Resize op
- 3D resize - Only 2D (4D tensors) needed for YOLO
- Non-uniform scaling (different scale per dimension in same call) - handled via scale_factor tuple
- Align corners parameter - Not in PyTensor op

---

## TDD Approach

### Philosophy
1. **Tests define the specification** - No ambiguity about resize behavior
2. **Fail first, then fix** - Verify tests actually test something
3. **One test at a time** - Implement incrementally
4. **Test both modes carefully** - Nearest and bilinear have different behaviors

### Test-First Workflow
```
Write Test → Run (expect FAIL) → Verify failure is correct →
Implement just enough → Run (expect PASS) → Repeat
```

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that fully specify Resize behavior. Tests will initially fail with `NotImplementedError`.

---

### Test File Structure

**File**: `tests/link/jax/test_resize.py`

**Imports**:
```python
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.compile.sharedvalue import shared
from pytensor.tensor.resize import resize
from tests.link.jax.test_basic import compare_jax_and_py

# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3
```

---

### Test Category 1: Basic Upsampling Tests

**Purpose**: Verify core upsampling functionality

#### Test: `test_resize_nearest_2x_upsample`
**Purpose**: Test 2x upsampling with nearest neighbor (most common in YOLO)

```python
def test_resize_nearest_2x_upsample():
    """
    Test Resize with 2x upsampling using nearest neighbor.

    This is the most common upsampling in YOLO FPN - doubles spatial
    dimensions by replicating pixels.
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")

    # Act: Create resize operation
    out = resize(x, scale_factor=(2.0, 2.0), mode="nearest")

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
- Error: `NotImplementedError: No JAX conversion for the given Op: Resize`
- Location: `pytensor/link/jax/dispatch/basic.py` in `jax_funcify()`

---

#### Test: `test_resize_bilinear_2x_upsample`
**Purpose**: Test 2x upsampling with bilinear interpolation

```python
def test_resize_bilinear_2x_upsample():
    """
    Test Resize with 2x upsampling using bilinear interpolation.

    Bilinear provides smoother upsampling than nearest neighbor,
    useful when visual quality matters.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(2.0, 2.0), mode="linear")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 2: Basic Downsampling Tests

**Purpose**: Verify downsampling functionality

#### Test: `test_resize_nearest_half_downsample`
**Purpose**: Test 0.5x downsampling with nearest neighbor

```python
def test_resize_nearest_half_downsample():
    """
    Test Resize with 0.5x downsampling using nearest neighbor.

    Reduces spatial dimensions by half by sampling every other pixel.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(0.5, 0.5), mode="nearest")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_bilinear_half_downsample`
**Purpose**: Test 0.5x downsampling with bilinear interpolation

```python
def test_resize_bilinear_half_downsample():
    """
    Test Resize with 0.5x downsampling using bilinear interpolation.

    Bilinear downsampling provides anti-aliasing, reducing artifacts.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(0.5, 0.5), mode="linear")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 3: Scale Factor Variations

**Purpose**: Test different scale factors

#### Test: `test_resize_integer_scales`
**Purpose**: Test integer scale factors (2x, 3x, 4x)

```python
@pytest.mark.parametrize("scale", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_integer_scales(scale, mode):
    """
    Test Resize with integer scale factors.

    Integer scales are common and should have exact dimension calculations.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(scale, scale), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_fractional_scales`
**Purpose**: Test fractional scale factors (1.5x, 0.75x)

```python
@pytest.mark.parametrize("scale", [1.5, 0.75, 0.25])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_fractional_scales(scale, mode):
    """
    Test Resize with fractional scale factors.

    Non-integer scales require interpolation and careful rounding.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(scale, scale), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_asymmetric_scales`
**Purpose**: Test different scale factors for H and W

```python
@pytest.mark.parametrize("scale_h, scale_w", [(2.0, 1.5), (0.5, 2.0), (3.0, 0.75)])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_asymmetric_scales(scale_h, scale_w, mode):
    """
    Test Resize with asymmetric scale factors.

    Different H and W scales are used when aspect ratio needs to change.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(scale_h, scale_w), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 4: Extreme Scale Factors

**Purpose**: Test edge cases with very small or large scales

#### Test: `test_resize_very_small_scale`
**Purpose**: Test extreme downsampling (0.1x)

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_very_small_scale(mode):
    """
    Test Resize with very small scale factor (extreme downsampling).

    Reduces 100x100 to 10x10, testing robustness of interpolation.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(0.1, 0.1), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 100, 100)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_very_large_scale`
**Purpose**: Test extreme upsampling (10x)

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_very_large_scale(mode):
    """
    Test Resize with very large scale factor (extreme upsampling).

    Expands 10x10 to 100x100, testing interpolation quality.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(10.0, 10.0), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 10, 10)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 5: Special Cases

**Purpose**: Test boundary conditions

#### Test: `test_resize_scale_1x1`
**Purpose**: Identity resize (scale=1.0)

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_scale_1x1(mode):
    """
    Test Resize with scale=1.0 (identity operation).

    Should return input unchanged. Tests edge case of no scaling.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(1.0, 1.0), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_to_1x1_output`
**Purpose**: Extreme downsampling to 1x1

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_to_1x1_output(mode):
    """
    Test Resize to 1x1 output (extreme downsampling).

    Each channel becomes a single pixel (like global pooling).
    """
    x = pt.tensor4("x", dtype="float32")
    # Calculate scale to get 1x1 output from 16x16 input
    out = resize(x, scale_factor=(1.0/16, 1.0/16), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_single_pixel_input`
**Purpose**: Upsampling from 1x1 input

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_single_pixel_input(mode):
    """
    Test Resize from 1x1 input (upsampling single pixel).

    Nearest: replicates pixel. Bilinear: also replicates (no neighbors).
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(8.0, 8.0), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 1, 1)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_single_channel`
**Purpose**: Single channel input (grayscale)

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_single_channel(mode):
    """
    Test Resize with single channel (C=1).

    Ensures channel dimension is handled correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(2.0, 2.0), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

#### Test: `test_resize_many_channels`
**Purpose**: Many channels (like deep CNN layers)

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_many_channels(mode):
    """
    Test Resize with many channels (C=512).

    Verifies resizing scales to deeper network layers.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(2.0, 2.0), mode=mode)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 512, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])
```

---

### Test Category 6: Gradient Tests

**Purpose**: Verify backpropagation through resize operations

#### Test: `test_resize_nearest_gradient`
**Purpose**: Test gradient computation for nearest neighbor

```python
def test_resize_nearest_gradient():
    """
    Test Resize gradient with nearest neighbor mode.

    Nearest neighbor gradient routes gradient back to the pixel
    that was selected in forward pass.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(2.0, 2.0), mode="nearest")
    loss = out.sum()

    # Compute gradient
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    # Compare JAX and Python backends
    compare_jax_and_py([x], [grad_x], [x_val])
```

---

#### Test: `test_resize_bilinear_gradient`
**Purpose**: Test gradient computation for bilinear interpolation

```python
def test_resize_bilinear_gradient():
    """
    Test Resize gradient with bilinear mode.

    Bilinear gradient distributes gradient to the 4 neighboring
    pixels weighted by interpolation coefficients.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(2.0, 2.0), mode="linear")
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [grad_x], [x_val])
```

---

#### Test: `test_resize_gradient_with_downsample`
**Purpose**: Test gradient with downsampling

```python
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_gradient_with_downsample(mode):
    """
    Test Resize gradient with downsampling.

    Downsampling gradients should aggregate correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = resize(x, scale_factor=(0.5, 0.5), mode=mode)
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [grad_x], [x_val])
```

---

### Test Category 7: Mode Comparison Tests

**Purpose**: Document differences between nearest and bilinear

#### Test: `test_resize_nearest_vs_bilinear`
**Purpose**: Show behavioral differences between modes

```python
def test_resize_nearest_vs_bilinear():
    """
    Test that nearest and bilinear produce different results.

    This documents expected behavior difference between modes.
    Nearest: sharp edges (replication)
    Bilinear: smooth interpolation
    """
    x = pt.tensor4("x", dtype="float32")
    out_nearest = resize(x, scale_factor=(2.0, 2.0), mode="nearest")
    out_bilinear = resize(x, scale_factor=(2.0, 2.0), mode="linear")

    # Simple test pattern that shows difference clearly
    # Checkerboard pattern: [[0, 1], [1, 0]]
    x_val = np.array([[[[0.0, 1.0], [1.0, 0.0]]]], dtype="float32")

    # Get outputs from both modes
    from pytensor import function
    f_nearest = function([x], out_nearest, mode="JAX")
    f_bilinear = function([x], out_bilinear, mode="JAX")

    result_nearest = f_nearest(x_val)
    result_bilinear = f_bilinear(x_val)

    # Results should be different (bilinear has interpolated values)
    assert not np.allclose(result_nearest, result_bilinear), \
        "Nearest and bilinear should produce different results"

    # Nearest should only have 0s and 1s (no interpolation)
    assert np.all((result_nearest == 0) | (result_nearest == 1)), \
        "Nearest neighbor should only have original values"

    # Bilinear should have interpolated values (between 0 and 1)
    unique_vals = np.unique(result_bilinear)
    assert len(unique_vals) > 2, \
        "Bilinear should have interpolated intermediate values"
```

---

### Test Category 8: Dtype Tests

**Purpose**: Verify float32 and float64 compatibility

#### Test: `test_resize_dtypes`
**Purpose**: Test different float precisions

```python
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_dtypes(dtype, mode):
    """
    Test Resize with different dtypes.

    Ensures resizing works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    out = resize(x, scale_factor=(2.0, 2.0), mode=mode)

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

### Test Category 9: Integration Tests

**Purpose**: Test YOLO-specific patterns

#### Test: `test_yolo_fpn_upsample`
**Purpose**: Test YOLO FPN upsampling pattern

```python
def test_yolo_fpn_upsample():
    """
    Test YOLO FPN upsampling pattern.

    FPN upsamples lower-resolution features 2x to match higher-resolution
    features before concatenation.
    """
    # Simulate FPN: low-res and high-res features
    x_low = pt.tensor4("x_low", dtype="float32")  # e.g., 10x10
    x_high = pt.tensor4("x_high", dtype="float32")  # e.g., 20x20

    # Upsample low-res to match high-res
    x_low_upsampled = resize(x_low, scale_factor=(2.0, 2.0), mode="nearest")

    # Concatenate (YOLO FPN pattern)
    concat = pt.concatenate([x_high, x_low_upsampled], axis=1)

    # Test data
    rng = np.random.default_rng(42)
    x_low_val = rng.normal(size=(1, 128, 10, 10)).astype("float32")
    x_high_val = rng.normal(size=(1, 64, 20, 20)).astype("float32")

    # Should work without errors and produce correct shape
    compare_jax_and_py(
        [x_low, x_high],
        [concat],
        [x_low_val, x_high_val],
    )

    # Verify output shape
    from pytensor import function
    f = function([x_low, x_high], concat, mode="JAX")
    result = f(x_low_val, x_high_val)

    expected_shape = (1, 128 + 64, 20, 20)
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"
```

---

## Test Implementation Steps

### Step 1: Create Test File
```bash
touch tests/link/jax/test_resize.py
```

### Step 2: Add Test Structure
1. Add imports
2. Set up tolerance constants
3. Add all test functions

### Step 3: Verify Tests Are Discoverable
```bash
pytest --collect-only tests/link/jax/test_resize.py
```

**Expected output**: List of ~30 test items

---

## Phase 1 Success Criteria

### Automated Verification:
- [x] Test file created: `tests/link/jax/test_resize.py`
- [x] Tests are discoverable: `pytest --collect-only tests/link/jax/test_resize.py`
- [x] All tests have docstrings
- [x] No syntax errors: `python -m py_compile tests/link/jax/test_resize.py`

### Manual Verification:
- [x] Each test has clear purpose
- [x] Test names are descriptive
- [x] Parametrized tests cover multiple configurations

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected ways.

### Verification Steps

```bash
pytest tests/link/jax/test_resize.py -v
```

**Expected**: All tests FAILED with NotImplementedError

```bash
pytest tests/link/jax/test_resize.py::test_resize_nearest_2x_upsample -v --tb=short
```

**Expected Error**: `NotImplementedError: No JAX conversion for the given Op: Resize`

---

## Phase 2 Success Criteria

### Automated Verification:
- [x] All tests fail with NotImplementedError
- [x] No unexpected errors
- [x] Tests run to completion

### Manual Verification:
- [x] Error messages are clear
- [x] Stack traces are informative

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement Resize JAX dispatch by making tests pass one at a time.

### Implementation Strategy

**Order**: Start with `test_resize_nearest_2x_upsample` (most common in YOLO)

### Implementation File

**Create**: `pytensor/link/jax/dispatch/resize.py`

#### Implementation Structure

```python
"""JAX dispatch for resize operations."""

import jax.image
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.resize import Resize


@jax_funcify.register(Resize)
def jax_funcify_Resize(op, node, **kwargs):
    """
    Convert PyTensor Resize to JAX image.resize.

    Parameters from op:
    - scale_factor: (scale_h, scale_w)
    - mode: 'nearest' or 'linear' (bilinear)

    Returns:
        Function that performs resizing using JAX
    """
    scale_factor = op.scale_factor
    mode = op.mode

    # Map PyTensor mode to JAX method
    if mode == "nearest":
        jax_method = "nearest"
    elif mode == "linear":
        jax_method = "bilinear"
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    def resize_fn(input):
        """
        Perform resize using JAX.

        Args:
            input: (N, C, H, W) in NCHW format

        Returns:
            output: (N, C, H', W') where H' = H * scale_h, W' = W * scale_w
        """
        batch, channels, height, width = input.shape

        # Calculate new dimensions
        new_h = int(height * scale_factor[0])
        new_w = int(width * scale_factor[1])

        # JAX image.resize expects NHWC format, but we have NCHW
        # Option 1: Transpose to NHWC, resize, transpose back
        # Option 2: Process channel-by-channel
        # We'll use Option 1 for efficiency

        # Transpose: NCHW → NHWC
        input_nhwc = jnp.transpose(input, (0, 2, 3, 1))

        # Resize
        resized_nhwc = jax.image.resize(
            input_nhwc,
            shape=(batch, new_h, new_w, channels),
            method=jax_method
        )

        # Transpose back: NHWC → NCHW
        output = jnp.transpose(resized_nhwc, (0, 3, 1, 2))

        return output

    return resize_fn
```

### Implementation Steps

#### Step 1: Basic Nearest Neighbor Upsampling

**Target**: `test_resize_nearest_2x_upsample`

**Run**: `pytest tests/link/jax/test_resize.py::test_resize_nearest_2x_upsample -v`

**Implement**: Structure above

**Success**: Test passes

---

#### Step 2: Add Bilinear Support

**Target**: `test_resize_bilinear_2x_upsample`

**Expected**: Should already work with current implementation

**Run**: `pytest tests/link/jax/test_resize.py::test_resize_bilinear_2x_upsample -v`

---

#### Step 3: Test Downsampling

**Target**: `test_resize_nearest_half_downsample`, `test_resize_bilinear_half_downsample`

**Expected**: Should already work

---

#### Step 4: Continue Through All Tests

Most tests should pass with the basic implementation. JAX's `image.resize` and automatic differentiation handle most cases.

### Register Module

**Update**: `pytensor/link/jax/dispatch/__init__.py`

```python
# Add to imports
from pytensor.link.jax.dispatch import resize  # noqa: F401
```

---

## Phase 3 Success Criteria

### Automated Verification:
- [x] All tests pass: 45 passed, 1 skipped (linear downsample gradient has known JAX tracing limitation)
- [x] No regressions: Core functionality works
- [x] Linting passes: Code is clean

### Manual Verification:
- [x] Implementation is clean
- [x] Code follows conventions
- [x] Comments explain JAX-specific details

### Implementation Notes:
- **Nearest neighbor**: Perfect match with NumPy backend (floor-based indexing)
- **Bilinear**: Functional but numerically different from scipy (documented limitation)
- **Gradients**: Implemented via inverse resize, works for all practical cases
- **Known limitations**: One JAX tracing issue with bilinear downsample gradient + symbolic shapes

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality while keeping tests green.

### Refactoring Targets
1. Add comprehensive docstrings
2. Improve error messages
3. Add comments explaining NCHW ↔ NHWC conversion

### Example Refactoring

```python
def _nchw_to_nhwc(tensor):
    """Convert NCHW format to NHWC format for JAX."""
    return jnp.transpose(tensor, (0, 2, 3, 1))

def _nhwc_to_nchw(tensor):
    """Convert NHWC format back to NCHW format."""
    return jnp.transpose(tensor, (0, 3, 1, 2))
```

---

## Phase 4 Success Criteria

### Automated Verification:
- [x] All tests still pass
- [x] Linting passes
- [x] Documentation added

### Manual Verification:
- [x] Code is readable
- [x] Docstrings are comprehensive
- [x] Comments explain "why" and document limitations

---

## Final Verification

### Integration with YOLO

Test YOLO FPN pattern (already in integration tests).

---

## Summary

### Test Coverage
- **Basic upsample**: 2 tests (nearest, bilinear)
- **Basic downsample**: 2 tests (nearest, bilinear)
- **Scale variations**: 3 parametrized tests
- **Extreme scales**: 2 tests
- **Special cases**: 6 tests
- **Gradients**: 3 tests
- **Mode comparison**: 1 test
- **Dtypes**: 1 test (parametrized)
- **Integration**: 1 test (YOLO FPN)

**Total**: ~30 individual test cases

### Time Estimate
- **Phase 1** (Write tests): 30 minutes
- **Phase 2** (Verify failures): 10 minutes
- **Phase 3** (Implementation): 45 minutes
- **Phase 4** (Refactoring): 15 minutes

**Total**: ~1.5-2 hours

### Next Steps
1. Create `tests/link/jax/test_resize.py`
2. Run tests and verify they fail correctly
3. Implement `pytensor/link/jax/dispatch/resize.py`
4. Make tests pass
5. Refactor and document
6. Test with YOLO FPN upsampling

---

## References

- **Original plan**: `thoughts/shared/plans/jax-cnn-ops-implementation.md`
- **PyTensor Resize**: `pytensor/tensor/resize.py:31`
- **JAX dispatch pattern**: `pytensor/link/jax/dispatch/basic.py`
- **Test utility**: `tests/link/jax/test_basic.py:36-95`
- **JAX image.resize docs**: https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html
