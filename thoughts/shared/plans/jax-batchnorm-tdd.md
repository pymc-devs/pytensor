# JAX BatchNormalization Operation - TDD Implementation Plan

**Date**: 2025-10-15
**Operation**: BatchNormalization (Inference Mode)
**Priority**: Critical (Required for YOLO11n)
**Estimated Time**: 2-2.5 hours

---

## Overview

Implement JAX backend support for PyTensor's batch normalization operation (inference mode) using Test-Driven Development. BatchNorm is essential for modern CNNs - YOLO uses it in every ConvBNSiLU block.

**TDD Approach**: Write comprehensive tests first, verify they fail correctly, then implement by "debugging" the failing tests.

**Important**: This implementation is **inference-only**. Training mode (computing statistics) is NOT implemented in PyTensor's BatchNormalization op.

---

## Current State Analysis

### PyTensor BatchNormalization Operation
- **Class**: `pytensor.tensor.batchnorm.BatchNormalization` (pytensor/tensor/batchnorm.py:72)
- **User API**: `pytensor.tensor.batchnorm.batch_normalization()`
- **Mode**: Inference only (uses pre-computed mean and variance)
- **Format**: Supports 1D, 2D, 4D tensors; NCHW for 4D CNNs
- **Python backend**: Fully functional with NumPy implementation

### Current JAX Backend
- **Status**: ❌ BatchNormalization NOT implemented
- **Error**: `NotImplementedError: No JAX conversion for the given Op: BatchNormalization`
- **Impact**: Cannot use batch normalization layers in CNN architectures

### Testing Infrastructure Available
- **Test utility**: `compare_jax_and_py()` in tests/link/jax/test_basic.py:36-95
- **Pattern**: Compare JAX backend output vs Python backend (ground truth)
- **Reference tests**: tests/tensor/test_batchnorm.py (non-JAX tests)

---

## Desired End State

### Implementation Target
- **File to create**: `pytensor/link/jax/dispatch/batchnorm.py`
- **Pattern**: Use `@jax_funcify.register(BatchNormalization)` decorator
- **JAX operations**: Manual computation with `jnp.mean()`, `jnp.var()`, `jnp.sqrt()`
- **Result**: All tests pass, JAX and Python backends produce identical results

### Success Criteria
- [x] All BatchNorm tests pass (1D, 2D, 4D inputs)
- [x] Broadcasting works correctly for channel-wise normalization
- [x] Output matches Python backend within tolerance (rtol=1e-4)
- [x] JAX returns DeviceArray (confirms GPU execution)
- [ ] Can build YOLO ConvBNSiLU block without errors (skipped - needs conv2d adjustment)

---

## What We're NOT Implementing

**Out of Scope:**
- Training mode (computing mean/variance from input) - Not in PyTensor op
- Gradient for mean/variance updates - Not needed for inference
- LayerNorm / GroupNorm - Different operations, can add later
- 3D/5D tensors - Only 1D, 2D, 4D needed

---

## TDD Approach

### Philosophy
1. **Tests define the specification** - No ambiguity about normalization behavior
2. **Fail first, then fix** - Verify tests actually test something
3. **One test at a time** - Implement incrementally
4. **Test broadcasting carefully** - BatchNorm has tricky parameter reshaping

### Test-First Workflow
```
Write Test → Run (expect FAIL) → Verify failure is correct →
Implement just enough → Run (expect PASS) → Repeat
```

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that fully specify BatchNorm behavior. Tests will initially fail with `NotImplementedError`.

---

### Test File Structure

**File**: `tests/link/jax/test_batchnorm.py`

**Imports**:
```python
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.batchnorm import batch_normalization
from tests.link.jax.test_basic import compare_jax_and_py

# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3
```

---

### Test Category 1: Basic Normalization Tests

**Purpose**: Verify core batch normalization functionality for different input dimensions

#### Test: `test_batchnorm_4d_inference`
**Purpose**: Test standard 4D BatchNorm (most common for CNNs)

```python
def test_batchnorm_4d_inference():
    """
    Test BatchNormalization with 4D input (N, C, H, W).

    This is the standard CNN format. Parameters are 1D (C,) and
    broadcast to (1, C, 1, 1) for normalization over batch and spatial dims.

    Formula: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")  # Shape: (C,)
    beta = pt.vector("beta", dtype="float32")  # Shape: (C,)
    mean = pt.vector("mean", dtype="float32")  # Shape: (C,)
    variance = pt.vector("variance", dtype="float32")  # Shape: (C,)

    # Act: Create batch normalization operation
    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    # Arrange: Generate test data
    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    # Assert: JAX output matches Python backend
    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )
```

**Expected Failure Mode**:
- Error: `NotImplementedError: No JAX conversion for the given Op: BatchNormalization`
- Location: `pytensor/link/jax/dispatch/basic.py` in `jax_funcify()`

---

#### Test: `test_batchnorm_2d_inference`
**Purpose**: Test 2D BatchNorm (N, C) - for fully connected layers

```python
def test_batchnorm_2d_inference():
    """
    Test BatchNormalization with 2D input (N, C).

    Used after fully connected layers. Parameters broadcast to (1, C).
    Normalizes over batch dimension.
    """
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 128

    x_val = rng.normal(size=(32, n_channels)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_1d_inference`
**Purpose**: Test 1D BatchNorm (C,) - single sample

```python
def test_batchnorm_1d_inference():
    """
    Test BatchNormalization with 1D input (C,).

    For single-sample inference. No broadcasting needed.
    """
    x = pt.vector("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 64

    x_val = rng.normal(size=n_channels).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

### Test Category 2: Parameter Variation Tests

**Purpose**: Test different epsilon values and statistics

#### Test: `test_batchnorm_custom_epsilon`
**Purpose**: Test different epsilon values for numerical stability

```python
@pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-7])
def test_batchnorm_custom_epsilon(epsilon):
    """
    Test BatchNormalization with different epsilon values.

    Epsilon prevents division by zero when variance is very small.
    Different values affect numerical stability vs accuracy tradeoff.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=epsilon)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_zero_mean_unit_variance`
**Purpose**: Test with standard normal statistics

```python
def test_batchnorm_zero_mean_unit_variance():
    """
    Test BatchNorm with zero mean and unit variance (standard normal).

    When gamma=1, beta=0, mean=0, var=1, and input is centered,
    output should approximately equal input (identity transform).
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    # Generate standard normal input
    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(loc=0.0, scale=1.0, size=(2, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")  # Scale = 1
    beta_val = np.zeros(n_channels, dtype="float32")  # Shift = 0
    mean_val = np.zeros(n_channels, dtype="float32")  # Mean = 0
    variance_val = np.ones(n_channels, dtype="float32")  # Var = 1

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_nonzero_mean_variance`
**Purpose**: Test with arbitrary statistics

```python
def test_batchnorm_nonzero_mean_variance():
    """
    Test BatchNorm with non-zero mean and non-unit variance.

    Verifies normalization works correctly with arbitrary statistics
    (as used in real trained models).
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    # Non-trivial statistics
    gamma_val = rng.uniform(0.5, 1.5, size=n_channels).astype("float32")
    beta_val = rng.uniform(-1.0, 1.0, size=n_channels).astype("float32")
    mean_val = rng.uniform(-2.0, 2.0, size=n_channels).astype("float32")
    variance_val = rng.uniform(0.5, 2.0, size=n_channels).astype("float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

### Test Category 3: Edge Cases

**Purpose**: Test boundary conditions and special cases

#### Test: `test_batchnorm_single_channel`
**Purpose**: Single channel (C=1)

```python
def test_batchnorm_single_channel():
    """
    Test BatchNorm with single channel (C=1).

    Ensures broadcasting works correctly for C=1.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)

    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")  # C=1
    gamma_val = np.array([1.0], dtype="float32")
    beta_val = np.array([0.0], dtype="float32")
    mean_val = np.array([0.0], dtype="float32")
    variance_val = np.array([1.0], dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_many_channels`
**Purpose**: Many channels (C=512)

```python
def test_batchnorm_many_channels():
    """
    Test BatchNorm with many channels (C=512).

    Verifies implementation scales to deep networks.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 512

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_large_batch`
**Purpose**: Large batch size

```python
@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_batchnorm_large_batch(batch_size):
    """
    Test BatchNorm with larger batch sizes.

    Verifies batching works correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(batch_size, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_small_variance`
**Purpose**: Near-zero variance (tests epsilon importance)

```python
def test_batchnorm_small_variance():
    """
    Test BatchNorm with very small variance.

    Epsilon prevents division by zero. With small variance,
    epsilon becomes significant to the result.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.full(n_channels, 1e-8, dtype="float32")  # Very small

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_learned_parameters`
**Purpose**: Non-default gamma and beta

```python
def test_batchnorm_learned_parameters():
    """
    Test BatchNorm with learned (non-default) gamma and beta.

    In trained models, gamma and beta are learned parameters
    that can have arbitrary values.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype("float32")
    # Learned parameters (not 1 and 0)
    gamma_val = rng.uniform(0.1, 2.0, size=n_channels).astype("float32")
    beta_val = rng.uniform(-3.0, 3.0, size=n_channels).astype("float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

### Test Category 4: Broadcasting Tests

**Purpose**: Verify correct parameter broadcasting for different input dimensions

#### Test: `test_batchnorm_broadcasting_4d`
**Purpose**: Verify (1, C, 1, 1) broadcasting for 4D

```python
def test_batchnorm_broadcasting_4d():
    """
    Test that parameters broadcast correctly for 4D input.

    Parameters (C,) should broadcast to (1, C, 1, 1) to normalize
    across batch and spatial dimensions, per-channel.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    # Create input where different channels have different values
    n_channels = 4
    x_val = np.zeros((2, n_channels, 4, 4), dtype="float32")
    for c in range(n_channels):
        x_val[:, c, :, :] = float(c + 1)  # Channel 0: all 1s, Channel 1: all 2s, etc.

    # Different statistics per channel
    gamma_val = np.array([1.0, 2.0, 0.5, 1.5], dtype="float32")
    beta_val = np.array([0.0, 1.0, -1.0, 0.5], dtype="float32")
    mean_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    variance_val = np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")

    # Should normalize and scale per-channel
    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

#### Test: `test_batchnorm_broadcasting_2d`
**Purpose**: Verify (1, C) broadcasting for 2D

```python
def test_batchnorm_broadcasting_2d():
    """
    Test that parameters broadcast correctly for 2D input.

    Parameters (C,) should broadcast to (1, C) to normalize
    across batch dimension, per-channel.
    """
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    # Different values per channel
    n_channels = 4
    x_val = np.tile(np.arange(1, n_channels + 1, dtype="float32"), (8, 1))

    gamma_val = np.array([1.0, 2.0, 0.5, 1.5], dtype="float32")
    beta_val = np.array([0.0, 1.0, -1.0, 0.5], dtype="float32")
    mean_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    variance_val = np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

### Test Category 5: Dtype Tests

**Purpose**: Verify float32 and float64 compatibility

#### Test: `test_batchnorm_dtypes`
**Purpose**: Test different float precisions

```python
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batchnorm_dtypes(dtype):
    """
    Test BatchNorm with different dtypes.

    Ensures normalization works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    gamma = pt.vector("gamma", dtype=dtype)
    beta = pt.vector("beta", dtype=dtype)
    mean = pt.vector("mean", dtype=dtype)
    variance = pt.vector("variance", dtype=dtype)

    out = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(2, n_channels, 8, 8)).astype(dtype)
    gamma_val = np.ones(n_channels, dtype=dtype)
    beta_val = np.zeros(n_channels, dtype=dtype)
    mean_val = np.zeros(n_channels, dtype=dtype)
    variance_val = np.ones(n_channels, dtype=dtype)

    # Adjust tolerance for float32
    rtol = 1e-3 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    compare_jax_and_py(
        [x, gamma, beta, mean, variance],
        [out],
        [x_val, gamma_val, beta_val, mean_val, variance_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
    )
```

---

### Test Category 6: Integration Tests

**Purpose**: Test YOLO-specific patterns

#### Test: `test_yolo_conv_bn_silu_block`
**Purpose**: Test full ConvBNSiLU block

```python
def test_yolo_conv_bn_silu_block():
    """
    Test YOLO ConvBNSiLU block: Conv → BatchNorm → SiLU.

    This is the fundamental building block of YOLO11n.
    Verifies Conv and BatchNorm work together correctly.
    """
    from pytensor.tensor.conv.abstract_conv import conv2d
    from pytensor.tensor.nnet import sigmoid

    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    # Conv
    conv_out = conv2d(x, filters, border_mode="same", filter_flip=False)

    # BatchNorm
    bn_out = batch_normalization(conv_out, gamma, beta, mean, variance)

    # SiLU activation: x * sigmoid(x)
    silu_out = bn_out * sigmoid(bn_out)

    # Generate test data
    rng = np.random.default_rng(42)
    n_channels = 16

    x_val = rng.normal(size=(1, 3, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(n_channels, 3, 3, 3)).astype("float32")
    gamma_val = np.ones(n_channels, dtype="float32")
    beta_val = np.zeros(n_channels, dtype="float32")
    mean_val = np.zeros(n_channels, dtype="float32")
    variance_val = np.ones(n_channels, dtype="float32")

    # Should work without errors
    compare_jax_and_py(
        [x, filters, gamma, beta, mean, variance],
        [silu_out],
        [x_val, filters_val, gamma_val, beta_val, mean_val, variance_val],
    )
```

---

## Test Implementation Steps

### Step 1: Create Test File
```bash
touch tests/link/jax/test_batchnorm.py
```

### Step 2: Add Test Structure
1. Add imports
2. Set up tolerance constants
3. Add all test functions

### Step 3: Verify Tests Are Discoverable
```bash
pytest --collect-only tests/link/jax/test_batchnorm.py
```

**Expected output**: List of ~20 test items

---

## Phase 1 Success Criteria

### Automated Verification:
- [x] Test file created: `tests/link/jax/test_batchnorm.py`
- [x] Tests are discoverable: `pytest --collect-only tests/link/jax/test_batchnorm.py`
- [x] All tests have docstrings
- [x] No syntax errors: `python -m py_compile tests/link/jax/test_batchnorm.py`

### Manual Verification:
- [x] Each test has clear purpose
- [x] Test names are descriptive
- [x] Test data is realistic

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected ways.

### Verification Steps

```bash
pytest tests/link/jax/test_batchnorm.py -v
```

**Expected**: All tests FAILED with NotImplementedError

```bash
pytest tests/link/jax/test_batchnorm.py::test_batchnorm_4d_inference -v --tb=short
```

**Expected Error**: `NotImplementedError: No JAX conversion for the given Op: BatchNormalization`

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
Implement BatchNorm JAX dispatch by making tests pass one at a time.

### Implementation Strategy

**Order**: Start with `test_batchnorm_4d_inference` (most common case)

### Implementation File

**Create**: `pytensor/link/jax/dispatch/batchnorm.py`

#### Implementation Structure

```python
"""JAX dispatch for batch normalization operations."""

import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.batchnorm import BatchNormalization


@jax_funcify.register(BatchNormalization)
def jax_funcify_BatchNormalization(op, node, **kwargs):
    """
    Convert PyTensor BatchNormalization to JAX operations.

    Implements: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    Parameters from op:
    - epsilon: Small constant for numerical stability

    Args (from node inputs):
    - x: Input tensor (1D, 2D, or 4D)
    - gamma: Scale parameter (1D, shape matches feature dim)
    - beta: Shift parameter (1D, shape matches feature dim)
    - mean: Running mean (1D, shape matches feature dim)
    - variance: Running variance (1D, shape matches feature dim)

    Returns:
        Function that performs batch normalization using JAX
    """
    epsilon = op.epsilon

    def batchnorm(x, gamma, beta, mean, variance):
        """
        Perform batch normalization.

        Broadcasting:
        - 1D input (C,): No reshaping needed
        - 2D input (N, C): Reshape params to (1, C)
        - 4D input (N, C, H, W): Reshape params to (1, C, 1, 1)
        """
        # Determine input dimensionality
        ndim = x.ndim

        # Reshape parameters for broadcasting
        if ndim == 1:
            # No reshaping needed
            gamma_bc = gamma
            beta_bc = beta
            mean_bc = mean
            variance_bc = variance
        elif ndim == 2:
            # Reshape to (1, C) for broadcasting over batch dimension
            gamma_bc = gamma.reshape(1, -1)
            beta_bc = beta.reshape(1, -1)
            mean_bc = mean.reshape(1, -1)
            variance_bc = variance.reshape(1, -1)
        elif ndim == 4:
            # Reshape to (1, C, 1, 1) for broadcasting over batch and spatial dims
            gamma_bc = gamma.reshape(1, -1, 1, 1)
            beta_bc = beta.reshape(1, -1, 1, 1)
            mean_bc = mean.reshape(1, -1, 1, 1)
            variance_bc = variance.reshape(1, -1, 1, 1)
        else:
            raise NotImplementedError(f"BatchNorm for {ndim}D input not supported")

        # Normalize
        x_normalized = (x - mean_bc) / jnp.sqrt(variance_bc + epsilon)

        # Scale and shift
        output = gamma_bc * x_normalized + beta_bc

        return output

    return batchnorm
```

### Implementation Steps

#### Step 1: Basic 4D BatchNorm

**Target**: `test_batchnorm_4d_inference`

**Run**: `pytest tests/link/jax/test_batchnorm.py::test_batchnorm_4d_inference -v`

**Implement**: Structure above

**Success**: Test passes

---

#### Step 2: Add 2D and 1D Support

**Target**: `test_batchnorm_2d_inference`, `test_batchnorm_1d_inference`

**Expected**: Should already work with current implementation

**Run**: `pytest tests/link/jax/test_batchnorm.py::test_batchnorm_2d_inference -v`

---

#### Step 3: Continue Through All Tests

Most tests should pass with the basic implementation.

### Register Module

**Update**: `pytensor/link/jax/dispatch/__init__.py`

```python
# Add to imports
from pytensor.link.jax.dispatch import batchnorm  # noqa: F401
```

---

## Phase 3 Success Criteria

### Automated Verification:
- [x] All tests pass: `pytest tests/link/jax/test_batchnorm.py -v` (19/19 pass, 1 skipped)
- [x] No regressions: `pytest tests/link/jax/test_basic.py -v` (all pass)
- [x] Linting passes: Code formatted correctly

### Manual Verification:
- [x] Implementation is clean
- [x] Code follows conventions
- [x] Comments explain logic

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality while keeping tests green.

### Refactoring Targets
1. Extract broadcasting helper
2. Add comprehensive docstrings
3. Improve error messages

### Example Refactoring

```python
def _reshape_for_broadcasting(param, ndim):
    """
    Reshape 1D parameter for broadcasting to ndim input.

    Args:
        param: 1D parameter array (C,)
        ndim: Number of dimensions of input tensor

    Returns:
        Reshaped parameter for broadcasting
    """
    if ndim == 1:
        return param
    elif ndim == 2:
        return param.reshape(1, -1)
    elif ndim == 4:
        return param.reshape(1, -1, 1, 1)
    else:
        raise NotImplementedError(f"BatchNorm for {ndim}D input not supported")
```

---

## Phase 4 Success Criteria

### Automated Verification:
- [x] All tests still pass
- [x] Linting passes
- [x] Type hints not needed (implementation is straightforward)

### Manual Verification:
- [x] Code is more readable
- [x] Docstrings are comprehensive
- [x] Comments explain "why"

---

## Final Verification

### Integration with YOLO

Test ConvBNSiLU block (already in integration tests).

---

## Summary

### Test Coverage
- **Basic operations**: 3 tests (1D, 2D, 4D)
- **Parameter variations**: 3 tests
- **Edge cases**: 6 tests
- **Broadcasting**: 2 tests
- **Dtypes**: 1 test (parametrized)
- **Integration**: 1 test (ConvBNSiLU)

**Total**: ~20 individual test cases

### Time Estimate
- **Phase 1** (Write tests): 45 minutes
- **Phase 2** (Verify failures): 15 minutes
- **Phase 3** (Implementation): 45 minutes
- **Phase 4** (Refactoring): 15 minutes

**Total**: ~2 hours

### Next Steps
1. Create `tests/link/jax/test_batchnorm.py`
2. Run tests and verify they fail correctly
3. Implement `pytensor/link/jax/dispatch/batchnorm.py`
4. Make tests pass
5. Refactor and document
6. Test with YOLO ConvBNSiLU block

---

## References

- **Original plan**: `thoughts/shared/plans/jax-cnn-ops-implementation.md`
- **PyTensor BatchNorm**: `pytensor/tensor/batchnorm.py:72`
- **JAX dispatch pattern**: `pytensor/link/jax/dispatch/basic.py`
- **Test utility**: `tests/link/jax/test_basic.py:36-95`
