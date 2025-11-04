# ONNX Conv2D Converter - TDD Implementation Plan

<!-- WORKSHOP NOTE: This is the MOST SUCCESSFUL plan document - it represents best practices for AI-first development through Test-Driven Development.

Why this plan stands out:
1. **Extremely detailed** - Every test is specified with exact code, test data, expected failures
2. **TDD methodology** - Write tests first, watch them fail, then implement
3. **Critical insight**: Uses ASYMMETRIC kernels (Sobel) to catch filter-flipping bugs that symmetric kernels hide
4. **Phase-based execution** - Clear progression from simple to complex
5. **Workshop gold**: Shows how planning can anticipate bugs before writing code

Execution results:
- All 21 tests written and passing ✅ (even more than planned!)
- Filter flipping bug was indeed critical - caught by asymmetric test ✅
- TDD approach caught 3 bugs during implementation that would have been missed otherwise ✅
- Implementation took ~6 hours (faster than estimated!) ✅

Key learning: THIS is how you write an AI-first plan for complex features:
- Specify tests completely (don't leave them vague)
- Anticipate failure modes
- Use domain knowledge (asymmetric kernels) to design better tests
- Break into phases that can each be verified

This plan was executed ALMOST exactly as written. Minor adjustments:
- Added a few extra edge case tests (empty tensors, unusual shapes)
- Skipped "full" padding mode (not needed for real models)
- Refactoring phase was lighter (code was already clean from TDD) -->

## Overview

Implement ONNX export support for PyTensor's 2D convolution operations (`AbstractConv2d`) following a strict Test-Driven Development approach. This enables exporting convolutional neural networks from PyTensor to ONNX format for deployment to browsers (WebAssembly/WebGPU), mobile devices, and edge hardware.

**Approach**: Write comprehensive tests first, verify they fail diagnostically, then implement features by making tests pass one at a time.

<!-- REALITY CHECK: This TDD approach WORKED. The tests really were written first, they really did fail with NotImplementedError, and implementation really did proceed by making them pass one-by-one. This isn't aspirational - this is how it actually happened. The discipline of TDD prevented bugs and made debugging trivial (test tells you exactly what's broken). -->

## Current State Analysis

### What Exists Now

**ONNX Backend Infrastructure** (✅ Working):
- Core dispatcher: `pytensor/link/onnx/dispatch/basic.py:29-70` - `@onnx_funcify.register()` pattern
- FunctionGraph converter: `pytensor/link/onnx/dispatch/basic.py:152-291`
- Test helper: `tests/link/onnx/test_basic.py:18-101` - `compare_onnx_and_py()` utility
- Element-wise ops: `pytensor/link/onnx/dispatch/elemwise.py` (Add, Mul, Exp, etc.)
- Matrix ops: `pytensor/link/onnx/dispatch/nlinalg.py` (Dot, MatMul)
- Activations: `pytensor/link/onnx/dispatch/special.py` (Softmax, ReLU via Maximum)
- Shape ops: `pytensor/link/onnx/dispatch/shape.py` (Reshape, DimShuffle, Flatten)

**PyTensor Conv2D Operations** (✅ Available):
- `AbstractConv2d` class: `pytensor/tensor/conv/abstract_conv.py:2654`
- `conv2d()` function: `pytensor/tensor/conv/abstract_conv.py:3514`
- Parameters: `border_mode`, `subsample`, `filter_flip`, `filter_dilation`, `num_groups`

### Current Testing Landscape

**Testing Framework**: pytest
**Test Pattern**: `compare_onnx_and_py([inputs], output, [test_values], tmp_path=tmp_path)`
**Available Test Utilities**:
- `tests/link/onnx/test_basic.py:18-101` - Core comparison helper
- `pytest.fixture` for `tmp_path` - Temporary directory for ONNX files
- `np.testing.assert_allclose` with `rtol=1e-4` - Default tolerance
- `onnx.checker.check_model()` - Model validation
- ONNX Runtime execution - Runtime verification

**Existing Test Patterns to Follow**:
- Simple ops: `tests/link/onnx/test_elemwise.py:20-29` (Add)
- Complex ops: `tests/link/onnx/test_nlinalg.py:58-72` (Linear layer)
- Parameterized: `tests/link/onnx/test_elemwise.py:130-148` (Different shapes)
- Multi-node: `tests/link/onnx/test_special.py:78-112` (2-layer network)

## Desired End State

After implementation, PyTensor users can export CNNs to ONNX:

```python
import pytensor.tensor as pt
from pytensor.tensor.nnet import conv2d
from pytensor.link.onnx import export_onnx

# Define CNN layer
x = pt.tensor4('x', dtype='float32')
kernel = shared(np.random.randn(32, 3, 3, 3).astype('float32'))
y = conv2d(x, kernel, border_mode='valid')

# Export to ONNX
f = pytensor.function([x], y)
export_onnx(f, 'cnn_model.onnx')

# Run in ONNX Runtime (browser, mobile, edge)
session = ort.InferenceSession('cnn_model.onnx')
result = session.run(None, {'x': input_data})
```

### Success Criteria

**Functional Requirements**:
- ✅ Conv2D with all padding modes (valid, same, explicit)
- ✅ Strided convolutions (subsample parameter)
- ✅ Dilated/atrous convolutions (filter_dilation)
- ✅ Grouped/depthwise convolutions (num_groups)
- ✅ **Filter flipping handled correctly** (most critical!)
- ✅ Multi-channel inputs and outputs
- ✅ Batch processing

**Quality Requirements**:
- 100% test pass rate
- Numerical accuracy: rtol=1e-4 vs PyTensor
- ONNX schema validation passes
- Clear error messages for unsupported features

## What We're NOT Testing/Implementing

**Explicitly out of scope**:
- ❌ Gradient operations (Conv2d_gradWeights, Conv2d_gradInputs) - training only
- ❌ 3D convolutions (AbstractConv3d) - separate feature
- ❌ 1D convolutions - separate feature
- ❌ Transposed/deconvolution operations
- ❌ Unshared convolutions (locally connected)
- ❌ Bias fusion optimization (Phase 2 feature)
- ❌ Graph optimizations (constant folding, etc.)

## TDD Approach

### Test Design Philosophy

**1. Tests Define Specification**
- Each test completely specifies expected behavior
- Test names clearly describe what they validate
- Docstrings explain "why" this test matters

**2. Fail Fast, Fail Clear**
- Tests fail with diagnostic error messages
- Failure points to exact location of missing feature
- Error types match expectations (NotImplementedError initially)

**3. Incremental Implementation**
- Start with simplest case (valid padding, no flip)
- Add complexity one parameter at a time
- Keep all previous tests passing

**4. Asymmetric Kernels for Flip Detection**
- Use Sobel/Prewitt edge detectors (asymmetric)
- Symmetric kernels hide flip bugs!
- This is THE critical test for correctness

---

## Phase 1: Test Design & Implementation

### Overview

Write comprehensive tests that define Conv2D ONNX export behavior. These tests will initially fail with `NotImplementedError` because the converter doesn't exist yet.

### Test File Structure

**File**: `tests/link/onnx/test_conv.py` (new file)

**Imports**:
```python
"""Tests for ONNX convolution operations."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from pytensor.tensor.nnet import conv2d

from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")
```

---

### Test Category 1: Basic Operation Tests

**Purpose**: Verify simple 2D convolution works end-to-end

#### Test 1.1: `test_conv2d_valid_single_channel`

**What it validates**: Most basic convolution - single channel, valid padding, no special parameters

**Test Data**:
- Input: (1, 1, 5, 5) - batch=1, channels=1, 5x5 spatial
- Kernel: (1, 1, 3, 3) - 1 filter, 1 input channel, 3x3 kernel
- Expected output: (1, 1, 3, 3) - valid padding reduces size

**Expected Behavior**: Convolution computes correctly, ONNX output matches PyTensor

**Test Code**:
```python
def test_conv2d_valid_single_channel(tmp_path):
    """
    Test basic 2D convolution with valid padding and single channel.

    This is the simplest convolution case - verifies:
    - Conv2D op is recognized and converted
    - Basic ONNX Conv node is created
    - Output shape is calculated correctly
    - Numerical results match PyTensor

    Configuration:
    - border_mode='valid' (no padding)
    - subsample=(1,1) (no stride)
    - filter_flip=False (cross-correlation, matches ONNX)
    - filter_dilation=(1,1) (no dilation)
    - num_groups=1 (standard convolution)
    """
    # Arrange: Create symbolic inputs
    x = pt.tensor4("x", dtype="float32")  # (batch, channels, height, width)
    kernel = pt.tensor4("kernel", dtype="float32")  # (filters, in_channels, kh, kw)

    # Define convolution operation
    y = conv2d(
        x, kernel,
        border_mode="valid",
        subsample=(1, 1),
        filter_flip=False,  # CRITICAL: Use cross-correlation to match ONNX
        filter_dilation=(1, 1),
        num_groups=1,
    )

    # Test data: Simple values for manual verification
    x_val = np.array([
        [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
          [21, 22, 23, 24, 25]]]
    ], dtype="float32")

    kernel_val = np.array([
        [[[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]]]
    ], dtype="float32")

    # Act & Assert: Compare ONNX Runtime output with PyTensor
    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- **Error type**: `NotImplementedError`
- **Error message**: `No ONNX conversion available for: AbstractConv2d`
- **Location**: Raised by `onnx_funcify` dispatcher (basic.py:57-70)

**Why this test matters**: If this fails, nothing else will work. This is the foundation test.

---

#### Test 1.2: `test_conv2d_output_shape`

**What it validates**: Output shape calculation is correct

**Test Data**: Various input/kernel sizes to verify shape math

**Test Code**:
```python
@pytest.mark.parametrize(
    "input_shape,kernel_shape,expected_output_shape",
    [
        ((1, 1, 5, 5), (1, 1, 3, 3), (1, 1, 3, 3)),  # Valid padding
        ((1, 1, 10, 10), (1, 1, 5, 5), (1, 1, 6, 6)),  # Larger input
        ((2, 1, 7, 7), (3, 1, 3, 3), (2, 3, 5, 5)),  # Batch + multiple filters
    ],
)
def test_conv2d_output_shape(tmp_path, input_shape, kernel_shape, expected_output_shape):
    """
    Test that Conv2D output shapes are calculated correctly.

    Output shape formula (valid padding):
    output_h = (input_h - kernel_h) + 1
    output_w = (input_w - kernel_w) + 1

    This test verifies ONNX Conv respects shape semantics.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random(input_shape).astype("float32")
    kernel_val = rng.random(kernel_shape).astype("float32")

    # Compare outputs
    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Verify output shape
    assert onnx_res[0].shape == expected_output_shape, \
        f"Expected shape {expected_output_shape}, got {onnx_res[0].shape}"
```

**Expected Failure Mode**: Same as Test 1.1 - converter doesn't exist yet

---

### Test Category 2: CRITICAL - Filter Flipping Tests

**Purpose**: Verify the critical `filter_flip` parameter is handled correctly

**⚠️ MOST IMPORTANT TESTS**: These catch the subtle convolution vs cross-correlation bug!

#### Test 2.1: `test_conv2d_filter_flip_false`

**What it validates**: Cross-correlation mode (filter_flip=False) works correctly

**Test Code**:
```python
def test_conv2d_filter_flip_false(tmp_path):
    """
    Test Conv2D with filter_flip=False (cross-correlation).

    When filter_flip=False:
    - PyTensor performs cross-correlation (no kernel flip)
    - ONNX Conv also performs cross-correlation (no flip)
    - Direct mapping should work correctly

    This is the simpler case and should work immediately.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError (converter doesn't exist)

---

#### Test 2.2: `test_conv2d_filter_flip_true_symmetric`

**What it validates**: True convolution with symmetric kernel (flipping doesn't matter)

**Test Code**:
```python
def test_conv2d_filter_flip_true_symmetric(tmp_path):
    """
    Test Conv2D with filter_flip=True and symmetric kernel.

    When kernel is symmetric (e.g., Gaussian blur), flipping doesn't change result.
    This test ensures filter_flip=True is recognized, even if flip is no-op.

    Note: This test will PASS even if flip logic is broken (symmetric kernel)!
    See test_conv2d_filter_flip_true_asymmetric for the critical test.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    # Symmetric Gaussian-like kernel
    kernel_val = np.array([
        [[[1, 2, 1],
          [2, 4, 2],
          [1, 2, 1]]]
    ], dtype="float32") / 16.0  # Normalized

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError for filter_flip=True support

---

#### Test 2.3: `test_conv2d_filter_flip_true_asymmetric` ⭐⭐⭐

**What it validates**: True convolution with ASYMMETRIC kernel - **THE CRITICAL TEST**

**Why this is critical**:
- Symmetric kernels hide flip bugs (same result flipped or not)
- Asymmetric kernels (Sobel, Prewitt) REQUIRE correct flipping
- This test will FAIL if flip logic is wrong, even if others pass

**Test Code**:
```python
def test_conv2d_filter_flip_true_asymmetric(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Conv2D with filter_flip=True and ASYMMETRIC kernel.

    This is THE most important test for Conv2D correctness!

    When filter_flip=True:
    - PyTensor flips kernel (mathematical convolution)
    - ONNX Conv does NOT flip (cross-correlation)
    - We MUST flip the kernel before passing to ONNX

    Using Sobel edge detector (asymmetric):
    - If we DON'T flip: Wrong results (detects edges in wrong direction)
    - If we DO flip correctly: Results match PyTensor

    Failure modes:
    - Test passes with symmetric kernel but fails here: Flip not implemented!
    - Results don't match: Flip implemented incorrectly
    - Error: Flip not supported yet (acceptable for Phase 1)

    References:
    - Gap analysis: lines 736-767 (filter flipping explanation)
    - ONNX Conv docs: Uses cross-correlation, not convolution
    - PyTensor filter_flip: Lines 2109-2114 in abstract_conv.py
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    # Sobel X edge detector (ASYMMETRIC!)
    # Detects vertical edges (left-to-right transitions)
    sobel_x = np.array([
        [[[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]]]
    ], dtype="float32")

    # Test image with vertical edge
    # Left side: bright (1.0), right side: dark (0.0)
    x_val = np.array([
        [[[1.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0]]]
    ], dtype="float32")

    # Expected: Strong response at the edge (column index 1-2)
    # If flip is wrong: Response will be inverted or at wrong location

    compare_onnx_and_py([x, kernel], y, [x_val, sobel_x], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- **Phase 1**: NotImplementedError with message "filter_flip=True requires kernel flipping, not yet implemented"
- **Phase 2**: Test should pass after implementing flip logic

**Debugging Strategy When Implementing**:
1. Run test: `pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -vv`
2. Read failure message carefully
3. Check if error or wrong result
4. If wrong result: Flip implementation is buggy
5. Print intermediate values to debug

---

### Test Category 3: Padding Mode Tests

**Purpose**: Verify all padding modes map correctly to ONNX

#### Test 3.1: `test_conv2d_valid_padding`

**What it validates**: border_mode='valid' (no padding) works

**Test Code**:
```python
def test_conv2d_valid_padding(tmp_path):
    """
    Test Conv2D with 'valid' padding (no padding).

    Valid padding:
    - PyTensor: border_mode='valid'
    - ONNX: auto_pad='VALID' or pads=[0,0,0,0]
    - Output size: (input_size - kernel_size) + 1

    This is the default and simplest padding mode.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Verify output shape: (8-3)+1 = 6
    assert onnx_res[0].shape == (1, 1, 6, 6)
```

**Expected Failure Mode**: NotImplementedError (converter doesn't exist)

---

#### Test 3.2: `test_conv2d_same_padding`

**What it validates**: border_mode='same' maintains input size (with stride=1)

**Test Code**:
```python
def test_conv2d_same_padding(tmp_path):
    """
    Test Conv2D with 'same' padding.

    Same padding:
    - PyTensor: border_mode='same' (or 'half')
    - ONNX: auto_pad='SAME_UPPER'
    - Output size: same as input (when stride=1)

    Padding amount: floor(kernel_size / 2) on each side
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="same", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Verify output shape: same as input
    assert onnx_res[0].shape == (1, 1, 8, 8)
```

**Expected Failure Mode**: NotImplementedError initially, then may fail if padding mapping is wrong

---

#### Test 3.3: `test_conv2d_explicit_symmetric_padding`

**What it validates**: Explicit symmetric padding (pad_h, pad_w) works

**Test Code**:
```python
def test_conv2d_explicit_symmetric_padding(tmp_path):
    """
    Test Conv2D with explicit symmetric padding.

    Symmetric padding:
    - PyTensor: border_mode=(pad_h, pad_w)
    - ONNX: pads=[pad_h, pad_w, pad_h, pad_w]
    - Same padding on all sides

    Example: (1, 1) adds 1 pixel padding on all 4 sides
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Add 1 pixel padding on each side
    y = conv2d(x, kernel, border_mode=(1, 1), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Output size: (5 + 2*1 - 3) + 1 = 5 (same as input)
    assert onnx_res[0].shape == (1, 1, 5, 5)
```

**Expected Failure Mode**: NotImplementedError, then potential padding calculation bugs

---

#### Test 3.4: `test_conv2d_explicit_asymmetric_padding`

**What it validates**: Asymmetric padding ((top,bottom), (left,right)) works

**Test Code**:
```python
def test_conv2d_explicit_asymmetric_padding(tmp_path):
    """
    Test Conv2D with explicit asymmetric padding.

    Asymmetric padding:
    - PyTensor: border_mode=((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
    - ONNX: pads=[pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]
    - Different padding on each side

    Example: ((1,2), (0,1)) adds:
    - 1 pixel top, 2 pixels bottom
    - 0 pixels left, 1 pixel right
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Asymmetric padding
    y = conv2d(x, kernel, border_mode=((1, 2), (0, 1)), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Output size:
    # height: (5 + 1 + 2 - 3) + 1 = 6
    # width: (5 + 0 + 1 - 3) + 1 = 4
    assert onnx_res[0].shape == (1, 1, 6, 4)
```

**Expected Failure Mode**: NotImplementedError, then padding calculation bugs

---

### Test Category 4: Stride Tests (subsample)

**Purpose**: Verify strided convolutions (downsampling) work

#### Test 4.1: `test_conv2d_stride_2x2`

**What it validates**: Strided convolution downsamples correctly

**Test Code**:
```python
def test_conv2d_stride_2x2(tmp_path):
    """
    Test Conv2D with stride 2x2 (downsampling).

    Strided convolution:
    - PyTensor: subsample=(stride_h, stride_w)
    - ONNX: strides=[stride_h, stride_w]
    - Output size: floor((input_size - kernel_size) / stride) + 1

    Common in CNNs for downsampling instead of pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", subsample=(2, 2), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Output size: floor((8-3)/2) + 1 = 3
    assert onnx_res[0].shape == (1, 1, 3, 3)
```

**Expected Failure Mode**: NotImplementedError, then stride mapping bugs

---

#### Test 4.2: `test_conv2d_asymmetric_stride`

**What it validates**: Different strides for height and width

**Test Code**:
```python
def test_conv2d_asymmetric_stride(tmp_path):
    """
    Test Conv2D with asymmetric stride (stride_h != stride_w).

    Asymmetric stride:
    - PyTensor: subsample=(2, 1)
    - ONNX: strides=[2, 1]
    - Different downsampling factors for H and W

    Less common but valid configuration.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", subsample=(2, 1), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 10, 10)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Output size: (floor((10-3)/2)+1, floor((10-3)/1)+1) = (4, 8)
    assert onnx_res[0].shape == (1, 1, 4, 8)
```

---

### Test Category 5: Dilation Tests (Atrous Convolution)

**Purpose**: Verify dilated convolutions (expanded receptive field) work

#### Test 5.1: `test_conv2d_dilation_2x2`

**What it validates**: Dilated convolution expands receptive field

**Test Code**:
```python
def test_conv2d_dilation_2x2(tmp_path):
    """
    Test Conv2D with dilation 2x2 (atrous convolution).

    Dilated convolution:
    - PyTensor: filter_dilation=(dilation_h, dilation_w)
    - ONNX: dilations=[dilation_h, dilation_w]
    - Expands receptive field without increasing parameters
    - Effective kernel size: kernel_size + (kernel_size - 1) * (dilation - 1)

    Example: 3x3 kernel with dilation=2 has effective size 5x5
    Common in semantic segmentation (DeepLab, etc.)
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_dilation=(2, 2), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 10, 10)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

    # Effective kernel: 3 + (3-1)*1 = 5
    # Output size: (10-5)+1 = 6
    assert onnx_res[0].shape == (1, 1, 6, 6)
```

**Expected Failure Mode**: NotImplementedError, then dilation mapping bugs

---

### Test Category 6: Grouped Convolution Tests

**Purpose**: Verify grouped and depthwise convolutions work

#### Test 6.1: `test_conv2d_grouped_convolution`

**What it validates**: Grouped convolution (num_groups > 1)

**Test Code**:
```python
def test_conv2d_grouped_convolution(tmp_path):
    """
    Test Conv2D with grouped convolution.

    Grouped convolution:
    - PyTensor: num_groups=2 (or other value)
    - ONNX: group=2
    - Divides input/output channels into groups
    - Each group processes independently
    - Reduces parameters and computation

    Example: 4 input channels, 8 output channels, 2 groups
    - Group 1: channels 0-1 → filters 0-3
    - Group 2: channels 2-3 → filters 4-7

    Common in efficient architectures (ResNeXt, ShuffleNet).
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", num_groups=2, filter_flip=False)

    rng = np.random.default_rng(42)
    # 4 input channels, 8 output filters, 2 groups
    x_val = rng.random((1, 4, 8, 8)).astype("float32")
    kernel_val = rng.random((8, 2, 3, 3)).astype("float32")  # 8 filters, 2 channels per group

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError, then group mapping bugs

---

#### Test 6.2: `test_conv2d_depthwise_convolution`

**What it validates**: Depthwise convolution (num_groups = num_channels)

**Test Code**:
```python
def test_conv2d_depthwise_convolution(tmp_path):
    """
    Test Conv2D with depthwise convolution (special case of grouped).

    Depthwise convolution:
    - PyTensor: num_groups = num_input_channels
    - ONNX: group = num_input_channels
    - Each input channel has its own filter
    - Extremely parameter-efficient
    - Common in MobileNet, EfficientNet

    Example: 16 input channels, 16 groups → 1 filter per channel
    Usually followed by 1x1 convolution (pointwise) → "Depthwise Separable"
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    num_channels = 8
    y = conv2d(x, kernel, border_mode="valid", num_groups=num_channels, filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, num_channels, 8, 8)).astype("float32")
    # Depthwise: num_filters = num_channels, channels_per_filter = 1
    kernel_val = rng.random((num_channels, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError, then group mapping bugs

---

### Test Category 7: Multi-Channel Tests

**Purpose**: Verify multi-channel inputs and outputs work correctly

#### Test 7.1: `test_conv2d_rgb_input`

**What it validates**: RGB-like 3-channel input

**Test Code**:
```python
def test_conv2d_rgb_input(tmp_path):
    """
    Test Conv2D with RGB-like 3-channel input.

    Multi-channel input:
    - Common for color images (RGB: 3 channels)
    - Kernel must have matching input channels
    - Each output filter convolves across ALL input channels

    Configuration:
    - Input: (batch, 3, H, W) - RGB image
    - Kernel: (num_filters, 3, kH, kW) - 3 input channels
    - Output: (batch, num_filters, H', W')
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 3, 8, 8)).astype("float32")  # batch=2, RGB
    kernel_val = rng.random((16, 3, 3, 3)).astype("float32")  # 16 filters

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

---

#### Test 7.2: `test_conv2d_batch_processing`

**What it validates**: Batch processing (batch_size > 1)

**Test Code**:
```python
def test_conv2d_batch_processing(tmp_path):
    """
    Test Conv2D with batch processing.

    Batch processing:
    - Multiple samples processed in parallel
    - Batch dimension is independent
    - Common in training (batch_size = 32, 64, etc.)

    Configuration:
    - Input: (batch, channels, H, W)
    - Kernel: (filters, channels, kH, kW)
    - Output: (batch, filters, H', W')

    Each sample in batch is convolved independently with same kernel.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((8, 1, 5, 5)).astype("float32")  # batch=8
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

---

### Test Category 8: Integration Tests

**Purpose**: Test complete CNN patterns (Conv + Activation + etc.)

#### Test 8.1: `test_conv2d_with_bias`

**What it validates**: Convolution followed by bias addition

**Test Code**:
```python
def test_conv2d_with_bias(tmp_path):
    """
    Test Conv2D followed by bias addition.

    Typical CNN layer:
    - Convolution computes weighted sum
    - Bias added to each output channel
    - Pattern: y = conv(x, kernel) + bias

    ONNX Conv can include bias as third input, but PyTensor
    typically does this as separate Add operation.

    This tests that pattern works correctly.
    Future optimization: Fuse bias into Conv node.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")
    bias = pt.vector("bias", dtype="float32")

    # Conv + bias
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    y = conv_out + bias.dimshuffle('x', 0, 'x', 'x')  # Broadcast bias

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((8, 1, 3, 3)).astype("float32")  # 8 filters
    bias_val = rng.random(8).astype("float32")  # 8 biases

    compare_onnx_and_py([x, kernel, bias], y, [x_val, kernel_val, bias_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Conv converter missing, then may work once Conv is implemented (Add already supported)

---

#### Test 8.2: `test_conv2d_relu_pattern`

**What it validates**: Conv → ReLU pattern (common in CNNs)

**Test Code**:
```python
def test_conv2d_relu_pattern(tmp_path):
    """
    Test Conv2D followed by ReLU activation.

    Standard CNN layer pattern:
    - Convolution
    - ReLU activation (non-linearity)
    - Often followed by pooling (when available)

    Configuration: Conv → ReLU

    This tests that Conv integrates with existing activation converters.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Conv + ReLU
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    y = pt.maximum(conv_out, 0)  # ReLU

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((8, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

---

#### Test 8.3: `test_simple_cnn_block`

**What it validates**: Complete CNN block (Conv → ReLU → [Flatten])

**Test Code**:
```python
def test_simple_cnn_block(tmp_path):
    """
    Test a simple CNN block: Conv → ReLU → Flatten.

    This simulates a typical CNN layer:
    1. Convolution extracts features
    2. ReLU adds non-linearity
    3. Flatten prepares for dense layer

    Integration test ensuring Conv works with rest of pipeline.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # CNN block
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    relu_out = pt.maximum(conv_out, 0)
    y = relu_out.flatten(2)  # Flatten spatial dimensions

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 1, 5, 5)).astype("float32")  # batch=2
    kernel_val = rng.random((4, 1, 3, 3)).astype("float32")  # 4 filters

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
```

---

### Test Implementation Steps

**Step 1: Create test file**
```bash
touch tests/link/onnx/test_conv.py
```

**Step 2: Write test file header and imports** (shown above)

**Step 3: Implement all test functions** (copy from test templates above)

**Step 4: Count test cases**
```bash
pytest tests/link/onnx/test_conv.py --collect-only
```

Expected: ~20 test cases

---

### Success Criteria

#### Automated Verification:
- [ ] Test file exists: `tests/link/onnx/test_conv.py`
- [ ] All tests are discovered: `pytest --collect-only tests/link/onnx/test_conv.py`
- [ ] Tests use `compare_onnx_and_py` helper correctly
- [ ] Test code follows project conventions: Passes `ruff check tests/link/onnx/test_conv.py`
- [ ] Each test has clear docstring explaining what it validates

#### Manual Verification:
- [ ] Test names clearly describe what they test
- [ ] Test data is appropriate (hardcoded for simple, random for complex)
- [ ] Asymmetric kernel test uses Sobel/Prewitt (not symmetric)
- [ ] Tests cover all major Conv2D parameters
- [ ] Tests are organized by category with clear comments

---

## Phase 2: Test Failure Verification

### Overview

Run the test suite and verify ALL tests fail in the expected, diagnostic way. This proves our tests actually test something and will catch regressions.

### Verification Steps

**Step 1: Run the full test suite**
```bash
cd C:\Users\armor\OneDrive\Desktop\cs\pytensor
pytest tests/link/onnx/test_conv.py -v
```

**Expected Output**:
```
tests/link/onnx/test_conv.py::test_conv2d_valid_single_channel FAILED
tests/link/onnx/test_conv.py::test_conv2d_output_shape FAILED
tests/link/onnx/test_conv.py::test_conv2d_filter_flip_false FAILED
...
=================== 20 failed in 2.34s ===================
```

**Step 2: Examine failure messages**

Run with more detail:
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_valid_single_channel -vv --tb=short
```

**Expected Failure Pattern**:
```
_________________________ test_conv2d_valid_single_channel __________________________

    def test_conv2d_valid_single_channel(tmp_path):
>       compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)

tests/link/onnx/test_conv.py:XX:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/link/onnx/test_basic.py:XX: in compare_onnx_and_py
    model = export_onnx(pytensor_fn, onnx_path)
pytensor/link/onnx/export.py:XX: in export_onnx
    model = onnx_funcify(fgraph, ...)
pytensor/link/onnx/dispatch/basic.py:XX: in onnx_funcify
    onnx_node = onnx_funcify(node.op, node=node, ...)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    @singledispatch
    def onnx_funcify(op, node=None, **kwargs):
>       raise NotImplementedError(
            f"No ONNX conversion available for: AbstractConv2d\n"
            ...
        )
E   NotImplementedError: No ONNX conversion available for: AbstractConv2d
```

**Key checks**:
- ✅ Error is `NotImplementedError`
- ✅ Error mentions `AbstractConv2d`
- ✅ Error message is clear and helpful
- ✅ Stack trace shows it's coming from dispatcher
- ✅ Not a syntax error or import error

**Step 3: Verify each test fails correctly**

Create a checklist:

```bash
# Save test results to file
pytest tests/link/onnx/test_conv.py --tb=line > test_failures.txt 2>&1
```

**Review checklist**:
- [ ] test_conv2d_valid_single_channel - NotImplementedError ✅
- [ ] test_conv2d_output_shape - NotImplementedError ✅
- [ ] test_conv2d_filter_flip_false - NotImplementedError ✅
- [ ] test_conv2d_filter_flip_true_symmetric - NotImplementedError ✅
- [ ] test_conv2d_filter_flip_true_asymmetric - NotImplementedError ✅
- [ ] test_conv2d_valid_padding - NotImplementedError ✅
- [ ] test_conv2d_same_padding - NotImplementedError ✅
- [ ] test_conv2d_explicit_symmetric_padding - NotImplementedError ✅
- [ ] test_conv2d_explicit_asymmetric_padding - NotImplementedError ✅
- [ ] test_conv2d_stride_2x2 - NotImplementedError ✅
- [ ] test_conv2d_asymmetric_stride - NotImplementedError ✅
- [ ] test_conv2d_dilation_2x2 - NotImplementedError ✅
- [ ] test_conv2d_grouped_convolution - NotImplementedError ✅
- [ ] test_conv2d_depthwise_convolution - NotImplementedError ✅
- [ ] test_conv2d_rgb_input - NotImplementedError ✅
- [ ] test_conv2d_batch_processing - NotImplementedError ✅
- [ ] test_conv2d_with_bias - NotImplementedError ✅
- [ ] test_conv2d_relu_pattern - NotImplementedError ✅
- [ ] test_simple_cnn_block - NotImplementedError ✅

**Step 4: Check failure diagnostics**

For critical test, check error message quality:
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -vv
```

**Verify error message includes**:
- ✅ "No ONNX conversion available for: AbstractConv2d"
- ✅ List of currently supported ops
- ✅ Suggestion for how to add support
- ✅ Clear indication this is expected (not a bug)

---

### Expected Failures Document

Create a reference document for expected failures:

**File**: `tests/link/onnx/CONV2D_TEST_FAILURES.md` (temporary, delete after Phase 3)

```markdown
# Expected Test Failures (Before Implementation)

All tests in test_conv.py should fail with:
- **Error type**: NotImplementedError
- **Error message**: "No ONNX conversion available for: AbstractConv2d"
- **Raised from**: pytensor/link/onnx/dispatch/basic.py (onnx_funcify singledispatch)

## Test Count: 19 tests

### Category 1: Basic (2 tests)
- test_conv2d_valid_single_channel ❌
- test_conv2d_output_shape ❌

### Category 2: Filter Flipping (3 tests)
- test_conv2d_filter_flip_false ❌
- test_conv2d_filter_flip_true_symmetric ❌
- test_conv2d_filter_flip_true_asymmetric ❌ (CRITICAL)

### Category 3: Padding (4 tests)
- test_conv2d_valid_padding ❌
- test_conv2d_same_padding ❌
- test_conv2d_explicit_symmetric_padding ❌
- test_conv2d_explicit_asymmetric_padding ❌

### Category 4: Stride (2 tests)
- test_conv2d_stride_2x2 ❌
- test_conv2d_asymmetric_stride ❌

### Category 5: Dilation (1 test)
- test_conv2d_dilation_2x2 ❌

### Category 6: Grouped (2 tests)
- test_conv2d_grouped_convolution ❌
- test_conv2d_depthwise_convolution ❌

### Category 7: Multi-Channel (2 tests)
- test_conv2d_rgb_input ❌
- test_conv2d_batch_processing ❌

### Category 8: Integration (3 tests)
- test_conv2d_with_bias ❌
- test_conv2d_relu_pattern ❌
- test_simple_cnn_block ❌

## After Phase 3 Implementation

Expected progression:
1. Basic tests pass first (valid padding, no flip)
2. Padding tests pass (border_mode mapping)
3. Stride/dilation tests pass (attribute mapping)
4. Grouped convolution tests pass (group parameter)
5. Filter flipping tests LAST (most complex)

Critical milestone: test_conv2d_filter_flip_true_asymmetric passes
```

---

### Adjustment Phase

**If tests don't fail as expected**, fix them:

#### Problem 1: Test passes unexpectedly
**Symptom**: Green checkmark when it should fail
**Cause**: Test is too lenient or testing wrong thing
**Fix**: Tighten assertions, verify test actually exercises Conv2D

#### Problem 2: Wrong error type
**Symptom**: ImportError, AttributeError, etc. instead of NotImplementedError
**Cause**: Missing imports, typos, wrong op class
**Fix**: Check imports, verify op names, fix typos

#### Problem 3: Cryptic error message
**Symptom**: Error doesn't explain what's missing
**Cause**: Poor error handling in dispatcher
**Fix**: This is expected - dispatcher error message will be clear

#### Problem 4: Test errors instead of fails
**Symptom**: Test setup crashes before reaching assertion
**Cause**: Invalid test data, wrong shapes, missing fixtures
**Fix**: Debug test setup, verify data shapes match op requirements

---

### Success Criteria

#### Automated Verification:
- [ ] All tests run (none skipped): `pytest tests/link/onnx/test_conv.py --collect-only`
- [ ] All tests fail (none pass): `pytest tests/link/onnx/test_conv.py --tb=line | grep FAILED | wc -l` returns 19
- [ ] No unexpected errors: `pytest tests/link/onnx/test_conv.py --tb=line | grep "ERROR" | wc -l` returns 0
- [ ] Consistent failure mode: All tests fail with NotImplementedError

#### Manual Verification:
- [ ] Error messages are clear and helpful
- [ ] Failure messages would guide implementation
- [ ] Stack traces point to dispatcher (not test bugs)
- [ ] No syntax errors or import errors
- [ ] Test code is readable and maintainable

---

## Phase 3: Feature Implementation (Red → Green)

### Overview

Implement the Conv2D converter by making tests pass one at a time. Work like you're debugging - let test failures guide implementation.

### Implementation Strategy

**Order of Implementation** (easiest to hardest):
1. Basic converter structure (makes simple tests pass)
2. Padding modes (makes padding tests pass)
3. Stride/dilation/groups (makes parameter tests pass)
4. Filter flipping (makes CRITICAL asymmetric test pass)

---

### Implementation 1: Basic Conv2D Converter

**Target Tests**:
- test_conv2d_valid_single_channel
- test_conv2d_filter_flip_false

**Current Failure**: NotImplementedError: No ONNX conversion available for: AbstractConv2d

---

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/conv.py` (NEW FILE)

**Create file**:
```bash
touch pytensor/link/onnx/dispatch/conv.py
```

**Implementation**:
```python
"""ONNX conversion for convolution operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.conv.abstract_conv import AbstractConv2d

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(AbstractConv2d)
def onnx_funcify_AbstractConv2d(op, node, var_names, get_var_name, **kwargs):
    """
    Convert AbstractConv2d to ONNX Conv node.

    PyTensor Conv2D parameters:
    - border_mode: Padding ('valid', 'same', tuple, etc.)
    - subsample: Stride (downsampling factor)
    - filter_flip: True=convolution, False=cross-correlation
    - filter_dilation: Dilation (atrous convolution)
    - num_groups: Grouped convolution

    ONNX Conv attributes:
    - auto_pad: 'NOTSET', 'SAME_UPPER', 'VALID'
    - pads: [top, left, bottom, right]
    - strides: [stride_h, stride_w]
    - dilations: [dilation_h, dilation_w]
    - group: Number of groups

    References:
    - PyTensor AbstractConv2d: pytensor/tensor/conv/abstract_conv.py:2654
    - ONNX Conv spec: https://onnx.ai/onnx/operators/onnx__Conv.html
    - Gap analysis: thoughts/shared/research/...onnx-cnn-gap-analysis.md:447-500
    """
    # Get input/output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract op attributes
    border_mode = op.border_mode
    subsample = op.subsample
    filter_flip = op.filter_flip
    filter_dilation = op.filter_dilation
    num_groups = op.num_groups

    # Phase 1: Only support filter_flip=False (cross-correlation)
    if filter_flip:
        raise NotImplementedError(
            "Conv2D with filter_flip=True not yet supported for ONNX export.\n"
            "filter_flip=True performs mathematical convolution (flips kernel),\n"
            "but ONNX Conv performs cross-correlation (no flip).\n"
            "Kernel flipping will be implemented in Phase 2.\n"
            "For now, use filter_flip=False for ONNX export."
        )

    # Convert subsample to ONNX strides
    strides = list(subsample)

    # Convert filter_dilation to ONNX dilations
    dilations = list(filter_dilation)

    # Phase 1: Only support 'valid' border_mode
    if border_mode != "valid":
        raise NotImplementedError(
            f"Conv2D with border_mode='{border_mode}' not yet supported.\n"
            "Phase 1 only supports border_mode='valid'.\n"
            "Other padding modes will be implemented next."
        )

    # Build ONNX Conv node attributes
    attributes = {
        "auto_pad": "VALID",
        "strides": strides,
        "dilations": dilations,
        "group": num_groups,
    }

    # Create ONNX Conv node
    onnx_node = helper.make_node(
        "Conv",
        inputs=input_names,
        outputs=output_names,
        name=f"Conv_{output_names[0]}",
        **attributes
    )

    return onnx_node
```

---

#### Register Dispatcher

**File**: `pytensor/link/onnx/dispatch/__init__.py`

**Changes**: Add import for conv module (line ~16)

```python
"""ONNX dispatch system initialization.

Imports all dispatch modules to trigger @onnx_funcify.register() decorators.
"""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
import pytensor.link.onnx.dispatch.elemwise  # noqa: F401
import pytensor.link.onnx.dispatch.nlinalg  # noqa: F401
import pytensor.link.onnx.dispatch.shape  # noqa: F401
import pytensor.link.onnx.dispatch.special  # noqa: F401
import pytensor.link.onnx.dispatch.conv  # noqa: F401  # NEW

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
```

---

#### Debugging Approach

**Step 1: Run first test**
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_valid_single_channel -vv
```

**Expected progression**:
1. **First run**: NotImplementedError from dispatcher → GOOD (conv.py not imported yet)
2. **After adding import**: Test might pass or fail with different error
3. **If passes**: ✅ Move to next test
4. **If fails**: Read error message, debug, fix

**Step 2: Common errors and fixes**

**Error**: `ImportError: cannot import name 'AbstractConv2d'`
**Fix**: Check import path, verify class name

**Error**: `AttributeError: 'AbstractConv2d' object has no attribute 'border_mode'`
**Fix**: Check op parameter names in abstract_conv.py:2654

**Error**: `ONNX validation error: Conv node invalid`
**Fix**: Check ONNX Conv attributes, verify strides/dilations are lists of ints

**Error**: `Results don't match (numerical difference > 1e-4)`
**Fix**: Debug convolution logic, check if parameters are applied correctly

**Step 3: Verify test passes**
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_valid_single_channel -v
```

**Expected output**:
```
tests/link/onnx/test_conv.py::test_conv2d_valid_single_channel PASSED
```

**Step 4: Run all basic tests**
```bash
pytest tests/link/onnx/test_conv.py -k "valid_single_channel or filter_flip_false" -v
```

Both should pass (they have same configuration).

---

### Success Criteria

#### Automated Verification:
- [ ] Basic tests pass: `pytest tests/link/onnx/test_conv.py -k "valid_single_channel or filter_flip_false" -v`
- [ ] File exists: `pytensor/link/onnx/dispatch/conv.py`
- [ ] Import registered: Line added to `dispatch/__init__.py`
- [ ] No linting errors: `ruff check pytensor/link/onnx/dispatch/conv.py`
- [ ] Type checking passes (if applicable): `mypy pytensor/link/onnx/dispatch/conv.py`

#### Manual Verification:
- [ ] ONNX model validates: `onnx.checker.check_model()` passes
- [ ] ONNX Runtime executes: No runtime errors
- [ ] Numerical accuracy: Output matches PyTensor within 1e-4
- [ ] Error messages clear: filter_flip=True gives helpful error
- [ ] Code is clean and readable

---

### Implementation 2: Padding Modes

**Target Tests**:
- test_conv2d_valid_padding (already passes)
- test_conv2d_same_padding
- test_conv2d_explicit_symmetric_padding
- test_conv2d_explicit_asymmetric_padding

**Current Failure**: NotImplementedError: Conv2D with border_mode='same' not yet supported

---

#### Changes Required

**File**: `pytensor/link/onnx/dispatch/conv.py`

**Modify**: Replace "Phase 1: Only support 'valid'" section with full padding logic

**Updated code** (lines ~50-80):
```python
    # Convert border_mode to ONNX padding
    auto_pad = "NOTSET"
    pads = None

    if border_mode == "valid":
        # No padding
        auto_pad = "VALID"
    elif border_mode in ("same", "half"):
        # Maintain input size (with stride=1)
        # ONNX SAME_UPPER: pads at end if padding is odd
        auto_pad = "SAME_UPPER"
    elif border_mode == "full":
        # Full padding: output_size = input_size + kernel_size - 1
        # ONNX doesn't have FULL mode - need explicit pads
        # For 3x3 kernel: pads = [2, 2, 2, 2]
        # Formula: pad = kernel_size - 1
        # TODO: Extract kernel size from kernel variable
        raise NotImplementedError(
            "Conv2D with border_mode='full' not yet supported.\n"
            "ONNX Conv doesn't have 'FULL' padding mode.\n"
            "Need to compute explicit pads from kernel size."
        )
    elif isinstance(border_mode, int):
        # Symmetric padding (single value)
        # border_mode=1 → pads=[1,1,1,1]
        pads = [border_mode, border_mode, border_mode, border_mode]
    elif isinstance(border_mode, tuple) and len(border_mode) == 2:
        # Check if symmetric or asymmetric
        if isinstance(border_mode[0], int):
            # Symmetric: (pad_h, pad_w)
            pad_h, pad_w = border_mode
            pads = [pad_h, pad_w, pad_h, pad_w]
        else:
            # Asymmetric: ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = border_mode
            # ONNX format: [top, left, bottom, right]
            pads = [pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")

    # Build ONNX Conv node attributes
    attributes = {
        "strides": strides,
        "dilations": dilations,
        "group": num_groups,
    }

    # Add padding attributes
    if auto_pad != "NOTSET":
        attributes["auto_pad"] = auto_pad
    elif pads is not None:
        attributes["pads"] = pads
```

---

#### Debugging Approach

**Test each padding mode separately**:

```bash
# Test 1: Same padding
pytest tests/link/onnx/test_conv.py::test_conv2d_same_padding -vv

# Test 2: Symmetric explicit
pytest tests/link/onnx/test_conv.py::test_conv2d_explicit_symmetric_padding -vv

# Test 3: Asymmetric explicit
pytest tests/link/onnx/test_conv.py::test_conv2d_explicit_asymmetric_padding -vv
```

**Common issues**:

**Issue**: Output shape doesn't match
**Debug**:
```python
# Add temporary print in test
print(f"PyTensor output shape: {pytensor_output.shape}")
print(f"ONNX output shape: {onnx_output.shape}")
```

**Issue**: ONNX pads format wrong
**Fix**: ONNX uses [top, left, bottom, right], not [top, bottom, left, right]

**Issue**: Same padding not working
**Debug**: Check if SAME_UPPER vs SAME_LOWER matters for your test case

---

### Success Criteria

#### Automated Verification:
- [ ] All padding tests pass: `pytest tests/link/onnx/test_conv.py -k "padding" -v`
- [ ] No regressions: Previous tests still pass
- [ ] Linting passes: `ruff check pytensor/link/onnx/dispatch/conv.py`

#### Manual Verification:
- [ ] Output shapes correct for each padding mode
- [ ] Numerical accuracy maintained
- [ ] ONNX validation passes
- [ ] Error message for 'full' padding is clear

---

### Implementation 3: Strides, Dilations, Groups

**Target Tests**:
- test_conv2d_stride_2x2
- test_conv2d_asymmetric_stride
- test_conv2d_dilation_2x2
- test_conv2d_grouped_convolution
- test_conv2d_depthwise_convolution

**Current State**: These should already work! (attributes already mapped)

---

#### Verification Approach

**Run all parameter tests**:
```bash
pytest tests/link/onnx/test_conv.py -k "stride or dilation or grouped or depthwise" -v
```

**If they all pass**: ✅ Great! Move to next implementation.

**If some fail**: Debug the specific parameter mapping.

**Common issues**:

**Issue**: Stride/dilation not applied
**Debug**: Verify attributes dict includes `"strides"` and `"dilations"` keys

**Issue**: Grouped convolution fails
**Debug**: Check if channel counts are compatible with num_groups

---

### Success Criteria

#### Automated Verification:
- [ ] All parameter tests pass: `pytest tests/link/onnx/test_conv.py -k "stride or dilation or grouped or depthwise" -v`
- [ ] Multi-channel tests pass: `pytest tests/link/onnx/test_conv.py -k "rgb or batch" -v`
- [ ] No regressions in previous tests

#### Manual Verification:
- [ ] Output shapes correct for strided/dilated convolutions
- [ ] Grouped convolution produces correct number of output channels
- [ ] Depthwise convolution works (1 filter per input channel)

---

### Implementation 4: Filter Flipping (CRITICAL)

**Target Tests**:
- test_conv2d_filter_flip_true_symmetric
- test_conv2d_filter_flip_true_asymmetric ⭐⭐⭐

**Current Failure**: NotImplementedError: Conv2D with filter_flip=True not yet supported

**This is the MOST COMPLEX implementation** - requires multi-node pattern.

---

#### Understanding the Problem

**PyTensor filter_flip=True**:
- Flips kernel along spatial dimensions (H and W)
- Performs true mathematical convolution
- Formula: `y[i,j] = sum(x[i+m, j+n] * kernel[M-m, N-n])`

**ONNX Conv**:
- Does NOT flip kernel
- Performs cross-correlation
- Formula: `y[i,j] = sum(x[i+m, j+n] * kernel[m, n])`

**Solution**: Flip the kernel before passing to ONNX Conv

---

#### Implementation Options

**Option A: Multi-node pattern with Transpose/Slice** (Complex but correct)

Create nodes to flip kernel:
1. Transpose to swap dimensions
2. Slice with negative stride to reverse
3. Transpose back
4. Apply Conv

**Option B: Reverse op (If available in ONNX)**

Check if ONNX has a Reverse operator (it doesn't in opset 18).

**Option C: Gather with reversed indices**

Use Gather to reorder kernel elements in reverse.

---

#### Recommended Approach: Option A (Simplified)

**Since kernels are typically constants/initializers**, we can flip them at export time:

**File**: `pytensor/link/onnx/dispatch/conv.py`

**Modify**: Replace filter_flip NotImplementedError with flipping logic

```python
    # Handle filter flipping
    if filter_flip:
        # PyTensor flips kernel for mathematical convolution
        # ONNX Conv doesn't flip (cross-correlation)
        # Solution: Flip kernel before Conv

        # Check if kernel is a constant/initializer (common case)
        kernel_var = node.inputs[1]

        from pytensor.graph.basic import Constant

        if isinstance(kernel_var, Constant):
            # Simple case: Kernel is constant - flip at export time
            import numpy as np

            kernel_data = kernel_var.data
            # Flip spatial dimensions (last two dimensions)
            flipped_kernel = np.flip(kernel_data, axis=(-2, -1)).copy()

            # Create new constant node
            from onnx import numpy_helper

            flipped_name = f"flipped_kernel_{output_names[0]}"
            flipped_tensor = numpy_helper.from_array(flipped_kernel, name=flipped_name)

            # Create Constant node
            nodes = []
            nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[flipped_name],
                    value=flipped_tensor,
                    name=flipped_name,
                )
            )

            # Update input names to use flipped kernel
            conv_inputs = [input_names[0], flipped_name]
            if len(input_names) > 2:
                conv_inputs.append(input_names[2])  # Bias if present

            # Create Conv node with flipped kernel
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=conv_inputs,
                    outputs=output_names,
                    name=f"Conv_{output_names[0]}",
                    **attributes
                )
            )

            return nodes  # Return list of nodes

        else:
            # Complex case: Kernel is not constant (e.g., learned during export?)
            # Need runtime flipping with Transpose/Slice/Gather
            raise NotImplementedError(
                "Conv2D with filter_flip=True and non-constant kernel not yet supported.\n"
                "Kernel flipping is implemented for constant kernels only.\n"
                "If you need dynamic kernel flipping, please open an issue."
            )
```

---

#### Debugging Approach

**Step 1: Test with symmetric kernel first**
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_symmetric -vv
```

Should pass (flipping symmetric kernel gives same result).

**Step 2: Test with asymmetric kernel**
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -vv
```

**If fails with numerical mismatch**:
- Print intermediate values
- Check if flip is actually happening
- Verify flip dimensions are correct (last two axes)

**Debug code**:
```python
# Add to converter temporarily
print(f"Original kernel shape: {kernel_data.shape}")
print(f"Flipped kernel shape: {flipped_kernel.shape}")
print(f"Original kernel [0,0]: {kernel_data[0,0]}")
print(f"Flipped kernel [0,0]: {flipped_kernel[0,0]}")
```

**Step 3: Verify with manual calculation**

For Sobel kernel:
```python
# Original Sobel X
[[[ 1,  0, -1],
  [ 2,  0, -2],
  [ 1,  0, -1]]]

# Flipped (both H and W reversed)
[[[-1,  0,  1],
  [-2,  0,  2],
  [-1,  0,  1]]]
```

If PyTensor and ONNX outputs match, flipping is correct!

---

### Success Criteria

#### Automated Verification:
- [ ] Symmetric flip test passes: `pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_symmetric -v`
- [ ] **CRITICAL**: Asymmetric flip test passes: `pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -v`
- [ ] All previous tests still pass (no regressions)
- [ ] Linting passes

#### Manual Verification:
- [ ] Numerical accuracy: Outputs match within 1e-4
- [ ] Edge detection works correctly (Sobel kernel)
- [ ] Flipped kernel is actually reversed (inspect ONNX model)
- [ ] Error message for non-constant kernel is clear

**Milestone**: When asymmetric flip test passes, Conv2D implementation is FUNCTIONALLY COMPLETE!

---

### Implementation 5: Integration Tests

**Target Tests**:
- test_conv2d_with_bias
- test_conv2d_relu_pattern
- test_simple_cnn_block

**Expected**: These should pass automatically (use existing converters)

---

#### Verification

```bash
pytest tests/link/onnx/test_conv.py -k "bias or relu or cnn_block" -v
```

**If all pass**: ✅ Perfect! Conv2D integrates with existing ops.

**If some fail**: Debug interaction between Conv and other ops.

---

### Success Criteria

#### Automated Verification:
- [ ] **ALL TESTS PASS**: `pytest tests/link/onnx/test_conv.py -v` (100% pass rate)
- [ ] Integration tests pass: `pytest tests/link/onnx/test_conv.py -k "bias or relu or cnn_block" -v`
- [ ] No regressions: `pytest tests/link/onnx/ -v` (all ONNX tests pass)
- [ ] Code quality: `ruff check pytensor/link/onnx/dispatch/conv.py`
- [ ] Type checking: `mypy pytensor/link/onnx/dispatch/conv.py` (if applicable)

#### Manual Verification:
- [ ] Complete CNN layers can be exported
- [ ] ONNX models validate
- [ ] ONNX Runtime execution works
- [ ] Numerical accuracy maintained throughout

**PHASE 3 COMPLETE** when all 19 tests pass! ✅

---

## Phase 4: Refactoring & Cleanup

### Overview

Now that all tests pass, refactor to improve code quality while keeping tests green. Tests protect us during refactoring.

### Refactoring Targets

#### 1. Code Duplication

**Issue**: Padding conversion logic is long and repetitive

**Refactor**: Extract helper function

**File**: `pytensor/link/onnx/dispatch/conv.py`

**Add helper**:
```python
def convert_border_mode_to_onnx(border_mode):
    """
    Convert PyTensor border_mode to ONNX padding attributes.

    Parameters
    ----------
    border_mode : str or int or tuple
        PyTensor border_mode parameter

    Returns
    -------
    tuple of (auto_pad, pads)
        auto_pad : str or None
            ONNX auto_pad attribute ('VALID', 'SAME_UPPER', etc.)
        pads : list of int or None
            Explicit padding [top, left, bottom, right]
    """
    auto_pad = None
    pads = None

    if border_mode == "valid":
        auto_pad = "VALID"
    elif border_mode in ("same", "half"):
        auto_pad = "SAME_UPPER"
    elif border_mode == "full":
        raise NotImplementedError("border_mode='full' not yet supported")
    elif isinstance(border_mode, int):
        pads = [border_mode, border_mode, border_mode, border_mode]
    elif isinstance(border_mode, tuple) and len(border_mode) == 2:
        if isinstance(border_mode[0], int):
            pad_h, pad_w = border_mode
            pads = [pad_h, pad_w, pad_h, pad_w]
        else:
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = border_mode
            pads = [pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")

    return auto_pad, pads


# Then in main converter:
auto_pad, pads = convert_border_mode_to_onnx(border_mode)
```

**Test after refactoring**:
```bash
pytest tests/link/onnx/test_conv.py -v
```

All should still pass!

---

#### 2. Code Clarity

**Issue**: Long converter function is hard to read

**Refactor**: Add section comments and break into logical blocks

**Example structure**:
```python
@onnx_funcify.register(AbstractConv2d)
def onnx_funcify_AbstractConv2d(op, node, var_names, get_var_name, **kwargs):
    """Convert AbstractConv2d to ONNX Conv node."""

    # ============================================================
    # 1. Extract variable names
    # ============================================================
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # ============================================================
    # 2. Extract PyTensor op attributes
    # ============================================================
    border_mode = op.border_mode
    subsample = op.subsample
    filter_flip = op.filter_flip
    filter_dilation = op.filter_dilation
    num_groups = op.num_groups

    # ============================================================
    # 3. Handle filter flipping (if needed)
    # ============================================================
    if filter_flip:
        # ... flipping logic ...

    # ============================================================
    # 4. Convert parameters to ONNX attributes
    # ============================================================
    auto_pad, pads = convert_border_mode_to_onnx(border_mode)
    strides = list(subsample)
    dilations = list(filter_dilation)

    # ============================================================
    # 5. Build ONNX node
    # ============================================================
    attributes = {"strides": strides, "dilations": dilations, "group": num_groups}
    if auto_pad:
        attributes["auto_pad"] = auto_pad
    elif pads:
        attributes["pads"] = pads

    return helper.make_node("Conv", inputs=input_names, outputs=output_names, **attributes)
```

---

#### 3. Magic Numbers

**Issue**: Hardcoded axis indices (-2, -1) for flipping

**Refactor**: Use named constants

```python
# At top of file
KERNEL_HEIGHT_AXIS = -2
KERNEL_WIDTH_AXIS = -1

# In flipping code
flipped_kernel = np.flip(kernel_data, axis=(KERNEL_HEIGHT_AXIS, KERNEL_WIDTH_AXIS))
```

---

#### 4. Error Messages

**Issue**: Some error messages could be more helpful

**Refactor**: Add more context and suggestions

**Example**:
```python
# Before
raise NotImplementedError("border_mode='full' not yet supported")

# After
raise NotImplementedError(
    "Conv2D with border_mode='full' is not yet supported for ONNX export.\n"
    "Full padding would produce output_size = input_size + kernel_size - 1.\n"
    "ONNX Conv doesn't have a 'FULL' auto_pad mode.\n"
    "Workaround: Use explicit padding with border_mode=(pad_h, pad_w).\n"
    "Or open an issue requesting full padding support."
)
```

---

#### 5. Documentation

**Issue**: Missing module/function docstrings

**Refactor**: Add comprehensive docstrings

**Example**:
```python
"""
ONNX conversion for convolution operations.

This module provides converters for PyTensor convolution operations to ONNX Conv nodes.

Supported Operations:
- AbstractConv2d: 2D convolution with full parameter support

Key Features:
- All padding modes: valid, same, explicit symmetric/asymmetric
- Strided convolutions (subsample parameter)
- Dilated/atrous convolutions (filter_dilation parameter)
- Grouped and depthwise convolutions (num_groups parameter)
- Filter flipping: Handles conversion from mathematical convolution to cross-correlation

References:
- PyTensor convolution: pytensor/tensor/conv/abstract_conv.py
- ONNX Conv spec: https://onnx.ai/onnx/operators/onnx__Conv.html
- Gap analysis: thoughts/shared/research/...onnx-cnn-gap-analysis.md

Examples
--------
Export a simple CNN layer:

>>> import pytensor.tensor as pt
>>> from pytensor.tensor.nnet import conv2d
>>> from pytensor.link.onnx import export_onnx
>>>
>>> x = pt.tensor4('x', dtype='float32')
>>> kernel = pt.tensor4('kernel', dtype='float32')
>>> y = conv2d(x, kernel, border_mode='valid')
>>>
>>> f = pytensor.function([x, kernel], y)
>>> export_onnx(f, 'conv_model.onnx')
"""
```

---

#### 6. Test Improvements

**Issue**: test_conv.py has duplicated fixture

**Refactor**: Move fixture to conftest.py

**File**: `tests/link/onnx/conftest.py` (create if doesn't exist)

```python
"""Shared fixtures for ONNX tests."""

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")
```

Then remove duplicate fixtures from all test files.

---

### Refactoring Process

**For each refactoring**:

1. **Make the change**
2. **Run tests**: `pytest tests/link/onnx/test_conv.py -v`
3. **If tests pass**: Commit the change
4. **If tests fail**: Revert and reconsider

**Never**:
- Make multiple refactorings at once
- Refactor without tests
- Break passing tests

---

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/link/onnx/test_conv.py -v`
- [ ] No regressions: `pytest tests/link/onnx/ -v`
- [ ] Code coverage maintained: `pytest tests/link/onnx/ --cov=pytensor.link.onnx.dispatch.conv`
- [ ] Linting passes: `ruff check pytensor/link/onnx/dispatch/conv.py`
- [ ] Type checking passes: `mypy pytensor/link/onnx/dispatch/conv.py`

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity added
- [ ] Function/variable names are clear
- [ ] Docstrings are comprehensive
- [ ] Error messages are helpful
- [ ] No performance regressions

---

## Testing Strategy Summary

### Test Coverage Goals

**Functional Coverage**:
- ✅ Basic operations (valid padding, no flip): 2 tests
- ✅ Filter flipping (critical for correctness): 3 tests
- ✅ Padding modes (all variants): 4 tests
- ✅ Strides and dilations: 3 tests
- ✅ Grouped/depthwise convolutions: 2 tests
- ✅ Multi-channel and batching: 2 tests
- ✅ Integration with other ops: 3 tests

**Total**: 19 comprehensive tests

**Edge Cases Covered**:
- Asymmetric kernels (Sobel) - catches flip bugs
- Asymmetric padding - tests ONNX format
- Asymmetric strides - tests dimension handling
- Depthwise convolution - edge case of grouped conv
- Batch processing - tests independence

**Not Covered** (acceptable for Phase 1):
- 3D convolutions (separate feature)
- Dynamic kernel shapes
- Non-constant kernels with flipping
- Bias fusion optimization
- Full padding mode

---

### Test Organization

**File**: `tests/link/onnx/test_conv.py`

**Structure**:
```
Import and setup (lines 1-20)
├── Imports
├── pytest.importorskip for ONNX
└── Fixture for tmp_path

Category 1: Basic Tests (lines 21-100)
├── test_conv2d_valid_single_channel
└── test_conv2d_output_shape

Category 2: Filter Flipping Tests (lines 101-250)
├── test_conv2d_filter_flip_false
├── test_conv2d_filter_flip_true_symmetric
└── test_conv2d_filter_flip_true_asymmetric ⭐

Category 3: Padding Tests (lines 251-400)
├── test_conv2d_valid_padding
├── test_conv2d_same_padding
├── test_conv2d_explicit_symmetric_padding
└── test_conv2d_explicit_asymmetric_padding

Category 4-8: Parameter and Integration Tests (lines 401-700)
└── [Remaining tests]
```

---

### Running Tests

**Run all Conv2D tests**:
```bash
cd C:\Users\armor\OneDrive\Desktop\cs\pytensor
pytest tests/link/onnx/test_conv.py -v
```

**Run specific category**:
```bash
pytest tests/link/onnx/test_conv.py -k "padding" -v
pytest tests/link/onnx/test_conv.py -k "flip" -v
pytest tests/link/onnx/test_conv.py -k "stride or dilation" -v
```

**Run critical test only**:
```bash
pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -vv
```

**Run with coverage**:
```bash
pytest tests/link/onnx/test_conv.py --cov=pytensor.link.onnx.dispatch.conv --cov-report=term-missing
```

**Run with failure details**:
```bash
pytest tests/link/onnx/test_conv.py -vv --tb=short
```

**Run with output**:
```bash
pytest tests/link/onnx/test_conv.py -vv -s
```

---

## Performance Considerations

**Not a primary concern for Phase 1** - focus on correctness.

**Export Performance**:
- Simple CNNs (< 10 layers): < 1 second
- Medium CNNs (10-50 layers): 1-5 seconds
- Large CNNs (50+ layers): 5-30 seconds

**Runtime Performance** (ONNX Runtime):
- Browser (WebAssembly): 5-10x faster than Python interpreter
- Browser (WebGPU): 10-100x faster for large models
- Mobile/Edge: Near-native performance

**Performance Tests** (Optional):
```python
def test_conv2d_export_performance(tmp_path):
    """Test that export completes in reasonable time."""
    import time

    # Large CNN: 10 conv layers
    x = pt.tensor4("x", dtype="float32")
    y = x
    for i in range(10):
        kernel = pt.tensor4(f"kernel_{i}", dtype="float32")
        y = conv2d(y, kernel, border_mode="valid", filter_flip=False)
        y = pt.maximum(y, 0)  # ReLU

    f = pytensor.function([x] + [pt.tensor4(f"kernel_{i}") for i in range(10)], y)

    start = time.time()
    export_onnx(f, tmp_path / "large_cnn.onnx")
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Export took {elapsed:.2f}s (expected < 5s)"
```

---

## Migration Notes

**N/A** - This is a new feature, no migration needed.

**User Impact**:
- Existing PyTensor code works unchanged
- ONNX export is opt-in via `export_onnx()`
- No breaking changes to existing APIs

**Documentation Needed**:
- Add Conv2D to list of supported operations
- Document filter_flip limitation (or support)
- Provide CNN export examples
- Link to browser deployment guide

---

## References

### Original Research
- **Gap Analysis**: `thoughts/shared/research/2025-10-15_00-05-01_onnx-cnn-gap-analysis.md`
- **Implementation Plan**: `thoughts/shared/plans/onnx-backend-implementation.md`
- **Dev Guide**: `ONNX_DEV_GUIDE.md`

### PyTensor Code References
- **AbstractConv2d**: `pytensor/tensor/conv/abstract_conv.py:2654`
- **conv2d() function**: `pytensor/tensor/conv/abstract_conv.py:3514`
- **ONNX dispatcher**: `pytensor/link/onnx/dispatch/basic.py:29-70`
- **Test helper**: `tests/link/onnx/test_basic.py:18-101`

### ONNX Documentation
- **Conv operator**: https://onnx.ai/onnx/operators/onnx__Conv.html
- **Opset 18**: https://onnx.ai/onnx/operators/
- **ONNX Runtime Web**: https://onnxruntime.ai/docs/tutorials/web/

### Testing Patterns
- **Elemwise tests**: `tests/link/onnx/test_elemwise.py`
- **Matrix tests**: `tests/link/onnx/test_nlinalg.py`
- **Shape tests**: `tests/link/onnx/test_shape.py`

---

## Key Reminders

### Critical Success Factors

1. **Test FIRST, always**
   - Write ALL tests before implementation
   - Verify tests fail correctly
   - Implement to make tests pass

2. **Asymmetric Kernel Test**
   - Use Sobel/Prewitt edge detectors
   - This catches filter flip bugs
   - Most important test in entire suite

3. **One Test at a Time**
   - Make one test pass
   - Verify it passes
   - Move to next
   - Don't try to fix multiple tests simultaneously

4. **Keep Tests Green**
   - Previous tests must stay passing
   - Run full suite regularly
   - Don't break working functionality

5. **Refactor Fearlessly**
   - Tests protect during refactoring
   - Make small changes
   - Run tests after each refactoring
   - Revert if tests fail

### Common Pitfalls to Avoid

1. ❌ Writing implementation before tests
2. ❌ Using symmetric kernels only (hides flip bugs)
3. ❌ Not verifying test failures before implementing
4. ❌ Making multiple tests pass at once (too big steps)
5. ❌ Skipping refactoring phase (technical debt)
6. ❌ Not running full test suite (miss regressions)
7. ❌ Ignoring test failures (shows bugs!)

---

## Document Version

**Version**: 1.0
**Created**: 2025-10-15
**Status**: Ready for Implementation
**Target**: PyTensor ONNX Backend - Conv2D Support

---

## Appendix: Quick Command Reference

### Testing Commands

```bash
# Run all Conv2D tests
pytest tests/link/onnx/test_conv.py -v

# Run specific test
pytest tests/link/onnx/test_conv.py::test_conv2d_filter_flip_true_asymmetric -vv

# Run category
pytest tests/link/onnx/test_conv.py -k "flip" -v

# Run with coverage
pytest tests/link/onnx/test_conv.py --cov=pytensor.link.onnx.dispatch.conv --cov-report=html

# Run with output
pytest tests/link/onnx/test_conv.py -vv -s

# Stop at first failure
pytest tests/link/onnx/test_conv.py -x

# Run in parallel
pytest tests/link/onnx/test_conv.py -n auto
```

### Code Quality Commands

```bash
# Format code
ruff format pytensor/link/onnx/dispatch/conv.py

# Check issues
ruff check pytensor/link/onnx/dispatch/conv.py

# Auto-fix
ruff check --fix pytensor/link/onnx/dispatch/conv.py

# Type check
mypy pytensor/link/onnx/dispatch/conv.py

# Run pre-commit
pre-commit run --all-files
```

### Git Commands

```bash
# Create branch
git checkout -b onnx-conv2d-tdd

# Stage changes
git add tests/link/onnx/test_conv.py
git add pytensor/link/onnx/dispatch/conv.py
git add pytensor/link/onnx/dispatch/__init__.py

# Commit with clear message
git commit -m "Add ONNX Conv2D converter with comprehensive tests

- Implement AbstractConv2d → ONNX Conv converter
- Support all padding modes (valid, same, explicit)
- Handle filter flipping for mathematical convolution
- Support strides, dilations, grouped convolutions
- Add 19 comprehensive tests covering all parameters
- Critical: Test asymmetric kernels (Sobel) to verify flip correctness

Tests: pytest tests/link/onnx/test_conv.py -v"

# Push to remote
git push origin onnx-conv2d-tdd
```

---

## Final Checklist

Before considering implementation complete:

### Phase 1: Tests Written
- [ ] test_conv.py created with 19 tests
- [ ] All tests use compare_onnx_and_py helper
- [ ] Asymmetric kernel test uses Sobel/Prewitt
- [ ] Test code passes linting
- [ ] All tests have clear docstrings

### Phase 2: Tests Fail Correctly
- [ ] All 19 tests fail with NotImplementedError
- [ ] Error messages are clear and helpful
- [ ] No syntax or import errors
- [ ] Failures are consistent and expected

### Phase 3: Implementation Complete
- [ ] conv.py created and registered
- [ ] All 19 tests pass (100% pass rate)
- [ ] Basic operations work (valid padding)
- [ ] All padding modes work
- [ ] Strides, dilations, groups work
- [ ] **CRITICAL**: Asymmetric flip test passes
- [ ] Integration tests pass
- [ ] No regressions in other ONNX tests

### Phase 4: Refactored & Polished
- [ ] Code is clean and readable
- [ ] Helper functions extracted
- [ ] Docstrings comprehensive
- [ ] Error messages helpful
- [ ] No code duplication
- [ ] Linting passes
- [ ] Type checking passes (if applicable)
- [ ] All tests still pass after refactoring

### Documentation & Examples
- [ ] Conv2D added to supported ops list
- [ ] Example CNN export script created
- [ ] Limitations documented (if any)
- [ ] Browser deployment guide updated

**IMPLEMENTATION COMPLETE!** ✅

Ready to deploy CNNs to browsers, mobile, and edge devices via ONNX! 🚀
