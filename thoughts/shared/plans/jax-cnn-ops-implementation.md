# JAX Backend CNN Operations Implementation Plan

**Date**: 2025-10-15
**Goal**: Enable GPU training for YOLO11n and other CNNs using PyTensor's JAX backend
**Status**: Planning

---

## Problem Statement

PyTensor's JAX backend **does not support CNN operations** required for YOLO11n training:
- ❌ Conv2D
- ❌ MaxPool / AvgPool
- ❌ BatchNormalization
- ❌ Resize/Upsample

**Impact**: Cannot use H100 GPU for YOLO training, forcing CPU-only training (2-4 hours vs 30-45 minutes)

---

## Required Implementations

### Priority 1: Critical for YOLO11n (Must Have)

#### 1. Conv2D Operation
**File to create**: `pytensor/link/jax/dispatch/conv.py`

**PyTensor Op**: `pytensor.tensor.conv.abstract_conv.BaseAbstractConv`
- Used in: `conv2d(input, filters, border_mode, subsample, filter_flip)`
- YOLO usage: ConvBNSiLU blocks (every layer uses this)

**JAX Implementation**: `jax.lax.conv_general_dilated()`

**Key Parameters**:
- `subsample` → `window_strides` in JAX
- `border_mode` ('valid', 'same', tuple) → `padding` in JAX
- `filter_dilation` → `rhs_dilation` in JAX
- `filter_flip` → handle via reversing kernel if needed

**Gradient**: JAX auto-differentiates convolutions natively

**Implementation complexity**: **Medium** (2-3 hours)
- Parameter mapping is straightforward
- JAX handles gradients automatically
- Need to handle NCHW format (JAX uses same)

---

#### 2. MaxPool Operation
**File to create**: `pytensor/link/jax/dispatch/pool.py`

**PyTensor Ops**:
- `pytensor.tensor.pool.Pool` (forward)
- `pytensor.tensor.pool.MaxPoolGrad` (backward)

**JAX Implementation**: `jax.lax.reduce_window()` with `jax.lax.max`

**Key Parameters**:
- `ws` (window size) → `window_dimensions`
- `stride` → `window_strides`
- `padding` → `padding`
- `mode='max'` → use `jax.lax.max` as reducer

**Gradient**: `jax.lax.max` is differentiable, JAX handles automatically

**Implementation complexity**: **Easy** (1-2 hours)
- Direct mapping to JAX primitives
- Auto-differentiation handles gradient

---

#### 3. BatchNormalization Operation
**File to create**: `pytensor/link/jax/dispatch/batchnorm.py`

**PyTensor Op**: `pytensor.tensor.batchnorm.BatchNormalization`

**JAX Implementation**: Manual computation using JAX arrays
```python
# Forward:
mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)  # Per-channel
var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
x_norm = (x - mean) / jnp.sqrt(var + epsilon)
output = gamma * x_norm + beta

# JAX handles gradient automatically
```

**Alternative**: Use `jax.nn.batch_norm()` if available

**Gradient**: JAX auto-differentiates through these operations

**Implementation complexity**: **Medium** (2-3 hours)
- Need to handle channel-wise normalization (NCHW format)
- Must support both training and inference modes
- Gradient computation automatic via JAX

**Note**: PyTensor BatchNorm gradient also needs to be implemented (separate task, Phase 1 of YOLO plan)

---

#### 4. Resize/Upsample Operation
**File to create**: `pytensor/link/jax/dispatch/resize.py`

**PyTensor Op**: `pytensor.tensor.resize.Resize`

**JAX Implementation**: `jax.image.resize()`

**Key Parameters**:
- `output_shape` → `shape` in JAX
- `method` ('nearest', 'bilinear') → `method` in JAX ('nearest', 'bilinear', 'bicubic')

**Gradient**: `jax.image.resize` is differentiable

**Implementation complexity**: **Easy** (1 hour)
- Direct mapping to JAX function
- Auto-differentiation included

---

### Priority 2: Already Implemented (No Work Needed) ✅

These operations are **already working** in JAX backend:

#### 1. Element-wise Operations
**File**: `pytensor/link/jax/dispatch/elemwise.py`
- ✅ Add, Subtract, Multiply, Divide, Power, etc.
- ✅ Maximum (for ReLU)
- ✅ All scalar operations

#### 2. Math Operations
**File**: `pytensor/link/jax/dispatch/math.py`
- ✅ Sigmoid, Tanh, Exp, Log, Sqrt
- ⚠️ **SiLU/Swish** - Need to verify if implemented

#### 3. Tensor Operations
**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
- ✅ Join/Concatenate (for skip connections)
- ✅ Reshape, Flatten
- ✅ Transpose, DimShuffle

#### 4. Reductions
**File**: `pytensor/link/jax/dispatch/elemwise.py`
- ✅ Sum, Mean, Max, Min
- ✅ Argmax

#### 5. Special Operations
**File**: `pytensor/link/jax/dispatch/elemwise.py`
- ✅ Softmax
- ✅ LogSoftmax

---

### Priority 3: Nice to Have (Optional)

#### 1. AvgPool Operation
**Use case**: Some architectures prefer average pooling
**Implementation**: Same as MaxPool but with `jax.lax.add` reducer + division
**Complexity**: **Easy** (30 minutes)

#### 2. GroupNorm / LayerNorm
**Use case**: Alternative normalization methods
**Complexity**: **Easy** (1 hour each)

#### 3. DepthwiseConv2D
**Use case**: Efficient mobile architectures (MobileNet, EfficientNet)
**Complexity**: **Medium** (add `feature_group_count` parameter to Conv2D)

---

## Implementation Plan

### Phase 1: Core Operations (Day 1)
**Time estimate**: 6-8 hours

1. **Conv2D** (2-3 hours)
   - Create `pytensor/link/jax/dispatch/conv.py`
   - Implement `jax_funcify` for `BaseAbstractConv`
   - Handle parameter mapping
   - Test with simple conv layer

2. **MaxPool** (1-2 hours)
   - Create `pytensor/link/jax/dispatch/pool.py`
   - Implement `jax_funcify` for `Pool` op
   - Implement `jax_funcify` for `MaxPoolGrad` op
   - Test with pooling layer

3. **Resize/Upsample** (1 hour)
   - Create `pytensor/link/jax/dispatch/resize.py`
   - Implement `jax_funcify` for `Resize` op
   - Test with upsample operation

4. **BatchNorm** (2-3 hours)
   - Create `pytensor/link/jax/dispatch/batchnorm.py`
   - Implement `jax_funcify` for `BatchNormalization` op
   - Handle training vs inference modes
   - Test with batchnorm layer

### Phase 2: Testing & Integration (Day 2)
**Time estimate**: 4-6 hours

1. **Unit Tests** (2-3 hours)
   - Create `tests/link/jax/test_conv.py`
   - Create `tests/link/jax/test_pool.py`
   - Create `tests/link/jax/test_batchnorm.py`
   - Create `tests/link/jax/test_resize.py`
   - Follow pattern from existing JAX tests

2. **Integration Tests** (1-2 hours)
   - Test Conv → BN → ReLU → Pool stack
   - Test on simple CNN (MNIST)
   - Verify gradients work correctly

3. **YOLO Block Tests** (1 hour)
   - Test ConvBNSiLU block
   - Test SPPF block (cascaded pooling)
   - Test FPN upsampling

### Phase 3: Optimization & Documentation (Day 3)
**Time estimate**: 2-4 hours

1. **Performance Testing** (1-2 hours)
   - Benchmark vs CPU backend
   - Ensure GPU is actually being used
   - Check memory usage

2. **Documentation** (1-2 hours)
   - Add docstrings to all functions
   - Update JAX backend documentation
   - Add examples

---

## File Structure

```
pytensor/link/jax/dispatch/
├── __init__.py          # Update to import new modules
├── conv.py              # NEW: Conv2D operations
├── pool.py              # NEW: Pooling operations (max, avg)
├── batchnorm.py         # NEW: Batch normalization
└── resize.py            # NEW: Resize/upsample operations

tests/link/jax/
├── test_conv.py         # NEW: Conv2D tests
├── test_pool.py         # NEW: Pooling tests
├── test_batchnorm.py    # NEW: BatchNorm tests
├── test_resize.py       # NEW: Resize tests
└── test_cnn_stack.py    # NEW: Integration tests for CNN stacks
```

---

## Implementation Details

### Conv2D Dispatch Implementation

```python
# pytensor/link/jax/dispatch/conv.py

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.conv.abstract_conv import BaseAbstractConv


@jax_funcify.register(BaseAbstractConv)
def jax_funcify_Conv2D(op, node, **kwargs):
    """
    Convert PyTensor Conv2D to JAX conv_general_dilated.

    Maps PyTensor's convolution parameters to JAX's format.
    """
    # Extract op parameters
    subsample = op.subsample  # (stride_h, stride_w)
    border_mode = op.border_mode  # 'valid', 'half', 'full', or tuple
    filter_dilation = getattr(op, 'filter_dilation', (1, 1))
    num_groups = getattr(op, 'num_groups', 1)

    # Convert border_mode to JAX padding format
    if border_mode == 'valid':
        padding = 'VALID'
    elif border_mode == 'same' or border_mode == 'half':
        padding = 'SAME'
    elif isinstance(border_mode, (tuple, list)):
        # Explicit padding: (pad_h, pad_w)
        padding = [(p, p) for p in border_mode]
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")

    # Dimension numbers: PyTensor uses NCHW format
    dimension_numbers = ('NCHW', 'OIHW', 'NCHW')

    def conv2d(input, filters):
        """
        JAX convolution implementation.

        Parameters
        ----------
        input : array (N, C_in, H, W)
        filters : array (C_out, C_in, K_h, K_w)

        Returns
        -------
        output : array (N, C_out, H', W')
        """
        # Handle filter_flip (PyTensor default is True, correlate not convolve)
        if op.filter_flip:
            # Flip kernel spatially (convert correlation to convolution)
            filters = jnp.flip(filters, axis=(-2, -1))

        # Call JAX convolution
        output = jax.lax.conv_general_dilated(
            lhs=input,
            rhs=filters,
            window_strides=subsample,
            padding=padding,
            lhs_dilation=(1, 1),  # Input dilation (not used in standard conv)
            rhs_dilation=filter_dilation,  # Filter dilation
            dimension_numbers=dimension_numbers,
            feature_group_count=num_groups,  # For grouped/depthwise convs
        )

        return output

    return conv2d
```

### MaxPool Dispatch Implementation

```python
# pytensor/link/jax/dispatch/pool.py

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.pool import Pool, MaxPoolGrad


@jax_funcify.register(Pool)
def jax_funcify_Pool(op, node, **kwargs):
    """
    Convert PyTensor Pool to JAX reduce_window.
    """
    ws = op.ws  # (pool_h, pool_w)
    stride = op.stride  # (stride_h, stride_w)
    padding = op.padding  # (pad_h, pad_w)
    mode = op.mode  # 'max' or 'average'

    # Convert padding to JAX format
    # PyTensor uses (pad_h, pad_w), JAX needs ((pad_h, pad_h), (pad_w, pad_w))
    jax_padding = [(0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])]

    if mode == 'max':
        init_value = -jnp.inf
        reducer = jax.lax.max
    elif mode == 'average':
        init_value = 0.0
        reducer = jax.lax.add
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    def pool(input):
        """
        JAX pooling implementation.

        Parameters
        ----------
        input : array (N, C, H, W)

        Returns
        -------
        output : array (N, C, H', W')
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

        # For average pooling, divide by pool area
        if mode == 'average':
            pool_area = ws[0] * ws[1]
            output = output / pool_area

        return output

    return pool


@jax_funcify.register(MaxPoolGrad)
def jax_funcify_MaxPoolGrad(op, node, **kwargs):
    """
    Gradient of max pooling.

    JAX handles this automatically through autodiff, but we can provide
    explicit implementation for efficiency.
    """
    # JAX's autodiff will handle this automatically
    # We just need to ensure the forward pass is differentiable

    def maxpool_grad(x, gz):
        # This will be handled by JAX's autodiff system
        # When we take grad of the forward pool operation
        raise NotImplementedError(
            "MaxPoolGrad should be handled by JAX autodiff. "
            "This should not be called directly."
        )

    return maxpool_grad
```

### BatchNorm Dispatch Implementation

```python
# pytensor/link/jax/dispatch/batchnorm.py

import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.batchnorm import BatchNormalization


@jax_funcify.register(BatchNormalization)
def jax_funcify_BatchNormalization(op, node, **kwargs):
    """
    Convert PyTensor BatchNormalization to JAX operations.

    Implements batch normalization with learnable scale (gamma) and shift (beta).
    """
    epsilon = op.epsilon

    def batchnorm(x, gamma, beta, mean, variance):
        """
        JAX batch normalization.

        Parameters
        ----------
        x : array (N, C, H, W)
            Input tensor
        gamma : array (C,)
            Scale parameter
        beta : array (C,)
            Shift parameter
        mean : array (C,)
            Running mean (for inference) or batch mean (for training)
        variance : array (C,)
            Running variance (for inference) or batch variance (for training)

        Returns
        -------
        output : array (N, C, H, W)
            Normalized tensor
        """
        # Reshape parameters for broadcasting: (C,) → (1, C, 1, 1)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        mean = mean.reshape(1, -1, 1, 1)
        variance = variance.reshape(1, -1, 1, 1)

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(variance + epsilon)

        # Scale and shift
        output = gamma * x_norm + beta

        return output

    return batchnorm
```

### Resize Dispatch Implementation

```python
# pytensor/link/jax/dispatch/resize.py

import jax.image
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.resize import Resize


@jax_funcify.register(Resize)
def jax_funcify_Resize(op, node, **kwargs):
    """
    Convert PyTensor Resize to JAX image.resize.
    """
    method = op.method  # 'nearest' or 'bilinear'

    # Map PyTensor method to JAX method
    if method == 'nearest':
        jax_method = 'nearest'
    elif method == 'bilinear':
        jax_method = 'bilinear'
    else:
        raise ValueError(f"Unsupported resize method: {method}")

    def resize(input, output_shape):
        """
        JAX resize implementation.

        Parameters
        ----------
        input : array (N, C, H, W)
        output_shape : tuple (H', W')

        Returns
        -------
        output : array (N, C, H', W')
        """
        batch, channels, _, _ = input.shape
        new_h, new_w = output_shape

        # JAX expects shape as (batch, height, width, channels)
        # So we need to transpose: NCHW → NHWC
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

    return resize
```

---

## Testing Strategy

### Unit Tests Pattern

```python
# tests/link/jax/test_conv.py

import numpy as np
import pytest
import pytensor.tensor as pt
from pytensor.tensor.conv.abstract_conv import conv2d
from tests.link.jax.test_basic import compare_jax_and_py


def test_conv2d_valid():
    """Test Conv2D with valid padding."""
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Compare JAX and Python backends
    compare_jax_and_py([x, filters], out, [x_val, filters_val])


def test_conv2d_same():
    """Test Conv2D with same padding."""
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="same", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], out, [x_val, filters_val])


def test_conv2d_stride():
    """Test Conv2D with stride."""
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, subsample=(2, 2), border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], out, [x_val, filters_val])


def test_conv2d_gradient():
    """Test Conv2D gradient computation."""
    import pytensor

    x = pt.tensor4("x", dtype="float32")
    filters = shared(np.random.randn(16, 3, 3, 3).astype("float32"))

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)
    loss = out.sum()

    # Compute gradient
    grad_x, grad_filters = pytensor.grad(loss, [x, filters])

    # Compile with JAX backend
    with pytensor.config.change_flags(mode="JAX"):
        f = pytensor.function([x], [loss, grad_x, grad_filters])

        x_val = np.random.randn(2, 3, 8, 8).astype("float32")
        loss_val, grad_x_val, grad_filters_val = f(x_val)

        # Verify gradients are not zero
        assert np.abs(grad_x_val).sum() > 0
        assert np.abs(grad_filters_val).sum() > 0
```

---

## Verification Checklist

### After Implementation

- [ ] Conv2D operation works on JAX backend
- [ ] MaxPool operation works on JAX backend
- [ ] BatchNorm operation works on JAX backend
- [ ] Resize operation works on JAX backend
- [ ] All unit tests pass
- [ ] Gradients compute correctly for all operations
- [ ] Can train simple CNN (MNIST) on JAX backend with GPU
- [ ] Can build YOLO11n ConvBNSiLU block on JAX backend
- [ ] Can build YOLO11n SPPF block on JAX backend
- [ ] GPU is actually being used (verify with `nvidia-smi`)
- [ ] Performance is significantly better than CPU

---

## Success Criteria

1. ✅ All 4 core operations implemented and tested
2. ✅ MNIST CNN trains successfully on JAX backend with GPU
3. ✅ YOLO11n architecture builds without errors
4. ✅ Training speed on H100 is 10-20x faster than CPU
5. ✅ All tests pass in CI/CD pipeline

---

## Timeline

**Total Estimated Time**: 2-3 days (16-24 hours)

- **Day 1** (6-8 hours): Implement all 4 core operations
- **Day 2** (4-6 hours): Write and run all tests
- **Day 3** (2-4 hours): Optimize, document, integrate

**After completion**: YOLO11n training on H100 becomes possible (30-45 min training time)

---

## Next Steps

1. Get approval to proceed with implementation
2. Start with Conv2D (most critical)
3. Add tests incrementally
4. Integrate with YOLO training pipeline
5. Measure performance improvements
