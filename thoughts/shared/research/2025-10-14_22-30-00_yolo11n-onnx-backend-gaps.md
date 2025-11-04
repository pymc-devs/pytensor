---
date: 2025-10-14T22:30:00-07:00
researcher: Claude (Sonnet 4.5)
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: onnx-backend
repository: pymc-devs/pytensor
topic: "What's missing from the current ONNX backend to support YOLO11n model architecture"
tags: [research, codebase, onnx, yolo11n, cnn, gap-analysis, object-detection]
status: complete
last_updated: 2025-10-14
last_updated_by: Claude (Sonnet 4.5)
---

# Research: What's Missing from the Current ONNX Backend to Support YOLO11n

**Date**: 2025-10-14T22:30:00-07:00
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: onnx-backend
**Repository**: pymc-devs/pytensor

## Research Question

What operations and features are missing from the current PyTensor ONNX backend to support exporting YOLO11n (YOLOv11 nano) model architecture to ONNX format?

## Summary

The current ONNX backend in PyTensor supports ~24 operations including Conv2D, elementwise ops, linear algebra, and shape operations. However, **6 critical operation categories are missing** for YOLO11n support:

**Critical Missing Operations:**
1. **MaxPool / Pooling** - Required by SPPF block
2. **Upsample / Resize** - Required by FPN head (2 instances)
3. **Concat / Join** - Required by skip connections throughout
4. **Batch Normalization** - Required by C3k2 and C2PSA blocks
5. **SiLU/Swish Activation** - Required by all modern YOLO blocks
6. **Attention Mechanisms** - Required by C2PSA blocks

The ONNX backend has excellent Conv2D support (21 tests) but lacks the compositional operations needed for modern CNN architectures like YOLO11n.

## Detailed Findings

### YOLO11n Architecture Overview

**Model Specs:**
- Input size: 320x320 (scalable)
- Parameters: 2.6M
- Layers: 181 total
- Scaling: depth=0.50, width=0.25

**Architecture Components:**

**BACKBONE (11 layers):**
1. Conv [64, 3, 2] - stride 2 downsample
2. Conv [128, 3, 2] - stride 2 downsample
3. C3k2 (×2) [256, False, 0.25] - CSP Bottleneck block
4. Conv [256, 3, 2]
5. C3k2 (×2) [512, False, 0.25]
6. Conv [512, 3, 2]
7. C3k2 (×2) [512, True]
8. Conv [1024, 3, 2]
9. C3k2 (×2) [1024, True]
10. SPPF [1024, 5] - **Spatial Pyramid Pooling Fast** (requires MaxPool)
11. C2PSA (×2) [1024] - **Parallel Spatial Attention** (requires attention ops)

**HEAD (Feature Pyramid Network):**
12. Upsample [None, 2, "nearest"] - **2x upsampling**
13. Concat [layer -1, layer 6] - **Skip connection**
14. C3k2 (×2) [512, False]
15. Upsample [None, 2, "nearest"] - **2x upsampling**
16. Concat [layer -1, layer 4] - **Skip connection**
17. C3k2 (×2) [256, False]
18-21. Conv + Concat layers for feature aggregation
22. Detect - Multi-scale detection head (3 scales: P3/8, P4/16, P5/32)

### Current ONNX Backend Implementation Status

**Architecture:**
- **Dispatch system**: Singledispatch-based converter registration (`pytensor/link/onnx/dispatch/basic.py:29-70`)
- **Target opset**: ONNX opset 18 (`basic.py:26`)
- **Mode**: Export-only (no training/gradients)
- **Test coverage**: ~95 tests with property-based testing

**✅ Currently Supported Operations (24 total):**

#### Elemwise Operations (13 ops)
**File**: `pytensor/link/onnx/dispatch/elemwise.py:14-29`
- Binary: Add, Mul, Sub, Div, Pow, Max, Min
- Unary: Neg, Exp, Log, Sqrt, Abs, Sqr
- Cast (with dtype mapping)

#### Shape Operations (5 ops)
**File**: `pytensor/link/onnx/dispatch/shape.py`
- DimShuffle (Unsqueeze/Squeeze/Transpose) - lines 188-385
- Reshape - lines 97-112
- Shape_i - lines 17-94
- AllocEmpty - lines 388-531
- DeepCopyOp - lines 534-549

#### Linear Algebra (3 ops)
**File**: `pytensor/link/onnx/dispatch/nlinalg.py`
- Dot - lines 13-29
- Dot22 - lines 32-45
- Gemv - lines 48-109

#### Convolution (1 op)
**File**: `pytensor/link/onnx/dispatch/conv.py:14-140`
- **AbstractConv2d** - Full support with:
  - All padding modes (valid, half, explicit symmetric/asymmetric)
  - Stride (subsample)
  - Dilation (filter_dilation)
  - Grouped convolution (num_groups)
  - Filter flipping for mathematical convolution
  - **Test coverage**: 21 dedicated tests (`tests/link/onnx/test_conv.py`)

#### Special Functions (2 ops)
**File**: `pytensor/link/onnx/dispatch/special.py`
- Softmax (with axis variations) - lines 12-88
- Maximum/Minimum (for ReLU via `pt.maximum(x, 0)`)

### Gap Analysis: Missing Operations for YOLO11n

#### 1. ❌ MaxPool / Pooling Operations - **CRITICAL**

**Required by:** SPPF (Spatial Pyramid Pooling Fast) block in backbone

**What SPPF does:**
- Applies multiple MaxPool operations with different kernel sizes (typically 5x5)
- Concatenates results to create multi-scale features
- Example: MaxPool(5×5) → MaxPool → MaxPool → Concat all intermediate outputs

**Current status:**
- PyTensor has: `MaxPool` and `AveragePool` ops exist in `pytensor/tensor/nnet/pool.py`
- ONNX backend: **No converter implemented**
- Test coverage: None

**What's needed:**
```python
# File: pytensor/link/onnx/dispatch/pool.py (NEW FILE)

@onnx_funcify.register(MaxPool)
def onnx_funcify_MaxPool(op, node, var_names, get_var_name, **kwargs):
    """Convert MaxPool to ONNX MaxPool node."""
    return helper.make_node(
        "MaxPool",
        inputs=input_names,
        outputs=output_names,
        kernel_shape=[pool_h, pool_w],
        strides=[stride_h, stride_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
    )
```

**Impact:** Without MaxPool, the SPPF block cannot be exported, blocking backbone completion.

#### 2. ❌ Upsample / Resize Operations - **CRITICAL**

**Required by:** Feature Pyramid Network (FPN) head - 2 upsample layers

**What it does:**
- Upsamples feature maps by 2x using nearest neighbor or bilinear interpolation
- Lines 12, 15 in YOLO11n head configuration
- Essential for multi-scale detection

**Current status:**
- PyTensor has: Limited upsampling support via `Resampler` or manual implementation
- ONNX backend: **No converter implemented**
- ONNX operator: `Resize` with modes (nearest, linear, cubic)

**What's needed:**
```python
# File: pytensor/link/onnx/dispatch/resize.py (NEW FILE)

@onnx_funcify.register(ResizeOp)  # Or appropriate PyTensor op
def onnx_funcify_Resize(op, node, var_names, get_var_name, **kwargs):
    """Convert resize/upsample to ONNX Resize node."""
    return helper.make_node(
        "Resize",
        inputs=[input_name, roi, scales],  # scales = [1, 1, 2, 2] for 2x
        outputs=output_names,
        mode="nearest",  # or "linear"
    )
```

**Impact:** Without Upsample, the entire head/neck section cannot be exported. This is a **complete blocker** for FPN-based architectures.

#### 3. ❌ Concat / Join Operations - **CRITICAL**

**Required by:** Skip connections throughout head (lines 13, 16, 19, 21 in YOLO11n)

**What it does:**
- Concatenates feature maps from different layers along channel dimension
- Enables skip connections between encoder and decoder
- Used in SPPF to combine multi-scale pooled features

**Current status:**
- PyTensor has: `Join` op exists in `pytensor/tensor/basic.py:2420`
- ONNX backend: **No converter implemented**
- ONNX uses Concat internally (seen in `shape.py:500-507` for shape vectors)

**What's needed:**
```python
# File: pytensor/link/onnx/dispatch/join.py (NEW FILE)

@onnx_funcify.register(Join)
def onnx_funcify_Join(op, node, var_names, get_var_name, **kwargs):
    """Convert Join to ONNX Concat node."""
    axis = op.view  # Join's axis parameter
    input_names = [get_var_name(inp) for inp in node.inputs[1:]]  # Skip axis input

    return helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=output_names,
        axis=axis,
    )
```

**Impact:** Without Concat, skip connections fail. YOLO11n has **6+ skip connections** in the head alone.

#### 4. ❌ Batch Normalization - **HIGH PRIORITY**

**Required by:** C3k2 blocks, C2PSA blocks (all modern CNN layers use BatchNorm)

**What it does:**
- Normalizes activations: `(x - mean) / sqrt(var + epsilon) * gamma + beta`
- Critical for training stability and inference accuracy
- Every Conv layer in YOLO11n is followed by BatchNorm + activation

**Current status:**
- PyTensor has: `BatchNormalization` op in `pytensor/tensor/nnet/bn.py`
- ONNX backend: **No converter implemented**
- ONNX operator: `BatchNormalization` with scale, bias, mean, variance

**What's needed:**
```python
# File: pytensor/link/onnx/dispatch/batchnorm.py (NEW FILE)

@onnx_funcify.register(BatchNorm)
def onnx_funcify_BatchNorm(op, node, var_names, get_var_name, **kwargs):
    """Convert BatchNorm to ONNX BatchNormalization node."""
    # Inputs: x, scale (gamma), bias (beta), mean, variance
    return helper.make_node(
        "BatchNormalization",
        inputs=input_names,
        outputs=output_names,
        epsilon=op.epsilon,
        momentum=op.momentum,
    )
```

**Impact:** Without BatchNorm, exported models will have **incorrect numerical behavior**. This is a correctness issue, not just a missing feature.

#### 5. ❌ SiLU / Swish Activation - **HIGH PRIORITY**

**Required by:** All C3k2 blocks, C2PSA blocks (modern YOLO uses SiLU everywhere)

**What it is:**
- SiLU(x) = x * Sigmoid(x)
- Also known as Swish activation
- Superior to ReLU for modern architectures

**Current status:**
- PyTensor: **Does not exist** - no SiLU/Swish op defined
- ONNX backend: No converter (can't convert what doesn't exist)
- ONNX has no direct SiLU op but can decompose: `Mul(x, Sigmoid(x))`

**What's needed:**

**Step 1:** Create PyTensor SiLU op
```python
# File: pytensor/scalar/math.py (ADD NEW OP)

class SiLU(UnaryScalarOp):
    """SiLU(x) = x * sigmoid(x), also known as Swish."""
    def impl(self, x):
        return x / (1 + np.exp(-x))

silu = SiLU(name="silu")
```

**Step 2:** Add ONNX converter with decomposition
```python
# File: pytensor/link/onnx/dispatch/elemwise.py (ADD TO MAPPING)

@onnx_funcify.register(SiLU)
def onnx_funcify_SiLU(op, node, var_names, get_var_name, **kwargs):
    """Convert SiLU to ONNX as x * Sigmoid(x)."""
    input_name = get_var_name(node.inputs[0])
    sigmoid_out = f"sigmoid_{output_names[0]}"

    nodes = [
        helper.make_node("Sigmoid", [input_name], [sigmoid_out]),
        helper.make_node("Mul", [input_name, sigmoid_out], output_names),
    ]
    return nodes
```

**Impact:** Without SiLU, YOLO11n would need to use ReLU instead, resulting in **degraded accuracy**. All 181 layers expect SiLU.

#### 6. ❌ Attention Mechanisms - **MEDIUM PRIORITY**

**Required by:** C2PSA (Convolutional with Parallel Spatial Attention) blocks

**What C2PSA does:**
- Applies spatial attention to emphasize important regions
- Typical pattern: Global pooling → FC layers → Sigmoid → Multiply with features
- May also use self-attention patterns with Q/K/V matrices

**Current status:**
- PyTensor: Has individual components (MatMul, Softmax, Reshape)
- ONNX backend: No attention patterns or composite converters
- Would need: MatMul ✅, Softmax ✅, Reshape ✅, but no pattern for combining them

**What's needed:**

Two approaches:

**Option A - Decompose to primitives:**
Let attention decompose naturally into MatMul, Softmax, etc. (already supported)

**Option B - Create attention pattern converter:**
```python
# File: pytensor/link/onnx/dispatch/attention.py (NEW FILE)

@onnx_funcify.register(SpatialAttention)  # If PyTensor adds this op
def onnx_funcify_SpatialAttention(op, node, var_names, get_var_name, **kwargs):
    """Convert spatial attention to ONNX sequence."""
    # Decompose into: GlobalAveragePool → Reshape → FC → Sigmoid → Mul
    # Or use ONNX's Attention operator for self-attention patterns
    pass
```

**Impact:** C2PSA blocks won't export. However, if attention is implemented using primitives (MatMul, Softmax, etc.), those **might work automatically**.

### Additional Missing Operations (Lower Priority)

#### 7. ❌ Global Pooling
- `GlobalAveragePool`, `GlobalMaxPool`
- Often used in detection heads and attention blocks
- PyTensor has: Can be implemented via reduce operations
- ONNX: Has dedicated global pooling operators

#### 8. ❌ Sigmoid Activation (Direct)
**Partial issue:** Sigmoid exists in PyTensor (`pytensor/scalar/math.py:1200`) but **not mapped to ONNX**

**Current workaround:** None - Sigmoid just isn't converted

**Easy fix:**
```python
# File: pytensor/link/onnx/dispatch/elemwise.py (ADD TO DICTIONARY)

SCALAR_OP_TO_ONNX = {
    # ... existing entries ...
    scalar.Sigmoid: "Sigmoid",  # ADD THIS LINE
}
```

**Test exists:** `tests/link/onnx/test_special.py:44-51` tests ReLU via maximum, but no Sigmoid test

#### 9. ❌ Tanh Activation
- Similar to Sigmoid - exists in PyTensor but not mapped to ONNX
- Less critical for YOLO11n but needed for completeness

### C3k2 and C2PSA Block Decomposition

Understanding what these blocks need helps prioritize:

**C3k2 (CSP Bottleneck with kernel 2):**
```
Input
  ├─> Conv(1×1) → BatchNorm → SiLU → Conv(3×3) → BatchNorm → SiLU → (bottleneck)
  └─> Conv(1×1) → BatchNorm → SiLU ──────────────────────────────> (shortcut)
  └─> Concat [bottleneck, shortcut] → Conv(1×1) → BatchNorm → SiLU → Output
```

**Needs:**
- Conv2D ✅
- BatchNorm ❌
- SiLU ❌
- Concat ❌
- Add (for residuals) ✅

**C2PSA (Parallel Spatial Attention):**
```
Input → Conv → BatchNorm → SiLU
  ├─> Spatial Attention (GlobalPool → FC → Sigmoid → Multiply)
  └─> Identity
  └─> Concat or Add → Conv → Output
```

**Needs:**
- Conv2D ✅
- BatchNorm ❌
- SiLU ❌
- GlobalPool ❌
- Softmax or Sigmoid (Sigmoid ⚠️ not mapped)
- Multiply ✅
- Concat ❌

## Code References

### Currently Implemented Operations

- `pytensor/link/onnx/dispatch/basic.py:29-70` - Main dispatcher system
- `pytensor/link/onnx/dispatch/conv.py:14-140` - Conv2D converter (✅ complete)
- `pytensor/link/onnx/dispatch/elemwise.py:14-29` - Elementwise ops mapping
- `pytensor/link/onnx/dispatch/shape.py` - Shape operations (Reshape, DimShuffle)
- `pytensor/link/onnx/dispatch/nlinalg.py` - Linear algebra ops
- `pytensor/link/onnx/dispatch/special.py:12-88` - Softmax

### Test Infrastructure

- `tests/link/onnx/test_basic.py:22-102` - `compare_onnx_and_py()` test helper
- `tests/link/onnx/test_conv.py:170-226` - Critical Conv2D filter flip test
- `tests/link/onnx/test_properties.py` - Property-based tests with Hypothesis
- `tests/link/onnx/strategies/operations.py:290-368` - Operation test strategies

### PyTensor Ops That Need ONNX Converters

- `pytensor/tensor/nnet/pool.py` - MaxPool, AveragePool ops
- `pytensor/tensor/basic.py:2420` - Join op (for Concat)
- `pytensor/tensor/nnet/bn.py` - BatchNormalization op
- `pytensor/scalar/math.py:1200` - Sigmoid op (exists but not mapped)

## Architecture Insights

### Current ONNX Backend Design Patterns

**1. Singledispatch Registration Pattern:**
```python
@onnx_funcify.register(OpClass)
def onnx_funcify_OpClass(op, node, var_names, get_var_name, **kwargs):
    # Convert PyTensor op to ONNX node(s)
    return onnx_node  # or list of nodes
```

**2. Multi-Node Decomposition:**
Complex ops can return lists of ONNX nodes:
- Shape_i: 5 nodes (Shape → Gather → Squeeze)
- Gemv: 4 nodes (MatMul → Mul → Mul → Add)
- Works for SiLU: 2 nodes (Sigmoid → Mul)

**3. Test-Driven Development:**
Every operation has:
- Unit test with `compare_onnx_and_py()`
- Property-based test (optional)
- Regression test for critical bugs

**4. Filter Flipping Pattern:**
Conv2D demonstrates sophisticated preprocessing:
- Pre-scans graph for `filter_flip=True` (`basic.py:207-218`)
- Flips kernel initializers before export
- Ensures mathematical convolution correctness

### Implementation Priority for YOLO11n

**Tier 1 - Complete Blockers (Cannot export without these):**
1. ✅ Conv2D - Already implemented with 21 tests
2. ❌ **Concat** - Used 6+ times in head
3. ❌ **Upsample** - Used 2 times in head
4. ❌ **MaxPool** - Used in SPPF block

**Tier 2 - Correctness Issues (Export works but incorrect behavior):**
5. ❌ **BatchNorm** - Every layer uses this
6. ❌ **SiLU** - Every activation uses this

**Tier 3 - Advanced Features:**
7. ❌ Attention mechanisms (C2PSA)
8. ❌ Global pooling
9. ⚠️ Sigmoid mapping (easy fix)

### Estimated Implementation Effort

**Easy (1-2 hours each):**
- Sigmoid mapping (just add to dictionary)
- Join/Concat converter (straightforward mapping)
- MaxPool converter (similar to Conv2D)

**Medium (1 day each):**
- Upsample/Resize (need to handle multiple modes)
- BatchNormalization (multiple parameters)
- SiLU (need to add to PyTensor first)

**Complex (2-3 days):**
- Global pooling (multiple variants)
- Attention patterns (if doing composite converters)

**Total estimated effort for Tier 1+2:** ~5-7 days of focused development

## Historical Context (from thoughts/)

### Related Implementation Plans

**1. Main ONNX Backend Plan** (`thoughts/shared/plans/onnx-backend-implementation.md`)
- Documents core dispatcher architecture
- Lists 24 currently supported operations
- Established testing patterns with Hypothesis

**2. Conv2D TDD Plan** (`thoughts/shared/plans/onnx-conv2d-tdd.md`)
- Completed Conv2D implementation with 21 tests
- Demonstrates successful TDD approach
- Filter flipping correctness verified with Sobel kernel test

**3. Coverage and Quality Plan** (`thoughts/shared/plans/onnx-backend-coverage-and-quality-improvements.md`)
- Current state: 8 implementation files (1,181 lines), 5 test files (706 lines)
- 27 tests (now 95+ with Conv2D and properties)
- Identified 5 completely untested operations (still true for pooling, etc.)

**4. Property-Based Testing Plan** (`thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`)
- Addressed test explosion problem (103 manual tests)
- Implemented 4 generic properties that test all operations
- Documents all supported operations

### Related Research Documents

**1. CNN Gap Analysis** (`thoughts/shared/research/2025-10-15_00-05-01_onnx-cnn-gap-analysis.md`)
- Previous CNN analysis likely identified pooling gaps

**2. Coverage Analysis** (`thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md`)
- Detailed operation support coverage

**3. WebAssembly Research** (`thoughts/shared/research/2025-10-15_onnx-backend-webassembly.md`)
- Target deployment: Browser with ONNX Runtime Web
- Motivates need for complete CNN support

**4. Open Questions** (`thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md`)
- Addresses dynamic shapes, custom ops, performance
- Question 1: How to handle dynamic shapes in ONNX export

## Related Research

- `thoughts/shared/research/2025-10-15_00-05-01_onnx-cnn-gap-analysis.md` - Previous CNN gap analysis
- `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md` - Operation coverage
- `thoughts/shared/plans/onnx-conv2d-tdd.md` - Conv2D implementation (completed)
- `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md` - Testing strategy

## Open Questions

### 1. Does PyTensor have a standard upsampling operation?

**Investigation needed:**
- Search for `Resampler`, `Upsample`, `Resize` operations in PyTensor
- Check if `resize` or `upsample` functions exist in `pytensor.tensor.nnet`
- May need to implement custom upsampling op

### 2. How should attention mechanisms be handled?

**Two approaches:**
- **Decompose to primitives**: Let attention blocks use MatMul, Softmax, etc. (already supported)
- **Composite converters**: Create attention-specific converters
- Which approach aligns better with PyTensor philosophy?

### 3. What is the priority order for implementation?

**Recommendation:**
1. **Concat** - Unblocks head section (many dependencies)
2. **Upsample** - Unblocks FPN head
3. **MaxPool** - Unblocks SPPF
4. **BatchNorm** - Correctness for all layers
5. **SiLU** - Correctness for activations
6. **Attention** - Advanced features

### 4. Should we implement all ONNX pooling variants?

**Options:**
- MaxPool only (minimum for YOLO11n)
- MaxPool + AveragePool (common duo)
- All variants (GlobalMaxPool, GlobalAvgPool, LpPool, etc.)

**Recommendation:** Start with MaxPool and AveragePool, add global variants as needed.

### 5. How to test composite blocks like C3k2?

**Testing strategy:**
- Unit tests for individual ops (Concat, BatchNorm, etc.)
- Integration test for complete C3k2 block?
- Property-based tests for block composition?

### 6. Can we use existing Hypothesis strategies for new ops?

**Current strategies** (`tests/link/onnx/strategies/operations.py:290-368`):
- Work for unary, binary, matmul, reshape, dimshuffle, conv2d
- Can extend for pooling, concat, upsample?
- Need new strategy patterns for attention?

## Conclusion

**To support YOLO11n architecture, the PyTensor ONNX backend needs 6 critical additions:**

1. ❌ **Concat** (Join converter) - HIGH PRIORITY, BLOCKER
2. ❌ **Upsample** (Resize converter) - HIGH PRIORITY, BLOCKER
3. ❌ **MaxPool** - HIGH PRIORITY, BLOCKER
4. ❌ **BatchNorm** - HIGH PRIORITY, CORRECTNESS
5. ❌ **SiLU** (requires PyTensor op + converter) - HIGH PRIORITY, CORRECTNESS
6. ❌ **Attention mechanisms** - MEDIUM PRIORITY

**Current strengths:**
- ✅ Excellent Conv2D support (21 tests, all features)
- ✅ Solid foundation (24 ops, ~95 tests)
- ✅ Good architecture (extensible, well-tested)

**Estimated effort:** ~5-7 days focused development for Tier 1+2 operations

**Recommended implementation order:** Concat → Upsample → MaxPool → BatchNorm → SiLU → Attention

The ONNX backend is well-architected and just needs these specific operations to support modern CNN architectures like YOLO11n.
