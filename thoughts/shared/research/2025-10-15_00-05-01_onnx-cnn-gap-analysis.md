---
date: 2025-10-15T00:05:01Z
researcher: Claude (AI Assistant)
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: onnx-backend
repository: pytensor
topic: "ONNX Backend Gap Analysis for CNN/MNIST Support"
tags: [research, codebase, onnx, cnn, mnist, gap-analysis, convolution, pooling]
status: complete
last_updated: 2025-10-15
last_updated_by: Claude
---

# Research: ONNX Backend Gap Analysis for CNN/MNIST Support

**Date**: 2025-10-15T00:05:01Z
**Researcher**: Claude (AI Assistant)
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: onnx-backend
**Repository**: pytensor (pymc-devs/pytensor)

## Research Question

What operations are missing from the current ONNX backend implementation to support building and exporting a simple convolutional neural network (CNN) for MNIST digit classification?

## Executive Summary

The current ONNX backend implementation (~916 lines) provides solid infrastructure and support for fully-connected neural networks, but **lacks critical CNN-specific operations** needed for typical convolutional architectures.

**Key Findings**:
- ✅ **Fully-connected networks**: Fully supported (Dense layers, activations, softmax)
- ❌ **Conv2D operations**: **MISSING** - Most critical gap
- ❌ **Pooling operations**: **MISSING** - PyTensor doesn't have built-in pooling ops
- ⚠️ **ReLU activation**: Works via `maximum(x, 0)` pattern but suboptimal
- ⚠️ **Flatten operation**: Likely works via Reshape but untested for CNN use

**Priority Gaps for MNIST CNN**:
1. 🔴 **CRITICAL**: Conv2D converter (AbstractConv2d → ONNX Conv operator)
2. 🔴 **CRITICAL**: Pooling support (requires investigating PyTensor pooling implementation)
3. 🟡 **Medium**: ReLU optimization (pattern detection for dedicated ONNX ReLU node)
4. 🟡 **Medium**: Flatten testing/verification

**Estimated Implementation Effort**: 2-4 days for Conv2D converter + pooling investigation

---

## Detailed Findings

### 1. Current ONNX Backend Implementation

#### 1.1 Architecture Overview

**Location**: `pytensor/link/onnx/`

**Structure**:
```
pytensor/link/onnx/
├── __init__.py                  # Public API
├── export.py                    # export_onnx() function (102 lines)
└── dispatch/
    ├── __init__.py              # Dispatcher loader (14 lines)
    ├── basic.py                 # Core infrastructure (292 lines)
    ├── elemwise.py              # Element-wise ops (180 lines)
    ├── nlinalg.py               # Linear algebra ops (110 lines)
    ├── special.py               # Activations (89 lines)
    └── shape.py                 # Shape operations (395 lines)

Total: ~916 lines core + ~554 test lines
```

**Key Components**:
- **Singledispatch architecture**: `@onnx_funcify.register(OpClass)` pattern (`basic.py:29-70`)
- **FunctionGraph converter**: Converts entire computation graph to ONNX ModelProto (`basic.py:152-291`)
- **Shared variable handling**: Converts shared variables to ONNX initializers (baked weights) (`basic.py:207-224`)
- **Type system**: Maps PyTensor dtypes to ONNX TensorProto types (`basic.py:121-132`)
- **Validation**: Uses `onnx.checker.check_model()` (`basic.py:286-289`)

**Target**: ONNX opset 18 (`basic.py:26`)

---

#### 1.2 Supported Operations

##### Element-wise Operations (`elemwise.py`)

**File**: `pytensor/link/onnx/dispatch/elemwise.py:15-28`

| PyTensor Op | ONNX Op | Status | Notes |
|-------------|---------|--------|-------|
| `Add` | Add | ✅ | Binary addition |
| `Mul` | Mul | ✅ | Binary multiplication |
| `Sub` | Sub | ✅ | Binary subtraction |
| `TrueDiv` | Div | ✅ | Binary division |
| `Neg` | Neg | ✅ | Unary negation |
| `Exp` | Exp | ✅ | Exponential |
| `Log` | Log | ✅ | Natural logarithm |
| `Sqrt` | Sqrt | ✅ | Square root |
| `Pow` | Pow | ✅ | Power |
| `Abs` | Abs | ✅ | Absolute value |
| `ScalarMaximum` | Max | ✅ | Element-wise max (ReLU pattern) |
| `ScalarMinimum` | Min | ✅ | Element-wise min |

**Additional Features**:
- **Cast operations**: `scalar.Cast` → ONNX Cast node (`elemwise.py:130-157`)
- **Composite ops**: Decomposes fused scalar operations into multiple ONNX nodes (`elemwise.py:31-113`)

**Dispatcher**: `@onnx_funcify.register(Elemwise)` at line 116

---

##### Linear Algebra Operations (`nlinalg.py`)

**File**: `pytensor/link/onnx/dispatch/nlinalg.py`

| PyTensor Op | ONNX Op | Implementation | Notes |
|-------------|---------|----------------|-------|
| `Dot` | MatMul | Single node (`nlinalg.py:13-29`) | Matrix multiplication for FC layers |
| `Dot22` | MatMul | Single node (`nlinalg.py:32-45`) | Optimized 2x2 dot |
| `Gemv` | MatMul+Mul+Add | Multi-node (`nlinalg.py:48-109`) | y = alpha*A@x + beta*y decomposed into 4 nodes |

**Critical for**: Dense/fully-connected layers in neural networks

---

##### Activation Functions (`special.py`)

**File**: `pytensor/link/onnx/dispatch/special.py`

| Activation | Implementation | Status | Notes |
|------------|---------------|--------|-------|
| **Softmax** | ONNX Softmax | ✅ (`special.py:12-88`) | Supports axis parameter |
| **Softmax (axis=None)** | Flatten→Softmax→Reshape | ✅ | 4-node decomposition for flattened softmax |
| **ReLU** | Via ScalarMaximum | ⚠️ Pattern-based | `maximum(x, 0)` works but creates Max node, not ReLU |

**Dispatcher**: `@onnx_funcify.register(Softmax)` at line 12

---

##### Shape Operations (`shape.py`)

**File**: `pytensor/link/onnx/dispatch/shape.py`

| PyTensor Op | ONNX Implementation | Complexity | Notes |
|-------------|---------------------|------------|-------|
| `Shape_i` | Shape→Gather→Squeeze | Multi-node (`shape.py:17-94`) | Extract single dimension from shape |
| `Reshape` | Reshape | Single node (`shape.py:97-112`) | Direct mapping |
| `DimShuffle` | Unsqueeze/Squeeze/Transpose | Conditional (`shape.py:115-230`) | Add/remove/reorder dimensions |
| `AllocEmpty` | ConstantOfShape | Multi-node (`shape.py:233-376`) | Allocate zero-filled tensor |
| `DeepCopyOp` | Identity | Single node (`shape.py:379-394`) | Copy maps to identity in ONNX |

**Critical for CNNs**: Reshape (for flatten operation), DimShuffle (for transpose)

---

### 2. PyTensor CNN Operations Available

#### 2.1 Convolution Operations

**Location**: `pytensor/tensor/conv/abstract_conv.py`

**Main Classes**:
- `BaseAbstractConv` (line 2059) - Base class for all convolution operations
- `AbstractConv` (line 2436) - Generic N-dimensional convolution
- **`AbstractConv2d`** (line 2654) - **2D convolution for CNNs** ⭐
- `AbstractConv3d` (line 2716) - 3D convolution
- Plus gradient operations for backpropagation

**User-facing Functions**:
- **`conv2d()`** (line 3514) - **Primary 2D convolution API** ⭐
- `conv2d_transpose()` (line 3629) - Transposed convolution (upsampling)
- `conv3d()` (line 971) - 3D convolution
- `separable_conv2d()` (line 706) - Depthwise separable convolution
- `causal_conv1d()` (line 1649) - 1D causal convolution

**Key Parameters** (AbstractConv2d):
- `border_mode`: Padding strategy ('valid', 'full', 'half', or tuple of ints)
- `subsample`: Stride (downsampling factor)
- `filter_dilation`: Dilation factor for atrous convolution
- **`filter_flip`**: **Boolean controlling convolution vs cross-correlation** (default: True)
- `num_groups`: Number of groups for grouped convolution

**Critical Finding**: PyTensor's `filter_flip=True` (default) performs **mathematical convolution** (kernel flipping), while ONNX Conv operator performs **cross-correlation** (no flipping). This requires weight transformation during export!

---

#### 2.2 Pooling Operations

**Status**: ❌ **NOT FOUND**

**Investigation Results**:
- No dedicated `MaxPool2D` or `AvgPool2D` operation classes in PyTensor
- Pooling operations are not built into the core tensor module
- Possible workarounds:
  1. Strided convolutions (via `conv2d` with `subsample` parameter)
  2. Manual implementation using slicing and reduction operations
  3. External libraries (if users implement custom pooling)

**Implication**: Even if ONNX backend adds pooling converters, PyTensor users would need to implement pooling operations separately or use alternative downsampling methods.

**Recommendation**: Investigate how PyTensor users typically implement pooling for CNNs. Check if there are external packages or common patterns.

---

#### 2.3 Activation Functions

**Softmax**: ✅ Fully supported (`pytensor/tensor/special.py:242`)

**Other Activations** (`pytensor/tensor/math.py`):
- `sigmoid()` (line 2455)
- `tanh()` (line 2183)
- `softplus()` (line 2463)

**ReLU**: No dedicated operation, implemented as `maximum(x, 0)` pattern

---

#### 2.4 Shape/Flatten Operations

**Reshape**: `pytensor/tensor/shape.py:615` (Reshape class)
**Flatten**: `pytensor/tensor/basic.py:3064` (flatten function)

**Status**: ⚠️ Likely works via Reshape (already supported in ONNX backend) but untested for CNN use cases

---

#### 2.5 Padding Operations

**Location**: `pytensor/tensor/pad.py:415`

**Classes**:
- `Pad` (line 415) - OpFromGraph-based padding

**Functions**:
- `pad()` (line 430) - Main padding function

**Status**: ❌ No ONNX converter implemented yet

---

### 3. ONNX Operators Required for CNNs

**Research Source**: ONNX official documentation, opset 18

#### 3.1 Conv Operator

**ONNX Operator**: `Conv` (opset 18, version 18)

**Official Docs**: https://onnx.ai/onnx/operators/onnx__Conv.html

**Purpose**: 2D/3D convolution operation (fundamental for CNNs)

**Key Attributes**:
- **`kernel_shape`** (list of ints): Convolution kernel dimensions
- **`strides`** (list of ints, default: 1): Stride along each spatial axis
- **`pads`** (list of ints): Padding values [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
- **`auto_pad`** (string, default: 'NOTSET'): Automatic padding strategy
  - `NOTSET`: Use explicit `pads` attribute
  - `VALID`: No padding
  - `SAME_UPPER`: Pad to maintain output size, extra padding at end
  - `SAME_LOWER`: Pad to maintain output size, extra padding at beginning
- **`dilations`** (list of ints, default: 1): Dilation factor for atrous convolution
- **`group`** (int, default: 1): Number of groups for grouped/depthwise convolution

**Inputs**:
1. **X** (required): Input tensor (N × C × H × W) for 2D
2. **W** (required): Weight tensor (M × C/group × kH × kW)
3. **B** (optional): Bias tensor (1D, length M)

**Outputs**:
- **Y**: Output tensor with convolution result

**Type Constraints**: bfloat16, double, float, float16

**Critical Conversion Issue**:
- **ONNX Conv uses cross-correlation, NOT mathematical convolution**
- PyTensor's default `filter_flip=True` performs mathematical convolution (flips kernel)
- **Must flip weight kernels during export** when `filter_flip=True`
- For symmetric kernels, this doesn't matter; for asymmetric kernels (trained weights), this is critical!

**Conversion Steps**:
1. Check PyTensor op's `filter_flip` parameter
2. If `filter_flip=True`: Flip weight tensor (reverse H and W dimensions)
3. Map `border_mode` → `pads` or `auto_pad`
4. Map `subsample` → `strides`
5. Map `filter_dilation` → `dilations`
6. Map `num_groups` → `group`

---

#### 3.2 MaxPool Operator

**ONNX Operator**: `MaxPool` (opset 18, version 18)

**Official Docs**: https://onnx.ai/onnx/operators/onnx__MaxPool.html

**Purpose**: Max pooling over spatial dimensions (downsampling)

**Key Attributes**:
- **`kernel_shape`** (list of ints, **required**): Pooling kernel size
- **`strides`** (list of ints): Stride along each spatial axis
- **`pads`** (list of ints): Explicit padding for spatial axes
- **`auto_pad`** (string, default: 'NOTSET'): Automatic padding strategy (same options as Conv)
- **`dilations`** (list of ints): Dilation for pooling kernel
- **`ceil_mode`** (int, default: 0): Use ceil (1) or floor (0) for output shape computation

**Inputs**:
- **X**: Input tensor (N × C × H × W) for 2D

**Outputs**:
1. **Y**: Output tensor after max pooling
2. **Indices** (optional): Indices of selected max values (int64)

**Type Constraints**: float types, 8-bit tensors

**PyTensor Status**: No built-in operation found

---

#### 3.3 AveragePool Operator

**ONNX Operator**: `AveragePool` (opset 18, version 18)

**Official Docs**: https://onnx.ai/onnx/operators/onnx__AveragePool.html

**Purpose**: Average pooling over spatial dimensions (alternative to MaxPool)

**Key Attributes**:
- **`kernel_shape`** (list of ints, **required**): Pooling kernel size
- **`strides`** (list of ints): Stride along each spatial axis
- **`pads`** (list of ints): Explicit padding
- **`auto_pad`** (string): Automatic padding strategy
- **`ceil_mode`** (int, default: 0): Ceil or floor for output shape
- **`count_include_pad`** (int, default: 0): Include pad pixels in average calculation

**Inputs**:
- **X**: Input tensor (N × C × H × W)

**Outputs**:
- **Y**: Output tensor after average pooling

**Type Constraints**: bfloat16, double, float, float16

**PyTensor Status**: No built-in operation found

---

#### 3.4 Relu Operator

**ONNX Operator**: `Relu` (opset 18, version 14)

**Official Docs**: https://onnx.ai/onnx/operators/onnx__Relu.html

**Purpose**: Rectified Linear Unit activation (y = max(0, x))

**Attributes**: None (simple elementwise operation)

**Inputs**:
- **X**: Input tensor

**Outputs**:
- **Y**: Output tensor (same shape as input)

**PyTensor Status**: Implemented as `maximum(x, 0)` pattern
- Current ONNX backend: Maps to `Max` operator with constant 0
- Better: Pattern detection to emit single `Relu` node

---

#### 3.5 Flatten Operator

**ONNX Operator**: `Flatten` (opset 18, version 13)

**Official Docs**: https://onnx.ai/onnx/operators/onnx__Flatten.html

**Purpose**: Flattens tensor into 2D matrix (commonly used before FC layers)

**Attributes**:
- **`axis`** (int, default: 1): First dimension of output tensor is [d_0, ..., d_{axis-1}], second is [d_axis, ..., d_n]

**Inputs**:
- **X**: Input tensor

**Outputs**:
- **Y**: 2D output tensor

**PyTensor Status**: `flatten()` function exists, likely uses Reshape internally (already supported)

---

### 4. Typical MNIST CNN Architecture Analysis

#### 4.1 Standard Architecture

```python
# Input: (batch=None, channels=1, height=28, width=28)

# Block 1
Conv2D(filters=32, kernel_size=(3,3), padding='valid')  # ❌ MISSING
ReLU()                                                    # ⚠️ Works via maximum(x,0)
MaxPool2D(pool_size=(2,2))                               # ❌ MISSING (no PyTensor op)
# Output: (batch, 32, 13, 13)

# Block 2
Conv2D(filters=64, kernel_size=(3,3), padding='valid')  # ❌ MISSING
ReLU()                                                    # ⚠️ Works via maximum(x,0)
MaxPool2D(pool_size=(2,2))                               # ❌ MISSING (no PyTensor op)
# Output: (batch, 64, 5, 5)

# Flatten
Flatten()                                                 # ⚠️ Likely works via Reshape
# Output: (batch, 1600)

# Classifier
Dense(128) = MatMul(W1) + Bias(b1)                       # ✅ Supported (Dot + Add)
ReLU()                                                    # ⚠️ Works via maximum(x,0)
# Output: (batch, 128)

Dense(10) = MatMul(W2) + Bias(b2)                        # ✅ Supported (Dot + Add)
Softmax()                                                 # ✅ Supported
# Output: (batch, 10)
```

#### 4.2 Operations Status Summary

| Operation | PyTensor Support | ONNX Converter | Priority | Complexity |
|-----------|------------------|----------------|----------|------------|
| **Conv2D** | ✅ `AbstractConv2d` | ❌ **MISSING** | 🔴 CRITICAL | Medium-High |
| **MaxPool2D** | ❌ Not built-in | ❌ **MISSING** | 🔴 CRITICAL | High (no PyTensor op) |
| **Flatten** | ✅ `flatten()` | ⚠️ Untested (via Reshape) | 🟡 Medium | Low |
| **ReLU** | ✅ `maximum(x,0)` | ⚠️ Via Max (suboptimal) | 🟡 Medium | Low-Medium |
| **Dense/FC** | ✅ `Dot` + `Add` | ✅ **Supported** | ✅ Done | - |
| **Softmax** | ✅ `Softmax` | ✅ **Supported** | ✅ Done | - |

**Summary**:
- **2 operations CRITICAL & MISSING**: Conv2D converter, Pooling
- **3 operations work but suboptimal**: ReLU, Flatten, Bias handling
- **3 operations fully supported**: MatMul, Add, Softmax

**Blocking Issue**: Cannot export typical CNN architectures without Conv2D converter.

---

### 5. Gap Analysis & Implementation Roadmap

#### 5.1 Critical Gaps

##### Gap 1: Conv2D Converter ❌ 🔴

**PyTensor Op**: `AbstractConv2d` (`pytensor/tensor/conv/abstract_conv.py:2654`)

**ONNX Target**: Conv operator

**Implementation Requirements**:

1. **Register dispatcher**:
```python
@onnx_funcify.register(AbstractConv2d)
def onnx_funcify_AbstractConv2d(op, node, var_names, get_var_name, **kwargs):
    # Implementation
```

2. **Parameter Mapping**:
   - `op.border_mode` → ONNX `pads` attribute
     - String modes: 'valid' → [0,0,0,0], 'full' → compute from kernel size
     - Tuple modes: (pad_h, pad_w) → [pad_h, pad_h, pad_w, pad_w]
   - `op.subsample` → ONNX `strides`
   - `op.filter_dilation` → ONNX `dilations`
   - `op.num_groups` → ONNX `group`

3. **Weight Handling** (CRITICAL):
```python
if op.filter_flip:
    # PyTensor uses mathematical convolution (flips kernel)
    # ONNX uses cross-correlation (no flip)
    # Must flip weights during export: W[:,:,::-1,::-1]
    # This requires creating a Constant node or modifying initializer
```

4. **Bias Handling**:
   - Check if bias is added separately (next node is Add)
   - Option to fuse bias into Conv node (third input)

**File to Create**: `pytensor/link/onnx/dispatch/conv.py`

**Estimated LOC**: 150-200 lines

**Complexity**: Medium-High
- Border mode conversion requires careful logic
- Filter flipping is critical for correctness
- Testing with trained weights essential

**Test Cases**:
- Valid padding, no dilation, no groups
- Same padding with different kernel sizes
- Strided convolutions
- Dilated convolutions (atrous)
- Grouped convolutions
- **Filter flipping with asymmetric kernels** (most important!)

---

##### Gap 2: Pooling Operations ❌ 🔴

**PyTensor Op**: ❌ **NOT FOUND IN PYTENSOR**

**ONNX Target**: MaxPool, AveragePool operators

**Investigation Needed**:

1. **Check user patterns**: How do PyTensor users implement pooling for CNNs?
   - Custom operations?
   - External libraries?
   - Workarounds using strided convolutions?

2. **Search for pooling in legacy Theano**:
   - Theano had `pool_2d` function
   - May be referenced in old PyMC or Theano-pymc codebases

3. **Options**:
   - **Option A**: PyTensor lacks pooling → document limitation, suggest workarounds
   - **Option B**: Add pooling Ops to PyTensor core (major undertaking, out of scope)
   - **Option C**: Detect pooling-like patterns in graph and convert (complex, unreliable)

**Recommendation**: Document as a known limitation in Phase 1. Users can:
- Use strided convolutions for downsampling
- Implement pooling using slicing + reduction operations (will export via existing ops)
- Wait for future PyTensor pooling Op implementation

**Priority**: 🔴 CRITICAL but **blocked by PyTensor core limitation**

---

#### 5.2 Medium-Priority Improvements

##### Improvement 1: ReLU Optimization ⚠️ 🟡

**Current Behavior**: `maximum(x, 0)` → ONNX Max node with constant 0

**Desired Behavior**: Direct ONNX Relu node (cleaner, more efficient)

**Implementation**:
1. Pattern detection in Elemwise converter
2. Check if `ScalarMaximum` has one input as constant 0
3. If yes, emit Relu node instead of Max

**Location**: Modify `pytensor/link/onnx/dispatch/elemwise.py:116-179`

**Estimated LOC**: 30-50 lines

**Complexity**: Low-Medium

---

##### Improvement 2: Flatten Verification ⚠️ 🟡

**Current Status**: Untested for CNN use case

**Tasks**:
1. Test PyTensor's `flatten()` function with CNN-like tensors
2. Verify it uses Reshape Op (already supported)
3. If different Op, add converter

**Estimated Effort**: 0.5 days (mostly testing)

**Complexity**: Low

---

##### Improvement 3: Explicit Flatten Converter (Optional) 🟢

**Alternative to Reshape**: Use ONNX Flatten operator directly

**Benefits**:
- Cleaner ONNX graph
- More explicit semantics
- Single node vs. potentially multiple Reshape/DimShuffle nodes

**Implementation**: Add converter for whatever Op PyTensor's `flatten()` uses

**Estimated LOC**: 50-80 lines

---

#### 5.3 Future Optimizations (Low Priority)

##### Optimization 1: Conv+Bias Fusion 🟢

**Current**: Conv → Separate Add for bias (2 nodes)

**Target**: Single Conv node with bias input (1 node)

**Requirements**:
- Graph pattern matching
- Detect: Conv output → Add with 1D constant bias
- Fuse bias into Conv node's third input

**Complexity**: Medium (requires graph analysis)

**Estimated LOC**: 100-150 lines

---

##### Optimization 2: Batch Normalization 🟢

**ONNX Operator**: BatchNormalization

**PyTensor Status**: Unknown if built-in op exists

**Future Work**: Add converter if PyTensor supports batch norm

---

### 6. Implementation Recommendations

#### 6.1 Phase 1: Enable Basic CNN Export (2-3 days)

**Priority 1**: Conv2D Converter (1.5-2 days)

**Tasks**:
1. Create `pytensor/link/onnx/dispatch/conv.py`
2. Implement `@onnx_funcify.register(AbstractConv2d)`
3. Handle all parameter mappings (border_mode, subsample, dilation, groups)
4. **Critical**: Implement filter flipping logic when `filter_flip=True`
5. Create comprehensive test suite (`tests/link/onnx/test_conv.py`)
6. Test with valid/same padding, strides, dilations, groups
7. **Verify with asymmetric kernels** to catch flip issues

**Test Cases**:
```python
def test_conv2d_valid_padding(tmp_path):
    # Basic convolution with valid padding

def test_conv2d_filter_flip_true(tmp_path):
    # Critical: test with asymmetric kernels

def test_conv2d_filter_flip_false(tmp_path):
    # Test cross-correlation mode

def test_conv2d_strided(tmp_path):
    # Test with subsample parameter

def test_conv2d_dilated(tmp_path):
    # Test atrous convolution

def test_conv2d_grouped(tmp_path):
    # Test grouped/depthwise convolution
```

**Deliverables**:
- `pytensor/link/onnx/dispatch/conv.py` (~150-200 lines)
- `tests/link/onnx/test_conv.py` (~200-300 lines)
- Update `pytensor/link/onnx/dispatch/__init__.py` to import conv module

---

**Priority 2**: Pooling Investigation (0.5-1 day)

**Tasks**:
1. Search PyTensor codebase for any pooling operations
2. Check external PyTensor/PyMC packages for pooling implementations
3. Research Theano legacy pooling (may give clues)
4. Document findings:
   - If pooling ops exist → implement converters
   - If no pooling ops → document limitation and workarounds
5. Update documentation with pooling status

**Deliverables**:
- Investigation report (add to this document or create new file)
- Documentation updates explaining pooling situation
- If pooling exists: converters and tests

---

**Priority 3**: ReLU Optimization + Flatten Testing (0.5-1 day)

**Tasks**:
1. Add pattern detection for `maximum(x, 0)` → Relu
2. Test PyTensor's `flatten()` function with 4D tensors (NCHW)
3. Verify Flatten works correctly for CNN use case
4. Add explicit tests for flatten in CNN context

**Deliverables**:
- Updated `pytensor/link/onnx/dispatch/elemwise.py` with ReLU pattern
- Test coverage for flatten operation
- Documentation clarifying flatten behavior

---

#### 6.2 Phase 2: Optimization & Polish (1-2 days)

**Tasks**:
1. Conv+Bias fusion optimization
2. Additional padding modes support
3. Performance testing with real CNN models
4. Documentation and examples
5. MNIST example script

**Deliverables**:
- Example: Train CNN on MNIST, export to ONNX, run in ONNX Runtime
- Performance benchmarks
- User guide for CNN export

---

### 7. Code References

#### 7.1 Current ONNX Backend

- `pytensor/link/onnx/export.py:1-102` - Main export API
- `pytensor/link/onnx/dispatch/basic.py:29-70` - Core onnx_funcify dispatcher
- `pytensor/link/onnx/dispatch/basic.py:152-291` - FunctionGraph to ModelProto converter
- `pytensor/link/onnx/dispatch/elemwise.py:116-179` - Elemwise operation converter
- `pytensor/link/onnx/dispatch/nlinalg.py:13-109` - Linear algebra converters
- `pytensor/link/onnx/dispatch/special.py:12-88` - Softmax converter
- `pytensor/link/onnx/dispatch/shape.py:97-112` - Reshape converter

#### 7.2 PyTensor CNN Operations

- `pytensor/tensor/conv/abstract_conv.py:2654` - AbstractConv2d class (target for converter)
- `pytensor/tensor/conv/abstract_conv.py:3514` - conv2d() user function
- `pytensor/tensor/basic.py:3064` - flatten() function
- `pytensor/tensor/shape.py:615` - Reshape class
- `pytensor/tensor/special.py:242` - Softmax class
- `pytensor/tensor/math.py:2759` - maximum() function (for ReLU)

#### 7.3 Test Patterns

- `tests/link/onnx/test_basic.py:48-82` - compare_onnx_and_py() helper function
- `tests/link/onnx/test_elemwise.py:*` - Elemwise test patterns
- `tests/link/onnx/test_nlinalg.py:*` - Matrix operation test patterns

---

### 8. Key Technical Considerations

#### 8.1 Filter Flipping (CRITICAL)

**Issue**: PyTensor's default `filter_flip=True` performs mathematical convolution (kernel flip), while ONNX Conv performs cross-correlation (no flip).

**Solution**:
```python
@onnx_funcify.register(AbstractConv2d)
def onnx_funcify_AbstractConv2d(op, node, var_names, get_var_name, **kwargs):
    # Get inputs
    input_var, weights_var = node.inputs[:2]

    if op.filter_flip:
        # Need to flip weights for ONNX
        # Option 1: If weights are a constant/initializer, flip in place
        # Option 2: Insert Flip/Reverse operation in ONNX graph

        # For initializers (trained weights), flip during export:
        if isinstance(weights_var, Constant) or is_shared(weights_var):
            # Flip last two dimensions (H and W)
            flipped_weights = weights_data[:, :, ::-1, ::-1]
            # Update initializer with flipped version
```

**Test Validation**: Use asymmetric kernels (e.g., edge detectors) to verify correctness:
```python
kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype='float32')  # Sobel kernel (asymmetric)
```

---

#### 8.2 Padding Conversion

**PyTensor border_mode → ONNX pads mapping**:

| PyTensor `border_mode` | ONNX Equivalent | Notes |
|------------------------|-----------------|-------|
| `'valid'` | `pads=[0,0,0,0]` or `auto_pad='VALID'` | No padding |
| `'full'` | Computed from kernel size | Pad such that output ≥ input |
| `'half'` | `auto_pad='SAME_UPPER'` or compute | Output size = ceil(input/stride) |
| `(ph, pw)` | `pads=[ph,pw,ph,pw]` | Symmetric padding |
| `((ph_top,ph_bottom), (pw_left,pw_right))` | `pads=[ph_top,pw_left,ph_bottom,pw_right]` | Asymmetric padding |

**Implementation**:
```python
def convert_border_mode_to_pads(border_mode, kernel_shape):
    if border_mode == 'valid':
        return [0, 0, 0, 0]
    elif border_mode == 'full':
        kh, kw = kernel_shape
        return [kh-1, kw-1, kh-1, kw-1]
    elif border_mode == 'half':
        kh, kw = kernel_shape
        return [kh//2, kw//2, kh//2, kw//2]
    elif isinstance(border_mode, tuple):
        # Handle (ph, pw) or ((ph_top,ph_bottom), (pw_left,pw_right))
        ...
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")
```

---

#### 8.3 Data Layout

**Standard**: ONNX uses NCHW (batch, channels, height, width) format

**PyTensor**: Also primarily uses NCHW format (inherited from Theano)

**Implication**: No transposition needed (compatible by default) ✅

---

### 9. Testing Strategy

#### 9.1 Unit Tests for Conv2D

**File**: `tests/link/onnx/test_conv.py`

**Test Coverage**:
1. **Basic convolution**: Valid padding, no dilation, no groups
2. **Filter flipping**: Test with asymmetric kernels (Sobel, Prewitt)
3. **Padding modes**: Valid, full, half, symmetric, asymmetric
4. **Strides**: Various stride values
5. **Dilations**: Atrous/dilated convolutions
6. **Groups**: Grouped and depthwise convolutions
7. **Bias handling**: Separate vs. fused bias
8. **Multiple channels**: RGB-like inputs (3 channels)
9. **Batch processing**: Batch size > 1

**Test Pattern**:
```python
def test_conv2d_filter_flip_asymmetric_kernel(tmp_path):
    """Test Conv2D with filter_flip=True and asymmetric kernel.

    This is the most critical test to catch flip issues!
    """
    # Create input: (1, 1, 5, 5)
    x = pt.tensor4('x', dtype='float32')

    # Asymmetric Sobel kernel
    kernel = np.array([
        [[[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]]]
    ], dtype='float32')

    # Create convolution with filter_flip=True (mathematical convolution)
    W = pt.constant(kernel, dtype='float32')
    y = pt.nnet.conv2d(x, W, border_mode='valid', filter_flip=True)

    f = pytensor.function([x], y)

    # Export to ONNX
    model = export_onnx(f, tmp_path / "conv_flip.onnx")

    # Test input
    x_val = np.random.randn(1, 1, 5, 5).astype('float32')

    # Compare PyTensor vs ONNX Runtime
    pytensor_output = f(x_val)
    onnx_output = run_onnx_model(model, x_val)

    np.testing.assert_allclose(onnx_output, pytensor_output, rtol=1e-4)
```

---

#### 9.2 Integration Tests

**End-to-End CNN**: Create simple CNN, export, run in ONNX Runtime

```python
def test_simple_cnn_export(tmp_path):
    """Test exporting a simple CNN architecture."""
    # Input: (batch, 1, 28, 28)
    x = pt.tensor4('x', dtype='float32')

    # Conv1: 32 filters, 3x3, valid padding
    W1 = shared(np.random.randn(32, 1, 3, 3).astype('float32'))
    b1 = shared(np.zeros(32, dtype='float32'))
    conv1 = pt.nnet.conv2d(x, W1, border_mode='valid')
    conv1 = conv1 + b1.dimshuffle('x', 0, 'x', 'x')
    relu1 = pt.maximum(conv1, 0)

    # TODO: Add pooling when available

    # Flatten
    flat = relu1.flatten(2)

    # Dense layer
    W2 = shared(np.random.randn(relu1.shape[1].eval() * 26 * 26, 10).astype('float32'))
    b2 = shared(np.zeros(10, dtype='float32'))
    logits = pt.dot(flat, W2) + b2
    output = pt.nnet.softmax(logits)

    f = pytensor.function([x], output)

    # Export and test
    model = export_onnx(f, tmp_path / "simple_cnn.onnx")

    x_val = np.random.randn(1, 1, 28, 28).astype('float32')
    compare_onnx_and_py([x], output, [x_val], tmp_path=tmp_path)
```

---

### 10. Documentation Requirements

#### 10.1 User Documentation

**Location**: `examples/onnx/export_cnn_model.py`

**Content**:
1. How to create CNN in PyTensor
2. How to export to ONNX
3. How to run in ONNX Runtime
4. Known limitations (pooling, etc.)
5. Workarounds for missing operations

---

#### 10.2 Developer Documentation

**Location**: `pytensor/link/onnx/README.md` or docstrings

**Content**:
1. Architecture overview
2. How to add new operation converters
3. Testing patterns
4. Filter flipping explanation
5. Padding conversion reference

---

### 11. Open Questions

#### 11.1 Pooling Operations

**Question**: Does PyTensor have any pooling operations, or how do users implement pooling?

**Investigation Needed**:
- Check legacy Theano code
- Search PyMC/PyTensor user examples
- Look for external pooling implementations

**Impact**: Critical for typical CNN architectures

---

#### 11.2 Batch Normalization

**Question**: Does PyTensor support batch normalization?

**Impact**: Common in modern CNNs, low priority for MNIST

---

#### 11.3 Alternative Pooling Implementations

**Question**: Can pooling be implemented using existing PyTensor operations?

**Options**:
- Strided convolutions (achievable)
- Slicing + reduction operations (possible but complex)
- Custom OpFromGraph (requires investigation)

---

### 12. Related Research

**Previous Research**:
- `thoughts/shared/plans/onnx-backend-implementation.md` - Original implementation plan
- `ONNX_BACKEND_ANALYSIS.md` - Initial analysis
- `ONNX_DEV_GUIDE.md` - Development guide

**External References**:
- [ONNX Conv Operator](https://onnx.ai/onnx/operators/onnx__Conv.html)
- [ONNX MaxPool Operator](https://onnx.ai/onnx/operators/onnx__MaxPool.html)
- [ONNX GitHub - Conv vs Cross-correlation](https://github.com/onnx/onnx/issues/1180)
- [PyTensor Conv Documentation](https://pytensor.readthedocs.io/)

---

## Architecture Insights

### Established Patterns

1. **Singledispatch Registration**: All op converters use `@onnx_funcify.register(OpClass)` pattern
2. **Multi-node Decomposition**: Some ops return `List[NodeProto]` instead of single node
3. **Conditional Conversion**: Ops can have different ONNX representations based on parameters (e.g., Softmax with axis=None)
4. **Shared Variables → Initializers**: Trained weights are baked into ONNX model at export time
5. **Type Mapping**: Clear dtype mapping from PyTensor to ONNX TensorProto types

### Design Decisions

1. **Target opset 18**: Mature, well-supported by ONNX Runtime
2. **Export-only backend**: Not an execution backend (unlike JAX/Numba)
3. **Graph validation**: All exported models validated with `onnx.checker.check_model()`
4. **Clear error messages**: Unsupported ops provide helpful error messages with supported op lists

---

## Conclusion

### Summary

The current ONNX backend provides excellent infrastructure and full support for fully-connected neural networks but **cannot export convolutional neural networks** due to missing Conv2D converter.

**Blocking Issues**:
1. ❌ **Conv2D converter** - Most critical, requires 150-200 LOC + testing
2. ❌ **Pooling operations** - PyTensor may not have built-in pooling ops (requires investigation)

**Minor Issues**:
3. ⚠️ **ReLU optimization** - Works but generates Max node instead of Relu node
4. ⚠️ **Flatten testing** - Likely works but untested for CNN use case

### Recommendations

**Immediate Actions** (Priority 1):
1. Implement Conv2D converter with filter flipping logic (1.5-2 days)
2. Investigate PyTensor pooling support (0.5-1 day)

**Short-term Actions** (Priority 2):
3. Add ReLU pattern detection (0.5 day)
4. Test and verify flatten operation (0.5 day)
5. Create MNIST CNN example (1 day)

**Total Estimated Effort**: 3-5 days for basic CNN export support

### Success Criteria

✅ **Phase 1 Complete When**:
- Can export PyTensor `conv2d()` operations to ONNX Conv operator
- Filter flipping handled correctly (tested with asymmetric kernels)
- Padding modes correctly converted
- Tests pass with 100% success rate
- Clear documentation of pooling limitations/workarounds

✅ **Overall Success**: User can train a simple CNN in PyTensor, export to ONNX, and run inference in ONNX Runtime with results matching PyTensor.

---

**Document Version**: 1.0
**Status**: Complete
**Next Steps**: Implement Conv2D converter and investigate pooling operations
