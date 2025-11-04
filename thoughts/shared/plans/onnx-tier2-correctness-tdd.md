# ONNX Tier 2 Correctness: BatchNorm, SiLU, Sigmoid - TDD Implementation Plan

## Overview

This plan implements Test-Driven Development for the **3 critical correctness operations** needed for YOLO11n support in PyTensor's ONNX backend. These operations are not blockers (models can export without them), but exported models will have **incorrect numerical behavior** without them.

**Operations covered:**
1. **Sigmoid activation** - Exists in PyTensor, just needs ONNX mapping (EASIEST)
2. **SiLU/Swish activation** - Must create PyTensor scalar op + ONNX converter
3. **BatchNormalization** - Must create PyTensor op + ONNX converter

**Why "Correctness" tier:**
- Without these: YOLO11n exports but produces wrong predictions
- With Sigmoid: Can export C2PSA attention blocks correctly
- With SiLU: All 181 layers get correct activation (not degraded ReLU)
- With BatchNorm: All layers get correct normalization (not incorrect scaling)

**Total estimated effort:** 2-3 days (Sigmoid: 2 hours, SiLU: 1 day, BatchNorm: 1 day)

## Current State Analysis

### Existing Infrastructure

**Test Infrastructure** (same as Tier 1):
- `compare_onnx_and_py()` helper in `tests/link/onnx/test_basic.py:22-102`
- Property-based testing with Hypothesis
- Dispatcher pattern: `@onnx_funcify.register(OpClass)`

**Scalar Op Pattern** (for SiLU):
- Reference: Sigmoid in `pytensor/scalar/math.py:1200-1239`
- Pattern: `UnaryScalarOp` with `impl()`, `grad()`, `c_code()`
- Tensor wrapper: `@scalar_elemwise` decorator in `pytensor/tensor/math.py`

### What Exists in PyTensor

1. **Sigmoid** ✅ - Fully implemented
   - Scalar op: `pytensor/scalar/math.py:1200-1239`
   - Tensor function: `pytensor/tensor/math.py:403-407`
   - **Just needs**: ONNX mapping (add to `SCALAR_OP_TO_ONNX` dict)

2. **SiLU/Swish** ❌ - Does NOT exist
   - No scalar op definition
   - No tensor function
   - **Must create**: Complete implementation + ONNX converter

3. **BatchNorm** ❌ - Does NOT exist
   - Research document incorrectly stated `pytensor/tensor/nnet/bn.py` exists
   - No `pytensor/tensor/nnet/` directory
   - **Must create**: Op class + ONNX converter

### ONNX Target Specifications

**ONNX Opset 18**:

1. **Sigmoid** - [ONNX Spec](https://onnx.ai/onnx/operators/onnx__Sigmoid.html)
   - Inputs: X (tensor)
   - Outputs: Y (tensor)
   - Formula: Y = 1 / (1 + exp(-X))
   - Simple 1:1 mapping

2. **SiLU** - No direct ONNX operator
   - Must decompose to: `Mul(X, Sigmoid(X))`
   - Requires multi-node conversion
   - Formula: Y = X * sigmoid(X)

3. **BatchNormalization** - [ONNX Spec](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)
   - Inputs: X, scale (gamma), B (beta), input_mean, input_var
   - Attributes:
     - `epsilon` (float, default=1e-5)
     - `momentum` (float, default=0.9) - for training only
   - Outputs: Y (normalized tensor)
   - Formula: Y = scale * (X - mean) / sqrt(var + epsilon) + B

## Desired End State

After implementation:

1. **Sigmoid ONNX mapping**:
   - File: `pytensor/link/onnx/dispatch/elemwise.py` (MODIFY)
   - Add `scalar.Sigmoid: "Sigmoid"` to `SCALAR_OP_TO_ONNX` dict
   - Test file: `tests/link/onnx/test_elemwise.py` (MODIFY or NEW test)
   - ~5 unit tests + property-based tests

2. **SiLU op + converter**:
   - Files:
     - `pytensor/scalar/math.py` (MODIFY) - Add SiLU scalar op
     - `pytensor/tensor/math.py` (MODIFY) - Add tensor wrapper
     - `pytensor/link/onnx/dispatch/elemwise.py` (MODIFY) - Add multi-node converter
   - Test files:
     - `tests/scalar/test_math.py` (MODIFY) - Scalar op tests
     - `tests/tensor/test_math.py` (MODIFY) - Tensor function tests
     - `tests/link/onnx/test_elemwise.py` (MODIFY) - ONNX converter tests
   - ~12 unit tests + property-based tests

3. **BatchNorm op + converter**:
   - Files:
     - `pytensor/tensor/batchnorm.py` (NEW) - Op definition
     - `pytensor/link/onnx/dispatch/batchnorm.py` (NEW) - ONNX converter
   - Test files:
     - `tests/tensor/test_batchnorm.py` (NEW) - PyTensor op tests
     - `tests/link/onnx/test_batchnorm.py` (NEW) - ONNX converter tests
   - ~15 unit tests + property-based tests

**Success criteria:**
- All 3 operations export to valid ONNX
- Numerical results match PyTensor within 1e-4 tolerance
- C3k2 block (Conv → BatchNorm → SiLU) exports correctly
- All tests pass

## What We're NOT Implementing

**Out of scope for this plan:**

1. **Training mode**: Only inference (no running mean/var updates for BatchNorm)
2. **Other activations**: Tanh, GELU, etc. (can be added later)
3. **Fused operations**: BatchNorm + ReLU fusion (optimization for later)
4. **Gradients**: ONNX export only (no backward pass)
5. **Learnable BatchNorm**: Assuming scale/bias are fixed at export time
6. **Dynamic BatchNorm**: Only static mean/variance (computed during training)

## TDD Approach

Same as Tier 1 plan:
1. **Red**: Write tests first
2. **Verify failure**: Confirm tests fail appropriately
3. **Green**: Implement to pass tests
4. **Refactor**: Clean up while keeping tests green

---

## Operation 1: Sigmoid (ONNX Mapping)

### Phase 1: Test Design & Implementation

#### Overview
Sigmoid already exists in PyTensor - we only need to add ONNX mapping. This is the EASIEST operation in this plan.

**Current situation:**
- Scalar op exists: `pytensor/scalar/math.py:1200`
- Tensor function exists: `pytensor/tensor/math.py:403`
- ONNX mapping missing: Not in `SCALAR_OP_TO_ONNX` dict

#### Test Categories

##### Category 1: Basic Sigmoid Tests
**Test File**: `tests/link/onnx/test_elemwise.py` (MODIFY or CREATE)
**Purpose**: Verify Sigmoid exports to ONNX correctly

**Test 1: `test_sigmoid_basic`**
```python
def test_sigmoid_basic(tmp_path):
    """
    Test basic sigmoid activation exports to ONNX.

    This is the fundamental test - verifies:
    - Sigmoid scalar op is recognized by ONNX converter
    - Output matches PyTensor sigmoid
    - Numerical stability for positive and negative values

    Sigmoid formula: y = 1 / (1 + exp(-x))
    - Maps any value to (0, 1) range
    - Used in attention mechanisms and gates
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector("x", dtype="float32")

    # Apply sigmoid
    y = pt.sigmoid(x)

    # Test data covering different ranges
    x_val = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype="float32")

    # Expected (manual calculation):
    # sigmoid(-10) ≈ 0.0000454
    # sigmoid(-1) ≈ 0.268941
    # sigmoid(0) = 0.5
    # sigmoid(1) ≈ 0.731059
    # sigmoid(10) ≈ 0.9999546

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- Error type: `KeyError` or similar
- Expected message: Sigmoid not found in `SCALAR_OP_TO_ONNX` mapping
- Points to: `elemwise.py` converter trying to map Sigmoid

**Test 2: `test_sigmoid_matrix`**
```python
def test_sigmoid_matrix(tmp_path):
    """Test sigmoid on 2D matrix."""
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix("x", dtype="float32")
    y = pt.sigmoid(x)

    x_val = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 3: `test_sigmoid_4d_tensor`**
```python
def test_sigmoid_4d_tensor(tmp_path):
    """
    Test sigmoid on 4D tensor (CNN feature maps).

    Used in attention mechanisms like C2PSA in YOLO11n.
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pt.sigmoid(x)

    # Typical CNN feature map
    x_val = np.random.randn(2, 64, 16, 16).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 4: `test_sigmoid_numerical_stability`**
```python
def test_sigmoid_numerical_stability(tmp_path):
    """
    Test sigmoid with extreme values (numerical stability).

    Sigmoid should:
    - Not overflow for large positive values (→ 1.0)
    - Not underflow for large negative values (→ 0.0)
    - Handle values near zero correctly
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector("x", dtype="float32")
    y = pt.sigmoid(x)

    # Extreme values
    x_val = np.array([-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

##### Category 2: Integration Tests

**Test 5: `test_sigmoid_in_attention_pattern`**
```python
def test_sigmoid_in_attention_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Sigmoid in attention mechanism (C2PSA pattern).

    Attention pattern:
    1. Compute attention scores
    2. Apply sigmoid to get attention weights (0 to 1)
    3. Multiply features by attention weights

    This is how C2PSA blocks use sigmoid in YOLO11n.
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Feature maps
    features = pt.tensor4("features", dtype="float32")
    # Attention scores (computed by some network)
    attention_scores = pt.tensor4("attention_scores", dtype="float32")

    # Apply sigmoid to attention scores
    attention_weights = pt.sigmoid(attention_scores)

    # Weighted features
    weighted_features = features * attention_weights

    # Test data
    features_val = np.random.randn(1, 256, 20, 20).astype("float32")
    attention_scores_val = np.random.randn(1, 256, 20, 20).astype("float32")

    compare_onnx_and_py(
        [features, attention_scores],
        weighted_features,
        [features_val, attention_scores_val],
        tmp_path=tmp_path
    )
```

#### Property-Based Tests

**Strategy** (in `strategies/operations.py`):
```python
# Add to existing ONNX_OPERATIONS registry

ONNX_OPERATIONS["sigmoid"] = OperationConfig(
    op_func=pt.sigmoid,
    input_strategy=unary_operation_inputs(),  # Already exists
    valid_dtypes=["float32", "float64"],
    category="elemwise",
    notes="Logistic sigmoid activation",
)
```

**Property test** (already covered by `test_onnx_matches_pytensor` in `test_properties.py`):
- Will automatically test sigmoid once added to registry
- Tests across random valid inputs
- Verifies numerical correctness

#### Test Implementation Steps

1. **Create or modify test file**: `tests/link/onnx/test_elemwise.py`
   - File might already exist for other elemwise ops
   - Add sigmoid tests to existing file

2. **Implement 5 unit tests** (see test cases above)

3. **Add to property-based test registry**

4. **Run tests to verify failures**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py::test_sigmoid_basic -xvs
   ```

#### Success Criteria

##### Automated Verification:
- [ ] All 5 sigmoid tests fail with expected error (KeyError or NotImplementedError)
- [ ] Tests are discovered: `pytest --collect-only tests/link/onnx/test_elemwise.py`
- [ ] Property test added to registry

##### Manual Verification:
- [ ] Failure messages clearly indicate Sigmoid is not mapped to ONNX
- [ ] Test data covers edge cases (extreme values, different tensor ranks)
- [ ] Attention pattern test accurately represents C2PSA usage

---

### Phase 2: Test Failure Verification

#### Verification Steps

1. **Run sigmoid tests**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py -k sigmoid -v
   ```

2. **Expected failure**:
   ```
   KeyError: <class 'pytensor.scalar.math.Sigmoid'>

   Or:

   NotImplementedError: No ONNX conversion for Sigmoid scalar op
   ```

3. **Verify stack trace**:
   - Should point to `elemwise.py` in the ONNX dispatcher
   - Should show lookup in `SCALAR_OP_TO_ONNX` dict failing

#### Success Criteria

##### Automated Verification:
- [ ] All sigmoid tests fail predictably
- [ ] No import errors or syntax errors

##### Manual Verification:
- [ ] Failure mode is clear: Sigmoid exists but ONNX mapping doesn't
- [ ] Error message guides implementation (add to SCALAR_OP_TO_ONNX)

---

### Phase 3: Feature Implementation (Red → Green)

#### Implementation Strategy

**Single-line fix!** Just add Sigmoid to the ONNX mapping dictionary.

#### Implementation

**File**: `pytensor/link/onnx/dispatch/elemwise.py` (MODIFY)

**Current state** (lines 15-29):
```python
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Sqr: "Mul",  # Special handling: x^2 -> x * x
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.ScalarMaximum: "Max",
    scalar.ScalarMinimum: "Min",
}
```

**Modified** (ADD ONE LINE):
```python
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Sqr: "Mul",  # Special handling: x^2 -> x * x
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.ScalarMaximum: "Max",
    scalar.ScalarMinimum: "Min",
    scalar.Sigmoid: "Sigmoid",  # ADD THIS LINE
}
```

**That's it!** The existing `onnx_funcify_Elemwise` converter (lines 161-224) already handles scalar ops via the dictionary lookup.

#### Testing Progression

```bash
# Should now pass all sigmoid tests
pytest tests/link/onnx/test_elemwise.py -k sigmoid -v
```

#### Success Criteria

##### Automated Verification:
- [ ] All 5 sigmoid tests pass: `pytest tests/link/onnx/test_elemwise.py -k sigmoid -v`
- [ ] Property test passes: `pytest tests/link/onnx/test_properties.py -k sigmoid -v`
- [ ] No regressions: `pytest tests/link/onnx/test_elemwise.py -v`
- [ ] Attention pattern test passes (critical for C2PSA)

##### Manual Verification:
- [ ] Sigmoid output matches PyTensor within tolerance
- [ ] Numerical stability verified (extreme values)
- [ ] Integration with other ops (multiply) works correctly

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Add Tanh** (bonus, if time permits):
   ```python
   scalar.Tanh: "Tanh",  # Also missing, easy to add
   ```

2. **Documentation**:
   - Add comment explaining which activations are supported
   - List unsupported activations (GELU, etc.)

#### Refactoring Steps

1. **Add comment to SCALAR_OP_TO_ONNX dict**:
   ```python
   # Supported activation functions:
   # - Sigmoid: Logistic sigmoid (1 / (1 + exp(-x)))
   # - Tanh: Hyperbolic tangent
   # - ReLU: Via ScalarMaximum pattern (pt.maximum(x, 0))
   #
   # Not yet supported:
   # - GELU, SiLU (requires multi-node decomposition)
   ```

2. **Optionally add Tanh** (same as Sigmoid):
   ```python
   scalar.Tanh: "Tanh",
   ```

3. **Run tests after refactoring**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py -v
   ```

#### Success Criteria

##### Automated Verification:
- [ ] All tests still pass
- [ ] Code is well-documented

##### Manual Verification:
- [ ] Dictionary is organized and readable
- [ ] Comments clearly explain supported operations

---

## Operation 2: SiLU/Swish Activation

### Phase 1: Test Design & Implementation

#### Overview
SiLU (Sigmoid Linear Unit), also known as Swish, doesn't exist in PyTensor. We need to:
1. Create scalar op in `pytensor/scalar/math.py`
2. Create tensor wrapper in `pytensor/tensor/math.py`
3. Write PyTensor op tests
4. Create ONNX converter with multi-node decomposition
5. Write ONNX converter tests

**SiLU formula**: `y = x * sigmoid(x)`

**ONNX decomposition**: Two nodes (Sigmoid → Mul)

#### Test Categories

##### Category 1: PyTensor Scalar Op Tests
**Test File**: `tests/scalar/test_math.py` (MODIFY)
**Purpose**: Verify SiLU scalar op works correctly in PyTensor

**Test 1: `test_silu_scalar_basic`**
```python
def test_silu_scalar_basic():
    """
    Test basic SiLU scalar operation.

    SiLU formula: y = x * sigmoid(x)
              = x / (1 + exp(-x))

    Properties:
    - Non-monotonic (has a minimum around x = -1.278)
    - Smooth everywhere (differentiable)
    - Range: approximately (-0.278, ∞)
    - Superior to ReLU for deep networks
    """
    import pytensor.scalar as ps
    from pytensor.scalar.math import silu

    x = ps.float32("x")
    y = silu(x)

    # Compile scalar function
    f = pytensor.function([x], y)

    # Test values
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for x_val in test_values:
        result = f(x_val)
        # Manual calculation: x * sigmoid(x)
        sigmoid_x = 1.0 / (1.0 + np.exp(-x_val))
        expected = x_val * sigmoid_x

        np.testing.assert_allclose(result, expected, rtol=1e-5)
```

**Expected Failure Mode**:
- `AttributeError: module 'pytensor.scalar.math' has no attribute 'silu'`

**Test 2: `test_silu_scalar_gradient`**
```python
def test_silu_scalar_gradient():
    """
    Test SiLU gradient computation.

    SiLU gradient: dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    This test verifies automatic differentiation works correctly.
    """
    import pytensor.scalar as ps
    from pytensor.scalar.math import silu
    from pytensor.gradient import grad

    x = ps.float32("x")
    y = silu(x)

    # Compute gradient
    dy_dx = grad(y, x)

    # Compile
    f_grad = pytensor.function([x], dy_dx)

    # Test gradient at x = 1.0
    x_val = 1.0
    grad_result = f_grad(x_val)

    # Manual calculation
    sigmoid_x = 1.0 / (1.0 + np.exp(-x_val))
    expected_grad = sigmoid_x * (1 + x_val * (1 - sigmoid_x))

    np.testing.assert_allclose(grad_result, expected_grad, rtol=1e-5)
```

**Test 3: `test_silu_scalar_edge_cases`**
```python
def test_silu_scalar_edge_cases():
    """Test SiLU with edge cases (extreme values)."""
    import pytensor.scalar as ps
    from pytensor.scalar.math import silu

    x = ps.float32("x")
    y = silu(x)

    f = pytensor.function([x], y)

    # Edge cases
    assert np.isfinite(f(-100.0))  # Large negative
    assert np.isfinite(f(100.0))   # Large positive
    assert f(0.0) == 0.0           # Zero input
```

##### Category 2: PyTensor Tensor Function Tests
**Test File**: `tests/tensor/test_math.py` (MODIFY)
**Purpose**: Verify SiLU tensor function works on multi-dimensional tensors

**Test 4: `test_silu_vector`**
```python
def test_silu_vector():
    """Test SiLU on 1D vector."""
    import pytensor.tensor as pt

    x = pt.vector("x", dtype="float32")
    y = pt.silu(x)

    f = pytensor.function([x], y)

    x_val = np.array([-2, -1, 0, 1, 2], dtype="float32")
    result = f(x_val)

    # Manual calculation
    sigmoid_x = 1.0 / (1.0 + np.exp(-x_val))
    expected = x_val * sigmoid_x

    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

**Test 5: `test_silu_4d_tensor`**
```python
def test_silu_4d_tensor():
    """
    Test SiLU on 4D CNN feature maps.

    This is how SiLU is used in YOLO11n - applied element-wise
    to feature maps after convolution and batch normalization.
    """
    import pytensor.tensor as pt

    x = pt.tensor4("x", dtype="float32")
    y = pt.silu(x)

    f = pytensor.function([x], y)

    # Typical CNN feature map
    x_val = np.random.randn(2, 64, 16, 16).astype("float32")
    result = f(x_val)

    # Manual calculation
    sigmoid_x = 1.0 / (1.0 + np.exp(-x_val))
    expected = x_val * sigmoid_x

    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

##### Category 3: ONNX Conversion Tests
**Test File**: `tests/link/onnx/test_elemwise.py` (MODIFY)
**Purpose**: Verify SiLU exports to ONNX with correct multi-node decomposition

**Test 6: `test_silu_onnx_basic`**
```python
def test_silu_onnx_basic(tmp_path):
    """
    Test SiLU exports to ONNX correctly.

    ONNX doesn't have a native SiLU operator (as of opset 18).
    We decompose to: Mul(X, Sigmoid(X))

    This creates 2 ONNX nodes:
    1. Sigmoid(X) → temp
    2. Mul(X, temp) → Y
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector("x", dtype="float32")
    y = pt.silu(x)

    x_val = np.array([-2, -1, 0, 1, 2], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Expected Failure Mode** (after PyTensor op exists):
- `NotImplementedError: No ONNX conversion for SiLU`

**Test 7: `test_silu_onnx_matrix`**
```python
def test_silu_onnx_matrix(tmp_path):
    """Test SiLU ONNX export on 2D matrix."""
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix("x", dtype="float32")
    y = pt.silu(x)

    x_val = np.random.randn(10, 20).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 8: `test_silu_onnx_4d_tensor`**
```python
def test_silu_onnx_4d_tensor(tmp_path):
    """Test SiLU ONNX export on 4D CNN feature maps."""
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pt.silu(x)

    x_val = np.random.randn(2, 64, 16, 16).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

##### Category 4: Integration Tests

**Test 9: `test_silu_in_c3k2_pattern`**
```python
def test_silu_in_c3k2_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: SiLU in C3k2 block pattern from YOLO11n.

    C3k2 pattern (simplified):
    1. Conv2D
    2. BatchNorm (will test once BatchNorm is implemented)
    3. SiLU activation ← This is what we're testing
    4. Output

    For this test, we simulate without BatchNorm:
    Conv → SiLU
    """
    import pytensor.tensor as pt
    from pytensor.tensor.conv import conv2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Conv2D
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    # SiLU activation (YOLO11n uses this instead of ReLU)
    activated = pt.silu(conv_out)

    # Test data
    x_val = np.random.randn(1, 3, 10, 10).astype("float32")
    kernel_val = np.random.randn(16, 3, 3, 3).astype("float32")

    compare_onnx_and_py(
        [x, kernel],
        activated,
        [x_val, kernel_val],
        tmp_path=tmp_path
    )
```

**Test 10: `test_silu_numerical_stability`**
```python
def test_silu_numerical_stability(tmp_path):
    """
    Test SiLU with extreme values.

    SiLU should be numerically stable:
    - Large positive: x * 1 ≈ x
    - Large negative: x * 0 ≈ 0
    - Zero: 0 * 0.5 = 0
    """
    import pytensor.tensor as pt
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector("x", dtype="float32")
    y = pt.silu(x)

    x_val = np.array([-100, -50, -10, 0, 10, 50, 100], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

#### Property-Based Tests

**Strategy** (in `strategies/operations.py`):
```python
ONNX_OPERATIONS["silu"] = OperationConfig(
    op_func=pt.silu,
    input_strategy=unary_operation_inputs(),
    valid_dtypes=["float32", "float64"],
    category="elemwise",
    notes="SiLU/Swish activation: x * sigmoid(x)",
)
```

#### Test Implementation Steps

1. **Create scalar tests**: Modify `tests/scalar/test_math.py` (3 tests)
2. **Create tensor tests**: Modify `tests/tensor/test_math.py` (2 tests)
3. **Create ONNX tests**: Modify `tests/link/onnx/test_elemwise.py` (5 tests)
4. **Add to property registry**
5. **Run tests to verify failures**

#### Success Criteria

##### Automated Verification:
- [ ] Scalar tests fail: `AttributeError: no attribute 'silu'`
- [ ] Tensor tests fail: `AttributeError: no attribute 'silu'`
- [ ] ONNX tests fail: `NotImplementedError` (after PyTensor op exists)

##### Manual Verification:
- [ ] Test progression logical (scalar → tensor → ONNX)
- [ ] C3k2 pattern test represents real YOLO11n usage
- [ ] Gradient test verifies automatic differentiation

---

### Phase 2: Test Failure Verification

Same process - verify tests fail appropriately at each stage.

---

### Phase 3: Feature Implementation (Red → Green)

#### Phase 3A: PyTensor SiLU Scalar Op

**File**: `pytensor/scalar/math.py` (MODIFY)

**Add after Softplus** (around line 1320):

```python
class SiLU(UnaryScalarOp):
    """
    SiLU (Sigmoid Linear Unit) activation function.

    Also known as Swish activation.

    Formula: y = x * sigmoid(x) = x / (1 + exp(-x))

    Properties:
    - Smooth and non-monotonic
    - Self-gated (gates input with its own sigmoid)
    - Superior to ReLU for deep networks
    - Used in modern architectures (EfficientNet, YOLO11n, etc.)

    References
    ----------
    .. [1] Ramachandran et al., "Searching for Activation Functions", 2017
           https://arxiv.org/abs/1710.05941
    """

    nfunc_spec = None  # No direct NumPy equivalent

    def impl(self, x):
        """Python/NumPy implementation of SiLU."""
        # Handle int8/uint8 to avoid float16 computation
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            x = np.asarray(x, dtype=np.float32)

        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        # Use numerically stable implementation
        return x / (1.0 + np.exp(-x))

    def grad(self, inp, grads):
        """
        Gradient of SiLU.

        d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        (x,) = inp
        (gz,) = grads

        sig_x = sigmoid(x)
        # Gradient: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        rval = gz * sig_x * (1 + x * (1 - sig_x))

        assert rval.type.dtype.find("float") != -1
        return [rval]

    def c_code(self, node, name, inp, out, sub):
        """C implementation of SiLU."""
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            # SiLU: x / (1 + exp(-x))
            if node.inputs[0].type == float64:
                return f"""
                {z} = {x} / (1.0 + exp(-{x}));
                """
            else:  # float32
                return f"""
                {z} = {x} / (1.0f + expf(-{x}));
                """
        else:
            raise NotImplementedError("SiLU only implemented for floating point")

    def c_code_cache_version(self):
        """Version for C code caching."""
        v = super().c_code_cache_version()
        if v:
            return (1, *v)
        else:
            return v


# Create instance
silu = SiLU(upgrade_to_float, name="silu")
```

**Export in** `pytensor/scalar/__init__.py`:
```python
from pytensor.scalar.math import silu  # ADD THIS LINE
```

#### Phase 3B: PyTensor SiLU Tensor Wrapper

**File**: `pytensor/tensor/math.py` (MODIFY)

**Add after sigmoid** (around line 2460):

```python
@scalar_elemwise
def silu(x):
    """
    SiLU (Sigmoid Linear Unit) activation function.

    Also known as Swish activation.

    Formula: y = x * sigmoid(x)

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.silu(x)

    Notes
    -----
    SiLU is used in modern CNN architectures as a replacement for ReLU:
    - Smooth and differentiable everywhere
    - Self-gated (input modulates itself)
    - Better gradient flow than ReLU
    - Used in YOLO11n, EfficientNet, and other modern models

    References
    ----------
    .. [1] Ramachandran et al., "Searching for Activation Functions", 2017
    """
    pass  # Implementation provided by @scalar_elemwise decorator


# Alias for Swish (same function, different name)
swish = silu
```

**Export in** `pytensor/tensor/__init__.py`:
```python
from pytensor.tensor.math import silu, swish  # ADD THIS LINE
```

**Testing progression for Phase 3A & 3B**:
```bash
# Should now pass PyTensor tests
pytest tests/scalar/test_math.py -k silu -v
pytest tests/tensor/test_math.py -k silu -v
```

#### Phase 3C: ONNX SiLU Converter

**File**: `pytensor/link/onnx/dispatch/elemwise.py` (MODIFY)

**Add converter after existing converters** (around line 225):

```python
@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node(s)."""

    # ... existing code ...

    # Check if this is a SiLU operation
    from pytensor.scalar.math import SiLU as ScalarSiLU

    if isinstance(scalar_op, ScalarSiLU):
        # SiLU requires multi-node decomposition: x * sigmoid(x)
        return onnx_funcify_SiLU_elemwise(op, node, var_names, get_var_name, **kwargs)

    # ... rest of existing code ...


def onnx_funcify_SiLU_elemwise(op, node, var_names, get_var_name, **kwargs):
    """
    Convert SiLU Elemwise to ONNX multi-node decomposition.

    SiLU(x) = x * sigmoid(x)

    ONNX decomposition:
    1. Sigmoid(x) → temp
    2. Mul(x, temp) → output

    Parameters
    ----------
    op : Elemwise
        Elemwise op with SiLU scalar op
    node : Apply
        Apply node
    var_names : dict
        Variable name mapping
    get_var_name : callable
        Name generator

    Returns
    -------
    list of onnx.NodeProto
        Two ONNX nodes (Sigmoid and Mul)
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    input_name = input_names[0]
    output_name = output_names[0]

    # Create intermediate name for sigmoid output
    sigmoid_out = f"silu_sigmoid_{output_name}"

    nodes = []

    # Node 1: Sigmoid(x)
    nodes.append(
        helper.make_node(
            "Sigmoid",
            inputs=[input_name],
            outputs=[sigmoid_out],
            name=f"Sigmoid_{output_name}",
        )
    )

    # Node 2: Mul(x, sigmoid(x))
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[input_name, sigmoid_out],
            outputs=[output_name],
            name=f"Mul_{output_name}",
        )
    )

    return nodes
```

**Testing progression for Phase 3C**:
```bash
# Should now pass ONNX tests
pytest tests/link/onnx/test_elemwise.py -k silu -v
```

#### Success Criteria

##### Automated Verification:
- [ ] All 10 SiLU tests pass
- [ ] Scalar op tests pass: Correct implementation and gradient
- [ ] Tensor tests pass: Works on multi-dimensional tensors
- [ ] ONNX tests pass: Correct multi-node decomposition
- [ ] C3k2 pattern test passes (critical for YOLO11n)
- [ ] Property-based tests pass

##### Manual Verification:
- [ ] SiLU produces correct values (verified against manual calculation)
- [ ] Gradient is correct (verified against analytical formula)
- [ ] ONNX export creates 2 nodes (Sigmoid + Mul)
- [ ] Numerical stability for extreme values

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Documentation**:
   - [ ] Add more examples to SiLU docstring
   - [ ] Document use in modern architectures

2. **Alternative implementation** (optional):
   - [ ] Consider: Is `x / (1 + exp(-x))` more stable than `x * sigmoid(x)`?
   - [ ] Benchmark both implementations

3. **Test quality**:
   - [ ] Add comparison with PyTorch SiLU (if available)

#### Success Criteria

Same as before - all tests pass, code is clean and well-documented.

---

## Operation 3: Batch Normalization

### Phase 1: Test Design & Implementation

#### Overview
Batch Normalization (BatchNorm) doesn't exist in PyTensor. We need to:
1. Create PyTensor BatchNorm op in `pytensor/tensor/batchnorm.py` (NEW)
2. Write PyTensor op tests
3. Create ONNX converter
4. Write ONNX converter tests

**BatchNorm formula** (inference mode):
```
y = scale * (x - mean) / sqrt(var + epsilon) + bias
```

Where:
- `x`: Input tensor
- `mean`: Pre-computed mean (from training)
- `var`: Pre-computed variance (from training)
- `scale` (gamma): Learnable scale parameter
- `bias` (beta): Learnable bias parameter
- `epsilon`: Small constant for numerical stability (typically 1e-5)

**Note**: We're only implementing inference mode (not training with running mean/var updates).

#### Test Categories

##### Category 1: PyTensor Op Tests
**Test File**: `tests/tensor/test_batchnorm.py` (NEW)
**Purpose**: Verify BatchNorm op works correctly in PyTensor

**Test 1: `test_batchnorm_basic`**
```python
def test_batchnorm_basic():
    """
    Test basic batch normalization in inference mode.

    BatchNorm formula (inference):
    y = scale * (x - mean) / sqrt(var + epsilon) + bias

    Configuration:
    - 4D input (NCHW): (batch, channels, height, width)
    - Per-channel normalization
    - Pre-computed mean and variance
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization

    # Input
    x = pt.tensor4("x", dtype="float32")

    # Per-channel statistics (for C channels)
    scale = pt.vector("scale", dtype="float32")  # gamma
    bias = pt.vector("bias", dtype="float32")    # beta
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    # Batch normalization
    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    # Compile
    f = pytensor.function([x, scale, bias, mean, var], y)

    # Test data: 2 channels
    x_val = np.array([[[[1, 2], [3, 4]],      # Channel 0
                       [[5, 6], [7, 8]]]],    # Channel 1
                     dtype="float32")  # Shape: (1, 2, 2, 2)

    scale_val = np.array([1.0, 1.0], dtype="float32")
    bias_val = np.array([0.0, 0.0], dtype="float32")
    mean_val = np.array([2.5, 6.5], dtype="float32")  # Mean of each channel
    var_val = np.array([1.25, 1.25], dtype="float32")  # Var of each channel

    result = f(x_val, scale_val, bias_val, mean_val, var_val)

    # Manual calculation for channel 0:
    # x_ch0 = [1, 2, 3, 4], mean = 2.5, var = 1.25
    # Normalized: (x - 2.5) / sqrt(1.25 + 1e-5)
    # = (x - 2.5) / 1.118...

    # Verify shape
    assert result.shape == x_val.shape
```

**Expected Failure Mode**:
- `ImportError: cannot import name 'batch_normalization'`

**Test 2: `test_batchnorm_with_scale_bias`**
```python
def test_batchnorm_with_scale_bias():
    """
    Test BatchNorm with non-identity scale and bias.

    This tests the full formula:
    y = scale * normalized + bias
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    f = pytensor.function([x, scale, bias, mean, var], y)

    # Test with specific scale and bias
    x_val = np.random.randn(2, 3, 4, 4).astype("float32")
    scale_val = np.array([0.5, 1.0, 2.0], dtype="float32")
    bias_val = np.array([0.1, 0.2, 0.3], dtype="float32")
    mean_val = np.array([0.0, 0.0, 0.0], dtype="float32")
    var_val = np.array([1.0, 1.0, 1.0], dtype="float32")

    result = f(x_val, scale_val, bias_val, mean_val, var_val)

    # Manual verification for channel 0
    normalized_ch0 = (x_val[:, 0, :, :] - 0.0) / np.sqrt(1.0 + 1e-5)
    expected_ch0 = 0.5 * normalized_ch0 + 0.1

    np.testing.assert_allclose(result[:, 0, :, :], expected_ch0, rtol=1e-5)
```

**Test 3: `test_batchnorm_multiple_batches`**
```python
def test_batchnorm_multiple_batches():
    """
    Test BatchNorm with multiple batches.

    BatchNorm normalizes each channel independently,
    but processes all batches simultaneously.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    f = pytensor.function([x, scale, bias, mean, var], y)

    # Multiple batches
    batch_size = 8
    channels = 16
    x_val = np.random.randn(batch_size, channels, 8, 8).astype("float32")

    scale_val = np.ones(channels, dtype="float32")
    bias_val = np.zeros(channels, dtype="float32")
    mean_val = np.zeros(channels, dtype="float32")
    var_val = np.ones(channels, dtype="float32")

    result = f(x_val, scale_val, bias_val, mean_val, var_val)

    assert result.shape == x_val.shape
```

##### Category 2: ONNX Conversion Tests
**Test File**: `tests/link/onnx/test_batchnorm.py` (NEW)

**Test 4: `test_batchnorm_onnx_basic`**
```python
def test_batchnorm_onnx_basic(tmp_path):
    """
    Test BatchNorm exports to ONNX correctly.

    ONNX BatchNormalization operator:
    - Inputs: X, scale, B, input_mean, input_var
    - Attributes: epsilon, momentum (training only)
    - Output: Y (normalized tensor)
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    # Test data
    x_val = np.random.randn(2, 3, 8, 8).astype("float32")
    scale_val = np.ones(3, dtype="float32")
    bias_val = np.zeros(3, dtype="float32")
    mean_val = np.zeros(3, dtype="float32")
    var_val = np.ones(3, dtype="float32")

    compare_onnx_and_py(
        [x, scale, bias, mean, var],
        y,
        [x_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

**Expected Failure Mode**:
- `NotImplementedError: No ONNX conversion for BatchNormalization`

**Test 5: `test_batchnorm_onnx_pretrained_weights`**
```python
def test_batchnorm_onnx_pretrained_weights(tmp_path):
    """
    Test BatchNorm with realistic pre-trained weights.

    Simulates a BatchNorm layer from a trained CNN:
    - Non-zero mean and variance (learned during training)
    - Scale and bias learned during training
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    # Realistic pre-trained weights
    channels = 64
    x_val = np.random.randn(1, channels, 16, 16).astype("float32")

    # Realistic learned parameters
    scale_val = np.random.uniform(0.8, 1.2, channels).astype("float32")
    bias_val = np.random.uniform(-0.1, 0.1, channels).astype("float32")
    mean_val = np.random.uniform(-0.5, 0.5, channels).astype("float32")
    var_val = np.random.uniform(0.5, 2.0, channels).astype("float32")

    compare_onnx_and_py(
        [x, scale, bias, mean, var],
        y,
        [x_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

**Test 6: `test_batchnorm_onnx_different_epsilon`**
```python
def test_batchnorm_onnx_different_epsilon(tmp_path):
    """
    Test BatchNorm with different epsilon values.

    Epsilon affects numerical stability - verify ONNX correctly
    passes this attribute.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    # Use larger epsilon
    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-3)

    x_val = np.random.randn(2, 8, 4, 4).astype("float32")
    scale_val = np.ones(8, dtype="float32")
    bias_val = np.zeros(8, dtype="float32")
    mean_val = np.zeros(8, dtype="float32")
    var_val = np.ones(8, dtype="float32")

    compare_onnx_and_py(
        [x, scale, bias, mean, var],
        y,
        [x_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

##### Category 3: Integration Tests

**Test 7: `test_batchnorm_onnx_after_conv`**
```python
def test_batchnorm_onnx_after_conv(tmp_path):
    """
    Test Conv2D → BatchNorm pattern (standard CNN layer).

    This is how BatchNorm is used in practice:
    1. Convolution
    2. Batch Normalization
    3. Activation (will add SiLU once implemented)
    """
    import pytensor.tensor as pt
    from pytensor.tensor.conv import conv2d
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    # Conv2D
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    # BatchNorm
    bn_out = batch_normalization(conv_out, scale, bias, mean, var, epsilon=1e-5)

    # Test data
    x_val = np.random.randn(1, 3, 10, 10).astype("float32")
    kernel_val = np.random.randn(16, 3, 3, 3).astype("float32")

    # BatchNorm parameters for 16 output channels
    scale_val = np.ones(16, dtype="float32")
    bias_val = np.zeros(16, dtype="float32")
    mean_val = np.zeros(16, dtype="float32")
    var_val = np.ones(16, dtype="float32")

    compare_onnx_and_py(
        [x, kernel, scale, bias, mean, var],
        bn_out,
        [x_val, kernel_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

**Test 8: `test_batchnorm_conv_silu_full_c3k2_layer`**
```python
def test_batchnorm_conv_silu_full_c3k2_layer(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Full C3k2 layer pattern from YOLO11n.

    Complete layer:
    1. Conv2D
    2. BatchNorm
    3. SiLU activation

    This is the exact pattern used in every C3k2 block in YOLO11n.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.conv import conv2d
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    # Conv2D
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    # BatchNorm
    bn_out = batch_normalization(conv_out, scale, bias, mean, var, epsilon=1e-5)

    # SiLU activation (requires SiLU to be implemented)
    activated = pt.silu(bn_out)

    # YOLO11n typical dimensions
    x_val = np.random.randn(1, 256, 20, 20).astype("float32")
    kernel_val = np.random.randn(512, 256, 3, 3).astype("float32")

    scale_val = np.ones(512, dtype="float32")
    bias_val = np.zeros(512, dtype="float32")
    mean_val = np.zeros(512, dtype="float32")
    var_val = np.ones(512, dtype="float32")

    compare_onnx_and_py(
        [x, kernel, scale, bias, mean, var],
        activated,
        [x_val, kernel_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

**Test 9: `test_batchnorm_numerical_stability`**
```python
def test_batchnorm_numerical_stability(tmp_path):
    """
    Test BatchNorm with small variance (numerical stability).

    When variance is very small, the division (x - mean) / sqrt(var)
    could cause numerical issues. Epsilon prevents division by zero.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    scale = pt.vector("scale", dtype="float32")
    bias = pt.vector("bias", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    var = pt.vector("var", dtype="float32")

    y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    x_val = np.random.randn(1, 3, 8, 8).astype("float32")
    scale_val = np.ones(3, dtype="float32")
    bias_val = np.zeros(3, dtype="float32")
    mean_val = np.zeros(3, dtype="float32")

    # Very small variance (tests epsilon effectiveness)
    var_val = np.array([1e-10, 1e-8, 1e-6], dtype="float32")

    compare_onnx_and_py(
        [x, scale, bias, mean, var],
        y,
        [x_val, scale_val, bias_val, mean_val, var_val],
        tmp_path=tmp_path
    )
```

#### Property-Based Tests

**Strategy**:
```python
@st.composite
def batchnorm_inputs(draw):
    """Generate valid inputs for BatchNorm."""
    # Input shape (NCHW)
    batch = draw(st.integers(1, 4))
    channels = draw(st.integers(1, 16))
    height = draw(st.integers(4, 20))
    width = draw(st.integers(4, 20))

    # Generate tensors
    x = draw(onnx_tensor(dtype=np.float32, shape=(batch, channels, height, width)))

    # Per-channel parameters
    scale = draw(onnx_tensor(dtype=np.float32, shape=(channels,)))
    bias = draw(onnx_tensor(dtype=np.float32, shape=(channels,)))
    mean = draw(onnx_tensor(dtype=np.float32, shape=(channels,)))

    # Variance must be positive
    var = np.abs(draw(onnx_tensor(dtype=np.float32, shape=(channels,)))) + 0.1

    return (x, scale, bias, mean, var)
```

#### Test Implementation Steps

1. **Create PyTensor op test file**: `tests/tensor/test_batchnorm.py` (3 tests)
2. **Create ONNX converter test file**: `tests/link/onnx/test_batchnorm.py` (6 tests)
3. **Run tests to verify failures**

#### Success Criteria

##### Automated Verification:
- [ ] All 9 tests fail with expected errors
- [ ] Full C3k2 layer test represents real YOLO11n usage

##### Manual Verification:
- [ ] Test progression is logical (PyTensor op → ONNX)
- [ ] Integration tests cover Conv → BatchNorm → SiLU pipeline

---

### Phase 2: Test Failure Verification

Same process - verify appropriate failures at each stage.

---

### Phase 3: Feature Implementation (Red → Green)

#### Phase 3A: PyTensor BatchNorm Op

**File**: `pytensor/tensor/batchnorm.py` (NEW)

```python
"""Batch Normalization operations for PyTensor."""

import numpy as np
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType
from pytensor.graph.basic import Apply
import pytensor.tensor as pt


class BatchNormalization(Op):
    """
    Batch Normalization operation (inference mode).

    Normalizes input by channel using pre-computed statistics.

    Formula:
        y = scale * (x - mean) / sqrt(var + epsilon) + bias

    Parameters
    ----------
    epsilon : float
        Small constant added to variance for numerical stability.
        Default: 1e-5

    Notes
    -----
    This implementation is for inference only (no training mode).
    Mean and variance are assumed to be pre-computed from training.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x")  # (batch, channels, height, width)
    >>> scale = pt.vector("scale")  # (channels,)
    >>> bias = pt.vector("bias")  # (channels,)
    >>> mean = pt.vector("mean")  # (channels,)
    >>> var = pt.vector("var")  # (channels,)
    >>> y = batch_normalization(x, scale, bias, mean, var)
    """

    __props__ = ("epsilon",)

    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def make_node(self, x, scale, bias, mean, var):
        """Create an Apply node for this operation."""
        x = pt.as_tensor_variable(x)
        scale = pt.as_tensor_variable(scale)
        bias = pt.as_tensor_variable(bias)
        mean = pt.as_tensor_variable(mean)
        var = pt.as_tensor_variable(var)

        # Validate input
        if x.type.ndim != 4:
            raise ValueError(
                f"BatchNormalization requires 4D input (NCHW format), "
                f"got {x.type.ndim}D tensor"
            )

        if scale.type.ndim != 1 or bias.type.ndim != 1 or mean.type.ndim != 1 or var.type.ndim != 1:
            raise ValueError(
                "scale, bias, mean, and var must be 1D vectors (per-channel)"
            )

        # Output has same type as input
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x, scale, bias, mean, var], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute batch normalization using NumPy."""
        x, scale, bias, mean, var = inputs

        # Normalize: (x - mean) / sqrt(var + epsilon)
        # Broadcasting: scale, bias, mean, var are (C,), x is (N, C, H, W)
        # Need to reshape to (1, C, 1, 1) for broadcasting

        # Reshape per-channel parameters for broadcasting
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)

        # Batch normalization formula
        normalized = (x - mean) / np.sqrt(var + self.epsilon)
        result = scale * normalized + bias

        output_storage[0][0] = result.astype(x.dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        """Output shape is same as input shape."""
        return [input_shapes[0]]


def batch_normalization(input, scale, bias, mean, var, epsilon=1e-5):
    """
    Apply batch normalization to a 4D tensor (inference mode).

    Parameters
    ----------
    input : TensorVariable
        4D tensor in NCHW format (batch, channels, height, width)
    scale : TensorVariable
        1D tensor of scale parameters (gamma), shape (channels,)
    bias : TensorVariable
        1D tensor of bias parameters (beta), shape (channels,)
    mean : TensorVariable
        1D tensor of pre-computed mean, shape (channels,)
    var : TensorVariable
        1D tensor of pre-computed variance, shape (channels,)
    epsilon : float, optional
        Small constant for numerical stability. Default: 1e-5

    Returns
    -------
    TensorVariable
        Normalized tensor, same shape as input

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x", dtype="float32")
    >>> scale = pt.vector("scale", dtype="float32")
    >>> bias = pt.vector("bias", dtype="float32")
    >>> mean = pt.vector("mean", dtype="float32")
    >>> var = pt.vector("var", dtype="float32")
    >>> y = batch_normalization(x, scale, bias, mean, var, epsilon=1e-5)

    Notes
    -----
    This is inference-mode batch normalization:
    - Mean and variance are pre-computed (frozen from training)
    - No running statistics updates
    - No learnable parameters (scale/bias are inputs)

    In typical usage (e.g., YOLO11n):
    - scale and bias are learned during training
    - mean and var are computed as moving averages during training
    - At inference, all four parameters are fixed
    """
    return BatchNormalization(epsilon=epsilon)(input, scale, bias, mean, var)
```

**Export**:

**File**: `pytensor/tensor/__init__.py` (MODIFY)

```python
from pytensor.tensor.batchnorm import batch_normalization  # ADD THIS LINE
```

**Testing progression**:
```bash
# Should pass PyTensor op tests
pytest tests/tensor/test_batchnorm.py -v
```

#### Phase 3B: ONNX BatchNorm Converter

**File**: `pytensor/link/onnx/dispatch/batchnorm.py` (NEW)

```python
"""ONNX conversion for batch normalization operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.batchnorm import BatchNormalization

from onnx import helper


@onnx_funcify.register(BatchNormalization)
def onnx_funcify_BatchNormalization(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor BatchNormalization op to ONNX BatchNormalization node.

    ONNX BatchNormalization operator:
    - Inputs: X, scale, B, input_mean, input_var
    - Attributes: epsilon, momentum (training only)
    - Outputs: Y

    Formula (same as PyTensor):
        Y = scale * (X - input_mean) / sqrt(input_var + epsilon) + B

    Parameters
    ----------
    op : BatchNormalization
        The BatchNormalization operation instance
    node : Apply
        The apply node
    var_names : dict
        Variable name mapping
    get_var_name : callable
        Name generator

    Returns
    -------
    onnx.NodeProto
        ONNX BatchNormalization node

    Notes
    -----
    ONNX BatchNormalization has optional outputs (running_mean, running_var)
    for training mode, but we only use inference mode, so we ignore those.

    PyTensor input order: [x, scale, bias, mean, var]
    ONNX input order: [X, scale, B, input_mean, input_var]
    (Same order, different names)
    """
    # Get input names
    # node.inputs = [x, scale, bias, mean, var]
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract epsilon
    epsilon = op.epsilon

    # Create ONNX BatchNormalization node
    return helper.make_node(
        "BatchNormalization",
        inputs=input_names,  # [X, scale, B, input_mean, input_var]
        outputs=output_names,
        epsilon=epsilon,
        name=f"BatchNormalization_{output_names[0]}",
    )
```

**Import registration**:

**File**: `pytensor/link/onnx/dispatch/__init__.py` (MODIFY)

```python
import pytensor.link.onnx.dispatch.batchnorm  # noqa: F401  # ADD THIS LINE
```

**Testing progression**:
```bash
# Should pass ONNX converter tests
pytest tests/link/onnx/test_batchnorm.py -v
```

#### Success Criteria

##### Automated Verification:
- [ ] All 9 BatchNorm tests pass
- [ ] PyTensor op produces correct output
- [ ] ONNX converter exports correctly
- [ ] Full C3k2 layer test passes (Conv → BatchNorm → SiLU)
- [ ] Property-based tests pass

##### Manual Verification:
- [ ] BatchNorm formula implemented correctly
- [ ] Epsilon parameter passed to ONNX correctly
- [ ] Integration with Conv2D and SiLU works

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Add 2D/3D BatchNorm** (optional):
   - Current: Only 4D (NCHW for images)
   - Could add: 2D (NC for fully connected), 3D (NCDHW for video)

2. **Performance**:
   - [ ] Current implementation uses NumPy (reasonable performance)
   - [ ] Consider: C code implementation for speed

3. **Documentation**:
   - [ ] Add more examples showing typical usage
   - [ ] Document relationship between training and inference modes

#### Success Criteria

Same as before - all tests pass, code is maintainable.

---

## Testing Strategy Summary

### Test Coverage Goals

**Operation 1: Sigmoid**
- [ ] Basic sigmoid (different tensor ranks)
- [ ] Numerical stability (extreme values)
- [ ] Integration with attention mechanisms (C2PSA pattern)
- [ ] Property-based testing

**Operation 2: SiLU**
- [ ] Scalar op (implementation, gradient, edge cases)
- [ ] Tensor function (different ranks)
- [ ] ONNX multi-node decomposition (Sigmoid + Mul)
- [ ] Integration with Conv2D (C3k2 pattern)
- [ ] Numerical stability
- [ ] Property-based testing

**Operation 3: BatchNorm**
- [ ] Basic normalization
- [ ] With scale and bias
- [ ] Multiple batches
- [ ] ONNX export
- [ ] Pre-trained weights (realistic scenario)
- [ ] Different epsilon values
- [ ] Integration with Conv2D
- [ ] Full C3k2 layer (Conv → BatchNorm → SiLU)
- [ ] Numerical stability (small variance)
- [ ] Property-based testing

### Test Organization

```
tests/
├── scalar/
│   └── test_math.py          # SiLU scalar op tests (MODIFY)
├── tensor/
│   ├── test_math.py          # SiLU tensor tests (MODIFY)
│   └── test_batchnorm.py     # BatchNorm op tests (NEW)
├── link/
│   └── onnx/
│       ├── test_elemwise.py  # Sigmoid + SiLU ONNX tests (MODIFY)
│       ├── test_batchnorm.py # BatchNorm ONNX tests (NEW)
│       ├── test_properties.py  # Property tests (MODIFY)
│       └── strategies/
│           └── operations.py  # Test strategies (MODIFY)
```

### Running Tests

**Per-operation testing**:
```bash
# Sigmoid
pytest tests/link/onnx/test_elemwise.py -k sigmoid -v

# SiLU
pytest tests/scalar/test_math.py -k silu -v        # Scalar op
pytest tests/tensor/test_math.py -k silu -v        # Tensor function
pytest tests/link/onnx/test_elemwise.py -k silu -v # ONNX converter

# BatchNorm
pytest tests/tensor/test_batchnorm.py -v           # PyTensor op
pytest tests/link/onnx/test_batchnorm.py -v        # ONNX converter
```

**Full test suite**:
```bash
# All Tier 2 tests
pytest tests/link/onnx/test_elemwise.py tests/tensor/test_batchnorm.py tests/link/onnx/test_batchnorm.py tests/scalar/test_math.py tests/tensor/test_math.py -v

# All ONNX tests (including Tier 1)
pytest tests/link/onnx/ -v
```

---

## Performance Considerations

**Sigmoid**: Already optimized in PyTensor (uses SciPy's expit)

**SiLU**:
- Two operations (sigmoid + multiply)
- Comparable to ReLU in speed
- ONNX Runtime will optimize

**BatchNorm**:
- Current: NumPy implementation (reasonable performance)
- Optimization: Could implement C code via `c_code()` method
- ONNX Runtime uses optimized kernels (faster than our NumPy)

---

## Migration Notes

**No migration needed** - these are new operations or new ONNX mappings.

**Integration with Tier 1**:

After implementing both Tier 1 and Tier 2, you can export complete YOLO11n layers:

```python
# Complete C3k2 block
x = pt.tensor4("input")

# Tier 1 operations (already implemented)
conv_out = conv2d(x, kernel)                    # ✅ Tier 1
pool_out = pool_2d(conv_out, ws=(5,5))          # ✅ Tier 1
upsampled = resize(conv_out, scale_factor=(2,2)) # ✅ Tier 1
skip = pt.join(1, upsampled, encoder_features)   # ✅ Tier 1

# Tier 2 operations (this plan)
bn_out = batch_normalization(conv_out, ...)     # ✅ Tier 2
activated = pt.silu(bn_out)                     # ✅ Tier 2

# Full layer with all operations
complete_layer = activated  # Ready for ONNX export!
```

---

## References

**ONNX Specifications**:
- Sigmoid: https://onnx.ai/onnx/operators/onnx__Sigmoid.html
- BatchNormalization: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html

**PyTensor Patterns**:
- Scalar ops: `pytensor/scalar/math.py:1200` (Sigmoid reference)
- Elemwise converters: `pytensor/link/onnx/dispatch/elemwise.py`

**Papers**:
- SiLU/Swish: Ramachandran et al., "Searching for Activation Functions", 2017

---

## Next Steps (After This Plan)

**Tier 3 operations** (lower priority):
- Tanh activation (easy - same as Sigmoid)
- Global pooling (GlobalMaxPool, GlobalAveragePool)
- Attention patterns (if not decomposed to primitives)

**Complete YOLO11n support**:
- Integration test: Full YOLO11n export end-to-end
- Performance benchmarking
- Documentation

---

## Success Metrics

**This plan is successful when:**

- [ ] All 3 Tier 2 operations implemented and tested
- [ ] ~32 unit tests pass (5 Sigmoid + 10 SiLU + 9 BatchNorm + 8 integration)
- [ ] Property-based tests pass for all operations
- [ ] Full C3k2 layer pattern exports to ONNX correctly
- [ ] Conv → BatchNorm → SiLU pipeline works end-to-end
- [ ] No regressions in existing tests
- [ ] Code coverage > 90% for new converters

**Verification command**:
```bash
# Run all Tier 2 tests
pytest tests/link/onnx/test_elemwise.py \
       tests/tensor/test_batchnorm.py tests/link/onnx/test_batchnorm.py \
       tests/scalar/test_math.py tests/tensor/test_math.py \
       -v --cov=pytensor/link/onnx/dispatch --cov=pytensor/tensor/batchnorm --cov=pytensor/scalar/math

# Verify no regressions
pytest tests/link/onnx/ tests/tensor/ tests/scalar/ -v
```

---

## Estimated Timeline

**Operation 1: Sigmoid** (EASIEST)
- Test design: 1 hour
- Test failure verification: 15 minutes
- Implementation: 15 minutes (one line!)
- Refactoring: 15 minutes
- **Total: ~2 hours**

**Operation 2: SiLU**
- Test design: 3 hours (scalar + tensor + ONNX)
- Test failure verification: 30 minutes
- PyTensor scalar op: 3 hours
- PyTensor tensor wrapper: 30 minutes
- ONNX converter (multi-node): 2 hours
- Refactoring: 1 hour
- **Total: ~10 hours (~1.5 days)**

**Operation 3: BatchNorm**
- Test design: 3 hours
- Test failure verification: 30 minutes
- PyTensor op: 4 hours
- ONNX converter: 2 hours
- Refactoring: 1 hour
- **Total: ~10.5 hours (~1.5 days)**

**Grand Total: ~22.5 hours (~2-3 days of focused development)**

---

**With Tier 1 + Tier 2 complete, PyTensor can export YOLO11n with correct numerical behavior!** 🚀
