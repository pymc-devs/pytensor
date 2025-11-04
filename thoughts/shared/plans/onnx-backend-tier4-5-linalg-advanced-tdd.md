---
date: 2025-11-04
status: ready
phase: "tier-4-5"
coverage: "Linear Algebra (Tier 4) & Advanced Operations (Tier 5)"
timeline: "Weeks 7-12"
tags: [tdd, onnx, backend, linear-algebra, advanced-ops, tier4, tier5]
related_research:
  - thoughts/shared/research/2025-11-04_11-52-15_onnx-backend-infrastructure-roadmap.md
  - thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md
related_plans:
  - thoughts/shared/plans/onnx-backend-phase1-3-infrastructure-tdd.md
  - thoughts/shared/plans/onnx-backend-tier2-3-shape-reductions-tdd.md
prerequisites:
  - "Tier 1-3 complete: 51 operations passing"
  - "Infrastructure: ONNXLinker, dispatch system, export API"
  - "Testing utilities: compare_onnx_and_py, tolerance helpers"
---

# ONNX Backend Tier 4-5: Linear Algebra & Advanced Operations - TDD Implementation Plan

## Overview

This TDD plan covers **Tier 4 (Linear Algebra, 20 ops)** and **Tier 5 (Advanced Operations, 43 ops)** of the ONNX backend. These are the most complex operations, including matrix decompositions, solvers, trigonometric functions, and control flow.

**TDD Approach**: Write comprehensive tests with appropriate numerical tolerances, verify they fail properly, then implement features by debugging the failing tests.

**Total Operations**: 63 operations across two tiers
**Timeline**: 5-6 weeks (2-3 weeks Tier 4, 2-3 weeks Tier 5)

**IMPORTANT NOTE**: Many linear algebra operations are **not in standard ONNX opset**. We'll need to either:
1. Use ONNX Runtime contrib ops (platform-specific)
2. Skip and document as unsupported
3. Implement as sequences of basic ops (complex, may be slow)

## Current State Analysis

### What Exists (Post-Tier 1-3):
- ✅ **ONNX backend infrastructure**: Complete with linker and dispatch system
- ✅ **Tier 1 (20 ops)**: Basic elemwise operations
- ✅ **Tier 2 (15 ops)**: Shape operations (Reshape, DimShuffle, Join, Subtensor)
- ✅ **Tier 3 (16 ops)**: Reductions (Sum, Max, Argmax) and Allocation (Alloc, ARange)
- ✅ **Testing infrastructure**: `compare_onnx_and_py`, 74+ passing tests
- ✅ **Export API**: Full export and compilation functionality

### Testing Landscape:
- **Testing framework**: pytest with comprehensive fixtures
- **Test patterns available**: From PyTensor linear algebra tests
  - Linalg tests: `tests/tensor/test_nlinalg.py`, `tests/tensor/test_slinalg.py`
  - JAX backend: `tests/link/jax/test_nlinalg.py`, `tests/link/jax/test_slinalg.py`
  - BLAS tests: `tests/tensor/test_blas.py`
- **Numerical tolerance patterns**: Dtype-dependent tolerances
  - Float64: `atol=1e-8, rtol=1e-8`
  - Float32: `atol=1e-4, rtol=1e-4`
  - Gradient tests: `abs_tol=0.05, rel_tol=0.05`

### Key Discoveries:
- **ONNX limitations**: Many linalg ops not in standard ONNX
  - SVD, QR, Cholesky: Not in standard opset
  - Eigendecomposition: Not supported
  - Matrix inverse: No direct operator
- **ONNX Runtime contrib ops**: May provide some operations
  - Platform-specific, not portable
  - Limited documentation
- **Test data generation critical**: Must use well-conditioned matrices
  - Positive definite: `A = X @ X.T`
  - Add identity: `A + 0.5 * I` for conditioning
- **Gradient testing requirements**: Float32 often too imprecise
  - Many gradient tests skip float32
  - Need `eps=2e-8` for float64 gradients

## Desired End State

After Tier 4-5 completion:

✅ **Linear Algebra Working** (Tier 4 - subset):
- Matrix multiplication: Dot, Gemm, BatchedDot
- Basic decompositions: SVD (if contrib op available)
- Matrix inverse: Via custom implementation
- Determinant: Via custom implementation
- Document unsupported ops clearly

✅ **Advanced Operations Working** (Tier 5 - 43 ops):
- Trigonometric: Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, etc.
- Comparison: LT, GT, LE, GE, EQ, NEQ
- Logical: AND, OR, XOR, Invert
- Special math: Sigmoid, Softplus, Erf, Log1p, Expm1, Clip
- Neural network: Softmax, LogSoftmax, Switch
- Extra ops: CumSum, Repeat, Unique, Pad

✅ **Comprehensive Testing**:
- 50+ new tests with appropriate tolerances
- Test data generation for stable tests
- Decomposition reconstruction tests
- Clear documentation of unsupported operations

✅ **Validation**:
- Can export matrix operations (multiplication, basic linalg)
- Can export neural network activations
- Can export complete models (MLPs, simple networks)
- Clear error messages for unsupported operations

## What We're NOT Testing/Implementing

❌ **Out of Scope**:
- **Complex decompositions**: Full QR, Cholesky may not be possible in portable ONNX
- **Eigendecomposition**: Not in ONNX standard
- **Matrix exponential**: Extremely complex, skip
- **Control flow**: Scan, IfElse - very complex, separate effort
- **Random variables**: Not in ONNX standard
- **Sparse operations**: Not in ONNX standard
- **Custom operators**: Avoid platform-specific code

**Strategy**: Focus on operations that can be implemented with standard ONNX ops or simple compositions. Document limitations clearly.

## TDD Approach

### Test Design Philosophy:
1. **Test with appropriate tolerances**: Float64 (1e-8), Float32 (1e-4)
2. **Generate well-conditioned matrices**: Avoid singular/ill-conditioned matrices
3. **Test reconstruction**: For decompositions, verify `A = U @ S @ V.T`
4. **Skip unsupported operations gracefully**: Use `pytest.skip` with clear messages
5. **Test both forward and gradient**: Where differentiable
6. **Compare against SciPy/NumPy**: Reference implementations

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests for linear algebra and advanced operations. Many will be marked as `pytest.skip` if operations aren't supported in ONNX.

---

## TIER 4: LINEAR ALGEBRA OPERATIONS

### Test Category 1: Matrix Multiplication

**Test File**: `tests/link/onnx/test_nlinalg.py`
**Purpose**: Test basic matrix multiplication operations

#### Test: `test_dot_2d`
**Purpose**: Test 2D matrix multiplication

```python
def test_dot_2d():
    """Test 2D matrix multiplication (Dot op)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    A = pt.matrix('A', dtype='float32')
    B = pt.matrix('B', dtype='float32')
    C = pt.dot(A, B)

    A_val = np.random.randn(3, 4).astype('float32')
    B_val = np.random.randn(4, 5).astype('float32')

    fn, result = compare_onnx_and_py([A, B], C, [A_val, B_val])

    expected = np.dot(A_val, B_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    # Verify ONNX uses MatMul
    from tests.link.onnx.test_basic import get_onnx_node_types
    node_types = get_onnx_node_types(fn)
    assert 'MatMul' in node_types, \
        f"Expected 'MatMul' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError: No ONNX conversion available for: Dot`

#### Test: `test_dot_1d_2d`
**Purpose**: Test vector-matrix multiplication

```python
def test_dot_1d_2d():
    """Test vector-matrix multiplication."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    v = pt.vector('v', dtype='float32')
    M = pt.matrix('M', dtype='float32')
    result = pt.dot(v, M)

    v_val = np.random.randn(4).astype('float32')
    M_val = np.random.randn(4, 5).astype('float32')

    fn, output = compare_onnx_and_py([v, M], result, [v_val, M_val])

    expected = np.dot(v_val, M_val)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    # Should be 1D output
    assert output.ndim == 1, f"Expected 1D output, got shape {output.shape}"
```

**Expected Failure Mode**: May need Reshape to handle 1D vectors

#### Test: `test_batched_dot`
**Purpose**: Test batched matrix multiplication

```python
def test_batched_dot():
    """Test batched matrix multiplication."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    A = pt.tensor3('A', dtype='float32')
    B = pt.tensor3('B', dtype='float32')
    C = pt.batched_dot(A, B)

    A_val = np.random.randn(2, 3, 4).astype('float32')
    B_val = np.random.randn(2, 4, 5).astype('float32')

    fn, result = compare_onnx_and_py([A, B], C, [A_val, B_val])

    expected = np.einsum('bij,bjk->bik', A_val, B_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    # ONNX MatMul handles batched operations natively
    node_types = get_onnx_node_types(fn)
    assert 'MatMul' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for BatchedDot

#### Test: `test_gemm`
**Purpose**: Test GEMM operation (General Matrix Multiply)

```python
def test_gemm():
    """Test GEMM: alpha*A@B + beta*C."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from pytensor.tensor.blas import gemm

    A = pt.matrix('A', dtype='float32')
    B = pt.matrix('B', dtype='float32')
    C = pt.matrix('C', dtype='float32')

    # GEMM: 2.0 * A @ B + 0.5 * C
    result = gemm(A, B, C, alpha=2.0, beta=0.5)

    A_val = np.random.randn(3, 4).astype('float32')
    B_val = np.random.randn(4, 5).astype('float32')
    C_val = np.random.randn(3, 5).astype('float32')

    fn, output = compare_onnx_and_py([A, B, C], result, [A_val, B_val, C_val])

    expected = 2.0 * np.dot(A_val, B_val) + 0.5 * C_val
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    # ONNX has Gemm operator
    node_types = get_onnx_node_types(fn)
    assert 'Gemm' in node_types, \
        f"Expected 'Gemm' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Gemm

---

### Test Category 2: Matrix Decompositions

**Test File**: `tests/link/onnx/test_nlinalg.py` (continued)
**Purpose**: Test matrix decompositions (SVD, QR, Cholesky)

**IMPORTANT**: Most decompositions are NOT in standard ONNX. Tests should be marked with `pytest.skip` or `pytest.xfail` with clear messages.

#### Test: `test_svd_not_supported`
**Purpose**: Document that SVD is not in standard ONNX

```python
@pytest.mark.skip(reason="SVD not in standard ONNX opset - requires contrib ops or custom implementation")
def test_svd_not_supported():
    """Test SVD - expected to be unsupported in standard ONNX.

    SVD decomposes A into U, S, V.T where A = U @ diag(S) @ V.T
    This is NOT available in standard ONNX opset.

    Options:
    1. Use ONNX Runtime contrib op (platform-specific)
    2. Implement as sequence of operations (very complex)
    3. Skip and document as unsupported

    This test documents the expected behavior if we choose to implement.
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.nlinalg import svd

    A = pt.matrix('A', dtype='float32')
    U, s, Vt = svd(A, full_matrices=False)

    # Well-conditioned test matrix
    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 3)).astype('float32')

    # This will raise NotImplementedError
    with pytest.raises(NotImplementedError, match="SVD not supported"):
        fn = pytensor.function([A], [U, s, Vt], mode=onnx_mode)
```

**Expected Failure Mode**: Test is skipped (not run)

#### Test: `test_cholesky_not_supported`
**Purpose**: Document that Cholesky is not in standard ONNX

```python
@pytest.mark.skip(reason="Cholesky not in standard ONNX opset")
def test_cholesky_not_supported():
    """Test Cholesky decomposition - not in standard ONNX.

    Cholesky decomposes positive definite A into L @ L.T
    where L is lower triangular.

    Not available in standard ONNX opset. ONNX Runtime may have
    contrib op: com.microsoft.Cholesky
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.slinalg import cholesky

    A = pt.matrix('A', dtype='float32')
    L = cholesky(A)

    # Positive definite matrix
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4, 4)).astype('float32')
    A_val = X @ X.T  # Positive definite

    with pytest.raises(NotImplementedError, match="Cholesky not supported"):
        fn = pytensor.function([A], L, mode=onnx_mode)
```

**Expected Failure Mode**: Test is skipped

---

### Test Category 3: Solving Linear Systems

**Test File**: `tests/link/onnx/test_slinalg.py`
**Purpose**: Test linear system solving operations

#### Test: `test_solve_not_supported`
**Purpose**: Document that Solve is not in standard ONNX

```python
@pytest.mark.skip(reason="Solve not in standard ONNX opset")
def test_solve_not_supported():
    """Test Solve operation - not in standard ONNX.

    Solve finds X such that A @ X = B.
    Not available in standard ONNX. Would require:
    - LU decomposition (not in ONNX)
    - Forward/backward substitution
    - Or matrix inverse + matmul
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.slinalg import solve

    A = pt.matrix('A', dtype='float32')
    B = pt.matrix('B', dtype='float32')
    X = solve(A, B)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype('float32')
    A_val = A_val + 0.5 * np.eye(4, dtype='float32')  # Well-conditioned
    B_val = rng.normal(size=(4, 3)).astype('float32')

    with pytest.raises(NotImplementedError, match="Solve not supported"):
        fn = pytensor.function([A, B], X, mode=onnx_mode)
```

**Expected Failure Mode**: Test is skipped

---

### Test Category 4: Matrix Properties

**Test File**: `tests/link/onnx/test_nlinalg.py` (continued)
**Purpose**: Test matrix property operations (determinant, inverse)

#### Test: `test_det_custom_implementation`
**Purpose**: Test determinant via custom implementation

```python
@pytest.mark.skip(reason="Det requires LU decomposition - complex custom implementation needed")
def test_det_custom_implementation():
    """Test matrix determinant - requires custom implementation.

    Determinant can be computed via:
    1. LU decomposition + product of diagonal (preferred)
    2. QR decomposition + product of R diagonal
    3. Direct computation for small matrices

    All approaches require operations not in standard ONNX.
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.nlinalg import det

    A = pt.matrix('A', dtype='float32')
    d = det(A)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype('float32')

    with pytest.raises(NotImplementedError, match="Det not supported"):
        fn = pytensor.function([A], d, mode=onnx_mode)
```

**Expected Failure Mode**: Test is skipped

#### Test: `test_matrix_inverse_not_supported`
**Purpose**: Document that matrix inverse is not in standard ONNX

```python
@pytest.mark.skip(reason="Matrix inverse not in standard ONNX opset")
def test_matrix_inverse_not_supported():
    """Test matrix inverse - not in standard ONNX.

    Matrix inverse could be implemented via:
    1. LU decomposition + solving (not available)
    2. Adjugate method (very complex)
    3. Gradient descent (iterative, expensive)

    Not practical for standard ONNX export.
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.nlinalg import matrix_inverse

    A = pt.matrix('A', dtype='float32')
    A_inv = matrix_inverse(A)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype('float32')
    A_val = A_val + 0.5 * np.eye(4, dtype='float32')

    with pytest.raises(NotImplementedError, match="Matrix inverse not supported"):
        fn = pytensor.function([A], A_inv, mode=onnx_mode)
```

**Expected Failure Mode**: Test is skipped

---

### Test Category 5: Extract Diagonal

**Test File**: `tests/link/onnx/test_nlinalg.py` (continued)
**Purpose**: Test diagonal extraction (this CAN be implemented)

#### Test: `test_extract_diag`
**Purpose**: Test extracting matrix diagonal

```python
def test_extract_diag():
    """Test extracting diagonal from matrix.

    This CAN be implemented in ONNX using:
    - Identity matrix of appropriate size
    - Element-wise multiply with input
    - ReduceSum along one axis

    Or using Gather operations.
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    A = pt.matrix('A', dtype='float32')
    d = pt.diag(A)  # Extract diagonal

    A_val = np.random.randn(4, 4).astype('float32')

    fn, result = compare_onnx_and_py([A], d, [A_val])

    expected = np.diag(A_val)
    np.testing.assert_array_equal(result, expected)
```

**Expected Failure Mode**: `NotImplementedError` for ExtractDiag (but implementable)

---

## TIER 5: ADVANCED OPERATIONS

### Test Category 6: Trigonometric Functions

**Test File**: `tests/link/onnx/test_special.py`
**Purpose**: Test trigonometric and hyperbolic functions

#### Test: `test_trigonometric_functions`
**Purpose**: Test all trig functions

```python
@pytest.mark.parametrize("pt_op,np_op,onnx_op", [
    (pt.sin, np.sin, 'Sin'),
    (pt.cos, np.cos, 'Cos'),
    (pt.tan, np.tan, 'Tan'),
    (pt.arcsin, np.arcsin, 'Asin'),
    (pt.arccos, np.arccos, 'Acos'),
    (pt.arctan, np.arctan, 'Atan'),
])
def test_trigonometric_functions(pt_op, np_op, onnx_op):
    """Test trigonometric functions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt_op(x)

    # Use values in appropriate domain
    if pt_op in [pt.arcsin, pt.arccos]:
        # Domain [-1, 1]
        x_val = np.linspace(-0.9, 0.9, 10).astype('float32')
    else:
        x_val = np.linspace(-3, 3, 10).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types, \
        f"Expected '{onnx_op}' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for trig functions (but they're in ONNX!)

#### Test: `test_hyperbolic_functions`
**Purpose**: Test hyperbolic functions

```python
@pytest.mark.parametrize("pt_op,np_op,onnx_op", [
    (pt.sinh, np.sinh, 'Sinh'),
    (pt.cosh, np.cosh, 'Cosh'),
    (pt.tanh, np.tanh, 'Tanh'),
    (pt.arcsinh, np.arcsinh, 'Asinh'),
    (pt.arccosh, np.arccosh, 'Acosh'),
    (pt.arctanh, np.arctanh, 'Atanh'),
])
def test_hyperbolic_functions(pt_op, np_op, onnx_op):
    """Test hyperbolic functions."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt_op(x)

    # Use values in appropriate domain
    if pt_op == pt.arccosh:
        # Domain [1, inf)
        x_val = np.linspace(1.1, 3, 10).astype('float32')
    elif pt_op == pt.arctanh:
        # Domain (-1, 1)
        x_val = np.linspace(-0.9, 0.9, 10).astype('float32')
    else:
        x_val = np.linspace(-2, 2, 10).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` initially

---

### Test Category 7: Comparison Operations

**Test File**: `tests/link/onnx/test_special.py` (continued)
**Purpose**: Test comparison operations

#### Test: `test_comparison_ops`
**Purpose**: Test all comparison operations

```python
@pytest.mark.parametrize("pt_op,np_op,onnx_op", [
    (pt.lt, np.less, 'Less'),
    (pt.gt, np.greater, 'Greater'),
    (pt.le, np.less_equal, 'LessOrEqual'),
    (pt.ge, np.greater_equal, 'GreaterOrEqual'),
    (pt.eq, np.equal, 'Equal'),
    (pt.neq, np.not_equal, 'Not'),  # Not + Equal
])
def test_comparison_ops(pt_op, np_op, onnx_op):
    """Test comparison operations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = pt_op(x, y)

    x_val = np.array([1, 2, 3, 4, 5], dtype='float32')
    y_val = np.array([2, 2, 2, 2, 2], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    expected = np_op(x_val, y_val)
    np.testing.assert_array_equal(result, expected)

    # Result should be boolean
    assert result.dtype == bool or result.dtype == np.bool_

    node_types = get_onnx_node_types(fn)
    # Check for expected ONNX op (may be combined with other ops)
```

**Expected Failure Mode**: `NotImplementedError` for comparison ops

---

### Test Category 8: Logical Operations

**Test File**: `tests/link/onnx/test_special.py` (continued)
**Purpose**: Test logical operations

#### Test: `test_logical_ops`
**Purpose**: Test AND, OR, XOR, NOT

```python
@pytest.mark.parametrize("pt_op,np_op,onnx_op", [
    (pt.and_, np.logical_and, 'And'),
    (pt.or_, np.logical_or, 'Or'),
    (pt.xor, np.logical_xor, 'Xor'),
    (pt.invert, np.logical_not, 'Not'),
])
def test_logical_ops(pt_op, np_op, onnx_op):
    """Test logical operations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    if pt_op == pt.invert:
        # Unary operation
        x = pt.vector('x', dtype='bool')
        y = pt_op(x)

        x_val = np.array([True, False, True, False, True], dtype=bool)

        fn, result = compare_onnx_and_py([x], y, [x_val])

        expected = np_op(x_val)
        np.testing.assert_array_equal(result, expected)
    else:
        # Binary operation
        x = pt.vector('x', dtype='bool')
        y_tensor = pt.vector('y', dtype='bool')
        z = pt_op(x, y_tensor)

        x_val = np.array([True, True, False, False], dtype=bool)
        y_val = np.array([True, False, True, False], dtype=bool)

        fn, result = compare_onnx_and_py([x, y_tensor], z, [x_val, y_val])

        expected = np_op(x_val, y_val)
        np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` for logical ops

---

### Test Category 9: Special Math Functions

**Test File**: `tests/link/onnx/test_special.py` (continued)
**Purpose**: Test special mathematical functions

#### Test: `test_sigmoid_softplus`
**Purpose**: Test activation functions

```python
@pytest.mark.parametrize("pt_op,onnx_op", [
    (pt.nnet.sigmoid, 'Sigmoid'),
    (pt.nnet.softplus, 'Softplus'),
])
def test_sigmoid_softplus(pt_op, onnx_op):
    """Test sigmoid and softplus activations."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt_op(x)

    x_val = np.linspace(-5, 5, 20).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Verify with manual computation
    if pt_op == pt.nnet.sigmoid:
        expected = 1 / (1 + np.exp(-x_val))
    else:  # softplus
        expected = np.log(1 + np.exp(x_val))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` initially

#### Test: `test_erf_erfc`
**Purpose**: Test error functions

```python
@pytest.mark.parametrize("pt_op,np_op,onnx_op", [
    (pt.erf, scipy.special.erf, 'Erf'),
    # Note: Erfc not in ONNX - would need to compute as 1 - Erf
])
def test_erf_erfc(pt_op, np_op, onnx_op):
    """Test error function."""
    import pytensor.tensor as pt
    import numpy as np
    from scipy import special
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt_op(x)

    x_val = np.linspace(-3, 3, 20).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert onnx_op in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Erf

#### Test: `test_log1p_expm1`
**Purpose**: Test log(1+x) and exp(x)-1

```python
@pytest.mark.parametrize("pt_op,np_op", [
    (pt.log1p, np.log1p),
    (pt.expm1, np.expm1),
])
def test_log1p_expm1(pt_op, np_op):
    """Test log1p and expm1 functions.

    These may not have direct ONNX ops, but can be composed:
    - log1p(x) = log(1 + x) using Add + Log
    - expm1(x) = exp(x) - 1 using Exp + Sub
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt_op(x)

    x_val = np.linspace(-0.5, 2, 20).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np_op(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
```

**Expected Failure Mode**: May fail if not composed correctly

#### Test: `test_clip`
**Purpose**: Test clipping values to range

```python
def test_clip():
    """Test clip operation (clamp values to range)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.clip(x, -1.0, 1.0)

    x_val = np.array([-2, -0.5, 0, 0.5, 2], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.clip(x_val, -1.0, 1.0)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Clip' in node_types, \
        f"Expected 'Clip' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Clip

---

### Test Category 10: Neural Network Operations

**Test File**: `tests/link/onnx/test_nnet.py`
**Purpose**: Test neural network specific operations

#### Test: `test_softmax`
**Purpose**: Test softmax activation

```python
@pytest.mark.parametrize("axis", [None, -1, 0, 1])
def test_softmax(axis):
    """Test softmax activation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from scipy.special import softmax as scipy_softmax

    x = pt.matrix('x', dtype='float32')
    y = pt.nnet.softmax(x, axis=axis)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    # Compute expected with scipy
    if axis is None:
        axis_np = 1  # PyTensor default
    else:
        axis_np = axis

    expected = scipy_softmax(x_val, axis=axis_np)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert 'Softmax' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Softmax

#### Test: `test_logsoftmax`
**Purpose**: Test log-softmax

```python
def test_logsoftmax():
    """Test log-softmax activation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py
    from scipy.special import log_softmax

    x = pt.matrix('x', dtype='float32')
    y = pt.nnet.logsoftmax(x)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = log_softmax(x_val, axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert 'LogSoftmax' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for LogSoftmax

#### Test: `test_switch`
**Purpose**: Test Switch (element-wise ternary conditional)

```python
def test_switch():
    """Test Switch operation (element-wise conditional).

    Switch(condition, then_value, else_value) returns:
    - then_value where condition is True
    - else_value where condition is False

    In ONNX this maps to Where operator.
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    condition = pt.vector('condition', dtype='bool')
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')

    result = pt.switch(condition, x, y)

    cond_val = np.array([True, False, True, False, True], dtype=bool)
    x_val = np.array([1, 2, 3, 4, 5], dtype='float32')
    y_val = np.array([10, 20, 30, 40, 50], dtype='float32')

    fn, output = compare_onnx_and_py([condition, x, y], result, [cond_val, x_val, y_val])

    expected = np.where(cond_val, x_val, y_val)
    np.testing.assert_array_equal(output, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Where' in node_types, \
        f"Expected 'Where' node, got {node_types}"
```

**Expected Failure Mode**: `NotImplementedError` for Switch

---

### Test Category 11: Extra Operations

**Test File**: `tests/link/onnx/test_extra_ops.py`
**Purpose**: Test extra utility operations

#### Test: `test_cumsum`
**Purpose**: Test cumulative sum

```python
@pytest.mark.parametrize("axis", [0, 1])
def test_cumsum(axis):
    """Test cumulative sum operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    y = pt.cumsum(x, axis=axis)

    x_val = np.random.randn(3, 4).astype('float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.cumsum(x_val, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    node_types = get_onnx_node_types(fn)
    assert 'CumSum' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for CumSum

#### Test: `test_repeat`
**Purpose**: Test repeat operation

```python
def test_repeat():
    """Test repeat operation (repeat elements)."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='float32')
    y = pt.repeat(x, repeats=3, axis=0)

    x_val = np.array([1, 2, 3], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.repeat(x_val, repeats=3, axis=0)
    np.testing.assert_array_equal(result, expected)

    # Repeat in ONNX can be done with Tile or Expand
```

**Expected Failure Mode**: `NotImplementedError` for Repeat

#### Test: `test_unique`
**Purpose**: Test unique operation

```python
def test_unique():
    """Test unique operation (find unique elements).

    Note: ONNX Unique has different semantics than NumPy.
    May need special handling.
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.vector('x', dtype='int64')
    y = pt.unique(x)

    x_val = np.array([1, 2, 3, 2, 1, 4, 3], dtype='int64')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.unique(x_val)

    # Result may be sorted differently
    np.testing.assert_array_equal(sorted(result), sorted(expected))

    node_types = get_onnx_node_types(fn)
    assert 'Unique' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Unique

#### Test: `test_pad`
**Purpose**: Test array padding

```python
def test_pad():
    """Test pad operation."""
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.matrix('x', dtype='float32')
    # Pad with 1 zero on each side
    y = pt.pad(x, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

    x_val = np.array([[1, 2], [3, 4]], dtype='float32')

    fn, result = compare_onnx_and_py([x], y, [x_val])

    expected = np.pad(x_val, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)
    np.testing.assert_array_equal(result, expected)

    node_types = get_onnx_node_types(fn)
    assert 'Pad' in node_types
```

**Expected Failure Mode**: `NotImplementedError` for Pad

---

### Integration Test: Complete Neural Network

**Test File**: `tests/link/onnx/test_integration.py` (continued)
**Purpose**: Test complete network using Tier 4-5 operations

#### Test: `test_simple_mlp`
**Purpose**: Test multi-layer perceptron

```python
def test_simple_mlp():
    """Test simple MLP using matmul, add, and activation.

    This integration test verifies that a complete neural network
    layer can be exported to ONNX.
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Input
    x = pt.matrix('x', dtype='float32')

    # Weights and biases
    W1 = pt.matrix('W1', dtype='float32')
    b1 = pt.vector('b1', dtype='float32')
    W2 = pt.matrix('W2', dtype='float32')
    b2 = pt.vector('b2', dtype='float32')

    # Layer 1: x @ W1 + b1, then ReLU
    h = pt.nnet.relu(pt.dot(x, W1) + b1)

    # Layer 2: h @ W2 + b2, then softmax
    logits = pt.dot(h, W2) + b2
    output = pt.nnet.softmax(logits)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(5, 10)).astype('float32')
    W1_val = rng.normal(size=(10, 20)).astype('float32')
    b1_val = rng.normal(size=(20,)).astype('float32')
    W2_val = rng.normal(size=(20, 3)).astype('float32')
    b2_val = rng.normal(size=(3,)).astype('float32')

    fn, result = compare_onnx_and_py(
        [x, W1, b1, W2, b2],
        output,
        [x_val, W1_val, b1_val, W2_val, b2_val]
    )

    # Verify output is valid probabilities
    assert result.shape == (5, 3), f"Expected shape (5, 3), got {result.shape}"
    assert np.allclose(result.sum(axis=1), 1.0), "Softmax should sum to 1"
    assert np.all(result >= 0) and np.all(result <= 1), "Probabilities should be in [0, 1]"
```

**Expected Failure Mode**: May fail if MatMul, Add, ReLU, or Softmax not implemented

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail appropriately. Many tests will be skipped for unsupported operations.

### Verification Steps

1. **Run linear algebra tests**:
   ```bash
   pytest tests/link/onnx/test_nlinalg.py -v --tb=short
   ```

   **Expected**:
   - Matrix multiplication tests: Fail with `NotImplementedError`
   - Decomposition tests: Skipped with clear messages
   - Property tests: Skipped (Det, Inverse)

2. **Run advanced operation tests**:
   ```bash
   pytest tests/link/onnx/test_special.py -v --tb=short
   pytest tests/link/onnx/test_nnet.py -v --tb=short
   pytest tests/link/onnx/test_extra_ops.py -v --tb=short
   ```

   **Expected**: All fail with `NotImplementedError` for their respective Ops

3. **Count tests**:
   ```bash
   pytest --collect-only tests/link/onnx/ | grep "test_"
   ```

   **Expected**: ~124 tests total (74 from Tiers 1-3 + 50 new)

### Success Criteria

#### Automated Verification:
- [ ] All new tests discovered
- [ ] Skipped tests show clear skip reasons
- [ ] Non-skipped tests fail with `NotImplementedError`
- [ ] Previous tier tests still pass

#### Manual Verification:
- [ ] Skip messages clearly explain why operation is unsupported
- [ ] Skip messages suggest alternatives if available
- [ ] Error messages for implementable ops are helpful

---

## Phase 3: Feature Implementation (Red → Green)

### Implementation Strategy

For Tier 4-5, we need to be selective about what to implement:

**Priority 1 - Implement**:
- Matrix multiplication (Dot, Gemm, BatchedDot) - in ONNX standard
- Trigonometric functions - in ONNX standard
- Comparison operations - in ONNX standard
- Logical operations - in ONNX standard
- Softmax, LogSoftmax - in ONNX standard
- Switch (→ Where) - in ONNX standard
- Clip, Erf - in ONNX standard
- CumSum, Pad - in ONNX standard

**Priority 2 - Compose from basic ops**:
- Log1p, Expm1 - can compose
- Sigmoid, Softplus - may already be in ONNX
- Repeat - can use Tile
- ExtractDiag - can implement with Gather

**Priority 3 - Skip/Document**:
- SVD, QR, Cholesky - not in standard ONNX
- Solve, Lstsq - complex, not in standard ONNX
- Det, Matrix Inverse - complex, not in standard ONNX
- Unique - different semantics in ONNX
- Advanced control flow (Scan, IfElse) - very complex

### Implementation Order

1. **Matrix multiplication** (simplest, most useful)
2. **Trigonometric functions** (direct mappings)
3. **Comparison and logical** (direct mappings)
4. **Neural network ops** (Softmax, Switch)
5. **Special math** (compose where needed)
6. **Extra operations** (CumSum, Pad, etc.)

---

### Implementation 1: Matrix Multiplication

**Target Tests**: `test_dot_*`, `test_batched_dot`, `test_gemm`

**File**: `pytensor/link/onnx/dispatch/nlinalg.py` (new file)

```python
"""ONNX conversion for linear algebra operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.blas import Dot, Gemm, BatchedDot
from pytensor.graph.basic import Constant

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Dot)
def onnx_funcify_Dot(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot op to ONNX MatMul node.

    Dot performs matrix multiplication. ONNX MatMul handles:
    - Matrix @ Matrix
    - Vector @ Matrix (with implicit unsqueeze)
    - Batched operations
    """
    input_a = get_var_name(node.inputs[0])
    input_b = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    # ONNX MatMul handles most cases directly
    matmul_node = helper.make_node(
        'MatMul',
        inputs=[input_a, input_b],
        outputs=[output_name],
        name=f"MatMul_{output_name}",
    )

    return matmul_node


@onnx_funcify.register(Gemm)
def onnx_funcify_Gemm(op, node, var_names, get_var_name, **kwargs):
    """Convert Gemm op to ONNX Gemm node.

    Gemm: C = alpha * A @ B + beta * C
    Direct mapping to ONNX Gemm operator.
    """
    input_a = get_var_name(node.inputs[0])
    input_b = get_var_name(node.inputs[1])
    input_c = get_var_name(node.inputs[2])
    output_name = get_var_name(node.outputs[0])

    # Get alpha and beta from op
    alpha = float(op.alpha) if hasattr(op, 'alpha') else 1.0
    beta = float(op.beta) if hasattr(op, 'beta') else 1.0

    gemm_node = helper.make_node(
        'Gemm',
        inputs=[input_a, input_b, input_c],
        outputs=[output_name],
        name=f"Gemm_{output_name}",
        alpha=alpha,
        beta=beta,
        transA=0,
        transB=0,
    )

    return gemm_node


@onnx_funcify.register(BatchedDot)
def onnx_funcify_BatchedDot(op, node, var_names, get_var_name, **kwargs):
    """Convert BatchedDot to ONNX MatMul.

    BatchedDot performs batched matrix multiplication.
    ONNX MatMul handles batching natively.
    """
    input_a = get_var_name(node.inputs[0])
    input_b = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    matmul_node = helper.make_node(
        'MatMul',
        inputs=[input_a, input_b],
        outputs=[output_name],
        name=f"MatMul_{output_name}",
    )

    return matmul_node
```

**Success Criteria**:
- [ ] `test_dot_2d` passes
- [ ] `test_dot_1d_2d` passes
- [ ] `test_batched_dot` passes
- [ ] `test_gemm` passes

---

### Implementation 2: Trigonometric Functions

These are already handled by the Elemwise dispatcher if we add them to the scalar op mapping.

**File**: `pytensor/link/onnx/dispatch/elemwise.py` (update)

Add to `SCALAR_OP_TO_ONNX` dictionary:

```python
# Trigonometric (add to existing dict)
scalar.Sin: "Sin",
scalar.Cos: "Cos",
scalar.Tan: "Tan",
scalar.ArcSin: "Asin",
scalar.ArcCos: "Acos",
scalar.ArcTan: "Atan",

# Hyperbolic
scalar.Sinh: "Sinh",
scalar.Cosh: "Cosh",
scalar.Tanh: "Tanh",
scalar.ArcSinh: "Asinh",
scalar.ArcCosh: "Acosh",
scalar.ArcTanh: "Atanh",

# Comparison
scalar.LT: "Less",
scalar.GT: "Greater",
scalar.LE: "LessOrEqual",
scalar.GE: "GreaterOrEqual",
scalar.EQ: "Equal",

# Logical
scalar.AND: "And",
scalar.OR: "Or",
scalar.XOR: "Xor",
scalar.Invert: "Not",

# Special
scalar.Sigmoid: "Sigmoid",
scalar.Erf: "Erf",
```

**Success Criteria**:
- [ ] All trig tests pass
- [ ] All comparison tests pass
- [ ] All logical tests pass

---

### Implementation 3-6: Remaining Operations

Continue implementing:
- Neural network ops (Softmax, LogSoftmax, Switch)
- Special math (Clip, compose Log1p/Expm1)
- Extra ops (CumSum, Pad, Repeat via Tile)

Each follows similar dispatch pattern:
1. Create dispatch function
2. Map to ONNX op or composition
3. Handle attributes/parameters
4. Test passes

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor to improve code quality while keeping tests green.

### Refactoring Targets

1. **Skip decorator helper**:
   - Create decorator for operations we're not implementing
   - Consistent skip messages

2. **Tolerance helper**:
   - Centralize dtype-dependent tolerance logic
   - Helper for choosing atol/rtol based on dtype

3. **Documentation**:
   - Create `UNSUPPORTED_OPERATIONS.md` listing what's not supported
   - Document alternatives where available

---

## Success Metrics

### Tier 4-5 Complete When:

- ✅ Matrix multiplication works (Dot, Gemm, BatchedDot)
- ✅ Trigonometric functions work (12 ops)
- ✅ Comparison and logical operations work (16 ops)
- ✅ Neural network ops work (Softmax, LogSoftmax, Switch)
- ✅ Special math works (Sigmoid, Clip, Erf, composed ops)
- ✅ Extra operations work (CumSum, Pad, subset of others)
- ✅ Unsupported operations clearly documented
- ✅ Integration test passes (simple MLP export)
- ✅ ~40-50 operations total implemented (realistically)
- ✅ Can export complete neural networks

### Documentation Deliverables

- ✅ `SUPPORTED_OPERATIONS.md`: List of working operations
- ✅ `UNSUPPORTED_OPERATIONS.md`: List of unsupported with explanations
- ✅ `ONNX_LIMITATIONS.md`: ONNX-specific constraints and workarounds

---

## References

### Test Pattern References
- Linear algebra: `tests/tensor/test_nlinalg.py`, `tests/tensor/test_slinalg.py`
- JAX backend: `tests/link/jax/test_nlinalg.py`, `tests/link/jax/test_slinalg.py`
- Special functions: `tests/tensor/test_special.py`

### ONNX Specification
- Matrix operations: MatMul, Gemm
- Trigonometric: Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, etc.
- Comparison: Less, Greater, Equal, etc.
- Neural network: Softmax, LogSoftmax, Sigmoid
- Utilities: CumSum, Pad, Clip, Where

### ONNX Limitations
- No standard ops for: SVD, QR, Cholesky, Eig, Solve, Det, Inverse
- ONNX Runtime contrib ops may help: https://github.com/microsoft/onnxruntime/tree/main/docs/ContribOperators.md
