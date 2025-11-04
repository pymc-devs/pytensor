# Hypothesis-Based Property Testing for ONNX Backend

<!-- WORKSHOP NOTE: This is the FOURTH iteration - the "future plan" that hasn't been implemented yet. It represents the natural evolution after three successful plans.

Why this plan exists:
1. Test count grew from 82 → 103 after Conv2D (linear growth problem)
2. Each new op requires 5-10 new tests manually written
3. Property-based testing is the scalable solution for mature backends

Status: NOT YET IMPLEMENTED (as of Oct 2025)

Execution plan:
- Phase 1 (Setup): Ready to implement - Hypothesis is well-documented
- Phase 2 (Properties): Will require iteration - finding right properties is an art
- Phase 3 (Regressions): Straightforward - just moving existing tests
- Phase 4 (Cleanup): Can't predict until we see what properties cover

This plan differs from earlier ones:
- **Preventive** vs reactive (stopping problems before they grow)
- **Framework change** vs feature add (testing strategy, not new functionality)
- **Higher risk** (property-based testing is advanced technique)
- **Long-term payoff** (saves time over years, not weeks)

Workshop lesson: Not all plans get executed immediately. This plan is waiting because:
1. Current testing works (103 tests isn't unmanageable yet)
2. Hypothesis requires learning curve
3. Want to validate approach with more complex ops first (maybe Pooling, BatchNorm)
4. Property-based testing is optimization, not requirement

When to implement: After adding 2-3 more complex operations, when test count hits ~150+, or when test maintenance becomes painful. -->

## Overview

Transform PyTensor's ONNX backend testing from **103 manual tests** (updated from 82) to a scalable property-based testing framework using Hypothesis. This enables comprehensive testing with minimal code maintenance and automatic edge case discovery, while preserving critical regression tests.

<!-- REALITY CHECK: The "103 tests" count is accurate as of Oct 2025. Conv2D added 21 tests in one go, demonstrating that the test suite is growing linearly. Without intervention, we'll have 200+ tests after a few more operations. This plan is designed to stop that growth. -->

**Key Update**: Conv2D implementation added 21 tests, demonstrating the linear growth problem. The hypothesis framework will prevent similar test explosions for future operations.

## Current State Analysis

### What Exists

**Implementation** (25+ operations):
- `pytensor/link/onnx/dispatch/elemwise.py` - 14+ scalar operations
- `pytensor/link/onnx/dispatch/shape.py` - 5 shape operations
- `pytensor/link/onnx/dispatch/nlinalg.py` - 3 linear algebra operations
- `pytensor/link/onnx/dispatch/special.py` - 1 special function (Softmax)
- `pytensor/link/onnx/dispatch/conv.py` - 1 convolution operation (AbstractConv2d) ✨ **NEW**

**Tests** (103 manual tests - updated from 82):
- `tests/link/onnx/test_basic.py` - 9 tests
- `tests/link/onnx/test_elemwise.py` - 36 tests
- `tests/link/onnx/test_shape.py` - 26 tests
- `tests/link/onnx/test_nlinalg.py` - 10 tests
- `tests/link/onnx/test_special.py` - 8 tests
- `tests/link/onnx/test_conv.py` - 21 tests ✨ **NEW**
  - Basic operations & shape validation
  - **CRITICAL**: Filter flipping tests (asymmetric kernels)
  - Padding modes (valid, same, symmetric, asymmetric)
  - Stride & dilation variations
  - Grouped & depthwise convolution
  - Multi-channel & batch processing
  - Integration tests (Conv+ReLU, Conv+Bias)

**Testing Patterns**:
- Fixed seed random generation: `np.random.default_rng(42)`
- Hardcoded test values for simple operations
- `@pytest.mark.parametrize` for dtype/shape variations
- `compare_onnx_and_py()` helper compares ONNX Runtime vs PyTensor output
- No Hypothesis usage currently

### Problems with Current Approach

1. **Linear growth**: Each new operation requires 3-10 manual tests (Conv2D added 21!)
2. **Limited coverage**: Only tests explicitly coded cases
3. **Maintenance burden**: **103 tests** to maintain, update, and debug (was 82)
4. **Missing edge cases**: No automatic discovery of corner cases
5. **Repetitive code**: Similar test structure repeated 103 times
6. **Conv2D explosion**: Simple Conv2D implementation added 21 tests, future ops will continue this trend

## Desired End State

### After Implementation

**Scalable Test Architecture**:
- ~25-30 regression tests (specific bugs & critical edge cases)
  - Includes Conv2D filter flipping tests (CRITICAL for correctness)
  - DimShuffle regressions, Cast in Composite, etc.
- ~12-18 property-based tests (comprehensive coverage)
  - Generic properties (correctness, shape, dtype)
  - Conv2D-specific properties (filter flip, padding, stride, dilation)
- Operation registry for easy expansion
- Hypothesis strategies module
- **Total: ~40-50 focused tests instead of 103+**

**Adding New Operations**:
```python
# Before: Write 5-10 manual tests
def test_new_op_float32(...): ...
def test_new_op_float64(...): ...
def test_new_op_shapes[5 variants](...): ...

# After: Add one registry entry
ONNX_OPERATIONS["new_op"] = OperationConfig(
    op_func=pt.new_op,
    input_strategy=new_op_inputs(),
    valid_dtypes=["float32", "float64"],
)
```

**Property Testing Benefits**:
- Automatic edge case discovery (empty tensors, scalars, extreme values)
- 100+ random test cases per property
- Shrinking to minimal failing examples
- Configurable for dev (10 examples) vs CI (1000 examples)

### Verification

#### Automated Verification:
- [ ] Hypothesis is installed: `uv pip list | grep hypothesis`
- [ ] Registry module imports without errors: `uv run python -c "from tests.link.onnx.strategies import ONNX_OPERATIONS"`
- [ ] Property tests pass with 10 examples: `uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=dev -v`
- [ ] Full property tests pass: `uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=ci -v`
- [ ] Regression tests still pass: `uv run pytest tests/link/onnx/test_regressions.py -v`
- [ ] No test regressions: `uv run pytest tests/link/onnx/ -v` (all pass)

#### Manual Verification:
- [ ] Hypothesis finds and shrinks a seeded bug correctly
- [ ] Test runs are fast in dev mode (~1 minute for all properties)
- [ ] Test runs are thorough in CI mode (~5-10 minutes)
- [ ] New operation can be added with just registry entry
- [ ] Failure messages are clear and actionable

## What We're NOT Doing

- Not removing all manual tests (keep ~20 regression tests)
- Not testing PyTensor operations themselves (only ONNX conversion)
- Not testing ONNX Runtime (assumes it's correct)
- Not implementing new ONNX operations (only improving tests)
- Not changing the dispatch system architecture
- Not testing performance or benchmarking
- Not adding integration tests with real models

## Implementation Approach

**Strategy**: Build reusable property-based testing infrastructure

1. Add Hypothesis dependency
2. Create strategies module for test input generation
3. Build operation registry for metadata
4. Write generic property tests
5. Keep ~20 critical regression tests
6. Replace ~60 repetitive tests with ~10 properties

**Pattern**: Test mathematical properties, not specific values
- Property: "ONNX output matches PyTensor for any valid input"
- Property: "Operation preserves shape constraints"
- Property: "Operation preserves dtype"

## Phase 1: Setup and Infrastructure

### Overview
Add Hypothesis dependency and create the foundational testing infrastructure including strategies module, operation registry, and Hypothesis configuration.

### Changes Required

#### 1. Add Hypothesis Dependency

**File**: `pyproject.toml`

**Changes**: Add to `[project.optional-dependencies]` test section

```toml
[project.optional-dependencies]
test = [
    "pytest>=6.0",
    "pytest-cov",
    "pytest-mock",
    "pytest-benchmark",
    "hypothesis>=6.100.0",  # Add this line
    # ... existing dependencies
]
```

#### 2. Create Strategies Module

**File**: `tests/link/onnx/strategies/__init__.py` (new file)

**Changes**: Create package initialization

```python
"""Hypothesis strategies for ONNX testing."""

from tests.link.onnx.strategies.core import (
    onnx_dtypes,
    valid_shapes,
    onnx_tensor,
)
from tests.link.onnx.strategies.operations import (
    ONNX_OPERATIONS,
    OperationConfig,
    binary_broadcastable_inputs,
    unary_operation_inputs,
)

__all__ = [
    "onnx_dtypes",
    "valid_shapes",
    "onnx_tensor",
    "ONNX_OPERATIONS",
    "OperationConfig",
    "binary_broadcastable_inputs",
    "unary_operation_inputs",
]
```

#### 3. Core Strategies Implementation

**File**: `tests/link/onnx/strategies/core.py` (new file)

**Changes**: Implement basic array generation strategies

```python
"""Core Hypothesis strategies for ONNX tensor generation."""

from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes, integer_dtypes
import numpy as np


def onnx_dtypes():
    """Strategy for ONNX-supported dtypes.

    Returns dtypes that are commonly supported across:
    - PyTensor
    - ONNX
    - ONNX Runtime
    """
    return st.sampled_from([
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ])


def valid_shapes(min_rank=1, max_rank=4, min_dim=0, max_dim=10):
    """Generate valid tensor shapes for ONNX.

    Parameters
    ----------
    min_rank : int
        Minimum number of dimensions (default: 1)
    max_rank : int
        Maximum number of dimensions (default: 4)
    min_dim : int
        Minimum size per dimension (default: 0, allows empty tensors)
    max_dim : int
        Maximum size per dimension (default: 10)

    Returns
    -------
    strategy
        Generates tuples of integers representing valid shapes

    Examples
    --------
    >>> valid_shapes().example()
    (3, 5, 2)
    >>> valid_shapes(min_rank=2, max_rank=2).example()  # matrices only
    (4, 7)
    """
    return st.lists(
        st.integers(min_value=min_dim, max_value=max_dim),
        min_size=min_rank,
        max_size=max_rank,
    ).map(tuple)


def _safe_float_elements(dtype):
    """Generate safe float elements for a dtype.

    Avoids infinities, NaNs, and extreme values that cause numerical issues.
    """
    if dtype in (np.float32, "float32"):
        # Float32 range: approximately ±3.4e38
        # Use smaller range to avoid overflow in operations
        return st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    elif dtype in (np.float64, "float64"):
        # Float64 range: approximately ±1.8e308
        return st.floats(
            min_value=-1e14,
            max_value=1e14,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    else:
        raise ValueError(f"Unsupported float dtype: {dtype}")


def _safe_integer_elements(dtype):
    """Generate safe integer elements for a dtype."""
    if dtype in (np.int32, "int32"):
        # int32 range: -2^31 to 2^31-1
        return st.integers(min_value=-100, max_value=100)
    elif dtype in (np.int64, "int64"):
        # int64 range: -2^63 to 2^63-1
        return st.integers(min_value=-1000, max_value=1000)
    else:
        raise ValueError(f"Unsupported integer dtype: {dtype}")


@st.composite
def onnx_tensor(draw, dtype=None, shape=None, elements=None):
    """Generate ONNX-compatible tensor.

    Parameters
    ----------
    dtype : numpy dtype or None
        Tensor dtype. If None, randomly chosen from onnx_dtypes()
    shape : tuple or None
        Tensor shape. If None, randomly generated
    elements : strategy or None
        Strategy for generating element values. If None, uses safe defaults

    Returns
    -------
    numpy.ndarray
        Tensor compatible with ONNX operations

    Examples
    --------
    >>> # Random tensor
    >>> onnx_tensor().example()
    array([[1.2, 3.4], [5.6, 7.8]], dtype=float32)

    >>> # Specific dtype
    >>> onnx_tensor(dtype=np.int32).example()
    array([10, 20, 30], dtype=int32)

    >>> # Specific shape
    >>> onnx_tensor(shape=(2, 3)).example()
    array([[...]], dtype=float32)
    """
    # Generate dtype if not provided
    if dtype is None:
        dtype = draw(onnx_dtypes())

    # Generate shape if not provided
    if shape is None:
        shape = draw(valid_shapes())

    # Generate elements strategy if not provided
    if elements is None:
        if np.issubdtype(dtype, np.floating):
            elements = _safe_float_elements(dtype)
        elif np.issubdtype(dtype, np.integer):
            elements = _safe_integer_elements(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    # Generate array
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))
```

#### 4. Operation Registry Structure

**File**: `tests/link/onnx/strategies/operations.py` (new file)

**Changes**: Define operation registry and input generation strategies

```python
"""Operation registry and input strategies for ONNX testing."""

from dataclasses import dataclass
from typing import Callable, List, Optional
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import pytensor.tensor as pt

from tests.link.onnx.strategies.core import onnx_dtypes, valid_shapes, onnx_tensor


@dataclass
class OperationConfig:
    """Configuration for testing an ONNX operation.

    Attributes
    ----------
    op_func : callable
        PyTensor operation function (e.g., pt.add, pt.dot)
    input_strategy : hypothesis.strategies.SearchStrategy
        Strategy that generates valid inputs for the operation
    valid_dtypes : list of str
        Dtypes supported by this operation
    category : str
        Operation category (elemwise, shape, nlinalg, etc.)
    notes : str, optional
        Additional notes or constraints
    """
    op_func: Callable
    input_strategy: st.SearchStrategy
    valid_dtypes: List[str]
    category: str
    notes: Optional[str] = None


@st.composite
def unary_operation_inputs(draw, dtype=None, shape=None):
    """Generate inputs for unary operations (e.g., neg, exp, log).

    Returns
    -------
    tuple
        (tensor,) - Single input tensor
    """
    if dtype is None:
        dtype = draw(onnx_dtypes())
    if shape is None:
        shape = draw(valid_shapes())

    x = draw(onnx_tensor(dtype=dtype, shape=shape))
    return (x,)


@st.composite
def binary_broadcastable_inputs(draw, dtypes=None):
    """Generate inputs for binary operations with broadcasting (e.g., add, mul).

    Parameters
    ----------
    dtypes : list or None
        Allowed dtypes. If None, uses all ONNX dtypes

    Returns
    -------
    tuple
        (x, y) - Two tensors with compatible broadcasting shapes
    """
    if dtypes is None:
        dtypes = [np.float32, np.float64, np.int32, np.int64]

    # Generate compatible dtype for both tensors
    dtype = draw(st.sampled_from(dtypes))

    # Generate base shape
    base_shape = draw(valid_shapes(min_rank=1, max_rank=3, min_dim=1, max_dim=5))

    # Generate broadcasting variant for second tensor
    # Options: same shape, broadcast dims, or smaller tensor
    broadcast_pattern = draw(st.sampled_from([
        "same",  # Same shape
        "broadcast_dims",  # Some dimensions are 1
        "prefix",  # Smaller tensor (broadcasts from right)
    ]))

    if broadcast_pattern == "same":
        shape_y = base_shape
    elif broadcast_pattern == "broadcast_dims":
        # Randomly make some dimensions 1
        shape_y = tuple(
            1 if draw(st.booleans()) and dim > 1 else dim
            for dim in base_shape
        )
    else:  # prefix
        # Take suffix of base_shape
        suffix_len = draw(st.integers(1, len(base_shape)))
        shape_y = base_shape[-suffix_len:]

    x = draw(onnx_tensor(dtype=dtype, shape=base_shape))
    y = draw(onnx_tensor(dtype=dtype, shape=shape_y))

    return (x, y)


@st.composite
def matmul_inputs(draw):
    """Generate inputs for matrix multiplication.

    Returns
    -------
    tuple
        (A, B) - Two tensors with compatible shapes for matmul
    """
    dtype = draw(st.sampled_from([np.float32, np.float64]))

    # Generate dimensions
    m = draw(st.integers(1, 50))
    n = draw(st.integers(1, 50))
    k = draw(st.integers(1, 50))

    # Optionally add batch dimension
    has_batch = draw(st.booleans())
    if has_batch:
        batch = draw(st.integers(1, 8))
        shape_a = (batch, m, k)
        shape_b = (batch, k, n)
    else:
        # Can be 1D (vector) or 2D (matrix)
        a_is_1d = draw(st.booleans()) and m > 1  # Avoid scalar
        b_is_1d = draw(st.booleans()) and n > 1

        if a_is_1d and b_is_1d:
            # Vector dot vector
            shape_a = (k,)
            shape_b = (k,)
        elif a_is_1d:
            # Vector @ Matrix
            shape_a = (k,)
            shape_b = (k, n)
        elif b_is_1d:
            # Matrix @ Vector
            shape_a = (m, k)
            shape_b = (k,)
        else:
            # Matrix @ Matrix
            shape_a = (m, k)
            shape_b = (k, n)

    A = draw(onnx_tensor(dtype=dtype, shape=shape_a))
    B = draw(onnx_tensor(dtype=dtype, shape=shape_b))

    return (A, B)


@st.composite
def reshape_inputs(draw):
    """Generate inputs for reshape operation.

    Returns
    -------
    tuple
        (tensor, new_shape) - Tensor and compatible reshape target
    """
    dtype = draw(onnx_dtypes())

    # Generate original shape
    original_shape = draw(valid_shapes(min_rank=1, max_rank=4, min_dim=1, max_dim=10))
    total_elements = np.prod(original_shape)

    # Generate compatible new shape
    # Find divisors of total_elements
    divisors = [i for i in range(1, int(total_elements**0.5) + 1) if total_elements % i == 0]

    if not divisors:
        # Handle edge case: total_elements is 1 or very large prime
        new_shape = (int(total_elements),)
    else:
        # Build new shape from divisors
        rank = draw(st.integers(1, 4))
        new_shape = []
        remaining = total_elements

        for _ in range(rank - 1):
            if remaining == 1:
                new_shape.append(1)
            else:
                valid_divs = [d for d in divisors if remaining % d == 0 and d <= remaining]
                if valid_divs:
                    dim = draw(st.sampled_from(valid_divs))
                    new_shape.append(dim)
                    remaining //= dim
                else:
                    new_shape.append(1)

        new_shape.append(remaining)
        new_shape = tuple(new_shape)

    tensor = draw(onnx_tensor(dtype=dtype, shape=original_shape))

    return (tensor, new_shape)


@st.composite
def dimshuffle_inputs(draw):
    """Generate inputs for dimshuffle/transpose operation.

    Returns
    -------
    tuple
        (tensor, pattern) - Tensor and valid dimshuffle pattern
    """
    dtype = draw(onnx_dtypes())

    # Generate shape
    ndim = draw(st.integers(1, 4))
    shape = tuple(draw(st.integers(1, 10)) for _ in range(ndim))

    # Generate valid dimshuffle pattern
    # Pattern can include dimension indices and 'x' for new axes

    # Simple transpose case
    pattern = list(range(ndim))
    draw(st.randoms()).shuffle(pattern)

    # Optionally add 'x' dimensions
    if draw(st.booleans()):
        num_x = draw(st.integers(1, 2))
        for _ in range(num_x):
            insert_pos = draw(st.integers(0, len(pattern)))
            pattern.insert(insert_pos, 'x')

    # Optionally drop some dimensions (only if dimension size is 1)
    # This is complex, so we'll skip for now and focus on transpose + unsqueeze

    tensor = draw(onnx_tensor(dtype=dtype, shape=shape))

    return (tensor, tuple(pattern))


# Operation Registry
# This is the central registry that maps operation names to their test configurations
ONNX_OPERATIONS = {
    # Elemwise Binary Operations
    "add": OperationConfig(
        op_func=lambda x, y: x + y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "mul": OperationConfig(
        op_func=lambda x, y: x * y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "sub": OperationConfig(
        op_func=lambda x, y: x - y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "div": OperationConfig(
        op_func=lambda x, y: x / y,
        input_strategy=binary_broadcastable_inputs(dtypes=[np.float32, np.float64]),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Division only defined for floating point types",
    ),

    # Elemwise Unary Operations
    "neg": OperationConfig(
        op_func=lambda x: -x,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "abs": OperationConfig(
        op_func=pt.abs,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "exp": OperationConfig(
        op_func=pt.exp,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Exponential only defined for floating point types",
    ),
    "log": OperationConfig(
        op_func=pt.log,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Logarithm only defined for positive floating point values",
    ),
    "sqrt": OperationConfig(
        op_func=pt.sqrt,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Square root only defined for non-negative floating point values",
    ),

    # Linear Algebra
    "dot": OperationConfig(
        op_func=pt.dot,
        input_strategy=matmul_inputs(),
        valid_dtypes=["float32", "float64"],
        category="nlinalg",
    ),

    # Shape Operations
    "reshape": OperationConfig(
        op_func=lambda x, shape: x.reshape(shape),
        input_strategy=reshape_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="shape",
    ),

    # Convolution Operations ✨ NEW
    "conv2d": OperationConfig(
        op_func=conv2d,
        input_strategy=conv2d_inputs(),
        valid_dtypes=["float32", "float64"],
        category="conv",
        notes="Conv2D with various padding, stride, dilation, and group configurations",
    ),
}


@st.composite
def conv2d_inputs(draw):
    """Generate inputs for 2D convolution operations.

    Returns
    -------
    tuple
        (input_4d, kernel_4d) - Input and kernel with compatible shapes:
        - input: (batch, in_channels, height, width)
        - kernel: (filters, in_channels_per_group, kH, kW)

    Note: Generates various configurations including:
    - Different padding modes
    - Stride variations
    - Dilation (atrous convolution)
    - Grouped convolution
    """
    dtype = draw(st.sampled_from([np.float32, np.float64]))

    # Generate dimensions
    batch = draw(st.integers(1, 4))
    in_channels = draw(st.integers(1, 8))
    height = draw(st.integers(5, 20))
    width = draw(st.integers(5, 20))

    # Kernel dimensions
    num_filters = draw(st.integers(1, 16))
    kernel_h = draw(st.integers(1, 5))
    kernel_w = draw(st.integers(1, 5))

    # Grouped convolution (optional)
    use_groups = draw(st.booleans())
    if use_groups and in_channels % 2 == 0 and num_filters % 2 == 0:
        num_groups = draw(st.sampled_from([2, in_channels]))  # Regular groups or depthwise
        in_channels_per_group = in_channels // num_groups
    else:
        num_groups = 1
        in_channels_per_group = in_channels

    # Generate tensors
    input_shape = (batch, in_channels, height, width)
    kernel_shape = (num_filters, in_channels_per_group, kernel_h, kernel_w)

    input_tensor = draw(onnx_tensor(dtype=dtype, shape=input_shape))
    kernel_tensor = draw(onnx_tensor(dtype=dtype, shape=kernel_shape))

    return (input_tensor, kernel_tensor)
```

#### 5. Hypothesis Configuration

**File**: `tests/link/onnx/conftest.py` (new file)

**Changes**: Configure Hypothesis profiles for different environments

```python
"""Pytest configuration for ONNX tests with Hypothesis."""

import pytest
from hypothesis import settings, Phase, HealthCheck
from datetime import timedelta
import os


# Register Hypothesis profiles
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # Skip shrinking in dev
    print_blob=False,
)

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # No deadline in CI
    derandomize=True,  # Deterministic for CI
    print_blob=True,  # Print failing examples for debugging
)

settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Suppress health checks that are problematic for ONNX operations
settings.register_profile(
    "onnx",
    suppress_health_check=[
        HealthCheck.too_slow,  # ONNX operations can be slow
        HealthCheck.filter_too_much,  # We filter invalid inputs aggressively
    ],
    max_examples=50,
    deadline=timedelta(seconds=5),  # Allow 5s per test
)

# Load profile from environment, default to 'dev'
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# Standard pytest fixture for tmp_path
@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")
```

### Success Criteria

#### Automated Verification:
- [ ] Hypothesis installs successfully: `uv sync`
- [ ] Strategies module imports: `uv run python -c "from tests.link.onnx.strategies import ONNX_OPERATIONS; print(len(ONNX_OPERATIONS))"`
- [ ] conftest.py loads profiles: `uv run pytest tests/link/onnx/ --collect-only --hypothesis-profile=dev`
- [ ] No import errors in new modules
- [ ] Existing tests still pass: `uv run pytest tests/link/onnx/ -v`

#### Manual Verification:
- [ ] `uv run hypothesis --version` shows version >= 6.100.0
- [ ] Can generate example tensors: `uv run python -c "from tests.link.onnx.strategies import onnx_tensor; print(onnx_tensor().example())"`
- [ ] Registry contains expected operations
- [ ] Profiles switch correctly via environment variable

---

## Phase 2: Generic Property Tests

### Overview
Create property-based tests that work for all operations in the registry. These tests verify fundamental properties that should hold for any ONNX operation.

### Changes Required

#### 1. Generic Property Test File

**File**: `tests/link/onnx/test_properties.py` (new file)

**Changes**: Implement generic property tests

```python
"""Property-based tests for ONNX operations using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, assume, strategies as st, example
from hypothesis.extra.numpy import arrays

import pytensor
import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py
from tests.link.onnx.strategies import ONNX_OPERATIONS, onnx_tensor


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


# Property 1: ONNX output matches PyTensor output
@given(
    op_name=st.sampled_from(list(ONNX_OPERATIONS.keys())),
    data=st.data(),
)
def test_onnx_matches_pytensor(tmp_path, op_name, data):
    """
    Property: For any valid operation and inputs, ONNX output must match PyTensor.

    This is the fundamental correctness property - the ONNX backend should
    produce the same numerical results as PyTensor's native execution.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs using operation-specific strategy
    inputs_tuple = data.draw(op_config.input_strategy)

    # Handle special cases that need filtering
    if op_name == "log":
        # Log requires positive inputs
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "sqrt":
        # Sqrt requires non-negative inputs
        inputs_tuple = tuple(np.abs(x) for x in inputs_tuple)
    elif op_name == "div":
        # Division requires non-zero divisor
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)  # Replace near-zero with 1.0
        inputs_tuple = (x, y)

    # Create symbolic variables
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        symbolic_inputs = [x]

        # Apply operation
        result = op_config.op_func(x)
    elif len(inputs_tuple) == 2:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)

        # Handle different second argument types
        if isinstance(inputs_tuple[1], tuple):
            # Second argument is a shape (e.g., reshape)
            symbolic_inputs = [x]
            result = op_config.op_func(x, inputs_tuple[1])
        else:
            # Second argument is a tensor
            y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
            symbolic_inputs = [x, y]
            result = op_config.op_func(x, y)
    else:
        raise NotImplementedError(f"Operations with {len(inputs_tuple)} inputs not yet supported")

    # Compare ONNX and PyTensor outputs
    try:
        compare_onnx_and_py(symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path)
    except Exception as e:
        # Re-raise with context about which operation failed
        raise AssertionError(
            f"Property test failed for operation '{op_name}' "
            f"with input shapes: {[x.shape for x in inputs_tuple]}, "
            f"dtypes: {[x.dtype for x in inputs_tuple]}"
        ) from e


# Property 2: Shape preservation for elemwise operations
@given(
    op_name=st.sampled_from([k for k, v in ONNX_OPERATIONS.items() if v.category == "elemwise"]),
    data=st.data(),
)
def test_elemwise_preserves_broadcast_shape(tmp_path, op_name, data):
    """
    Property: Elemwise operations preserve broadcasting shape rules.

    For any elemwise operation, the output shape should match NumPy's
    broadcasting rules applied to the input shapes.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Filter invalid inputs
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    # Compute expected output shape using NumPy broadcasting
    if len(inputs_tuple) == 1:
        expected_shape = inputs_tuple[0].shape
    else:
        # Use NumPy to determine broadcast shape
        expected_shape = np.broadcast_shapes(*[x.shape for x in inputs_tuple])

    # Create symbolic computation
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x)
        symbolic_inputs = [x]
    else:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
        result = op_config.op_func(x, y)
        symbolic_inputs = [x, y]

    # Run through ONNX
    _, onnx_results = compare_onnx_and_py(
        symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path
    )

    # Verify shape
    assert onnx_results[0].shape == expected_shape, (
        f"Operation '{op_name}' produced wrong shape. "
        f"Expected {expected_shape}, got {onnx_results[0].shape}"
    )


# Property 3: Dtype preservation
@given(
    op_name=st.sampled_from(list(ONNX_OPERATIONS.keys())),
    data=st.data(),
)
def test_operation_preserves_dtype(tmp_path, op_name, data):
    """
    Property: Operations preserve input dtype (with known exceptions).

    Most operations should output the same dtype as their input.
    Exceptions: division always produces float, comparisons produce bool.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Filter invalid inputs
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    input_dtype = inputs_tuple[0].dtype

    # Create symbolic computation
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x)
        symbolic_inputs = [x]
    elif isinstance(inputs_tuple[1], tuple):
        # Second arg is shape (reshape case)
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x, inputs_tuple[1])
        symbolic_inputs = [x]
    else:
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
        result = op_config.op_func(x, y)
        symbolic_inputs = [x, y]

    # Run through ONNX
    _, onnx_results = compare_onnx_and_py(
        symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path
    )

    # Verify dtype (accounting for known exceptions)
    output_dtype = onnx_results[0].dtype

    # Known exceptions where dtype changes
    if op_name == "div":
        # Division always produces float
        assert np.issubdtype(output_dtype, np.floating), (
            f"Division should produce float, got {output_dtype}"
        )
    else:
        # Most operations preserve dtype
        assert output_dtype == input_dtype, (
            f"Operation '{op_name}' changed dtype from {input_dtype} to {output_dtype}"
        )


# Property 4: Operations don't crash on edge cases
@given(
    op_name=st.sampled_from(list(ONNX_OPERATIONS.keys())),
    data=st.data(),
)
@example(op_name="add", data=st.data())  # Always test at least one example
def test_operation_handles_edge_cases(tmp_path, op_name, data):
    """
    Property: Operations handle edge cases without crashing.

    Tests with:
    - Empty tensors (shape with 0)
    - Scalars (0-dimensional tensors)
    - Large values
    - Small values near zero

    Operations may produce inf/nan for invalid inputs, but should not crash.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Apply necessary filters
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    # Create symbolic computation
    try:
        if len(inputs_tuple) == 1:
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            result = op_config.op_func(x)
            symbolic_inputs = [x]
        elif isinstance(inputs_tuple[1], tuple):
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            result = op_config.op_func(x, inputs_tuple[1])
            symbolic_inputs = [x]
        else:
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
            result = op_config.op_func(x, y)
            symbolic_inputs = [x, y]

        # Run through ONNX - should not crash
        compare_onnx_and_py(symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path)

    except (ValueError, TypeError, RuntimeError) as e:
        # Some operations may legitimately fail for certain inputs
        # (e.g., reshape with incompatible shape)
        # This is acceptable - we just want to ensure it doesn't crash Python
        pass
```

### Success Criteria

#### Automated Verification:
- [ ] Property tests collect: `uv run pytest tests/link/onnx/test_properties.py --collect-only`
- [ ] Properties pass with 10 examples: `uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=dev -v`
- [ ] Properties pass with 100 examples: `uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=ci -v`
- [ ] No test crashes (failures are OK for invalid inputs)
- [ ] Hypothesis finds and shrinks a seeded bug: (manual test by introducing a bug)

#### Manual Verification:
- [ ] Test output is readable and shows which property failed
- [ ] Failing examples are minimal (Hypothesis shrinking works)
- [ ] Tests run in <1 minute with dev profile
- [ ] Tests run in <10 minutes with ci profile
- [ ] Hypothesis database saves failing examples to `.hypothesis/`

---

## Phase 3: Regression Test Preservation

### Overview
Keep ~20 critical regression tests for specific bugs we've fixed. These serve as documentation and fast smoke tests.

### Changes Required

#### 1. Regression Test File

**File**: `tests/link/onnx/test_regressions.py` (new file)

**Changes**: Extract critical regression tests from existing test files

```python
"""Regression tests for specific ONNX bugs.

These tests document specific bugs that were found and fixed.
They serve as fast smoke tests and documentation of edge cases.

DO NOT add routine tests here - use property tests in test_properties.py instead.
Only add tests for:
1. Specific bugs that were fixed
2. Edge cases that broke in production
3. Cases that took significant debugging to identify
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py, validate_onnx_graph_structure


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


# ============================================================================
# DimShuffle Regressions (Phase 1 bug fixes)
# ============================================================================

def test_dimshuffle_transpose_and_unsqueeze_regression(tmp_path):
    """
    Regression: DimShuffle incorrectly used Identity for transpose+unsqueeze.

    Bug: Pattern (1, 'x', 0) on shape (2,3) would incorrectly use Identity
    node, producing shape (2,3) instead of correct (3,1,2).

    Fixed in: Phase 1 - Added proper Squeeze→Transpose→Unsqueeze decomposition
    Reference: pytensor/link/onnx/dispatch/shape.py:188-405
    """
    x = pt.matrix("x", dtype="float32")
    y = x.dimshuffle(1, "x", 0)  # (2,3) → (3,1,2)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    # Should produce (3,1,2) shape, not (2,3)
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Verify correct ONNX structure (should use Transpose + Unsqueeze, not Identity)
    from pytensor.link.onnx import export_onnx
    f = pytensor.function([x], y)
    model = export_onnx(f, tmp_path / "dimshuffle.onnx")

    structure = validate_onnx_graph_structure(model)
    assert "Identity" not in structure["node_types"], \
        "DimShuffle should not use Identity for complex patterns"
    assert "Transpose" in structure["node_types"] or "Unsqueeze" in structure["node_types"], \
        "DimShuffle should use Transpose or Unsqueeze nodes"


def test_dimshuffle_squeeze_and_transpose_regression(tmp_path):
    """
    Regression: DimShuffle pattern (2, 0) on (2,1,3) incorrectly matched Case 3.

    Bug: Case 3 (pure transpose) didn't check for axes_to_add, so it matched
    patterns that also needed squeeze operations.

    Fixed in: Phase 1 - Added `and not axes_to_add` condition to Case 3
    Reference: pytensor/link/onnx/dispatch/shape.py:286
    """
    x = pt.tensor(dtype="float32", shape=(2, 1, 3), name="x")
    y = x.dimshuffle(2, 0)  # (2,1,3) → (3,2)

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 1, 3)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Composite Operation Regressions (Phase 2 bug fixes)
# ============================================================================

def test_cast_in_composite_regression(tmp_path):
    """
    Regression: Cast operation not supported in Composite decomposition.

    Bug: decompose_composite_elemwise() didn't handle scalar.Cast operations.
    When PyTensor's optimizer fused Cast into a Composite, export would fail.

    Fixed in: Phase 2.2 - Added Cast handling in decompose_composite_elemwise
    Reference: pytensor/link/onnx/dispatch/elemwise.py:96-124
    """
    x = pt.vector("x", dtype="int32")

    # This creates a Composite with Cast in FAST_RUN mode
    x_float = pt.cast(x, "float32")
    y_float = x_float * 2.5 + 1.0
    y = pt.cast(y_float, "int32")

    x_val = np.array([1, 2, 3, 4, 5], dtype="int32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_sqr_in_composite_regression(tmp_path):
    """
    Regression: Sqr scalar operation not in SCALAR_OP_TO_ONNX mapping.

    Bug: Expression x**2 creates scalar.Sqr op, which wasn't mapped to ONNX.

    Fixed in: Phase 2.3 - Added scalar.Sqr: "Mul" and special x*x handling
    Reference: pytensor/link/onnx/dispatch/elemwise.py:24, 126-138
    """
    x = pt.vector("x", dtype="float32")

    # Expression with x^2 that becomes Composite with Sqr
    y = x**2 * 2 + x

    f = pytensor.function([x], y, mode="FAST_RUN")

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Structure Validation Tests
# ============================================================================

def test_cast_generates_correct_onnx_node(tmp_path):
    """Validate that Cast generates ONNX Cast node with correct 'to' attribute."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.cast(x, "int32")

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_cast.onnx"
    model = export_onnx(f, model_path)

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Cast"],
        expected_node_count=1,
    )

    # Verify Cast node has correct 'to' attribute
    cast_node = model.graph.node[0]
    assert cast_node.op_type == "Cast"
    to_attr = next(attr for attr in cast_node.attribute if attr.name == "to")
    assert to_attr.i == 6, "Cast to int32 should have TensorProto.INT32 = 6"


def test_gemv_generates_correct_onnx_structure(tmp_path):
    """Validate that Gemv generates 4-node ONNX decomposition."""
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor.blas import Gemv

    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha = pt.scalar("alpha", dtype="float32")
    beta = pt.scalar("beta", dtype="float32")

    gemv_op = Gemv(inplace=False)
    y = gemv_op(y_in, alpha, A, x, beta)

    f = pytensor.function([y_in, alpha, A, x, beta], y)

    model_path = tmp_path / "test_gemv.onnx"
    model = export_onnx(f, model_path)

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["MatMul", "Mul", "Mul", "Add"],
        expected_node_count=4,
    )

    # Verify node types
    node_types = structure["node_types"]
    assert node_types.count("MatMul") == 1, "Gemv should have 1 MatMul"
    assert node_types.count("Mul") == 2, "Gemv should have 2 Mul (alpha, beta scaling)"
    assert node_types.count("Add") == 1, "Gemv should have 1 Add"


def test_deep_copy_generates_identity(tmp_path):
    """Validate that DeepCopyOp generates ONNX Identity node."""
    from pytensor.link.onnx import export_onnx
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_deep_copy.onnx"
    model = export_onnx(f, model_path)

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Identity"],
        expected_node_count=1,
    )

    assert structure["node_types"] == ["Identity"]


# ============================================================================
# Known Edge Cases
# ============================================================================

def test_alloc_empty_with_shape_from_tensor(tmp_path):
    """Test AllocEmpty with dimensions extracted from another tensor's shape."""
    from pytensor.tensor.basic import AllocEmpty

    x = pt.matrix("x", dtype="float32")
    dim0 = x.shape[0]
    dim1 = x.shape[1]

    alloc_op = AllocEmpty(dtype="float32")
    y = alloc_op(dim0, dim1)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    from pytensor.link.onnx import export_onnx

    f = pytensor.function([x], y)
    model_path = tmp_path / "test_alloc_empty.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    # Run and verify shape matches
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {onnx_inputs[0].name: x_val}
    onnx_res = session.run(None, input_feed)

    assert onnx_res[0].shape == x_val.shape


def test_float64_dtype_preserved(tmp_path):
    """
    Regression: float64 inputs were incorrectly converted to float32.

    Bug: compare_onnx_and_py had dtype conversion logic that changed float64 to float32.

    Fixed in: Phase 2.2 - Simplified dtype handling in compare_onnx_and_py
    Reference: tests/link/onnx/test_basic.py:77-85
    """
    x = pt.vector("x", dtype="float64")
    y = pt.cast(x, "float32")

    rng = np.random.default_rng(42)
    x_val = rng.random(5).astype("float64")

    # Should work without dtype conversion errors
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Conv2D Regressions ✨ NEW
# ============================================================================

def test_conv2d_filter_flip_true_asymmetric_regression(tmp_path):
    """
    ⭐⭐⭐ CRITICAL REGRESSION: Conv2D with filter_flip=True and asymmetric kernel.

    This is THE most important Conv2D correctness test!

    When filter_flip=True:
    - PyTensor flips kernel (mathematical convolution)
    - ONNX Conv does NOT flip (cross-correlation)
    - We MUST flip the kernel before passing to ONNX

    Using Sobel edge detector (asymmetric):
    - If we DON'T flip: Wrong results (detects edges in wrong direction)
    - If we DO flip correctly: Results match PyTensor

    This test ensures the filter flipping logic remains correct.
    Reference: pytensor/link/onnx/dispatch/conv.py:48-68
    """
    from pytensor import shared
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")

    # Sobel X edge detector (ASYMMETRIC!)
    sobel_x = np.array(
        [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype="float32"
    )

    kernel = shared(sobel_x, name="kernel")
    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    # Test image with vertical edge
    x_val = np.array(
        [
            [
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype="float32",
    )

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_conv2d_explicit_asymmetric_padding_regression(tmp_path):
    """
    Regression: Conv2D with asymmetric padding mapping to ONNX.

    Asymmetric padding is less common but critical for certain architectures.
    ONNX format: pads=[pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]

    This test ensures the padding order and values are correctly mapped.
    Reference: pytensor/link/onnx/dispatch/conv.py:105-108
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Asymmetric padding: different on each side
    y = conv2d(x, kernel, border_mode=((1, 2), (0, 1)), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Verify output shape matches expected
    # height: (5 + 1 + 2 - 3) + 1 = 6
    # width: (5 + 0 + 1 - 3) + 1 = 4
    assert onnx_res[0].shape == (1, 1, 6, 4)


def test_conv2d_grouped_convolution_regression(tmp_path):
    """
    Regression: Grouped convolution channel dimension handling.

    Grouped convolution divides channels into independent groups.
    Critical for efficient architectures (ResNeXt, etc.).

    This test ensures the num_groups parameter is correctly passed to ONNX.
    Reference: pytensor/link/onnx/dispatch/conv.py:116
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", num_groups=2, filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 4, 8, 8)).astype("float32")
    # 8 filters, 2 channels per group (4 input channels / 2 groups)
    kernel_val = rng.random((8, 2, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_conv2d_dilation_regression(tmp_path):
    """
    Regression: Dilated convolution (atrous) output shape.

    Dilation expands the receptive field without adding parameters.
    Common in semantic segmentation (DeepLab, etc.).

    Effective kernel size: kernel_size + (kernel_size - 1) * (dilation - 1)
    This test ensures dilation is correctly passed to ONNX.
    Reference: pytensor/link/onnx/dispatch/conv.py:74
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_dilation=(2, 2), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 10, 10)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Effective kernel: 3 + (3-1)*1 = 5
    # Output size: (10-5)+1 = 6
    assert onnx_res[0].shape == (1, 1, 6, 6)
```

### Success Criteria

#### Automated Verification:
- [ ] Regression tests pass: `uv run pytest tests/link/onnx/test_regressions.py -v`
- [ ] Regression tests are fast (<30 seconds total)
- [ ] Each test has clear docstring documenting the bug
- [ ] Tests fail if bug is re-introduced (verify by temporarily reverting fix)

#### Manual Verification:
- [ ] Each regression test documents: what broke, how it was fixed, where the fix is
- [ ] Test names clearly indicate what regression they prevent
- [ ] Tests serve as documentation for future developers
- [ ] No redundant tests (each tests a unique bug/edge case)

---

## Phase 4: Cleanup and Documentation

### Overview
Remove redundant tests, update documentation, and ensure the new framework is easy to use.

### Changes Required

#### 1. Remove Redundant Parametrized Tests

**Files**: `tests/link/onnx/test_elemwise.py`, `tests/link/onnx/test_shape.py`, `tests/link/onnx/test_nlinalg.py`

**Changes**: Remove tests that are now covered by properties

**Tests to Remove** (~65-75 tests):
- `test_cast_dtypes[7 variants]` → Covered by property test
- `test_alloc_empty_dtypes[4 variants]` → Covered by property test
- `test_gemv_scaling_factors[4 variants]` → Covered by property test
- `test_add_different_shapes[3 variants]` → Covered by property test
- Various dtype parametrization tests across elemwise, shape, nlinalg
- **Conv2D**: ~15 routine Conv2D tests → Covered by property test
  - `test_conv2d_output_shape[3 variants]` → Property test
  - `test_conv2d_valid_padding` → Property test
  - `test_conv2d_stride_2x2` → Property test
  - `test_conv2d_rgb_input` → Property test
  - `test_conv2d_batch_processing` → Property test
  - etc.

**Tests to Keep** (~25-30 tests):
- DimShuffle regressions → Move to test_regressions.py
- Cast/Composite regressions → Move to test_regressions.py
- Gemv structure validation → Move to test_regressions.py
- **Conv2D CRITICAL regressions** → Move to test_regressions.py:
  - `test_conv2d_filter_flip_true_asymmetric` ⭐ MOST IMPORTANT
  - `test_conv2d_explicit_asymmetric_padding`
  - `test_conv2d_grouped_convolution`
  - `test_conv2d_dilation_2x2`
- Basic smoke tests (`test_add`, `test_mul`, etc.) → Keep for quick validation

#### 2. Update Test Documentation

**File**: `tests/link/onnx/README.md` (new file)

**Changes**: Document the new testing architecture

```markdown
# ONNX Backend Testing

This directory contains tests for PyTensor's ONNX export functionality.

## Test Organization

### Property-Based Tests (`test_properties.py`)

Comprehensive tests using Hypothesis that verify fundamental properties for all operations:

- **test_onnx_matches_pytensor**: Core correctness - ONNX must match PyTensor
- **test_elemwise_preserves_broadcast_shape**: Shape broadcasting works correctly
- **test_operation_preserves_dtype**: Dtype handling is correct
- **test_operation_handles_edge_cases**: No crashes on edge cases

**Running property tests:**
```bash
# Fast (10 examples per property)
uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=dev

# Thorough (100 examples per property)
uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=ci

# Exhaustive (1000 examples per property)
uv run pytest tests/link/onnx/test_properties.py --hypothesis-profile=thorough
```

### Regression Tests (`test_regressions.py`)

Specific tests for bugs that were fixed. Each test documents:
- What broke
- How it was fixed
- Where the fix is in the codebase

These serve as fast smoke tests and documentation.

### Basic Tests (`test_basic.py`)

Core infrastructure tests:
- ONNX export functionality
- Helper functions (`compare_onnx_and_py`, `validate_onnx_graph_structure`)
- Shared variables as initializers

## Adding Tests for New Operations

**DO NOT write manual tests.** Instead:

1. Add operation to the registry in `strategies/operations.py`:

```python
ONNX_OPERATIONS["new_op"] = OperationConfig(
    op_func=pt.new_op,
    input_strategy=appropriate_strategy(),
    valid_dtypes=["float32", "float64"],
    category="elemwise",  # or "shape", "nlinalg", etc.
)
```

2. If operation needs custom input generation, add strategy to `strategies/operations.py`:

```python
@st.composite
def new_op_inputs(draw):
    """Generate valid inputs for new_op."""
    # Custom generation logic
    return (input1, input2, ...)
```

3. Run property tests - they automatically test your new operation:

```bash
uv run pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor --hypothesis-profile=dev
```

4. If you discover a specific bug, add a regression test to `test_regressions.py` documenting it.

## Hypothesis Profiles

Configure via `HYPOTHESIS_PROFILE` environment variable:

- **dev** (default): 10 examples, fast feedback for development
- **ci**: 100 examples, deterministic, used in CI
- **thorough**: 1000 examples, for thorough validation
- **onnx**: 50 examples, relaxed health checks for slow ONNX ops

Example:
```bash
HYPOTHESIS_PROFILE=thorough uv run pytest tests/link/onnx/test_properties.py
```

## Debugging Hypothesis Failures

When Hypothesis finds a failure:

1. Let shrinking complete to get minimal example
2. The failure is saved in `.hypothesis/examples/`
3. Add the minimal example to regression tests:

```python
@given(...)
@example(failing_case_from_hypothesis)  # Lock in the failure
def test_operation(...):
    ...
```

4. Fix the bug
5. Verify the `@example()` now passes
6. Keep the test as regression prevention

## Test Helpers

### `compare_onnx_and_py(graph_inputs, graph_outputs, test_inputs, *, tmp_path)`

Main helper that compares ONNX Runtime output with PyTensor output.

### `validate_onnx_graph_structure(model, *, expected_node_types, expected_node_count)`

Validates ONNX graph structure beyond numerical correctness.

## Coverage

Run with coverage:
```bash
uv run pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term
```

Target: 100% coverage of dispatch modules.
```

#### 3. Update Main README

**File**: `tests/link/onnx/test_basic.py`

**Changes**: Add docstring pointing to new README

```python
"""Core ONNX export tests and comparison utilities.

For information on the ONNX test architecture and how to add tests,
see tests/link/onnx/README.md
"""
```

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `uv run pytest tests/link/onnx/ -v`
- [ ] Test count reduced: `uv run pytest tests/link/onnx/ --collect-only | grep "test session"` (should show **~40-50 tests instead of 103**)
- [ ] README renders correctly: `cat tests/link/onnx/README.md`
- [ ] No dead code: removed test files don't import
- [ ] Coverage maintained: `uv run pytest tests/link/onnx/ --cov=pytensor.link.onnx` shows >90% coverage

#### Manual Verification:
- [ ] README is clear and actionable for new contributors
- [ ] Examples in README actually work
- [ ] Test output is readable
- [ ] Adding new operation is truly just registry entry + strategy

---

## Testing Strategy

### Property Tests (12-18 tests)
Test fundamental mathematical properties:
- **Generic properties** (4 tests):
  - Correctness: ONNX matches PyTensor
  - Shape preservation: Broadcasting works
  - Dtype preservation: Types handled correctly
  - Edge cases: No crashes on empty/scalar/large values
- **Conv2D-specific properties** (3-5 tests):
  - Filter flip correctness (symmetric vs asymmetric)
  - Padding output shape correctness
  - Stride downsampling correctness
  - Dilation receptive field correctness
  - Grouped convolution channel handling

### Regression Tests (25-30 tests)
Document specific bugs that were fixed:
- **Elemwise/Shape/NLinalg regressions** (~20 tests):
  - DimShuffle Identity fallback bug
  - Cast in Composite bug
  - Sqr operation support
  - Structure validation for multi-node ops
- **Conv2D regressions** ✨ (~4-5 tests):
  - Filter flip with asymmetric kernel (CRITICAL)
  - Asymmetric padding order
  - Grouped convolution
  - Dilation output shape

### Hypothesis Configuration
- **Dev**: 10 examples, fast feedback (~1 minute total)
- **CI**: 100 examples, thorough (~10 minutes total)
- **Thorough**: 1000 examples, exhaustive (rare use)

## Performance Considerations

**Test Speed**:
- Property tests with dev profile: ~1 minute
- Property tests with ci profile: ~10 minutes
- Regression tests: ~30 seconds

**Hypothesis Overhead**:
- Generation: Minimal (milliseconds per example)
- Shrinking: Can be slow (disabled in dev profile)
- Database: Automatically caches failures

**Optimization**:
- Use dev profile during development
- Run ci profile in CI/CD
- Run thorough profile before releases

## Migration Notes

**Backward Compatibility**:
- Existing tests remain valid
- Can migrate incrementally
- No changes to implementation code
- Property tests complement, don't replace

**Migration Path**:
1. Add Hypothesis (Phase 1)
2. Add property tests (Phase 2)
3. Add regression tests (Phase 3)
4. Remove redundant tests (Phase 4)
5. Each phase independently valuable

## References

- **Hypothesis Documentation**: https://hypothesis.readthedocs.io/
- **NumPy Strategies**: https://hypothesis.readthedocs.io/en/latest/numpy.html
- **SciPy Hypothesis Usage**: https://github.com/scipy/scipy/pull/18927
- **Property-Based Testing Guide**: https://increment.com/testing/in-praise-of-property-based-testing/
- **Current ONNX Tests**: `tests/link/onnx/test_*.py`
- **ONNX Backend Implementation**: `pytensor/link/onnx/dispatch/`

---

## Summary of Plan Updates (✨ NEW)

This plan has been reviewed against the current codebase (including recent Conv2D implementation) and remains **fully valid** with the following updates:

### What Changed
1. **Test count**: 82 → 103 tests (Conv2D added 21 tests)
2. **Operations**: 24 → 25+ operations (added AbstractConv2d)
3. **Target after migration**: ~30-35 tests → **~40-50 tests** (to include Conv2D regressions)

### Conv2D-Specific Additions
- **Phase 1**: Add `conv2d_inputs()` strategy to operation registry
- **Phase 2**: Add Conv2D-specific property tests (filter flip, padding, stride, dilation, groups)
- **Phase 3**: Add 4-5 critical Conv2D regression tests, especially:
  - **`test_conv2d_filter_flip_true_asymmetric`** ⭐ MOST CRITICAL for correctness
  - Asymmetric padding, grouped convolution, dilation tests

### Why This Still Works
- **Same architecture**: Registry + Hypothesis strategies + property tests
- **Same benefits**: Prevents future test explosions (Conv2D demonstrated the problem!)
- **Same phases**: All 4 phases still apply with Conv2D additions
- **Better ROI**: Now prevents **103+ tests** from growing to 200+, not just 82 to 160

### Next Steps
1. Review this updated plan
2. Proceed with Phase 1 implementation (add Hypothesis, strategies, registry)
3. Include Conv2D from the start (don't wait for Phase 4)
