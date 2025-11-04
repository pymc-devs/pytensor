# ONNX Tier 1 Blockers: Concat, MaxPool, Upsample - TDD Implementation Plan

## Overview

This plan implements Test-Driven Development for the **3 critical blocker operations** needed for YOLO11n support in PyTensor's ONNX backend. These operations completely block YOLO11n export and must be implemented first.

**Operations covered:**
1. **Concat (Join → ONNX Concat)** - Used 6+ times in YOLO11n head for skip connections
2. **MaxPool** - Used in SPPF block in backbone
3. **Upsample/Resize** - Used 2 times in FPN head for 2x upsampling

**Total estimated effort:** 3-4 days (1-1.5 days per operation)

## Current State Analysis

### Existing Infrastructure

**Test Infrastructure:**
- **Helper**: `compare_onnx_and_py()` in `tests/link/onnx/test_basic.py:22-102`
  - Compiles PyTensor function
  - Exports to ONNX
  - Runs both PyTensor and ONNX Runtime
  - Compares outputs with `np.testing.assert_allclose(rtol=1e-4)`
- **Fixtures**: `tmp_path` pytest fixture for ONNX file storage
- **Property-based testing**: Hypothesis strategies in `tests/link/onnx/strategies/`

**Dispatcher Pattern:**
```python
# pytensor/link/onnx/dispatch/basic.py:29-70
@onnx_funcify.register(OpClass)
def onnx_funcify_OpName(op, node, var_names, get_var_name, **kwargs):
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]
    return helper.make_node("ONNXOpName", inputs=..., outputs=..., **attributes)
```

**Converter Examples:**
- **Simple**: Dot → MatMul (10 lines) in `nlinalg.py:13-29`
- **Complex**: Conv2D (140 lines) in `conv.py:14-140`
- **Multi-node**: Gemv (60 lines) in `nlinalg.py:48-109`

### What Exists in PyTensor

1. **Join Op** ✅ - `pytensor/tensor/basic.py:2420`
   - Concatenates tensors along an axis
   - Takes axis as first argument
   - Already fully implemented
   - **Just needs ONNX converter**

2. **MaxPool Op** ❌ - Does NOT exist
   - Research document incorrectly stated `pytensor/tensor/nnet/pool.py` exists
   - No `pytensor/tensor/nnet/` directory exists
   - **Must create Op class + ONNX converter**

3. **Upsample Op** ⚠️ - Partial
   - `bilinear_upsampling()` function exists in `pytensor/tensor/conv/abstract_conv.py:1933-2053`
   - Only supports bilinear mode
   - YOLO11n needs **nearest neighbor** mode
   - **Must create general Resize Op + ONNX converter**

### ONNX Target Specifications

**ONNX Opset 18** (current target in `basic.py:26`):

1. **Concat** - [ONNX Spec](https://onnx.ai/onnx/operators/onnx__Concat.html)
   - Inputs: List of tensors (2+)
   - Attributes: `axis` (int)
   - Output: Single concatenated tensor

2. **MaxPool** - [ONNX Spec](https://onnx.ai/onnx/operators/onnx__MaxPool.html)
   - Inputs: X (tensor)
   - Attributes:
     - `kernel_shape` (list of ints, required)
     - `strides` (list of ints, default=[1,1,...])
     - `pads` (list of ints, default=[0,0,...,0,0])
     - `auto_pad` (string, default="NOTSET")
     - `dilations` (list of ints, default=[1,1,...])
   - Outputs: Y (tensor)

3. **Resize** - [ONNX Spec](https://onnx.ai/onnx/operators/onnx__Resize.html)
   - Inputs: X, roi (optional), scales (optional), sizes (optional)
   - Attributes:
     - `mode` (string: "nearest", "linear", "cubic")
     - `coordinate_transformation_mode` (string, default="half_pixel")
     - `nearest_mode` (string, default="round_prefer_floor")
   - Output: Y (tensor)

## Desired End State

After implementation:

1. **Concat converter implemented**:
   - File: `pytensor/link/onnx/dispatch/join.py` (NEW)
   - Converts `Join` op to ONNX `Concat`
   - Test file: `tests/link/onnx/test_join.py` (NEW)
   - ~10 unit tests + property-based tests

2. **MaxPool op + converter implemented**:
   - Files:
     - `pytensor/tensor/pool.py` (NEW) - Op definition
     - `pytensor/link/onnx/dispatch/pool.py` (NEW) - ONNX converter
   - Test files:
     - `tests/tensor/test_pool.py` (NEW) - PyTensor op tests
     - `tests/link/onnx/test_pool.py` (NEW) - ONNX conversion tests
   - ~15 unit tests + property-based tests

3. **Resize op + converter implemented**:
   - Files:
     - `pytensor/tensor/resize.py` (NEW) - Op definition
     - `pytensor/link/onnx/dispatch/resize.py` (NEW) - ONNX converter
   - Test files:
     - `tests/tensor/test_resize.py` (NEW) - PyTensor op tests
     - `tests/link/onnx/test_resize.py` (NEW) - ONNX conversion tests
   - ~12 unit tests + property-based tests

**Success criteria:**
- All 3 operations export to valid ONNX
- Numerical results match PyTensor within 1e-4 tolerance
- All tests pass in both PyTensor and ONNX modes
- Property-based tests validate correctness across random inputs

## What We're NOT Implementing

**Out of scope for this plan:**

1. **Other pooling variants**: AveragePool, GlobalMaxPool, GlobalAveragePool (Phase 2)
2. **All resize modes**: Only implementing `nearest` and `linear` (bilinear)
3. **Advanced resize features**: ROI (region of interest) support, all coordinate transformation modes
4. **Training/gradients**: ONNX export only (no backward pass)
5. **Dynamic shapes**: Focus on static shapes first
6. **Other blockers**: BatchNorm, SiLU, Sigmoid mapping (separate plan)

## TDD Approach

### Testing Philosophy

**Write tests first, verify they fail, then implement:**

1. **Red**: Write comprehensive tests that define expected behavior
2. **Verify failure**: Run tests and confirm they fail in expected ways
3. **Green**: Implement just enough to make tests pass
4. **Refactor**: Clean up code while keeping tests green

**Test quality standards:**
- Clear, descriptive docstrings explaining what's being tested
- Simple test data that can be manually verified
- Informative failure messages with actual vs expected values
- Both unit tests (specific cases) and property tests (random inputs)

---

## Operation 1: Concat (Join → ONNX Concat)

### Phase 1: Test Design & Implementation

#### Overview
Write comprehensive tests for the Join-to-Concat converter. Since Join already exists in PyTensor, we only need to test ONNX conversion.

#### Test Categories

##### Category 1: Basic Concatenation Tests
**Test File**: `tests/link/onnx/test_join.py` (NEW)
**Purpose**: Verify basic concatenation along different axes

**Test 1: `test_join_axis0_two_tensors`**
```python
def test_join_axis0_two_tensors(tmp_path):
    """
    Test Join along axis 0 (row concatenation) with two 2D tensors.

    This is the simplest join case - verifies:
    - Join op is recognized and converted to ONNX Concat
    - Axis parameter is correctly passed
    - Output shape is calculated correctly ([3+2, 4] = [5, 4])
    - Numerical results match PyTensor

    Configuration:
    - axis=0 (concatenate rows)
    - 2 input tensors
    - Same shape except axis 0: (3,4) and (2,4)
    """
    import pytensor.tensor as pt

    # Arrange: Create symbolic inputs
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    # Define join operation
    z = pt.join(0, x, y)  # Concatenate along axis 0

    # Test data: Simple values for manual verification
    x_val = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype="float32")

    y_val = np.array([[13, 14, 15, 16],
                      [17, 18, 19, 20]], dtype="float32")

    # Expected output (manual verification):
    # [[1, 2, 3, 4],
    #  [5, 6, 7, 8],
    #  [9, 10, 11, 12],
    #  [13, 14, 15, 16],
    #  [17, 18, 19, 20]]

    # Act & Assert
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: "No ONNX conversion for <class 'pytensor.tensor.basic.Join'>"
- Points to: `pytensor/link/onnx/dispatch/basic.py` default handler

**Test 2: `test_join_axis1_two_tensors`**
```python
def test_join_axis1_two_tensors(tmp_path):
    """
    Test Join along axis 1 (column concatenation) with two 2D tensors.

    Verifies axis parameter handling - same operation, different axis.

    Configuration:
    - axis=1 (concatenate columns)
    - 2 input tensors
    - Same shape except axis 1: (3,2) and (3,3)
    """
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(1, x, y)  # Concatenate along axis 1

    x_val = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype="float32")

    y_val = np.array([[7, 8, 9],
                      [10, 11, 12],
                      [13, 14, 15]], dtype="float32")

    # Expected: (3, 5) output
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Test 3: `test_join_three_tensors`**
```python
def test_join_three_tensors(tmp_path):
    """
    Test Join with three input tensors.

    Verifies:
    - ONNX Concat supports variable number of inputs (not just 2)
    - Multiple inputs are concatenated in correct order

    Configuration:
    - axis=0
    - 3 input tensors
    """
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.matrix("z", dtype="float32")

    result = pt.join(0, x, y, z)

    x_val = np.array([[1, 2]], dtype="float32")
    y_val = np.array([[3, 4]], dtype="float32")
    z_val = np.array([[5, 6]], dtype="float32")

    # Expected: [[1,2], [3,4], [5,6]]
    compare_onnx_and_py([x, y, z], result, [x_val, y_val, z_val], tmp_path=tmp_path)
```

##### Category 2: Different Data Types
**Purpose**: Verify dtype handling (float32, float64, int32, int64)

**Test 4: `test_join_float64`**
```python
def test_join_float64(tmp_path):
    """Test Join with float64 dtype."""
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="float64")
    y = pt.matrix("y", dtype="float64")

    z = pt.join(0, x, y)

    x_val = np.array([[1.5, 2.5]], dtype="float64")
    y_val = np.array([[3.5, 4.5]], dtype="float64")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Test 5: `test_join_int32`**
```python
def test_join_int32(tmp_path):
    """Test Join with int32 dtype."""
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="int32")
    y = pt.matrix("y", dtype="int32")

    z = pt.join(0, x, y)

    x_val = np.array([[1, 2]], dtype="int32")
    y_val = np.array([[3, 4]], dtype="int32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

##### Category 3: Different Tensor Ranks
**Purpose**: Verify Join works with 1D, 3D, 4D tensors

**Test 6: `test_join_vectors_axis0`**
```python
def test_join_vectors_axis0(tmp_path):
    """Test Join with 1D vectors."""
    import pytensor.tensor as pt

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")

    z = pt.join(0, x, y)

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5], dtype="float32")

    # Expected: [1, 2, 3, 4, 5]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Test 7: `test_join_4d_tensors_axis1`**
```python
def test_join_4d_tensors_axis1(tmp_path):
    """
    Test Join with 4D tensors (NCHW format, typical for CNNs).

    This is THE critical test for YOLO11n - skip connections join
    feature maps from different layers along the channel dimension.

    Configuration:
    - 4D tensors: (batch, channels, height, width)
    - axis=1 (channel dimension)
    - Simulates skip connection in FPN head
    """
    import pytensor.tensor as pt

    x = pt.tensor4("x", dtype="float32")
    y = pt.tensor4("y", dtype="float32")

    z = pt.join(1, x, y)  # Concatenate along channel axis

    # Batch=1, different channels, same H and W
    x_val = np.random.rand(1, 3, 8, 8).astype("float32")
    y_val = np.random.rand(1, 5, 8, 8).astype("float32")

    # Expected output shape: (1, 8, 8, 8)
    session, onnx_res = compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)

    assert onnx_res[0].shape == (1, 8, 8, 8), \
        f"Expected shape (1, 8, 8, 8), got {onnx_res[0].shape}"
```

##### Category 4: Edge Cases

**Test 8: `test_join_negative_axis`**
```python
def test_join_negative_axis(tmp_path):
    """
    Test Join with negative axis indexing.

    ONNX Concat supports negative axes (e.g., axis=-1 for last dimension).
    Verify PyTensor's negative axis is correctly converted.
    """
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(-1, x, y)  # axis=-1 means last axis (columns for 2D)

    x_val = np.array([[1], [2]], dtype="float32")
    y_val = np.array([[3], [4]], dtype="float32")

    # Expected: [[1, 3], [2, 4]]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Test 9: `test_join_single_element_tensors`**
```python
def test_join_single_element_tensors(tmp_path):
    """Test Join with tensors containing single elements."""
    import pytensor.tensor as pt

    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(0, x, y)

    x_val = np.array([[1.0]], dtype="float32")
    y_val = np.array([[2.0]], dtype="float32")

    # Expected: [[1.0], [2.0]]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

##### Category 5: Integration Tests

**Test 10: `test_join_after_conv2d`**
```python
def test_join_after_conv2d(tmp_path):
    """
    Test Join combined with Conv2D (typical YOLO11n pattern).

    Pattern:
    - Two parallel convolution paths
    - Concatenate outputs along channel axis
    - This is the C3k2 block pattern
    """
    import pytensor.tensor as pt
    from pytensor.tensor.conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel1 = pt.tensor4("kernel1", dtype="float32")
    kernel2 = pt.tensor4("kernel2", dtype="float32")

    # Two conv paths
    conv1 = conv2d(x, kernel1, border_mode="valid", filter_flip=False)
    conv2 = conv2d(x, kernel2, border_mode="valid", filter_flip=False)

    # Concatenate along channel axis
    result = pt.join(1, conv1, conv2)

    x_val = np.random.rand(1, 3, 10, 10).astype("float32")
    kernel1_val = np.random.rand(4, 3, 3, 3).astype("float32")
    kernel2_val = np.random.rand(8, 3, 3, 3).astype("float32")

    # Expected: (1, 12, 8, 8) - 4+8 channels
    compare_onnx_and_py(
        [x, kernel1, kernel2],
        result,
        [x_val, kernel1_val, kernel2_val],
        tmp_path=tmp_path
    )
```

#### Property-Based Tests

**File**: `tests/link/onnx/strategies/operations.py` (ADD)

```python
@st.composite
def join_inputs(draw, max_inputs=5, max_rank=4):
    """
    Generate valid inputs for Join operation.

    Strategy:
    1. Choose axis, number of inputs, and base shape
    2. Generate tensors with same shape except along join axis
    3. Vary dimension along join axis for each input
    """
    # Choose parameters
    num_inputs = draw(st.integers(2, max_inputs))
    rank = draw(st.integers(1, max_rank))
    axis = draw(st.integers(-rank, rank - 1))

    # Normalize negative axis
    normalized_axis = axis if axis >= 0 else rank + axis

    # Generate base shape (same for all inputs except join axis)
    base_shape = draw(st.lists(
        st.integers(1, 10),
        min_size=rank,
        max_size=rank
    ))

    # Generate inputs with varying dimension along join axis
    inputs = []
    for _ in range(num_inputs):
        shape = list(base_shape)
        # Vary dimension along join axis
        shape[normalized_axis] = draw(st.integers(1, 10))

        tensor = draw(onnx_tensor(dtype=np.float32, shape=tuple(shape)))
        inputs.append(tensor)

    return (axis, tuple(inputs))


# Add to ONNX_OPERATIONS registry
ONNX_OPERATIONS["join"] = OperationConfig(
    op_func=lambda axis, *tensors: pt.join(axis, *tensors),
    input_strategy=join_inputs(),
    valid_dtypes=["float32", "float64", "int32", "int64"],
    category="shape",
    notes="Join/concatenate tensors along an axis",
)
```

**Property Test** (in `tests/link/onnx/test_properties.py`):
```python
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=50,
)
@given(data=st.data())
def test_join_property_matches_pytensor(tmp_path, data):
    """
    Property: Join with any valid inputs should produce same results in ONNX and PyTensor.

    This tests Join across:
    - Different axes (positive and negative)
    - Different numbers of inputs (2-5 tensors)
    - Different ranks (1D to 4D)
    - Different shapes along join axis
    """
    axis, inputs_tuple = data.draw(join_inputs(max_inputs=4, max_rank=3))

    # Create symbolic variables
    symbolic_inputs = []
    for i, inp in enumerate(inputs_tuple):
        var = pt.tensor(f"x{i}", dtype=inp.dtype, shape=inp.shape)
        symbolic_inputs.append(var)

    # Join operation
    result = pt.join(axis, *symbolic_inputs)

    # Compare ONNX and PyTensor
    try:
        compare_onnx_and_py(symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path)
    except Exception as e:
        shapes = [x.shape for x in inputs_tuple]
        raise AssertionError(
            f"Property test failed for join with axis={axis}, "
            f"input shapes: {shapes}"
        ) from e
```

#### Test Implementation Steps

1. **Create test file**: `tests/link/onnx/test_join.py`
   ```python
   import numpy as np
   import pytest
   import pytensor.tensor as pt
   from tests.link.onnx.test_basic import compare_onnx_and_py

   # Import necessary for ONNX
   pytest.importorskip("onnx")
   pytest.importorskip("onnxruntime")
   ```

2. **Implement all 10 unit tests** (see test cases above)

3. **Add to property-based test registry** in `strategies/operations.py`

4. **Run tests to verify they fail**:
   ```bash
   pytest tests/link/onnx/test_join.py -v
   ```

#### Success Criteria

##### Automated Verification:
- [ ] Test file created: `tests/link/onnx/test_join.py`
- [ ] All 10 tests discovered: `pytest --collect-only tests/link/onnx/test_join.py`
- [ ] All tests fail with `NotImplementedError`: `pytest tests/link/onnx/test_join.py`
- [ ] Strategy added to operations registry
- [ ] Property test runs and fails: `pytest tests/link/onnx/test_properties.py::test_join_property_matches_pytensor -v`

##### Manual Verification:
- [ ] Each test has clear docstring explaining what it validates
- [ ] Test names clearly describe the scenario (e.g., `test_join_axis0_two_tensors`)
- [ ] Failure messages are informative (show axis, shapes, expected behavior)
- [ ] Test data is simple enough to manually verify expected output
- [ ] Edge cases are covered (negative axis, single elements, 4D tensors)

---

### Phase 2: Test Failure Verification

#### Verification Steps

1. **Run the full test suite**:
   ```bash
   pytest tests/link/onnx/test_join.py -v
   ```

2. **Verify each test fails correctly**:
   - Check error type is `NotImplementedError`
   - Check message mentions "No ONNX conversion for <class 'pytensor.tensor.basic.Join'>"
   - Check stack trace points to `pytensor/link/onnx/dispatch/basic.py`

3. **Run property-based test**:
   ```bash
   pytest tests/link/onnx/test_properties.py::test_join_property_matches_pytensor -v --hypothesis-seed=12345
   ```

4. **Document failures**:

**Expected Failure Log**:
```
tests/link/onnx/test_join.py::test_join_axis0_two_tensors FAILED
tests/link/onnx/test_join.py::test_join_axis1_two_tensors FAILED
tests/link/onnx/test_join.py::test_join_three_tensors FAILED
tests/link/onnx/test_join.py::test_join_float64 FAILED
tests/link/onnx/test_join.py::test_join_int32 FAILED
tests/link/onnx/test_join.py::test_join_vectors_axis0 FAILED
tests/link/onnx/test_join.py::test_join_4d_tensors_axis1 FAILED
tests/link/onnx/test_join.py::test_join_negative_axis FAILED
tests/link/onnx/test_join.py::test_join_single_element_tensors FAILED
tests/link/onnx/test_join.py::test_join_after_conv2d FAILED

All failures with: NotImplementedError: No ONNX conversion for <class 'pytensor.tensor.basic.Join'>
```

#### Success Criteria

##### Automated Verification:
- [ ] All 10 tests fail (not pass or error): `pytest tests/link/onnx/test_join.py --tb=line`
- [ ] No import errors or syntax errors: Tests run but fail as expected
- [ ] Property test fails with same error: `pytest tests/link/onnx/test_properties.py -k join`

##### Manual Verification:
- [ ] Error messages clearly indicate Join is not supported
- [ ] Stack traces point to dispatcher in `basic.py:29-70`
- [ ] No unexpected errors (e.g., ONNX Runtime crashes, segfaults)
- [ ] Failure output is clean and diagnostic

---

### Phase 3: Feature Implementation (Red → Green)

#### Implementation Strategy

**Goal**: Make tests pass one at a time by implementing the Join → Concat converter.

**Implementation file**: `pytensor/link/onnx/dispatch/join.py` (NEW)

#### Implementation: Join → ONNX Concat Converter

**File**: `pytensor/link/onnx/dispatch/join.py` (NEW)

```python
"""ONNX conversion for Join (Concat) operation."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.basic import Join

from onnx import helper


@onnx_funcify.register(Join)
def onnx_funcify_Join(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Join op to ONNX Concat node.

    PyTensor Join concatenates multiple tensors along a specified axis.
    ONNX Concat performs the same operation.

    Parameters
    ----------
    op : Join
        The Join operation instance
    node : Apply
        The apply node containing inputs and outputs
    var_names : dict
        Mapping of variables to ONNX names
    get_var_name : callable
        Function to get ONNX name for a variable

    Returns
    -------
    onnx.NodeProto
        ONNX Concat node

    Notes
    -----
    PyTensor Join takes axis as the first input (runtime value),
    but ONNX Concat requires axis as a compile-time attribute.

    In PyTensor graphs, the axis is typically a Constant, so we extract
    its value and pass it as an ONNX attribute.

    Join inputs: [axis (scalar constant), tensor1, tensor2, ...]
    Concat inputs: [tensor1, tensor2, ...]
    Concat attributes: axis=<int>
    """
    # Extract inputs
    # node.inputs[0] is the axis (should be a Constant)
    # node.inputs[1:] are the tensors to concatenate

    from pytensor.graph.basic import Constant

    axis_input = node.inputs[0]
    tensor_inputs = node.inputs[1:]

    # Extract axis value
    if not isinstance(axis_input, Constant):
        raise NotImplementedError(
            "ONNX Concat requires axis to be a compile-time constant. "
            f"Got: {axis_input}"
        )

    axis = int(axis_input.data)

    # Get ONNX names for tensor inputs
    input_names = [get_var_name(inp) for inp in tensor_inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX Concat node
    return helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=output_names,
        axis=axis,
        name=f"Concat_{output_names[0]}",
    )
```

**Debugging Approach**:
1. Run simplest test first: `pytest tests/link/onnx/test_join.py::test_join_axis0_two_tensors -xvs`
2. Read failure message to understand what's missing
3. Implement just enough to address the failure
4. Re-run test until it passes
5. Move to next test

#### Import Registration

**File**: `pytensor/link/onnx/dispatch/__init__.py` (MODIFY)

Add import to trigger registration:
```python
import pytensor.link.onnx.dispatch.basic  # noqa: F401
import pytensor.link.onnx.dispatch.conv  # noqa: F401
import pytensor.link.onnx.dispatch.elemwise  # noqa: F401
import pytensor.link.onnx.dispatch.join  # noqa: F401  # ADD THIS LINE
import pytensor.link.onnx.dispatch.nlinalg  # noqa: F401
import pytensor.link.onnx.dispatch.shape  # noqa: F401
import pytensor.link.onnx.dispatch.special  # noqa: F401
```

#### Testing Progression

**Step 1: Make `test_join_axis0_two_tensors` pass**
```bash
pytest tests/link/onnx/test_join.py::test_join_axis0_two_tensors -xvs
```

**Expected initial failure**:
- Still `NotImplementedError` (converter not imported)

**Fix**: Add import to `__init__.py`, re-run.

**Expected second failure** (if axis handling is wrong):
- ONNX validation error or shape mismatch

**Fix**: Ensure axis is correctly extracted and passed.

**Success**: Test passes!

**Step 2: Make `test_join_axis1_two_tensors` pass**
```bash
pytest tests/link/onnx/test_join.py::test_join_axis1_two_tensors -xvs
```

Should pass immediately if axis handling is generic.

**Step 3: Make `test_join_three_tensors` pass**
```bash
pytest tests/link/onnx/test_join.py::test_join_three_tensors -xvs
```

Verifies multiple inputs work correctly.

**Steps 4-10**: Continue with remaining tests.

#### Success Criteria

##### Automated Verification:
- [ ] All unit tests pass: `pytest tests/link/onnx/test_join.py -v`
- [ ] Property-based test passes: `pytest tests/link/onnx/test_properties.py -k join -v`
- [ ] No regressions: `pytest tests/link/onnx/ -v` (all other tests still pass)
- [ ] Code lints cleanly: `ruff check pytensor/link/onnx/dispatch/join.py`
- [ ] Type checking passes (if enabled): `mypy pytensor/link/onnx/dispatch/join.py`

##### Manual Verification:
- [ ] Implementation handles all test cases correctly
- [ ] Axis parameter is correctly extracted from Constant input
- [ ] Multiple inputs (2+) are handled correctly
- [ ] Negative axis values work (if ONNX supports them)
- [ ] Error message is clear if axis is not a constant

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Code clarity**:
   - [ ] Add detailed docstring with examples
   - [ ] Add inline comments for non-obvious logic (axis extraction)
   - [ ] Ensure variable names are descriptive

2. **Error handling**:
   - [ ] Clear error if axis is dynamic (not Constant)
   - [ ] Consider: Should we support dynamic axis via graph rewriting?

3. **Test quality**:
   - [ ] Extract common test fixtures if tests have duplication
   - [ ] Consider adding test for edge case: axis out of bounds (should fail at ONNX validation)

#### Refactoring Steps

1. **Ensure all tests pass**: `pytest tests/link/onnx/test_join.py -v`

2. **Improve docstring**:
   ```python
   """
   Convert PyTensor Join op to ONNX Concat node.

   Examples
   --------
   PyTensor:
   >>> x = pt.matrix("x")
   >>> y = pt.matrix("y")
   >>> z = pt.join(0, x, y)  # Concatenate along axis 0

   ONNX equivalent:
   >>> Concat(inputs=[x, y], axis=0)

   Notes
   -----
   - PyTensor Join takes axis as first input (runtime value)
   - ONNX Concat requires axis as compile-time attribute
   - We extract axis from Constant input at export time
   """
   ```

3. **Add error handling test**:
   ```python
   def test_join_dynamic_axis_raises(tmp_path):
       """Test that Join with dynamic axis raises informative error."""
       import pytensor.tensor as pt

       axis = pt.scalar("axis", dtype="int32")  # Dynamic axis
       x = pt.matrix("x", dtype="float32")
       y = pt.matrix("y", dtype="float32")

       z = pt.join(axis, x, y)

       # Should raise NotImplementedError with clear message
       with pytest.raises(NotImplementedError, match="compile-time constant"):
           from pytensor.link.onnx.export import export_onnx
           export_onnx(z, [axis, x, y], tmp_path / "test.onnx")
   ```

4. **Run tests after each refactoring**:
   ```bash
   pytest tests/link/onnx/test_join.py -v
   ```

#### Success Criteria

##### Automated Verification:
- [ ] All tests still pass after refactoring: `pytest tests/link/onnx/test_join.py -v`
- [ ] Linting passes: `ruff check pytensor/link/onnx/dispatch/join.py`
- [ ] Code coverage maintained: `pytest tests/link/onnx/test_join.py --cov=pytensor/link/onnx/dispatch/join`

##### Manual Verification:
- [ ] Code is more readable than initial implementation
- [ ] Docstring clearly explains PyTensor vs ONNX differences
- [ ] Error messages help users debug issues
- [ ] No unnecessary complexity

---

## Operation 2: MaxPool

### Phase 1: Test Design & Implementation

#### Overview
MaxPool doesn't exist in PyTensor yet. We need to:
1. Create the PyTensor MaxPool op in `pytensor/tensor/pool.py` (NEW)
2. Write PyTensor op tests in `tests/tensor/test_pool.py` (NEW)
3. Create ONNX converter in `pytensor/link/onnx/dispatch/pool.py` (NEW)
4. Write ONNX converter tests in `tests/link/onnx/test_pool.py` (NEW)

This is more complex than Join because we're creating a new op from scratch.

#### Test Categories

##### Category 1: PyTensor Op Tests (Non-ONNX)
**Test File**: `tests/tensor/test_pool.py` (NEW)
**Purpose**: Verify MaxPool op works correctly in PyTensor (before ONNX)

**Test 1: `test_maxpool2d_basic`**
```python
def test_maxpool2d_basic():
    """
    Test basic MaxPool2D operation in PyTensor.

    Configuration:
    - 4D input: (batch, channels, height, width)
    - Kernel size: 2x2
    - Stride: 2 (default, same as kernel size)
    - No padding
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d  # Function we'll create

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 2x2 kernel
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Compile PyTensor function
    f = pytensor.function([x], y)

    # Test data: 4x4 input
    x_val = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype="float32")

    # Expected: 2x2 output with max of each 2x2 region
    # [[6, 8],
    #  [14, 16]]
    expected = np.array([[[[6, 8],
                           [14, 16]]]], dtype="float32")

    result = f(x_val)

    np.testing.assert_allclose(result, expected)
```

**Expected Failure Mode** (before implementation):
- Error type: `ImportError` or `AttributeError`
- Expected message: "cannot import name 'pool_2d'" or "module 'pytensor.tensor' has no attribute 'pool'"

**Test 2: `test_maxpool2d_stride`**
```python
def test_maxpool2d_stride():
    """
    Test MaxPool2D with stride different from kernel size.

    Configuration:
    - Kernel: 3x3
    - Stride: 1 (overlapping pools)
    - Verifies stride parameter works independently
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 3x3 kernel, stride 1
    y = pool_2d(x, ws=(3, 3), stride=(1, 1), mode="max")

    f = pytensor.function([x], y)

    # 5x5 input
    x_val = np.arange(25, dtype="float32").reshape(1, 1, 5, 5)

    result = f(x_val)

    # Expected shape: (1, 1, 3, 3) with stride 1
    assert result.shape == (1, 1, 3, 3)
```

**Test 3: `test_maxpool2d_padding`**
```python
def test_maxpool2d_padding():
    """
    Test MaxPool2D with padding.

    Configuration:
    - Kernel: 2x2
    - Padding: (1, 1) - add 1 pixel border
    - Padding value: -inf (or very negative) so max ignores it
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with padding
    y = pool_2d(x, ws=(2, 2), padding=(1, 1), mode="max")

    f = pytensor.function([x], y)

    x_val = np.ones((1, 1, 4, 4), dtype="float32")

    result = f(x_val)

    # With padding (1,1), output should be larger
    assert result.shape == (1, 1, 3, 3)
```

##### Category 2: ONNX Conversion Tests
**Test File**: `tests/link/onnx/test_pool.py` (NEW)
**Purpose**: Verify MaxPool exports to ONNX correctly

**Test 4: `test_maxpool2d_onnx_basic`**
```python
def test_maxpool2d_onnx_basic(tmp_path):
    """
    Test MaxPool2D exports to ONNX and produces same results.

    This is THE fundamental test - verifies:
    - MaxPool op is recognized by ONNX converter
    - Kernel size is correctly converted
    - Numerical results match PyTensor
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 2x2 kernel
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Test data
    x_val = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype="float32")

    # Compare ONNX and PyTensor outputs
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: "No ONNX conversion for <class 'pytensor.tensor.pool.Pool'>"

**Test 5: `test_maxpool2d_onnx_3x3_kernel`**
```python
def test_maxpool2d_onnx_3x3_kernel(tmp_path):
    """Test MaxPool with 3x3 kernel (different from 2x2)."""
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(3, 3), mode="max")

    x_val = np.random.rand(1, 1, 10, 10).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 6: `test_maxpool2d_onnx_stride`**
```python
def test_maxpool2d_onnx_stride(tmp_path):
    """
    Test MaxPool with stride parameter in ONNX.

    ONNX MaxPool has 'strides' attribute that must match PyTensor stride.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(2, 2), stride=(2, 2), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 7: `test_maxpool2d_onnx_multiple_channels`**
```python
def test_maxpool2d_onnx_multiple_channels(tmp_path):
    """
    Test MaxPool with multiple channels (typical CNN scenario).

    MaxPool operates independently on each channel.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Batch=2, Channels=16, 10x10 spatial
    x_val = np.random.rand(2, 16, 10, 10).astype("float32")

    session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Verify output shape: (2, 16, 5, 5)
    assert onnx_res[0].shape == (2, 16, 5, 5)
```

**Test 8: `test_maxpool2d_onnx_yolo_sppf_pattern`**
```python
def test_maxpool2d_onnx_yolo_sppf_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: SPPF pattern from YOLO11n.

    SPPF (Spatial Pyramid Pooling Fast):
    - Apply MaxPool multiple times with same kernel
    - Concatenate all intermediate results
    - Creates multi-scale features

    Pattern:
    x → MaxPool → MaxPool → MaxPool
    └─────┴─────────┴─────────┴──> Concat all 4
    """
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # SPPF pattern: cascade of 5x5 MaxPool
    pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))

    # Concatenate original + all pooled versions
    result = pt.join(1, x, pool1, pool2, pool3)

    # Test with YOLO-like feature map
    x_val = np.random.rand(1, 256, 20, 20).astype("float32")

    compare_onnx_and_py([x], result, [x_val], tmp_path=tmp_path)
```

##### Category 3: Edge Cases

**Test 9: `test_maxpool2d_1x1_kernel`**
```python
def test_maxpool2d_1x1_kernel(tmp_path):
    """Test MaxPool with 1x1 kernel (identity operation)."""
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(1, 1), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    # Output should equal input (1x1 max pool is identity)
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 10: `test_maxpool2d_large_kernel`**
```python
def test_maxpool2d_large_kernel(tmp_path):
    """Test MaxPool with kernel larger than input (global pooling)."""
    import pytensor.tensor as pt
    from pytensor.tensor.pool import pool_2d
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # 8x8 kernel on 8x8 input = global max pooling
    y = pool_2d(x, ws=(8, 8), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Output should be (1, 3, 1, 1) - single value per channel
    assert onnx_res[0].shape == (1, 3, 1, 1)
```

#### Property-Based Tests

**Strategy** (in `strategies/operations.py`):
```python
@st.composite
def maxpool2d_inputs(draw):
    """
    Generate valid inputs for MaxPool2D.

    Strategy:
    1. Generate input tensor (NCHW format)
    2. Generate kernel size (must be <= input spatial dimensions)
    3. Generate stride (reasonable range)
    4. Optionally generate padding
    """
    # Input shape: (batch, channels, height, width)
    batch = draw(st.integers(1, 4))
    channels = draw(st.integers(1, 16))
    height = draw(st.integers(4, 20))
    width = draw(st.integers(4, 20))

    # Kernel size (must fit in input)
    kernel_h = draw(st.integers(2, min(height, 8)))
    kernel_w = draw(st.integers(2, min(width, 8)))

    # Stride (default to kernel size for non-overlapping)
    stride_h = draw(st.integers(1, kernel_h))
    stride_w = draw(st.integers(1, kernel_w))

    # Generate input tensor
    input_tensor = draw(onnx_tensor(
        dtype=np.float32,
        shape=(batch, channels, height, width)
    ))

    return (input_tensor, (kernel_h, kernel_w), (stride_h, stride_w))
```

#### Test Implementation Steps

1. **Create PyTensor op test file**: `tests/tensor/test_pool.py`
2. **Create ONNX converter test file**: `tests/link/onnx/test_pool.py`
3. **Implement all tests** (10 tests total: 3 PyTensor op + 7 ONNX)
4. **Run tests to verify failures**:
   ```bash
   pytest tests/tensor/test_pool.py -v  # Should fail: module not found
   pytest tests/link/onnx/test_pool.py -v  # Should fail: module not found
   ```

#### Success Criteria

##### Automated Verification:
- [ ] PyTensor test file created: `tests/tensor/test_pool.py`
- [ ] ONNX test file created: `tests/link/onnx/test_pool.py`
- [ ] All tests fail with expected errors (ImportError, AttributeError, NotImplementedError)
- [ ] Property-based strategy added

##### Manual Verification:
- [ ] Test progression makes sense (PyTensor op first, then ONNX)
- [ ] SPPF pattern test accurately represents YOLO11n usage
- [ ] Tests cover different kernel sizes, strides, and padding

---

### Phase 2: Test Failure Verification

#### Verification Steps

1. **Run PyTensor op tests**:
   ```bash
   pytest tests/tensor/test_pool.py -v
   ```

   **Expected failures**:
   - `ImportError: cannot import name 'pool_2d' from 'pytensor.tensor.pool'`
   - `ModuleNotFoundError: No module named 'pytensor.tensor.pool'`

2. **Run ONNX converter tests**:
   ```bash
   pytest tests/link/onnx/test_pool.py -v
   ```

   **Expected failures**:
   - Same import errors as above
   - Once PyTensor op exists: `NotImplementedError: No ONNX conversion for Pool`

3. **Document failure progression**:

**Failure Log**:
```
Phase 1: Before PyTensor Op Implementation
- All tests fail with ImportError (module doesn't exist)

Phase 2: After PyTensor Op, Before ONNX Converter
- tests/tensor/test_pool.py: PASS (op works in PyTensor)
- tests/link/onnx/test_pool.py: FAIL with NotImplementedError (no ONNX converter)

Phase 3: After ONNX Converter
- All tests: PASS
```

#### Success Criteria

##### Automated Verification:
- [ ] PyTensor op tests fail predictably: Import errors before implementation
- [ ] ONNX tests fail predictably: NotImplementedError after PyTensor op exists
- [ ] No unexpected errors (segfaults, ONNX Runtime crashes)

##### Manual Verification:
- [ ] Failure messages clearly indicate what's missing
- [ ] Test failures guide implementation (clear next steps)

---

### Phase 3: Feature Implementation (Red → Green)

#### Implementation Strategy

**Two-phase implementation:**
1. **Phase 3A**: Create PyTensor MaxPool op (make `tests/tensor/test_pool.py` pass)
2. **Phase 3B**: Create ONNX converter (make `tests/link/onnx/test_pool.py` pass)

#### Phase 3A: PyTensor MaxPool Op

**File**: `pytensor/tensor/pool.py` (NEW)

```python
"""Pooling operations for PyTensor."""

import numpy as np
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType


class Pool(Op):
    """
    Pooling operation for tensors.

    Applies a pooling function (max, average, etc.) over spatial dimensions.

    Parameters
    ----------
    ws : tuple of int
        Window size (kernel size) for pooling. For 2D: (height, width).
    stride : tuple of int, optional
        Stride for pooling window. Defaults to ws (non-overlapping).
    padding : tuple of int, optional
        Padding to add to input. For 2D: (pad_h, pad_w). Defaults to (0, 0).
    mode : {'max', 'average'}
        Pooling mode. Currently only 'max' is implemented.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x")
    >>> y = pool_2d(x, ws=(2, 2), mode="max")
    """

    __props__ = ("ws", "stride", "padding", "mode")

    def __init__(self, ws, stride=None, padding=(0, 0), mode="max"):
        self.ws = tuple(ws)
        self.stride = tuple(stride) if stride is not None else self.ws
        self.padding = tuple(padding)
        self.mode = mode

        if mode != "max":
            raise NotImplementedError(f"Only 'max' pooling is implemented, got: {mode}")

    def make_node(self, x):
        """Create an Apply node for this operation."""
        from pytensor.tensor.type import TensorType

        x = pt.as_tensor_variable(x)

        # Validate input
        if x.type.ndim != 4:
            raise ValueError(
                f"Pool requires 4D input (NCHW format), got {x.type.ndim}D tensor"
            )

        # Output has same type as input
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute the pooling operation using NumPy."""
        (x,) = inputs

        if self.mode == "max":
            result = self._perform_max_pool(x)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        output_storage[0][0] = result

    def _perform_max_pool(self, x):
        """Perform max pooling using NumPy."""
        batch, channels, height, width = x.shape
        pool_h, pool_w = self.ws
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=-np.inf,  # Max pooling ignores -inf
            )
            height += 2 * pad_h
            width += 2 * pad_w

        # Calculate output dimensions
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1

        # Initialize output
        output = np.zeros((batch, channels, out_height, out_width), dtype=x.dtype)

        # Perform max pooling
        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        h_end = h_start + pool_h
                        w_end = w_start + pool_w

                        # Extract pool region and compute max
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(pool_region)

        return output

    def infer_shape(self, fgraph, node, input_shapes):
        """Infer output shape from input shape."""
        (x_shape,) = input_shapes

        batch, channels, height, width = x_shape
        pool_h, pool_w = self.ws
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Calculate output shape
        if height is not None:
            out_height = (height + 2 * pad_h - pool_h) // stride_h + 1
        else:
            out_height = None

        if width is not None:
            out_width = (width + 2 * pad_w - pool_w) // stride_w + 1
        else:
            out_width = None

        return [(batch, channels, out_height, out_width)]


def pool_2d(input, ws, stride=None, padding=(0, 0), mode="max"):
    """
    Apply 2D pooling to a 4D tensor.

    Parameters
    ----------
    input : TensorVariable
        4D tensor in NCHW format (batch, channels, height, width)
    ws : tuple of 2 ints
        Window size (kernel size): (height, width)
    stride : tuple of 2 ints, optional
        Stride for pooling window. Defaults to ws (non-overlapping).
    padding : tuple of 2 ints, optional
        Padding to add: (pad_height, pad_width). Defaults to (0, 0).
    mode : {'max', 'average'}
        Pooling mode. Currently only 'max' is supported.

    Returns
    -------
    TensorVariable
        Pooled tensor, same rank as input with reduced spatial dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x", dtype="float32")
    >>> # Max pool with 2x2 kernel
    >>> y = pool_2d(x, ws=(2, 2), mode="max")
    >>> # Max pool with 3x3 kernel and stride 1
    >>> y = pool_2d(x, ws=(3, 3), stride=(1, 1), mode="max")
    """
    return Pool(ws=ws, stride=stride, padding=padding, mode=mode)(input)
```

**Missing imports**:
```python
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
```

**Export function**:

**File**: `pytensor/tensor/__init__.py` (MODIFY)

Add to exports:
```python
from pytensor.tensor.pool import pool_2d  # ADD THIS LINE
```

**Testing progression for Phase 3A**:
```bash
# Should now pass PyTensor op tests
pytest tests/tensor/test_pool.py -v
```

#### Phase 3B: ONNX MaxPool Converter

**File**: `pytensor/link/onnx/dispatch/pool.py` (NEW)

```python
"""ONNX conversion for pooling operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.pool import Pool

from onnx import helper


@onnx_funcify.register(Pool)
def onnx_funcify_Pool(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Pool op to ONNX MaxPool node.

    Parameters
    ----------
    op : Pool
        The Pool operation instance
    node : Apply
        The apply node containing inputs and outputs
    var_names : dict
        Mapping of variables to ONNX names
    get_var_name : callable
        Function to get ONNX name for a variable

    Returns
    -------
    onnx.NodeProto
        ONNX MaxPool node

    Notes
    -----
    ONNX MaxPool operator:
    - Inputs: X (4D tensor in NCHW format)
    - Attributes:
      - kernel_shape (required): [pool_h, pool_w]
      - strides (optional): [stride_h, stride_w]
      - pads (optional): [pad_top, pad_left, pad_bottom, pad_right]
    - Outputs: Y (pooled tensor)

    PyTensor Pool op stores:
    - op.ws: window size (kernel_shape)
    - op.stride: stride for pooling
    - op.padding: (pad_h, pad_w) -> ONNX uses [pad_h, pad_w, pad_h, pad_w]
    - op.mode: 'max' or 'average'
    """
    if op.mode != "max":
        raise NotImplementedError(
            f"Only max pooling is supported for ONNX export, got: {op.mode}"
        )

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract pooling parameters
    kernel_shape = list(op.ws)
    strides = list(op.stride)

    # ONNX pads format: [pad_top, pad_left, pad_bottom, pad_right]
    # PyTensor padding: (pad_h, pad_w) - same padding on both sides
    pad_h, pad_w = op.padding
    pads = [pad_h, pad_w, pad_h, pad_w]

    # Build attributes
    attributes = {
        "kernel_shape": kernel_shape,
    }

    # Add strides if different from kernel size
    if strides != kernel_shape:
        attributes["strides"] = strides

    # Add pads if non-zero
    if any(p > 0 for p in pads):
        attributes["pads"] = pads

    # Create ONNX MaxPool node
    return helper.make_node(
        "MaxPool",
        inputs=input_names,
        outputs=output_names,
        name=f"MaxPool_{output_names[0]}",
        **attributes,
    )
```

**Import registration**:

**File**: `pytensor/link/onnx/dispatch/__init__.py` (MODIFY)

```python
import pytensor.link.onnx.dispatch.pool  # noqa: F401  # ADD THIS LINE
```

**Testing progression for Phase 3B**:
```bash
# Should now pass ONNX converter tests
pytest tests/link/onnx/test_pool.py -v
```

#### Success Criteria

##### Automated Verification:
- [ ] PyTensor op tests pass: `pytest tests/tensor/test_pool.py -v`
- [ ] ONNX converter tests pass: `pytest tests/link/onnx/test_pool.py -v`
- [ ] SPPF pattern test passes (critical for YOLO11n)
- [ ] No regressions: `pytest tests/link/onnx/ -v`
- [ ] Linting passes: `ruff check pytensor/tensor/pool.py pytensor/link/onnx/dispatch/pool.py`

##### Manual Verification:
- [ ] MaxPool op produces correct output in PyTensor
- [ ] ONNX exported model produces same results as PyTensor
- [ ] Kernel size, stride, and padding are correctly converted
- [ ] SPPF cascade pattern works correctly

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Performance optimization** (PyTensor op):
   - [ ] Current implementation uses nested loops (slow)
   - [ ] Consider: Use `as_strided` or other NumPy tricks for speed
   - [ ] Or: Implement C code via `c_code()` method (advanced)

2. **Code clarity**:
   - [ ] Add more examples to docstrings
   - [ ] Document edge cases (padding with -inf for max pooling)

3. **Test quality**:
   - [ ] Consider adding benchmark test (performance regression detection)

#### Refactoring Steps

1. **Optimize PyTensor op** (optional for MVP, but good practice):
   ```python
   def _perform_max_pool_optimized(self, x):
       """Optimized max pooling using im2col trick."""
       # Use numpy stride tricks to avoid nested loops
       # This is MUCH faster for large tensors
       from numpy.lib.stride_tricks import as_strided

       # TODO: Implement im2col-based max pooling
       # For now, keep simple loop-based version
       pass
   ```

2. **Add gradient** (out of scope for ONNX export, but mentioned for completeness):
   ```python
   def grad(self, inputs, output_grads):
       """Gradient of max pooling (max unpooling)."""
       # Not needed for ONNX export (inference only)
       raise NotImplementedError("MaxPool gradient not implemented")
   ```

3. **Run tests after refactoring**:
   ```bash
   pytest tests/tensor/test_pool.py tests/link/onnx/test_pool.py -v
   ```

#### Success Criteria

##### Automated Verification:
- [ ] All tests still pass after refactoring
- [ ] Performance hasn't regressed (if optimizations added)
- [ ] Code coverage maintained

##### Manual Verification:
- [ ] Code is maintainable and well-documented
- [ ] No unnecessary complexity added

---

## Operation 3: Upsample/Resize

### Phase 1: Test Design & Implementation

#### Overview
Like MaxPool, Resize doesn't exist in PyTensor as a dedicated op. We have `bilinear_upsampling()` function, but it only supports bilinear mode. YOLO11n needs **nearest neighbor** mode for 2x upsampling in the FPN head.

We'll create a general `Resize` op supporting multiple modes.

#### Test Categories

##### Category 1: PyTensor Op Tests
**Test File**: `tests/tensor/test_resize.py` (NEW)

**Test 1: `test_resize_nearest_2x`**
```python
def test_resize_nearest_2x():
    """
    Test nearest neighbor resizing with 2x scale factor.

    Nearest neighbor:
    - Each pixel is duplicated
    - No interpolation
    - Fast but creates blocky output

    Configuration:
    - Mode: nearest
    - Scale: 2x (both H and W)
    """
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize  # Function we'll create

    x = pt.tensor4("x", dtype="float32")

    # Resize with 2x nearest neighbor
    y = resize(x, scale_factor=(2, 2), mode="nearest")

    f = pytensor.function([x], y)

    # Test data: 2x2 input
    x_val = np.array([[[[1, 2],
                        [3, 4]]]], dtype="float32")

    # Expected: 4x4 output, each pixel duplicated
    # [[1, 1, 2, 2],
    #  [1, 1, 2, 2],
    #  [3, 3, 4, 4],
    #  [3, 3, 4, 4]]
    expected = np.array([[[[1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [3, 3, 4, 4],
                           [3, 3, 4, 4]]]], dtype="float32")

    result = f(x_val)

    np.testing.assert_allclose(result, expected)
```

**Expected Failure Mode**:
- `ImportError: cannot import name 'resize'`

**Test 2: `test_resize_bilinear_2x`**
```python
def test_resize_bilinear_2x():
    """
    Test bilinear resizing with 2x scale factor.

    Bilinear interpolation:
    - Smooth interpolation between pixels
    - Creates intermediate values
    - Higher quality than nearest neighbor
    """
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize

    x = pt.tensor4("x", dtype="float32")

    # Resize with 2x bilinear interpolation
    y = resize(x, scale_factor=(2, 2), mode="linear")

    f = pytensor.function([x], y)

    # Simple test case
    x_val = np.array([[[[1.0, 2.0],
                        [3.0, 4.0]]]], dtype="float32")

    result = f(x_val)

    # Output should be (1, 1, 4, 4) with interpolated values
    assert result.shape == (1, 1, 4, 4)

    # Check corners match input
    np.testing.assert_allclose(result[0, 0, 0, 0], 1.0, rtol=1e-3)
    np.testing.assert_allclose(result[0, 0, -1, -1], 4.0, rtol=1e-3)
```

**Test 3: `test_resize_fractional_scale`**
```python
def test_resize_fractional_scale():
    """
    Test resize with non-integer scale factor.

    Example: 1.5x upsampling (6x6 -> 9x9)
    """
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize

    x = pt.tensor4("x", dtype="float32")

    # Resize with 1.5x scale
    y = resize(x, scale_factor=(1.5, 1.5), mode="nearest")

    f = pytensor.function([x], y)

    x_val = np.random.rand(1, 3, 6, 6).astype("float32")

    result = f(x_val)

    # Expected shape: (1, 3, 9, 9)
    assert result.shape == (1, 3, 9, 9)
```

##### Category 2: ONNX Conversion Tests
**Test File**: `tests/link/onnx/test_resize.py` (NEW)

**Test 4: `test_resize_onnx_nearest_2x`**
```python
def test_resize_onnx_nearest_2x(tmp_path):
    """
    Test nearest neighbor resize exports to ONNX correctly.

    This is THE critical test for YOLO11n FPN head.
    """
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # 2x nearest neighbor upsampling (YOLO11n pattern)
    y = resize(x, scale_factor=(2, 2), mode="nearest")

    x_val = np.array([[[[1, 2],
                        [3, 4]]]], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- `NotImplementedError: No ONNX conversion for Resize`

**Test 5: `test_resize_onnx_yolo_fpn_pattern`**
```python
def test_resize_onnx_yolo_fpn_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: FPN pattern from YOLO11n head.

    FPN (Feature Pyramid Network) pattern:
    - Low-resolution feature map (e.g., 20x20)
    - Upsample 2x using nearest neighbor (→ 40x40)
    - Concatenate with skip connection from encoder
    - This pattern repeats twice in YOLO11n head
    """
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    # Two feature maps: low-res and skip connection
    low_res = pt.tensor4("low_res", dtype="float32")
    skip = pt.tensor4("skip", dtype="float32")

    # Upsample low-res by 2x
    upsampled = resize(low_res, scale_factor=(2, 2), mode="nearest")

    # Concatenate with skip connection along channel axis
    result = pt.join(1, upsampled, skip)

    # YOLO11n FPN dimensions:
    # low_res: (1, 512, 20, 20) -> upsampled: (1, 512, 40, 40)
    # skip: (1, 512, 40, 40)
    # result: (1, 1024, 40, 40)
    low_res_val = np.random.rand(1, 512, 20, 20).astype("float32")
    skip_val = np.random.rand(1, 512, 40, 40).astype("float32")

    session, onnx_res = compare_onnx_and_py(
        [low_res, skip],
        result,
        [low_res_val, skip_val],
        tmp_path=tmp_path
    )

    # Verify output shape
    assert onnx_res[0].shape == (1, 1024, 40, 40)
```

**Test 6: `test_resize_onnx_bilinear`**
```python
def test_resize_onnx_bilinear(tmp_path):
    """Test bilinear resize exports to ONNX."""
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = resize(x, scale_factor=(2, 2), mode="linear")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 7: `test_resize_onnx_different_scales_hw`**
```python
def test_resize_onnx_different_scales_hw(tmp_path):
    """Test resize with different scale factors for H and W."""
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # 2x height, 3x width
    y = resize(x, scale_factor=(2, 3), mode="nearest")

    x_val = np.random.rand(1, 3, 10, 10).astype("float32")

    session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Expected shape: (1, 3, 20, 30)
    assert onnx_res[0].shape == (1, 3, 20, 30)
```

##### Category 3: Edge Cases

**Test 8: `test_resize_1x_scale`**
```python
def test_resize_1x_scale(tmp_path):
    """Test resize with 1x scale (identity operation)."""
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")
    y = resize(x, scale_factor=(1, 1), mode="nearest")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    # Output should equal input
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Test 9: `test_resize_downsampling`**
```python
def test_resize_downsampling(tmp_path):
    """Test resize with scale < 1 (downsampling)."""
    import pytensor.tensor as pt
    from pytensor.tensor.resize import resize
    from tests.link.onnx.test_basic import compare_onnx_and_py

    x = pt.tensor4("x", dtype="float32")

    # 0.5x downsampling
    y = resize(x, scale_factor=(0.5, 0.5), mode="nearest")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Expected shape: (1, 3, 4, 4)
    assert onnx_res[0].shape == (1, 3, 4, 4)
```

#### Property-Based Tests

**Strategy** (in `strategies/operations.py`):
```python
@st.composite
def resize_inputs(draw):
    """
    Generate valid inputs for Resize operation.

    Strategy:
    1. Generate input tensor (NCHW format)
    2. Generate scale factors (reasonable range)
    3. Choose mode (nearest or linear)
    """
    # Input shape
    batch = draw(st.integers(1, 4))
    channels = draw(st.integers(1, 16))
    height = draw(st.integers(4, 20))
    width = draw(st.integers(4, 20))

    # Scale factors (0.5x to 4x)
    scale_h = draw(st.floats(0.5, 4.0))
    scale_w = draw(st.floats(0.5, 4.0))

    # Mode
    mode = draw(st.sampled_from(["nearest", "linear"]))

    # Generate input tensor
    input_tensor = draw(onnx_tensor(
        dtype=np.float32,
        shape=(batch, channels, height, width)
    ))

    return (input_tensor, (scale_h, scale_w), mode)
```

#### Test Implementation Steps

1. **Create PyTensor test file**: `tests/tensor/test_resize.py` (3 tests)
2. **Create ONNX test file**: `tests/link/onnx/test_resize.py` (6 tests)
3. **Run tests to verify failures**

#### Success Criteria

##### Automated Verification:
- [ ] All tests fail with expected errors (ImportError, NotImplementedError)
- [ ] FPN pattern test accurately represents YOLO11n

##### Manual Verification:
- [ ] Tests cover nearest and bilinear modes
- [ ] Tests cover upsampling and downsampling
- [ ] YOLO11n FPN pattern is correctly represented

---

### Phase 2: Test Failure Verification

Same process as MaxPool - verify tests fail appropriately before implementation.

---

### Phase 3: Feature Implementation (Red → Green)

#### Phase 3A: PyTensor Resize Op

**File**: `pytensor/tensor/resize.py` (NEW)

```python
"""Resize (upsample/downsample) operations for PyTensor."""

import numpy as np
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType


class Resize(Op):
    """
    Resize operation for tensors (upsampling or downsampling).

    Supports multiple interpolation modes:
    - 'nearest': Nearest neighbor (fast, blocky)
    - 'linear': Bilinear interpolation (smooth)

    Parameters
    ----------
    scale_factor : tuple of float
        Scale factors for spatial dimensions. For 2D: (scale_h, scale_w).
        Values > 1 upsample, values < 1 downsample.
    mode : {'nearest', 'linear'}
        Interpolation mode.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x")
    >>> # 2x nearest neighbor upsampling
    >>> y = resize(x, scale_factor=(2, 2), mode="nearest")
    >>> # 1.5x bilinear upsampling
    >>> y = resize(x, scale_factor=(1.5, 1.5), mode="linear")
    """

    __props__ = ("scale_factor", "mode")

    def __init__(self, scale_factor, mode="nearest"):
        self.scale_factor = tuple(scale_factor)
        self.mode = mode

        if mode not in ("nearest", "linear"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'nearest' or 'linear'.")

    def make_node(self, x):
        """Create an Apply node for this operation."""
        import pytensor.tensor as pt
        from pytensor.tensor.type import TensorType
        from pytensor.graph.basic import Apply

        x = pt.as_tensor_variable(x)

        if x.type.ndim != 4:
            raise ValueError(
                f"Resize requires 4D input (NCHW format), got {x.type.ndim}D tensor"
            )

        # Output has same type as input (shape will be different)
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute the resize operation using NumPy."""
        (x,) = inputs

        if self.mode == "nearest":
            result = self._perform_nearest(x)
        elif self.mode == "linear":
            result = self._perform_linear(x)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        output_storage[0][0] = result

    def _perform_nearest(self, x):
        """Perform nearest neighbor resize using NumPy."""
        batch, channels, height, width = x.shape
        scale_h, scale_w = self.scale_factor

        # Calculate output dimensions
        out_height = int(height * scale_h)
        out_width = int(width * scale_w)

        # Create coordinate mappings
        # For each output pixel, find nearest input pixel
        out_h_coords = np.floor(np.arange(out_height) / scale_h).astype(np.int32)
        out_w_coords = np.floor(np.arange(out_width) / scale_w).astype(np.int32)

        # Clip to valid range
        out_h_coords = np.clip(out_h_coords, 0, height - 1)
        out_w_coords = np.clip(out_w_coords, 0, width - 1)

        # Index into input using nearest neighbor
        # Use advanced indexing: x[:, :, h_coords[:, None], w_coords]
        result = x[:, :, out_h_coords[:, None], out_w_coords]

        return result.astype(x.dtype)

    def _perform_linear(self, x):
        """Perform bilinear interpolation using NumPy."""
        batch, channels, height, width = x.shape
        scale_h, scale_w = self.scale_factor

        # Calculate output dimensions
        out_height = int(height * scale_h)
        out_width = int(width * scale_w)

        # Use scipy for bilinear interpolation
        # This is simpler than implementing bilinear from scratch
        from scipy.ndimage import zoom

        # Zoom operates on each batch and channel independently
        # zoom factors: [batch, channels, height, width]
        result = zoom(x, (1, 1, scale_h, scale_w), order=1)  # order=1 = bilinear

        return result.astype(x.dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        """Infer output shape from input shape."""
        (x_shape,) = input_shapes

        batch, channels, height, width = x_shape
        scale_h, scale_w = self.scale_factor

        # Calculate output shape
        if height is not None:
            out_height = int(height * scale_h)
        else:
            out_height = None

        if width is not None:
            out_width = int(width * scale_w)
        else:
            out_width = None

        return [(batch, channels, out_height, out_width)]


def resize(input, scale_factor, mode="nearest"):
    """
    Resize a 4D tensor using interpolation.

    Parameters
    ----------
    input : TensorVariable
        4D tensor in NCHW format (batch, channels, height, width)
    scale_factor : tuple of 2 floats
        Scale factors for spatial dimensions: (scale_height, scale_width)
        Values > 1 upsample, values < 1 downsample
    mode : {'nearest', 'linear'}
        Interpolation mode:
        - 'nearest': Nearest neighbor (fast, blocky output)
        - 'linear': Bilinear interpolation (smooth output)

    Returns
    -------
    TensorVariable
        Resized tensor with shape (batch, channels, H*scale_h, W*scale_w)

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x", dtype="float32")
    >>> # 2x upsampling with nearest neighbor (YOLO11n FPN pattern)
    >>> y = resize(x, scale_factor=(2, 2), mode="nearest")
    >>> # 1.5x upsampling with bilinear interpolation
    >>> y = resize(x, scale_factor=(1.5, 1.5), mode="linear")
    """
    return Resize(scale_factor=scale_factor, mode=mode)(input)
```

**Export function**:

**File**: `pytensor/tensor/__init__.py` (MODIFY)

```python
from pytensor.tensor.resize import resize  # ADD THIS LINE
```

#### Phase 3B: ONNX Resize Converter

**File**: `pytensor/link/onnx/dispatch/resize.py` (NEW)

```python
"""ONNX conversion for resize operations."""

import numpy as np
from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.resize import Resize

from onnx import helper, numpy_helper


@onnx_funcify.register(Resize)
def onnx_funcify_Resize(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Resize op to ONNX Resize node.

    ONNX Resize operator (opset 18):
    - Inputs:
      1. X: Input tensor
      2. roi: Region of interest (optional, we don't use)
      3. scales: Scale factors (what we use)
      4. sizes: Output sizes (alternative to scales, we don't use)
    - Attributes:
      - mode: "nearest" or "linear"
      - coordinate_transformation_mode: How to map coordinates
      - nearest_mode: Rounding mode for nearest neighbor

    Parameters
    ----------
    op : Resize
        The Resize operation instance
    node : Apply
        The apply node
    var_names : dict
        Variable name mapping
    get_var_name : callable
        Name generator

    Returns
    -------
    list of onnx.NodeProto
        ONNX nodes (Resize requires Constant nodes for scales)
    """
    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    input_name = input_names[0]
    output_name = output_names[0]

    # Map PyTensor mode to ONNX mode
    mode_mapping = {
        "nearest": "nearest",
        "linear": "linear",  # ONNX 'linear' = bilinear for 2D
    }

    onnx_mode = mode_mapping.get(op.mode)
    if onnx_mode is None:
        raise ValueError(f"Unsupported resize mode: {op.mode}")

    # ONNX Resize requires scales as a Constant input
    # scales format: [batch_scale, channel_scale, height_scale, width_scale]
    # We don't scale batch or channels, only spatial dimensions
    scale_h, scale_w = op.scale_factor
    scales = np.array([1.0, 1.0, scale_h, scale_w], dtype=np.float32)

    # Create Constant node for scales
    scales_name = f"scales_{output_name}"
    scales_tensor = numpy_helper.from_array(scales, name=scales_name)

    nodes = []

    # Constant node for scales
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[scales_name],
            value=scales_tensor,
            name=f"Const_{scales_name}",
        )
    )

    # ONNX Resize node
    # Inputs: X, roi (empty), scales
    # We create an empty roi tensor since we don't use it
    roi_name = f"roi_{output_name}"
    roi_tensor = numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name)

    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[roi_name],
            value=roi_tensor,
            name=f"Const_{roi_name}",
        )
    )

    # Create Resize node
    nodes.append(
        helper.make_node(
            "Resize",
            inputs=[input_name, roi_name, scales_name],
            outputs=[output_name],
            mode=onnx_mode,
            coordinate_transformation_mode="asymmetric",  # Matches PyTorch default
            nearest_mode="floor" if onnx_mode == "nearest" else None,
            name=f"Resize_{output_name}",
        )
    )

    return nodes
```

**Import registration**:

**File**: `pytensor/link/onnx/dispatch/__init__.py` (MODIFY)

```python
import pytensor.link.onnx.dispatch.resize  # noqa: F401  # ADD THIS LINE
```

#### Success Criteria

##### Automated Verification:
- [ ] PyTensor op tests pass: `pytest tests/tensor/test_resize.py -v`
- [ ] ONNX converter tests pass: `pytest tests/link/onnx/test_resize.py -v`
- [ ] FPN pattern test passes (critical for YOLO11n)
- [ ] No regressions in other tests

##### Manual Verification:
- [ ] Nearest neighbor produces blocky output (correct behavior)
- [ ] Bilinear produces smooth output (correct behavior)
- [ ] YOLO11n FPN pattern works end-to-end

---

### Phase 4: Refactoring & Cleanup

#### Refactoring Targets

1. **Coordinate transformation modes**:
   - [ ] Document why we chose "asymmetric" mode
   - [ ] Consider: Should we make it configurable?

2. **Alternative to scipy dependency**:
   - [ ] Current bilinear uses `scipy.ndimage.zoom`
   - [ ] Consider: Implement pure NumPy version to avoid scipy dependency

3. **Test quality**:
   - [ ] Add visual test (optional): Plot input and output to verify correctness

#### Success Criteria

Same as previous operations - all tests pass, code is clean and maintainable.

---

## Testing Strategy Summary

### Test Coverage Goals

**Operation 1: Concat**
- [ ] Basic concatenation (axis 0, axis 1)
- [ ] Multiple inputs (2, 3, 5 tensors)
- [ ] Different dtypes (float32, float64, int32, int64)
- [ ] Different ranks (1D, 2D, 3D, 4D)
- [ ] Negative axis indexing
- [ ] Integration with Conv2D (C3k2 pattern)
- [ ] Property-based testing (random valid inputs)

**Operation 2: MaxPool**
- [ ] Basic pooling (2x2, 3x3 kernels)
- [ ] Different strides (overlapping vs non-overlapping)
- [ ] Padding (valid, same)
- [ ] Multiple channels and batches
- [ ] SPPF cascade pattern (YOLO11n)
- [ ] Edge cases (1x1 kernel, global pooling)
- [ ] Property-based testing

**Operation 3: Resize**
- [ ] Nearest neighbor upsampling (2x, 1.5x, fractional)
- [ ] Bilinear upsampling (2x, different H/W scales)
- [ ] Downsampling (0.5x)
- [ ] FPN pattern with concat (YOLO11n)
- [ ] Edge cases (1x scale = identity)
- [ ] Property-based testing

### Test Organization

**Test file structure**:
```
tests/
├── tensor/
│   ├── test_pool.py          # PyTensor MaxPool op tests (non-ONNX)
│   └── test_resize.py        # PyTensor Resize op tests (non-ONNX)
├── link/
│   └── onnx/
│       ├── test_join.py      # Join → Concat ONNX converter tests
│       ├── test_pool.py      # MaxPool ONNX converter tests
│       ├── test_resize.py    # Resize ONNX converter tests
│       ├── test_properties.py  # Property-based tests (MODIFY)
│       └── strategies/
│           └── operations.py  # Test strategies (MODIFY)
```

### Running Tests

**Per-operation testing**:
```bash
# Concat
pytest tests/link/onnx/test_join.py -v

# MaxPool
pytest tests/tensor/test_pool.py -v                # PyTensor op
pytest tests/link/onnx/test_pool.py -v             # ONNX converter

# Resize
pytest tests/tensor/test_resize.py -v              # PyTensor op
pytest tests/link/onnx/test_resize.py -v           # ONNX converter
```

**Full test suite**:
```bash
# All new tests
pytest tests/link/onnx/test_join.py tests/tensor/test_pool.py tests/link/onnx/test_pool.py tests/tensor/test_resize.py tests/link/onnx/test_resize.py -v

# All ONNX tests (including existing)
pytest tests/link/onnx/ -v

# Property-based tests
pytest tests/link/onnx/test_properties.py -v --hypothesis-seed=12345
```

**With coverage**:
```bash
pytest tests/link/onnx/ --cov=pytensor/link/onnx/dispatch --cov-report=term-missing
```

---

## Performance Considerations

**MaxPool optimization**:
- Current implementation uses nested loops (slow for large tensors)
- Consider: Implement C code via `c_code()` method
- Or: Use NumPy stride tricks (im2col)
- Benchmark: Compare with NumPy/PyTorch implementations

**Resize optimization**:
- Nearest neighbor is already fast (pure NumPy indexing)
- Bilinear uses scipy.ndimage.zoom (reasonably fast)
- Consider: Pure NumPy implementation to avoid scipy dependency

**ONNX Runtime performance**:
- ONNX Runtime uses optimized kernels (faster than our NumPy implementations)
- Focus on correctness first, then optimize if needed

---

## Migration Notes

**No migration needed** - these are new operations, not replacing existing ones.

**Integration points**:
- Join/Concat: Used with Conv2D in C3k2 blocks
- MaxPool: Used in SPPF block
- Resize: Used in FPN head with Concat

**YOLO11n full pipeline** (after this plan):
```python
# Pseudo-code for YOLO11n backbone + head
x = pt.tensor4("input")

# Backbone
x = conv2d(x, kernel1)  # ✅ Already works
x = pool_2d(x, ws=(5,5))  # ✅ After this plan
x = pool_2d(x, ws=(5,5))
x = pool_2d(x, ws=(5,5))
backbone_out = pt.join(1, x, pool1, pool2, pool3)  # ✅ After this plan

# Head (FPN)
upsampled1 = resize(low_res, scale_factor=(2,2))  # ✅ After this plan
fpn1 = pt.join(1, upsampled1, skip1)  # ✅ After this plan
upsampled2 = resize(fpn1, scale_factor=(2,2))
fpn2 = pt.join(1, upsampled2, skip2)

# At this point, we can export backbone + head to ONNX!
# Still missing: BatchNorm, SiLU, Sigmoid (Tier 2)
```

---

## References

**Original Research**:
- Gap analysis: `thoughts/shared/research/2025-10-14_22-30-00_yolo11n-onnx-backend-gaps.md`
- Identifies 6 missing operations for YOLO11n

**ONNX Specifications**:
- Concat: https://onnx.ai/onnx/operators/onnx__Concat.html
- MaxPool: https://onnx.ai/onnx/operators/onnx__MaxPool.html
- Resize: https://onnx.ai/onnx/operators/onnx__Resize.html

**PyTensor Patterns**:
- Existing converters: `pytensor/link/onnx/dispatch/`
- Conv2D reference: `pytensor/link/onnx/dispatch/conv.py`
- Test patterns: `tests/link/onnx/test_conv.py`

**Related Plans**:
- Conv2D TDD: `thoughts/shared/plans/onnx-conv2d-tdd.md`
- Property-based testing: `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`

---

## Next Steps (After This Plan)

**Tier 2 operations** (separate plan needed):
1. **BatchNorm** - Required by all CNN layers
2. **SiLU** - Required by all activations
3. **Sigmoid** - Simple mapping to ONNX (add to dictionary)

**Tier 3 operations** (lower priority):
4. **Attention mechanisms** (C2PSA blocks)
5. **Global pooling** (detection heads)

**After all 6 operations**:
- Full YOLO11n export to ONNX
- End-to-end integration test
- Performance benchmarking
- Documentation update

---

## Success Metrics

**This plan is successful when:**

- [ ] All 3 Tier 1 blocker operations implemented and tested
- [ ] ~35 unit tests pass (10 Concat + 10 MaxPool + 9 Resize + 6 integration)
- [ ] Property-based tests pass for all operations
- [ ] YOLO11n SPPF pattern exports to ONNX correctly
- [ ] YOLO11n FPN pattern exports to ONNX correctly
- [ ] No regressions in existing ONNX backend tests
- [ ] Code coverage > 90% for new converters
- [ ] All code passes linting and type checking

**Verification command**:
```bash
# Run full test suite
pytest tests/link/onnx/test_join.py \
       tests/tensor/test_pool.py tests/link/onnx/test_pool.py \
       tests/tensor/test_resize.py tests/link/onnx/test_resize.py \
       tests/link/onnx/test_properties.py \
       -v --cov=pytensor/link/onnx/dispatch --cov-report=term-missing

# Verify no regressions
pytest tests/link/onnx/ -v
```

---

## Estimated Timeline

**Operation 1: Concat (Join converter)**
- Test design: 2 hours
- Test failure verification: 30 minutes
- Implementation: 1 hour
- Refactoring: 30 minutes
- **Total: ~4 hours**

**Operation 2: MaxPool**
- Test design: 3 hours (PyTensor op + ONNX tests)
- Test failure verification: 30 minutes
- PyTensor op implementation: 4 hours
- ONNX converter implementation: 2 hours
- Refactoring: 1 hour
- **Total: ~10.5 hours (~1.5 days)**

**Operation 3: Resize**
- Test design: 3 hours (PyTensor op + ONNX tests)
- Test failure verification: 30 minutes
- PyTensor op implementation: 4 hours (nearest + bilinear)
- ONNX converter implementation: 2 hours (multi-node with Constants)
- Refactoring: 1 hour
- **Total: ~10.5 hours (~1.5 days)**

**Grand Total: ~25 hours (~3-4 days of focused development)**

---

**Let's build modern CNN support for PyTensor's ONNX backend!** 🚀
