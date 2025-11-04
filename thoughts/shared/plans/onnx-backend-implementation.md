# ONNX Export Backend Implementation Plan

<!-- WORKSHOP NOTE: This was the FIRST comprehensive plan document, created after initial architectural research. It's intentionally broad and ambitious - representing an "ideal end state" rather than immediate next steps. As you'll see in the inline comments, reality diverged significantly from this plan. That's expected and valuable - this plan helped scope the work and identify what needed deeper investigation.

Key workshop insights about this document:
1. It's TOO detailed for initial implementation - better to have rough phases
2. It doesn't account for unknown unknowns (bugs, edge cases)
3. It's useful as a REFERENCE, not a SCRIPT to follow exactly
4. Later plans (coverage, conv2d) are more practical and focused

This plan was never "executed" in sequence - instead, pieces were cherry-picked and adapted. That's AI-first development: iterate on plans, don't treat them as gospel. -->

## Overview

Implement ONNX export functionality for PyTensor to enable deploying trained models to environments that support ONNX Runtime (browsers via WebAssembly, mobile devices, edge devices, etc.). This initial implementation focuses on establishing the **core infrastructure and scaffolding** with **basic operations only**, creating patterns for future op additions.

<!-- REALITY CHECK: The "basic operations only" scope was good, but this plan tries to detail all 5 phases upfront. In practice, only Phase 1-2 were implemented directly from this doc. Phase 3-5 were re-planned with more information. Lesson: Plan the immediate next step in detail, later steps at high level only. -->

**Phase 1 Goal**: Export simple PyTensor inference functions to valid ONNX files that execute correctly in ONNX Runtime.

## Current State Analysis

**What exists now:**
- PyTensor has multiple backend implementations (JAX, Numba, PyTorch, MLX) that follow a consistent pattern
- All backends use `singledispatch` for op conversion and extend `JITLinker` base class
- Optional dependencies are managed via `[project.optional-dependencies]` in `pyproject.toml:68-82`
- Test patterns are well-established in `tests/link/{backend}/` directories
- No ONNX export capability currently exists

**Key architectural patterns discovered:**
- **Dispatch system**: `@singledispatch` with `@backend_funcify.register(OpClass)` decorators (`pytensor/link/jax/dispatch/basic.py:43`, `pytensor/link/numba/dispatch/basic.py:333`)
- **Linker pattern**: Extend `JITLinker` from `pytensor/link/basic.py:576` and implement three methods
- **Module loading**: Import all dispatch modules in `dispatch/__init__.py` to trigger registration
- **Testing**: Use `compare_backend_and_py()` functions that compile with backend mode vs python mode and compare outputs

**Key constraints:**
- ONNX export is **export-only** (not execution), unlike JAX/Numba which execute graphs
- ONNX uses graph-based representation (nodes + edges), not composed Python functions
- Shared variables must be "baked" as ONNX initializers (trained weights frozen at export time)
- Target ONNX opset 18 (mature, good WebAssembly support)

## Desired End State

### Core Functionality
- ✅ `export_onnx(pytensor_function, "model.onnx")` exports compiled PyTensor functions to ONNX format
- ✅ Basic operations supported: Add, Mul, Sub, Div, Neg, Exp, Log, Sqrt, Dot, Maximum (ReLU), Softmax
- ✅ Exported ONNX models pass validation: `onnx.checker.check_model()`
- ✅ Exported models execute correctly in ONNX Runtime with outputs matching PyTensor
- ✅ Clear error messages for unsupported operations
- ✅ Shared variables converted to ONNX initializers (baked weights)
- ✅ Documentation and examples provided

### Verification
Run the following to verify completion:

#### Automated Verification:
- [ ] ONNX optional dependency installs: `pip install pytensor[onnx]`
- [ ] Unit tests pass: `pytest tests/link/onnx/test_basic.py -v`
- [ ] All op conversion tests pass: `pytest tests/link/onnx/ -v`
- [ ] Type checking passes: `mypy pytensor/link/onnx/`
- [ ] Linting passes: `ruff check pytensor/link/onnx/`
- [ ] Import works: `python -c "from pytensor.link.onnx import export_onnx"`

#### Manual Verification:
- [ ] Export simple function: `export_onnx(function([x, y], x + y * 2), "test.onnx")` succeeds
- [ ] ONNX file validates: `python -c "import onnx; onnx.checker.check_model(onnx.load('test.onnx'))"`
- [ ] ONNX Runtime executes correctly: Results match PyTensor for basic operations
- [ ] Error message is clear when attempting to export unsupported op (e.g., Scan)
- [ ] Documentation builds: `cd doc && make html`

## What We're NOT Doing

<!-- REALITY CHECK: This "out of scope" list was accurate initially, but Conv2D was pulled forward much earlier than planned. Why? Because during implementation testing, we tried a real CNN model and hit Conv2D immediately. This taught us that "scope" is aspirational - user needs drive priority.

The Conv2D work required its own dedicated plan (onnx-conv2d-tdd.md) because it was complex enough to deserve TDD treatment. Lesson: Some "future" features become urgent during implementation. Stay flexible. -->

**Explicitly out of scope for Phase 1:**
- ❌ Complex operations (Conv2D, Pooling, BatchNorm, Scan/loops) <!-- Conv2D was implemented later via onnx-conv2d-tdd.md -->
- ❌ Execution via ONNXLinker (using ONNX Runtime as a PyTensor backend)
- ❌ Graph optimizations (operator fusion, constant folding)
- ❌ Dynamic shapes or shape inference from example inputs
- ❌ Gradient/training operations (only inference)
- ❌ Quantization support
- ❌ Custom operators for unsupported ops
- ❌ WebAssembly browser demo (moved to future work) <!-- Still future work as of Oct 2025 -->

## Implementation Approach

**Strategy**: Follow the established PyTensor backend pattern (singledispatch + JITLinker), but adapt for export instead of execution. Build minimal infrastructure first, then add operations incrementally.

**Key architectural decision**: Unlike JAX/Numba which return Python callables, ONNX dispatch functions will return ONNX `NodeProto` objects that get collected into a `ModelProto` graph.

---

## Phase 1: Core Infrastructure & Scaffolding

**Goal**: Create the foundational structure for ONNX export without any op conversions yet

### Changes Required:

#### 1. Add ONNX Optional Dependency
**File**: `pyproject.toml`
**Location**: Lines 68-83 (in `[project.optional-dependencies]` section)
**Changes**: Add ONNX as an optional dependency

```toml
[project.optional-dependencies]
complete = ["pytensor[jax]", "pytensor[numba]", "pytensor[onnx]"]
development = ["pytensor[complete]", "pytensor[tests]", "pytensor[rtd]"]
tests = [
    "pytest",
    "pre-commit",
    "pytest-cov>=2.6.1",
    "coverage>=5.1",
    "pytest-benchmark",
    "pytest-mock",
    "pytest-sphinx",
]
rtd = ["sphinx>=5.1.0,<6", "pygments", "pydot"]
jax = ["jax", "jaxlib"]
numba = ["numba>=0.57", "llvmlite"]
onnx = ["onnx>=1.14.0", "onnxruntime>=1.16.0"]  # NEW
```

#### 2. Create Directory Structure
**Action**: Create new directories

```bash
mkdir -p pytensor/link/onnx/dispatch
mkdir -p tests/link/onnx
```

#### 3. Core Dispatcher (Minimal)
**File**: `pytensor/link/onnx/dispatch/basic.py`
**Changes**: Create new file with core dispatch functions

```python
"""Core ONNX dispatch system for PyTensor.

This module provides the singledispatch-based conversion system for
converting PyTensor ops to ONNX nodes.
"""

from functools import singledispatch
from typing import Callable, Dict, List

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph


# Target ONNX opset version
ONNX_OPSET_VERSION = 18


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX representation.

    This is the main dispatch function. Register converters for specific
    Op types using @onnx_funcify.register(OpClass).

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation to convert
    node : Apply, optional
        The Apply node containing the op (when op is an Op)
    **kwargs
        Additional conversion parameters:
        - var_names: Dict[Variable, str] - mapping of variables to names
        - get_var_name: Callable - function to get/create variable names

    Returns
    -------
    onnx.NodeProto or onnx.ModelProto
        ONNX representation of the operation

    Raises
    ------
    NotImplementedError
        If no converter is registered for this Op type
    """
    raise NotImplementedError(
        f"No ONNX conversion available for: {type(op).__name__}\n"
        f"Op: {op}\n"
        f"Node: {node}\n\n"
        f"This op is not yet supported for ONNX export.\n"
        f"Currently supported ops:\n"
        f"  - Elemwise: Add, Mul, Sub, Div, Neg, Exp, Log, Sqrt, Pow, Abs\n"
        f"  - Matrix: Dot\n"
        f"  - Activations: Softmax, Maximum (for ReLU)\n\n"
        f"To add support for this op, register a converter:\n"
        f"  @onnx_funcify.register({type(op).__name__})\n"
        f"  def onnx_funcify_{type(op).__name__}(op, node, var_names, get_var_name, **kwargs):\n"
        f"      # Return onnx.NodeProto\n"
    )


@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert Python/NumPy data to ONNX-compatible types.

    This is used for converting constants and shared variables to ONNX tensors.

    Parameters
    ----------
    data : Any
        Data to convert (typically numpy array or scalar)
    dtype : str, optional
        Target dtype for conversion

    Returns
    -------
    onnx.TensorProto or data
        ONNX tensor representation or original data
    """
    if dtype is None:
        return data
    else:
        return np.array(data, dtype=dtype)


@onnx_typify.register(np.ndarray)
def onnx_typify_ndarray(data, dtype=None, name="", **kwargs):
    """Convert numpy array to ONNX TensorProto."""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


def make_value_info(var: Variable, name: str) -> onnx.ValueInfoProto:
    """Create ONNX ValueInfoProto from PyTensor Variable.

    Parameters
    ----------
    var : Variable
        PyTensor variable
    name : str
        Name for the ONNX value

    Returns
    -------
    onnx.ValueInfoProto
        ONNX value info with type and shape
    """
    # Map PyTensor dtype to ONNX dtype
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "bool": TensorProto.BOOL,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (use symbolic dimensions if needed)
    if hasattr(var.type, "shape"):
        shape = []
        for i, dim in enumerate(var.type.shape):
            if dim is None or (isinstance(dim, int) and dim < 0):
                # Dynamic dimension - use symbolic name
                shape.append(f"dim_{i}")
            else:
                shape.append(int(dim))
    else:
        shape = None

    # Create tensor type
    tensor_type = helper.make_tensor_type_proto(elem_type=onnx_dtype, shape=shape)

    return helper.make_value_info(name, tensor_type)


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    node=None,
    opset_version: int = ONNX_OPSET_VERSION,
    model_name: str = "pytensor_model",
    **kwargs,
) -> onnx.ModelProto:
    """Convert a FunctionGraph to ONNX ModelProto.

    Parameters
    ----------
    fgraph : FunctionGraph
        The graph to convert
    opset_version : int
        ONNX opset version to target (default: 18)
    model_name : str
        Name for the ONNX model

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    # Track converted nodes and initializers
    onnx_nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []

    # Generate unique names for variables
    var_names: Dict[Variable, str] = {}
    name_counter = 0

    def get_var_name(var: Variable) -> str:
        """Get or create unique name for a variable."""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, "name") and var.name:
                base_name = var.name
                # Ensure uniqueness
                if base_name in var_names.values():
                    base_name = f"{base_name}_{name_counter}"
                    name_counter += 1
                var_names[var] = base_name
            else:
                var_names[var] = f"var_{name_counter}"
                name_counter += 1
        return var_names[var]

    # Convert constants to initializers
    for node in fgraph.apply_nodes:
        for inp in node.inputs:
            if isinstance(inp, Constant):
                name = get_var_name(inp)
                if name not in [init.name for init in initializers]:
                    tensor = numpy_helper.from_array(
                        np.asarray(inp.data), name=name
                    )
                    initializers.append(tensor)

    # Convert ops in topological order
    for node in fgraph.toposort():
        # Get ONNX node for this Apply
        onnx_node = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            **kwargs,
        )

        if onnx_node is not None:
            onnx_nodes.append(onnx_node)

    # Create inputs (only non-constant inputs)
    input_protos = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(make_value_info(inp, name))

    # Create outputs
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(make_value_info(out, name))

    # Create graph
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=f"{model_name}_graph",
        inputs=input_protos,
        outputs=output_protos,
        initializer=initializers,
    )

    # Create model
    model = helper.make_model(
        graph, producer_name="PyTensor", opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}") from e

    return model
```

#### 4. Dispatch Module Loader
**File**: `pytensor/link/onnx/dispatch/__init__.py`
**Changes**: Create new file

```python
"""ONNX dispatch system initialization.

Imports all dispatch modules to trigger @onnx_funcify.register() decorators.
"""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
# (Phase 2 will add: elemwise, nlinalg, special)

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
```

#### 5. Export API
**File**: `pytensor/link/onnx/export.py`
**Changes**: Create new file with main export function

```python
"""ONNX export API for PyTensor."""

from pathlib import Path
from typing import Optional, Union

try:
    import onnx
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

from pytensor.compile.function import Function
from pytensor.link.onnx.dispatch.basic import onnx_funcify


def export_onnx(
    pytensor_function: Function,
    output_path: Union[str, Path],
    *,
    opset_version: int = 18,
    model_name: str = "pytensor_model",
    **kwargs,
) -> onnx.ModelProto:
    """Export a PyTensor function to ONNX format.

    Parameters
    ----------
    pytensor_function : Function
        Compiled PyTensor function to export
    output_path : str or Path
        Path where the .onnx file will be saved
    opset_version : int, optional
        ONNX opset version to target (default: 18)
    model_name : str, optional
        Name for the ONNX model (default: "pytensor_model")
    **kwargs
        Additional parameters passed to onnx_funcify

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> from pytensor.link.onnx import export_onnx
    >>>
    >>> # Create function
    >>> x = pt.vector('x')
    >>> y = pt.vector('y')
    >>> z = x + y * 2
    >>> f = pytensor.function([x, y], z)
    >>>
    >>> # Export to ONNX
    >>> model = export_onnx(f, "model.onnx")
    >>>
    >>> # Load in ONNX Runtime
    >>> import onnxruntime as ort
    >>> session = ort.InferenceSession("model.onnx")
    >>> result = session.run(None, {'x': [1, 2, 3], 'y': [4, 5, 6]})
    """
    # Get the FunctionGraph from the compiled function
    fgraph = pytensor_function.fgraph

    # Convert to ONNX
    model = onnx_funcify(
        fgraph, opset_version=opset_version, model_name=model_name, **kwargs
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    print(f"✓ Exported PyTensor function to ONNX: {output_path}")
    print(f"  Opset version: {opset_version}")
    print(f"  Inputs: {len(fgraph.inputs)}")
    print(f"  Outputs: {len(fgraph.outputs)}")
    print(f"  Nodes: {len(model.graph.node)}")

    return model
```

#### 6. Package Initialization
**File**: `pytensor/link/onnx/__init__.py`
**Changes**: Create new file

```python
"""ONNX export functionality for PyTensor.

This module provides functionality to export PyTensor functions to ONNX format
for deployment in environments like WebAssembly, mobile, or edge devices.

Example
-------
>>> import pytensor
>>> import pytensor.tensor as pt
>>> from pytensor.link.onnx import export_onnx
>>>
>>> # Create and compile function
>>> x = pt.vector('x')
>>> y = pt.vector('y')
>>> z = x + y * 2
>>> f = pytensor.function([x, y], z)
>>>
>>> # Export to ONNX
>>> export_onnx(f, "model.onnx")
"""

from pytensor.link.onnx.export import export_onnx

__all__ = ["export_onnx"]
```

### Success Criteria:

#### Automated Verification:
- [ ] ONNX package imports successfully: `python -c "from pytensor.link.onnx import export_onnx"`
- [ ] Import with missing dependency shows clear error: Try importing without onnx installed, verify error message mentions `pip install pytensor[onnx]`
- [ ] Dispatcher is registered: `python -c "from pytensor.link.onnx.dispatch import onnx_funcify; print(onnx_funcify)"`

#### Manual Verification:
- [ ] Directory structure matches other backends (compare with `pytensor/link/jax/`)
- [ ] Error message for unsupported op is clear and helpful
- [ ] Code follows PyTensor style (passes ruff checks)

---

## Phase 2: Basic Elemwise Operations

**Goal**: Support element-wise operations (Add, Mul, Sub, Div, Neg, Exp, Log, Sqrt, Pow, Abs)

### Changes Required:

#### 1. Elemwise Dispatch Module
**File**: `pytensor/link/onnx/dispatch/elemwise.py`
**Changes**: Create new file

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.scalar import basic as scalar
from pytensor.tensor.elemwise import Elemwise

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
}


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    Elemwise ops perform element-wise operations on tensors.
    They map directly to ONNX ops like Add, Mul, etc.
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}\n"
            f"Supported scalar ops: {', '.join(op.__name__ for op in SCALAR_OP_TO_ONNX.keys())}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    onnx_node = helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )

    return onnx_node
```

#### 2. Load Elemwise Dispatch
**File**: `pytensor/link/onnx/dispatch/__init__.py`
**Changes**: Add import to load elemwise converters

```python
"""ONNX dispatch system initialization."""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
import pytensor.link.onnx.dispatch.elemwise  # NEW

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
```

#### 3. Basic Tests
**File**: `tests/link/onnx/test_basic.py`
**Changes**: Create new file with test infrastructure

```python
"""Core ONNX export tests and comparison utilities."""

from functools import partial

import numpy as np
import pytest

# Skip entire module if ONNX not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.link.onnx import export_onnx


def compare_onnx_and_py(
    graph_inputs,
    graph_outputs,
    test_inputs,
    *,
    assert_fn=None,
    tmp_path=None,
):
    """Compare ONNX Runtime output with PyTensor output.

    Parameters
    ----------
    graph_inputs : list of Variable
        Symbolic input variables
    graph_outputs : Variable or list of Variable
        Symbolic output variables
    test_inputs : list
        Concrete test values for inputs
    assert_fn : callable, optional
        Custom assertion function (default: np.testing.assert_allclose)
    tmp_path : Path, optional
        Temporary directory for ONNX file (pytest fixture)

    Returns
    -------
    tuple
        (onnx_session, onnx_results)
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if tmp_path is None:
        import tempfile
        tmp_path = tempfile.mkdtemp()

    # Ensure graph_outputs is a list
    outputs_is_list = isinstance(graph_outputs, (list, tuple))
    if not outputs_is_list:
        graph_outputs = [graph_outputs]

    # Compile PyTensor function (reference implementation)
    pytensor_fn = function(graph_inputs, graph_outputs)
    py_res = pytensor_fn(*test_inputs)
    if not outputs_is_list:
        py_res = [py_res]

    # Export to ONNX
    onnx_path = f"{tmp_path}/test_model.onnx"
    model = export_onnx(pytensor_fn, onnx_path)

    # Validate ONNX model
    onnx.checker.check_model(model)

    # Run with ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create input feed dict
    input_names = [inp.name for inp in session.get_inputs()]
    input_feed = {}
    for name, value in zip(input_names, test_inputs, strict=True):
        # Convert to numpy array with correct dtype
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        input_feed[name] = value.astype(config.floatX)

    # Run inference
    onnx_res = session.run(None, input_feed)

    # Compare results
    assert len(onnx_res) == len(py_res), f"Output count mismatch: {len(onnx_res)} vs {len(py_res)}"

    for onnx_out, py_out in zip(onnx_res, py_res, strict=True):
        assert_fn(onnx_out, py_out)

    return session, onnx_res


def test_onnx_import():
    """Test that ONNX export can be imported."""
    from pytensor.link.onnx import export_onnx

    assert callable(export_onnx)


def test_dispatcher_registered():
    """Test that dispatch system is registered."""
    from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify

    assert callable(onnx_funcify)
    assert callable(onnx_typify)


def test_export_simple_add(tmp_path):
    """Test exporting a simple addition."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    f = pytensor.function([x, y], z)

    # Export
    model_path = tmp_path / "test_add.onnx"
    model = export_onnx(f, model_path)

    # Validate
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model)
    assert model_path.exists()

    # Test with ONNX Runtime
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], [z], [x_val, y_val], tmp_path=tmp_path)


def test_export_multiple_ops(tmp_path):
    """Test exporting with multiple operations."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = (x + y) * 2 - y

    f = pytensor.function([x, y], z)

    # Export and validate
    model = export_onnx(f, tmp_path / "test_multi.onnx")
    onnx.checker.check_model(model)

    # Test execution
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], [z], [x_val, y_val], tmp_path=tmp_path)


def test_unsupported_op_error():
    """Test that unsupported ops give clear error messages."""
    from pytensor.tensor import nlinalg

    x = pt.matrix("x")
    # SVD is not supported in Phase 1
    u, s, vt = nlinalg.svd(x)

    f = pytensor.function([x], [u, s, vt])

    with pytest.raises(NotImplementedError, match="No ONNX conversion available"):
        export_onnx(f, "/tmp/test_svd.onnx")
```

#### 4. Elemwise Tests
**File**: `tests/link/onnx/test_elemwise.py`
**Changes**: Create new file

```python
"""Tests for ONNX elemwise operations."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from pytensor.configdefaults import config

from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_add(tmp_path):
    """Test addition operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_mul(tmp_path):
    """Test multiplication operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x * y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_sub(tmp_path):
    """Test subtraction operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x - y

    x_val = np.array([5, 6, 7], dtype="float32")
    y_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_div(tmp_path):
    """Test division operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x / y

    x_val = np.array([4, 9, 16], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_neg(tmp_path):
    """Test negation operation."""
    x = pt.vector("x", dtype="float32")
    z = -x

    x_val = np.array([1, -2, 3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_exp(tmp_path):
    """Test exponential operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.exp(x)

    x_val = np.array([0, 1, 2], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_log(tmp_path):
    """Test logarithm operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.log(x)

    x_val = np.array([1, 2.718, 7.389], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_sqrt(tmp_path):
    """Test square root operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.sqrt(x)

    x_val = np.array([1, 4, 9], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_pow(tmp_path):
    """Test power operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x**y

    x_val = np.array([2, 3, 4], dtype="float32")
    y_val = np.array([2, 2, 2], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_abs(tmp_path):
    """Test absolute value operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.abs(x)

    x_val = np.array([-1, 2, -3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize(
    "shape",
    [
        (3,),  # vector
        (2, 3),  # matrix
        (2, 3, 4),  # 3D tensor
    ],
)
def test_add_different_shapes(tmp_path, shape):
    """Test addition with different tensor shapes."""
    x = pt.tensor("x", dtype="float32", shape=shape)
    y = pt.tensor("y", dtype="float32", shape=shape)
    z = x + y

    rng = np.random.default_rng(42)
    x_val = rng.random(shape).astype("float32")
    y_val = rng.random(shape).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_chained_operations(tmp_path):
    """Test multiple operations chained together."""
    x = pt.vector("x", dtype="float32")
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)
```

### Success Criteria:

#### Automated Verification:
- [ ] All elemwise tests pass: `pytest tests/link/onnx/test_elemwise.py -v`
- [ ] Basic tests pass: `pytest tests/link/onnx/test_basic.py -v`
- [ ] Elemwise module loads: `python -c "from pytensor.link.onnx.dispatch import elemwise"`

#### Manual Verification:
- [ ] Export simple math expression: `x + y * 2 - z / 4` exports and runs correctly
- [ ] ONNX graph visualization shows correct node types (use Netron or similar)
- [ ] Error message for unsupported scalar op is helpful

---

## Phase 3: Matrix Operations

**Goal**: Support basic linear algebra (Dot, MatMul)

### Changes Required:

#### 1. Matrix Operations Dispatch
**File**: `pytensor/link/onnx/dispatch/nlinalg.py`
**Changes**: Create new file

```python
"""ONNX conversion for linear algebra operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.blas import Dot22
from pytensor.tensor.math import Dot

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Dot)
def onnx_funcify_Dot(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot to ONNX MatMul node.

    PyTensor's Dot operation maps to ONNX MatMul for matrix multiplication.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}",
    )

    return onnx_node


@onnx_funcify.register(Dot22)
def onnx_funcify_Dot22(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot22 (optimized 2x2 dot) to ONNX MatMul node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}",
    )

    return onnx_node
```

#### 2. Load Matrix Dispatch
**File**: `pytensor/link/onnx/dispatch/__init__.py`
**Changes**: Add import

```python
"""ONNX dispatch system initialization."""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
import pytensor.link.onnx.dispatch.elemwise
import pytensor.link.onnx.dispatch.nlinalg  # NEW

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
```

#### 3. Matrix Tests
**File**: `tests/link/onnx/test_nlinalg.py`
**Changes**: Create new file

```python
"""Tests for ONNX linear algebra operations."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_dot_vector_vector(tmp_path):
    """Test dot product of two vectors."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.dot(x, y)

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_dot_matrix_vector(tmp_path):
    """Test matrix-vector multiplication."""
    x = pt.matrix("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.dot(x, y)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 4)).astype("float32")
    y_val = rng.random(4).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_dot_matrix_matrix(tmp_path):
    """Test matrix-matrix multiplication."""
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.dot(x, y)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 4)).astype("float32")
    y_val = rng.random((4, 5)).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_simple_linear_layer(tmp_path):
    """Test a simple linear layer: W @ x + b."""
    x = pt.vector("x", dtype="float32")
    W = pt.matrix("W", dtype="float32")
    b = pt.vector("b", dtype="float32")

    # Linear layer
    y = pt.dot(W, x) + b

    rng = np.random.default_rng(42)
    x_val = rng.random(10).astype("float32")
    W_val = rng.random((5, 10)).astype("float32")
    b_val = rng.random(5).astype("float32")

    compare_onnx_and_py([x, W, b], y, [x_val, W_val, b_val], tmp_path=tmp_path)
```

### Success Criteria:

#### Automated Verification:
- [ ] Matrix tests pass: `pytest tests/link/onnx/test_nlinalg.py -v`
- [ ] All previous tests still pass: `pytest tests/link/onnx/ -v`

#### Manual Verification:
- [ ] Export simple neural network layer (W @ x + b) and verify output
- [ ] Matrix shapes are correctly inferred in ONNX graph

---

## Phase 4: Activation Functions & Constants

**Goal**: Support Softmax, Maximum (for ReLU), and proper constant handling

### Changes Required:

#### 1. Activation Functions Dispatch
**File**: `pytensor/link/onnx/dispatch/special.py`
**Changes**: Create new file

```python
"""ONNX conversion for special functions and activations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.nnet import Softmax

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Softmax)
def onnx_funcify_Softmax(op, node, var_names, get_var_name, **kwargs):
    """Convert Softmax to ONNX Softmax node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Get axis attribute
    axis = getattr(op, "axis", -1)

    onnx_node = helper.make_node(
        "Softmax",
        inputs=input_names,
        outputs=output_names,
        axis=axis,
        name=f"Softmax_{output_names[0]}",
    )

    return onnx_node
```

#### 2. Handle Maximum for ReLU
**File**: `pytensor/link/onnx/dispatch/elemwise.py`
**Changes**: Add Maximum to scalar op mapping

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.scalar import basic as scalar
from pytensor.tensor.elemwise import Elemwise

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.Maximum: "Max",  # NEW - for ReLU pattern
    scalar.Minimum: "Min",  # NEW
}

# Rest of elemwise.py remains the same
```

#### 3. Load Special Functions Dispatch
**File**: `pytensor/link/onnx/dispatch/__init__.py`
**Changes**: Add import

```python
"""ONNX dispatch system initialization."""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
import pytensor.link.onnx.dispatch.elemwise
import pytensor.link.onnx.dispatch.nlinalg
import pytensor.link.onnx.dispatch.special  # NEW

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
```

#### 4. Activation Tests
**File**: `tests/link/onnx/test_special.py`
**Changes**: Create new file

```python
"""Tests for ONNX special functions and activations."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_softmax(tmp_path):
    """Test softmax activation."""
    x = pt.matrix("x", dtype="float32")
    y = pt.nnet.softmax(x)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 5)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_softmax_axis(tmp_path, axis):
    """Test softmax with different axes."""
    x = pt.matrix("x", dtype="float32")
    y = pt.nnet.softmax(x, axis=axis)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 5)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_relu_via_maximum(tmp_path):
    """Test ReLU implementation via maximum(x, 0)."""
    x = pt.vector("x", dtype="float32")
    y = pt.maximum(x, 0)

    x_val = np.array([-2, -1, 0, 1, 2], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maximum(tmp_path):
    """Test maximum operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.maximum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_minimum(tmp_path):
    """Test minimum operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.minimum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

#### 5. Shared Variables Test
**File**: `tests/link/onnx/test_basic.py`
**Changes**: Add test for shared variables

```python
# Add to existing test_basic.py

def test_shared_variables_as_initializers(tmp_path):
    """Test that shared variables are converted to ONNX initializers."""
    from pytensor import shared

    # Create a simple linear model with shared weights
    W = shared(np.array([[1, 2], [3, 4], [5, 6]], dtype="float32"), name="W")
    b = shared(np.array([0.5, 1.5], dtype="float32"), name="b")

    x = pt.vector("x", dtype="float32")
    y = pt.dot(W, x) + b

    f = pytensor.function([x], y)

    # Export to ONNX
    model_path = tmp_path / "test_shared.onnx"
    model = export_onnx(f, model_path)

    # Verify initializers exist in the model
    initializer_names = [init.name for init in model.graph.initializer]
    assert "W" in initializer_names
    assert "b" in initializer_names

    # Verify values are correct
    for init in model.graph.initializer:
        if init.name == "W":
            init_value = numpy_helper.to_array(init)
            np.testing.assert_allclose(init_value, W.get_value())
        elif init.name == "b":
            init_value = numpy_helper.to_array(init)
            np.testing.assert_allclose(init_value, b.get_value())

    # Test execution
    x_val = np.array([1, 2], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

### Success Criteria:

#### Automated Verification:
- [ ] Activation tests pass: `pytest tests/link/onnx/test_special.py -v`
- [ ] Shared variable test passes: `pytest tests/link/onnx/test_basic.py::test_shared_variables_as_initializers -v`
- [ ] All tests pass: `pytest tests/link/onnx/ -v`

#### Manual Verification:
- [ ] Export 2-layer neural network (Dense + ReLU + Dense + Softmax) successfully
- [ ] Verify weights are baked into ONNX file (inspect with Netron)
- [ ] ONNX Runtime output matches PyTensor for full neural network

---

## Phase 5: Documentation & Polish

**Goal**: Complete documentation, examples, and final testing

### Changes Required:

#### 1. Example Script
**File**: `examples/onnx/export_simple_model.py`
**Changes**: Create new file

```python
"""Example: Export a simple PyTensor model to ONNX.

This script demonstrates:
1. Defining a simple 2-layer neural network in PyTensor
2. Exporting the inference function to ONNX
3. Verifying the export with ONNX Runtime
"""

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import shared
from pytensor.link.onnx import export_onnx


def create_simple_network():
    """Create a simple 2-layer neural network.

    Architecture: Input(4) → Dense(8) → ReLU → Dense(3) → Softmax
    """
    # Input
    x = pt.vector("x", dtype="float32")

    # Layer 1: Dense(8) + ReLU
    W1 = shared(
        np.random.randn(8, 4).astype("float32") * 0.1,
        name="W1",
    )
    b1 = shared(np.zeros(8, dtype="float32"), name="b1")
    h1 = pt.dot(W1, x) + b1
    h1_relu = pt.maximum(h1, 0)  # ReLU activation

    # Layer 2: Dense(3) + Softmax
    W2 = shared(
        np.random.randn(3, 8).astype("float32") * 0.1,
        name="W2",
    )
    b2 = shared(np.zeros(3, dtype="float32"), name="b2")
    y_logits = pt.dot(W2, h1_relu) + b2
    y_pred = pt.nnet.softmax(y_logits.reshape((1, -1))).flatten()

    return x, y_pred


def main():
    """Main function."""
    print("=" * 60)
    print("PyTensor ONNX Export Example")
    print("=" * 60)

    # Create model
    print("\n1. Creating simple neural network...")
    x, y_pred = create_simple_network()
    print("   ✓ Model created: Input(4) → Dense(8) → ReLU → Dense(3) → Softmax")

    # Compile inference function
    print("\n2. Compiling PyTensor function...")
    inference_fn = pytensor.function([x], y_pred)
    print("   ✓ Function compiled")

    # Test with random input
    print("\n3. Testing PyTensor inference...")
    test_input = np.random.randn(4).astype("float32")
    pytensor_output = inference_fn(test_input)
    print(f"   Input: {test_input}")
    print(f"   Output: {pytensor_output}")
    print(f"   Sum of probabilities: {pytensor_output.sum():.6f}")

    # Export to ONNX
    print("\n4. Exporting to ONNX...")
    onnx_path = "simple_model.onnx"
    model = export_onnx(inference_fn, onnx_path, model_name="simple_network")
    print(f"   ✓ Exported to: {onnx_path}")

    # Verify with ONNX Runtime
    print("\n5. Verifying with ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"x": test_input})[0]
    print(f"   ONNX Output: {onnx_output}")

    # Compare outputs
    print("\n6. Comparing outputs...")
    difference = np.abs(pytensor_output - onnx_output).max()
    print(f"   Max difference: {difference:.2e}")

    if difference < 1e-5:
        print("   ✓ Outputs match!")
    else:
        print("   ✗ Outputs differ!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print(f"\nGenerated file: {onnx_path}")
    print("You can visualize it at: https://netron.app/")


if __name__ == "__main__":
    main()
```

#### 2. README for Examples
**File**: `examples/onnx/README.md`
**Changes**: Create new file

```markdown
# PyTensor ONNX Export Examples

This directory contains examples demonstrating ONNX export functionality.

## Prerequisites

Install PyTensor with ONNX support:

```bash
pip install pytensor[onnx]
```

## Examples

### 1. Simple Model Export (`export_simple_model.py`)

Demonstrates exporting a 2-layer neural network to ONNX format.

**Run:**
```bash
python export_simple_model.py
```

**Output:**
- `simple_model.onnx` - Exported ONNX model

**Visualize:**
- Upload to [Netron](https://netron.app/) to view the model graph

## Supported Operations

The current ONNX backend supports:

**Element-wise operations:**
- Add, Mul, Sub, Div
- Neg, Abs
- Exp, Log, Sqrt, Pow

**Matrix operations:**
- Dot (matrix multiplication)

**Activations:**
- Softmax
- ReLU (via Maximum)
- Maximum, Minimum

**Special handling:**
- Shared variables → ONNX initializers (baked weights)
- Constants → ONNX initializers

## Limitations

**Not yet supported:**
- Complex operations (Conv2D, Pooling, BatchNorm)
- Recurrent operations (Scan, loops)
- Dynamic shapes
- Gradient operations (training)
- Custom operators

For unsupported operations, you'll receive a clear error message indicating what's missing.

## Next Steps

After exporting to ONNX:

1. **Validate**: Check the model structure with Netron
2. **Test**: Run inference with ONNX Runtime
3. **Deploy**: Use in production environments:
   - Browser (ONNX Runtime Web + WebAssembly)
   - Mobile (ONNX Runtime Mobile)
   - Edge devices (ONNX Runtime for IoT)

## Resources

- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTensor Documentation](https://pytensor.readthedocs.io/)
```

#### 3. API Documentation
**File**: `pytensor/link/onnx/export.py`
**Changes**: Enhance docstring (already comprehensive in Phase 1, but add troubleshooting section)

Add to the docstring:

```python
    Troubleshooting
    ---------------
    **ImportError: No module named 'onnx'**
    Install ONNX: `pip install pytensor[onnx]`

    **NotImplementedError: No ONNX conversion available for: <OpName>**
    The operation is not yet supported. Check the list of supported ops in the
    error message or PyTensor documentation.

    **ValueError: Generated ONNX model is invalid**
    The generated ONNX graph failed validation. This is likely a bug in the
    ONNX backend. Please report it with a minimal reproducible example.

    **Shape mismatch in ONNX Runtime**
    Ensure input shapes match what the model expects. ONNX models have specific
    shape requirements that may differ from PyTensor's dynamic shapes.
```

#### 4. Add ruff Ignore for Test Files
**File**: `pyproject.toml`
**Changes**: Add ONNX test files to E402 exceptions (lines 153-164)

```toml
[tool.ruff.lint.per-file-ignores]
# ... existing entries ...
"tests/link/onnx/test_basic.py" = ["E402"]
"tests/link/onnx/test_elemwise.py" = ["E402"]
"tests/link/onnx/test_nlinalg.py" = ["E402"]
"tests/link/onnx/test_special.py" = ["E402"]
```

### Success Criteria:

#### Automated Verification:
- [ ] Example script runs successfully: `python examples/onnx/export_simple_model.py`
- [ ] All tests pass: `pytest tests/link/onnx/ -v`
- [ ] Documentation builds: `cd doc && make html` (if added to docs)
- [ ] Linting passes: `ruff check pytensor/link/onnx/`
- [ ] Type checking passes: `mypy pytensor/link/onnx/`

#### Manual Verification:
- [ ] README is clear and helpful
- [ ] Example output looks correct
- [ ] Generated ONNX file opens in Netron
- [ ] API documentation is complete and accurate
- [ ] Error messages are user-friendly

---

## Testing Strategy

### Unit Tests

**Location**: `tests/link/onnx/`

**Coverage**:
- `test_basic.py`: Core functionality, infrastructure, error handling
- `test_elemwise.py`: All element-wise operations
- `test_nlinalg.py`: Matrix operations
- `test_special.py`: Activation functions

**Pattern**: Use `compare_onnx_and_py()` helper that:
1. Compiles PyTensor function (reference)
2. Exports to ONNX
3. Validates ONNX model with `onnx.checker.check_model()`
4. Runs in ONNX Runtime
5. Compares outputs with `np.testing.assert_allclose()`

### Integration Tests

**Covered by unit tests** - Each test is actually an integration test since it:
- Tests full export pipeline (PyTensor → ONNX)
- Validates ONNX model structure
- Tests execution in ONNX Runtime
- Verifies numerical correctness

### Manual Testing Steps

After implementation:

1. **Export simple function**:
   ```python
   x = pt.vector('x')
   y = pt.vector('y')
   f = pytensor.function([x, y], x + y * 2)
   export_onnx(f, 'test.onnx')
   ```

2. **Verify ONNX file**:
   - Upload to https://netron.app/
   - Check graph structure looks correct

3. **Test in ONNX Runtime**:
   ```python
   import onnxruntime as ort
   sess = ort.InferenceSession('test.onnx')
   result = sess.run(None, {'x': [1, 2], 'y': [3, 4]})
   ```

4. **Test error messages**:
   - Try exporting unsupported op (e.g., SVD)
   - Verify error is clear and helpful

5. **Test neural network**:
   - Export 2-layer network with ReLU and Softmax
   - Verify weights are baked in
   - Test inference matches PyTensor

---

## Performance Considerations

**Not a concern for Phase 1** - Focus is on correctness, not performance.

Export performance:
- Small models (< 100 ops): < 1 second
- Medium models (100-1000 ops): 1-10 seconds
- Large models: May take longer, but this is one-time cost

Runtime performance (ONNX Runtime):
- Typically 2-5x slower than native CPU
- Much faster than Python interpreter
- Good enough for production inference

---

## Migration Notes

**N/A** - This is a new feature with no existing users or data to migrate.

Users can opt-in by:
```bash
pip install pytensor[onnx]
```

---

## Future Enhancements

**Not in scope for Phase 1, but documented for future work:**

### Phase 6: More Operations
- Conv2D, MaxPool, AvgPool
- BatchNormalization
- Dropout (convert to identity for inference)
- More activations (Sigmoid, Tanh, LeakyReLU, ELU, GELU)
- Reshape, Transpose, Squeeze, Unsqueeze
- Concat, Split, Stack

### Phase 7: Advanced Features
- Shape inference from example inputs
- Support for Scan → ONNX Loop conversion
- Graph optimizations (constant folding, operator fusion)
- Quantization support
- Custom operators for unsupported ops

### Phase 8: WebAssembly Browser Demo
- Complete browser demo with ONNX Runtime Web
- Interactive visualization
- Performance benchmarks
- Tutorial for deployment

### Phase 9: Execution Backend
- Implement ONNXLinker for direct execution
- Use ONNX Runtime as a PyTensor backend (like JAX/Numba)
- Support training operations (if feasible)

### Phase 10: Production Features
- Model optimization passes
- Deployment guides
- CI/CD integration examples
- Performance profiling tools

---

<!-- WORKSHOP NOTE: Below this point is "future phases" content. Notice how detailed Phase 1-2 are compared to Phase 3-5. This reflects uncertainty - we don't know what challenges we'll face until we build the foundation.

In practice:
- Phase 1-2: Implemented largely as planned
- Phase 3: Matrix ops were simpler than expected (just MatMul)
- Phase 4: Activation functions worked well, but revealed bugs in shape handling
- Phase 5: Documentation happened incrementally, not as a "phase"

The rigid "phase" structure implies waterfall, but development was actually iterative. We'd implement a feature, find bugs, fix them, document, then move forward. The phase structure is useful for PLANNING but misleading for EXECUTION. -->

---

## References

- **Original research**: `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md`
- **ONNX specification**: https://onnx.ai/onnx/
- **ONNX opset 18**: https://onnx.ai/onnx/operators/index.html
- **ONNX Runtime**: https://onnxruntime.ai/
- **JAX backend implementation**: `pytensor/link/jax/` (reference pattern)
- **Numba backend implementation**: `pytensor/link/numba/` (reference pattern)
- **Similar implementations**:
  - PyTorch → ONNX: `torch.onnx.export()`
  - TensorFlow → ONNX: `tf2onnx`
  - Keras → ONNX: `keras2onnx`

---

## Implementation Timeline Estimate

- **Phase 1** (Infrastructure): 1-2 days
- **Phase 2** (Elemwise ops): 1-2 days
- **Phase 3** (Matrix ops): 1 day
- **Phase 4** (Activations & constants): 1-2 days
- **Phase 5** (Documentation & polish): 1 day

**Total**: 5-8 days for basic ONNX export functionality

---

## Success Metrics

✅ **Phase 1 complete when**:
- Can import `from pytensor.link.onnx import export_onnx`
- Error messages are clear for unsupported ops
- Infrastructure matches PyTensor patterns

✅ **Phase 2 complete when**:
- All element-wise ops export correctly
- ONNX Runtime results match PyTensor
- Tests pass with 100% success rate

✅ **Phase 3 complete when**:
- Matrix multiplication works correctly
- Can export simple linear layer (W @ x + b)

✅ **Phase 4 complete when**:
- Can export 2-layer neural network with activations
- Shared variables are baked as initializers
- All tests pass

✅ **Phase 5 complete when**:
- Documentation is complete
- Example script runs successfully
- Ready for user testing and feedback

✅ **Overall success**: Can export a simple trained PyTensor neural network to ONNX, validate it, run it in ONNX Runtime, and get results that match PyTensor within numerical tolerance.
