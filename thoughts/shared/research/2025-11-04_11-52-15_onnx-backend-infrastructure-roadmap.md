---
date: 2025-11-04T11:52:15Z
researcher: Claude
git_commit: b556aec588e2f55a347e5e30ed955d3a611f8a20
branch: onnx-backend
repository: pytensor-workshop-demo
topic: "ONNX Backend Infrastructure Roadmap: Linker, Dispatch, Export API, and Testing"
tags: [research, onnx, backend, infrastructure, linker, dispatch, api, testing]
status: complete
last_updated: 2025-11-04
last_updated_by: Claude
---

# Research: ONNX Backend Infrastructure Roadmap

**Date**: 2025-11-04T11:52:15Z
**Researcher**: Claude
**Git Commit**: b556aec588e2f55a347e5e30ed955d3a611f8a20
**Branch**: onnx-backend
**Repository**: pytensor-workshop-demo

## Research Question

What infrastructure components (linker, dispatch system, export API, testing framework, etc.) are needed for an ONNX backend in PyTensor, and how should they be implemented?

## Executive Summary

**Purpose**: This document complements the operations roadmap by detailing the infrastructure needed to build a production ONNX backend. While the operations roadmap focuses on *which* PyTensor operations to implement (the "what"), this document focuses on *how* to build the supporting infrastructure (the "how").

**Key Finding**: An ONNX backend requires **7 major infrastructure components** that must be built before or alongside operation implementations:

1. **Linker Architecture** - Handles graph-to-ONNX conversion and execution (1-2 weeks)
2. **Dispatch System** - Maps PyTensor Ops to ONNX operators (1 week, foundational)
3. **Export API** - User-facing interface for ONNX export (1 week)
4. **Module Structure** - File organization and packaging (1 day, foundational)
5. **Testing Infrastructure** - Validation framework and test utilities (1 week)
6. **Build & CI Integration** - Dependency management and continuous integration (2-3 days)
7. **Documentation** - User guides and API reference (1-2 weeks)

**Timeline**: 4-6 weeks for complete infrastructure, can be done in parallel with operation implementation

**Critical Path**: Module Structure → Dispatch System → Linker → Export API → Testing

---

## Implementation Roadmap Overview

### Phase 1: Foundation (Week 1)
- ✅ Module structure and file organization
- ✅ Basic dispatch system (`onnx_funcify`, `onnx_typify`)
- ✅ Linker stub with FunctionGraph conversion
- ✅ Basic test utilities (`compare_onnx_and_py`)

### Phase 2: Core Infrastructure (Weeks 2-3)
- ✅ Complete linker implementation
- ✅ Export API (`export_onnx`, Mode integration)
- ✅ Graph traversal and variable naming
- ✅ Type system integration
- ✅ Comprehensive testing framework

### Phase 3: Polish & Integration (Weeks 4-6)
- ✅ CI/CD integration
- ✅ Documentation and examples
- ✅ Performance benchmarking
- ✅ Error handling and validation

---

## Detailed Infrastructure Components

## 1. Linker Architecture

### 1.1 Overview

The **linker** is the core component that converts a PyTensor `FunctionGraph` into an executable ONNX model. For ONNX, this means generating an ONNX `ModelProto` that can be:
1. Saved to disk as `.onnx` file
2. Executed by ONNX Runtime
3. Deployed to various platforms

**Key Difference from JAX/Numba Linkers**: Unlike JIT backends that return Python callables, the ONNX linker produces a **static graph representation** (ONNX ModelProto).

### 1.2 Linker Class Hierarchy

**Base Class Pattern** (from `pytensor/link/basic.py:144-229`):
```python
from pytensor.link.basic import Linker

class Linker(ABC):
    """Abstract base class for all linkers"""

    @abstractmethod
    def make_thunk(self, **kwargs) -> tuple[Callable, InputStorageType, OutputStorageType]:
        """Return (function, input_storage, output_storage) triplet"""
        pass

    def schedule(self, fgraph: FunctionGraph) -> list[Apply]:
        """Returns execution order of nodes"""
        pass
```

**ONNX Linker Options**:

#### Option A: Extend JITLinker (Recommended for Development)

Allows testing via ONNX Runtime execution:

```python
# pytensor/link/onnx/linker.py

from pytensor.link.basic import JITLinker
from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify
from functools import singledispatch

class ONNXLinker(JITLinker):
    """A Linker that converts PyTensor graphs to ONNX models"""

    def __init__(self, opset_version=18, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opset_version = opset_version
        self.onnx_model = None

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        """Convert FunctionGraph to ONNX ModelProto

        Returns
        -------
        onnx_model : onnx.ModelProto
            Complete ONNX model
        """
        # Use dispatch system to convert graph
        self.onnx_model = onnx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            opset_version=self.opset_version,
            **kwargs
        )

        # Return wrapper function that executes via ONNX Runtime
        return self._create_onnx_runtime_function(self.onnx_model)

    def _create_onnx_runtime_function(self, onnx_model):
        """Create ONNX Runtime inference session"""
        import onnxruntime as ort

        # Serialize model to bytes
        model_bytes = onnx_model.SerializeToString()

        # Create inference session
        session = ort.InferenceSession(model_bytes)

        def onnx_runtime_fn(*inputs):
            """Execute ONNX model via ONNX Runtime"""
            # Map inputs to ONNX input names
            input_names = [inp.name for inp in session.get_inputs()]
            input_dict = {name: inp for name, inp in zip(input_names, inputs)}

            # Run inference
            output_names = [out.name for out in session.get_outputs()]
            outputs = session.run(output_names, input_dict)

            return outputs if len(outputs) > 1 else outputs[0]

        return onnx_runtime_fn

    def jit_compile(self, fn):
        """No-op for ONNX (already compiled as static graph)"""
        return fn

    def create_thunk_inputs(self, storage_map):
        """Standard input preparation"""
        return [storage_map[n] for n in self.fgraph.inputs]

    def export_to_file(self, filename):
        """Export ONNX model to file"""
        if self.onnx_model is None:
            raise RuntimeError("No ONNX model has been generated yet")

        import onnx
        onnx.save(self.onnx_model, filename)
```

**Key Methods**:
1. `fgraph_convert()` - Converts graph to ONNX ModelProto
2. `_create_onnx_runtime_function()` - Wraps ONNX model for execution
3. `export_to_file()` - Saves ONNX model to disk

#### Option B: Direct Linker Implementation (Simpler, Export-Only)

For pure export without execution:

```python
class ONNXExportLinker(Linker):
    """Simplified linker for ONNX export only"""

    def __init__(self, opset_version=18, allow_gc=None, scheduler=None):
        super().__init__(allow_gc=allow_gc, scheduler=scheduler)
        self.opset_version = opset_version
        self.onnx_model = None

    def accept(self, fgraph, no_recycling=None, profile=None):
        """Associate FunctionGraph with this linker"""
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_thunk(self, input_storage=None, output_storage=None, storage_map=None):
        """Create ONNX model and return stub thunk"""
        # Convert graph to ONNX
        self.onnx_model = onnx_funcify(
            self.fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            opset_version=self.opset_version
        )

        # Return stub function (not meant to be executed)
        def stub_thunk():
            raise NotImplementedError(
                "ONNX export linker is for export only, not execution. "
                "Use ONNXLinker with ONNX Runtime for execution."
            )

        # Create empty storage containers
        if input_storage is None:
            input_storage = [[None] for _ in self.fgraph.inputs]
        if output_storage is None:
            output_storage = [[None] for _ in self.fgraph.outputs]

        return stub_thunk, input_storage, output_storage
```

### 1.3 FunctionGraph to ONNX Conversion

The core conversion logic in `fgraph_convert()` / `make_thunk()`:

```python
@singledispatch
def onnx_funcify(op, node=None, storage_map=None, **kwargs):
    """Convert PyTensor Op/FunctionGraph to ONNX"""
    raise NotImplementedError(f"No ONNX conversion for: {op}")

@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph,
    node=None,
    input_storage=None,
    storage_map=None,
    opset_version=18,
    model_name="pytensor_model",
    **kwargs
):
    """Convert FunctionGraph to ONNX ModelProto"""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    # Track ONNX nodes and initializers
    onnx_nodes = []
    initializers = []

    # Variable name management
    var_names = {}
    name_counter = 0

    def get_var_name(var):
        """Get or create unique name for variable"""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, 'name') and var.name:
                base_name = var.name
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

    # Convert operations in topological order
    for node in fgraph.toposort():
        # Convert this node to ONNX node(s)
        onnx_node_or_nodes = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            opset_version=opset_version,
            **kwargs
        )

        # Add to ONNX graph
        if onnx_node_or_nodes is not None:
            if isinstance(onnx_node_or_nodes, list):
                onnx_nodes.extend(onnx_node_or_nodes)
            else:
                onnx_nodes.append(onnx_node_or_nodes)

    # Create input protos (non-constant inputs only)
    input_protos = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(make_value_info(inp, name))

    # Create output protos
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(make_value_info(out, name))

    # Create ONNX graph
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=f"{model_name}_graph",
        inputs=input_protos,
        outputs=output_protos,
        initializer=initializers
    )

    # Create ONNX model
    model = helper.make_model(
        graph,
        producer_name="PyTensor",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}") from e

    return model
```

**Key Components**:
1. **Variable Naming**: Unique name generation for all variables
2. **Constant Handling**: Convert PyTensor Constants to ONNX initializers
3. **Node Conversion**: Dispatch to op-specific converters
4. **Type Mapping**: PyTensor types to ONNX TensorProto types
5. **Validation**: ONNX checker to validate generated model

### 1.4 Type Mapping Utilities

```python
def make_value_info(var: Variable, name: str) -> onnx.ValueInfoProto:
    """Create ONNX ValueInfoProto from PyTensor Variable"""
    # Map PyTensor dtype to ONNX dtype
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
        "bool": TensorProto.BOOL,
        "complex64": TensorProto.COMPLEX64,
        "complex128": TensorProto.COMPLEX128,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (handle symbolic dimensions)
    if hasattr(var.type, 'shape'):
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
    tensor_type = helper.make_tensor_type_proto(
        elem_type=onnx_dtype, shape=shape
    )

    return helper.make_value_info(name, tensor_type)
```

### 1.5 Linker File Structure

```
pytensor/link/onnx/
├── __init__.py          # Exports ONNXLinker
├── linker.py            # ONNXLinker class
└── utils.py             # Helper functions (make_value_info, etc.)
```

**Timeline**: 1-2 weeks
- Week 1: Basic linker structure, FunctionGraph conversion
- Week 2: Type mapping, validation, ONNX Runtime integration

**Dependencies**: Dispatch system must exist first

---

## 2. Dispatch System

### 2.1 Overview

The **dispatch system** maps PyTensor operations to ONNX operators. It uses Python's `singledispatch` decorator for extensible, type-based dispatch.

**Pattern Reference**: JAX backend (`pytensor/link/jax/dispatch/basic.py:27-46`)

### 2.2 Core Dispatch Functions

**File**: `pytensor/link/onnx/dispatch/basic.py`

```python
"""ONNX dispatch system for PyTensor operations"""

from functools import singledispatch
from typing import Dict, List, Callable
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph


# Target ONNX opset version
ONNX_OPSET_VERSION = 18


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX node(s).

    This is the main dispatch function. Register converters for specific
    Op types using @onnx_funcify.register(OpClass).

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation to convert
    node : Apply, optional
        The Apply node containing the op
    **kwargs
        Additional conversion parameters:
        - var_names: Dict[Variable, str] - variable name mapping
        - get_var_name: Callable - function to get/create variable names
        - opset_version: int - target ONNX opset version

    Returns
    -------
    onnx.NodeProto or List[onnx.NodeProto]
        ONNX node(s) representing the operation

    Raises
    ------
    NotImplementedError
        If no converter is registered for this Op type
    """
    raise NotImplementedError(
        f"No ONNX conversion available for: {type(op).__name__}\n"
        f"Op: {op}\n"
        f"This operation is not yet supported for ONNX export.\n\n"
        f"Currently supported operations:\n"
        f"  Tier 1: Add, Mul, Sub, Div, Neg, Abs, Exp, Log, Sqrt, Pow\n"
        f"  Tier 2: Reshape, DimShuffle, Join, Split, Subtensor\n"
        f"  Tier 3: Sum, Prod, Max, Min, Argmax, Argmin, Alloc\n"
        f"  See operations roadmap for complete list.\n\n"
        f"To add support for this operation, register a converter:\n"
        f"  @onnx_funcify.register({type(op).__name__})\n"
        f"  def onnx_funcify_{type(op).__name__}(op, node, var_names, get_var_name, **kwargs):\n"
        f"      # Return onnx.NodeProto or list of onnx.NodeProto\n"
    )


@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert Python/NumPy data to ONNX-compatible types.

    This is used for converting constants and inputs to ONNX tensors.

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
    """Convert numpy array to ONNX TensorProto"""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


@onnx_funcify.register(Constant)
def onnx_funcify_Constant(op, node, **kwargs):
    """Constants are handled as initializers, not as nodes"""
    return None


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(fgraph, **kwargs):
    """Convert entire FunctionGraph - implemented in linker.py"""
    # This is implemented in the linker's fgraph_convert method
    # Placeholder here for documentation
    raise NotImplementedError(
        "FunctionGraph conversion should be handled by ONNXLinker.fgraph_convert()"
    )
```

### 2.3 Operation Registration Pattern

Each operation category gets its own dispatch file:

**File**: `pytensor/link/onnx/dispatch/elemwise.py`

```python
"""ONNX conversion for elementwise operations"""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise, DimShuffle
from pytensor.scalar import basic as scalar

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    # Arithmetic (Tier 1)
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.IntDiv: "Div",  # Map to Div with type casting

    # Math (Tier 1)
    scalar.Abs: "Abs",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",
    scalar.Floor: "Floor",
    scalar.Ceil: "Ceil",
    scalar.Round: "Round",

    # Min/Max (Tier 1)
    scalar.Maximum: "Max",
    scalar.Minimum: "Min",

    # Trigonometric (Tier 5)
    scalar.Sin: "Sin",
    scalar.Cos: "Cos",
    scalar.Tan: "Tan",
    scalar.ArcSin: "Asin",
    scalar.ArcCos: "Acos",
    scalar.ArcTan: "Atan",

    # Hyperbolic (Tier 5)
    scalar.Sinh: "Sinh",
    scalar.Cosh: "Cosh",
    scalar.Tanh: "Tanh",
    scalar.ArcSinh: "Asinh",
    scalar.ArcCosh: "Acosh",
    scalar.ArcTanh: "Atanh",

    # Comparison (Tier 5)
    scalar.LT: "Less",
    scalar.GT: "Greater",
    scalar.LE: "LessOrEqual",
    scalar.GE: "GreaterOrEqual",
    scalar.EQ: "Equal",

    # Logical (Tier 5)
    scalar.AND: "And",
    scalar.OR: "Or",
    scalar.XOR: "Xor",
    scalar.Invert: "Not",

    # Special (Tier 5)
    scalar.Sigmoid: "Sigmoid",
    scalar.Erf: "Erf",
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
        name=f"{onnx_op_type}_{output_names[0]}"
    )

    return onnx_node


@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(op, node, var_names, get_var_name, **kwargs):
    """Convert DimShuffle to ONNX Transpose/Squeeze/Unsqueeze.

    DimShuffle handles:
    - Transpose: permuting dimensions
    - Squeeze: removing singleton dimensions
    - Unsqueeze: adding singleton dimensions
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    new_order = op.new_order

    # Case 1: Pure transpose (no 'x' in new_order)
    if 'x' not in new_order:
        # Simple transpose
        onnx_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            perm=list(new_order),
            name=f"Transpose_{output_name}"
        )
        return onnx_node

    # Case 2: Has 'x' (unsqueeze operations)
    # This requires multiple ONNX nodes
    nodes = []
    current_name = input_name

    # First, handle any transpose
    non_x_order = [i for i in new_order if i != 'x']
    if non_x_order != sorted(non_x_order):
        # Need transpose
        temp_name = f"{output_name}_transposed"
        nodes.append(helper.make_node(
            "Transpose",
            inputs=[current_name],
            outputs=[temp_name],
            perm=non_x_order,
            name=f"Transpose_{temp_name}"
        ))
        current_name = temp_name

    # Then add unsqueeze for 'x' positions
    unsqueeze_axes = [i for i, val in enumerate(new_order) if val == 'x']
    if unsqueeze_axes:
        nodes.append(helper.make_node(
            "Unsqueeze",
            inputs=[current_name],
            outputs=[output_name],
            axes=unsqueeze_axes,
            name=f"Unsqueeze_{output_name}"
        ))

    return nodes if len(nodes) > 1 else nodes[0]
```

### 2.4 Dispatch Module Organization

**File**: `pytensor/link/onnx/dispatch/__init__.py`

```python
"""ONNX dispatch system for PyTensor operations"""

# Import core dispatch functions
from pytensor.link.onnx.dispatch.basic import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

# Import all dispatch modules to trigger registration
# Order matters: basic ops before complex ops
import pytensor.link.onnx.dispatch.elemwise      # Tier 1 + 5
import pytensor.link.onnx.dispatch.shape         # Tier 2
import pytensor.link.onnx.dispatch.tensor_basic  # Tier 2 + 3
import pytensor.link.onnx.dispatch.math          # Tier 3
import pytensor.link.onnx.dispatch.nlinalg       # Tier 4
import pytensor.link.onnx.dispatch.subtensor     # Tier 2
# Import others as implemented...

__all__ = [
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

### 2.5 Dispatch System Timeline

**Week 1: Foundation**
- Day 1-2: `basic.py` with core dispatch functions
- Day 3-4: `elemwise.py` with Tier 1 operations
- Day 5: Module organization and imports

**Dependencies**: None (foundational component)

**Priority**: Critical path - needed before any operation implementations

---

## 3. Export API

### 3.1 Overview

The **export API** provides user-facing functions for exporting PyTensor graphs to ONNX format. It should support multiple use cases:
1. Export a PyTensor function to `.onnx` file
2. Export a symbolic graph without compilation
3. Integration with PyTensor's `Mode` system

### 3.2 Primary Export Function

**File**: `pytensor/link/onnx/export.py`

```python
"""User-facing API for ONNX export"""

from pathlib import Path
from typing import Iterable, Union
import onnx

from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.compile.function import function
from pytensor.link.onnx.linker import ONNXLinker
from pytensor.link.onnx.dispatch import onnx_funcify


def export_onnx(
    inputs: Iterable[Variable],
    outputs: Union[Variable, Iterable[Variable]],
    filename: Union[str, Path],
    *,
    opset_version: int = 18,
    model_name: str = "pytensor_model",
    doc_string: str = "",
    optimize: bool = True,
) -> onnx.ModelProto:
    """Export a PyTensor computation graph to ONNX format.

    Parameters
    ----------
    inputs : list of Variable
        Input variables for the computation graph
    outputs : Variable or list of Variable
        Output variables to compute
    filename : str or Path
        Path to save the ONNX model (.onnx extension)
    opset_version : int, optional
        ONNX opset version to target (default: 18)
    model_name : str, optional
        Name for the ONNX model (default: "pytensor_model")
    doc_string : str, optional
        Documentation string for the model
    optimize : bool, optional
        Apply PyTensor graph optimizations before export (default: True)

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model

    Examples
    --------
    Export a simple computation:

    >>> import pytensor.tensor as pt
    >>> from pytensor.link.onnx import export_onnx
    >>> x = pt.vector('x')
    >>> y = pt.vector('y')
    >>> z = (x + y) * 2
    >>> export_onnx([x, y], z, 'model.onnx')

    Export with multiple outputs:

    >>> import pytensor.tensor as pt
    >>> x = pt.matrix('x')
    >>> mean = pt.mean(x, axis=0)
    >>> std = pt.std(x, axis=0)
    >>> export_onnx([x], [mean, std], 'stats.onnx')
    """
    # Validate inputs
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of Variables")

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    # Create FunctionGraph
    from pytensor.compile.builders import construct_nominal_fgraph
    from pytensor.compile.mode import ONNX  # Mode defined below

    fgraph = construct_nominal_fgraph(inputs, outputs)

    # Apply optimizations if requested
    if optimize:
        optimizer = ONNX._optimizer
        fgraph = optimizer.rewrite(fgraph)

    # Convert to ONNX
    onnx_model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name=model_name,
    )

    # Add doc string
    if doc_string:
        onnx_model.doc_string = doc_string

    # Save to file
    onnx.save(onnx_model, str(filename))

    print(f"ONNX model exported to: {filename}")
    print(f"  Opset version: {opset_version}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    return onnx_model


def export_function_onnx(
    fn,
    filename: Union[str, Path],
    *,
    opset_version: int = 18,
) -> onnx.ModelProto:
    """Export a compiled PyTensor function to ONNX.

    Parameters
    ----------
    fn : pytensor.compile.function_module.Function
        Compiled PyTensor function
    filename : str or Path
        Path to save the ONNX model
    opset_version : int, optional
        ONNX opset version (default: 18)

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector('x')
    >>> y = x ** 2
    >>> fn = pytensor.function([x], y)
    >>> from pytensor.link.onnx import export_function_onnx
    >>> export_function_onnx(fn, 'square.onnx')
    """
    # Extract FunctionGraph from compiled function
    fgraph = fn.maker.fgraph

    # Get inputs and outputs
    inputs = fgraph.inputs
    outputs = fgraph.outputs

    # Convert to ONNX
    onnx_model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name="pytensor_function",
    )

    # Save
    onnx.save(onnx_model, str(filename))

    return onnx_model


def compile_onnx(
    inputs: Iterable[Variable],
    outputs: Union[Variable, Iterable[Variable]],
    *,
    opset_version: int = 18,
    **kwargs
):
    """Compile a PyTensor graph using ONNX backend.

    This returns a function that executes via ONNX Runtime.

    Parameters
    ----------
    inputs : list of Variable
        Input variables
    outputs : Variable or list of Variable
        Output variables
    opset_version : int, optional
        ONNX opset version (default: 18)
    **kwargs
        Additional arguments passed to pytensor.function()

    Returns
    -------
    Function
        Compiled function that executes via ONNX Runtime

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> from pytensor.link.onnx import compile_onnx
    >>> x = pt.vector('x')
    >>> y = pt.sum(x ** 2)
    >>> fn = compile_onnx([x], y)
    >>> fn([1, 2, 3])
    array(14.)
    """
    from pytensor.compile.mode import ONNX

    # Use ONNX mode for compilation
    return function(inputs, outputs, mode=ONNX, **kwargs)
```

### 3.3 Mode Integration

**File**: `pytensor/compile/mode.py` (additions)

```python
# Add to existing mode.py file

from pytensor.link.onnx.linker import ONNXLinker
from pytensor.graph import RewriteDatabaseQuery

# Register ONNX linker
predefined_linkers["onnx"] = ONNXLinker()

# Define ONNX mode
ONNX = Mode(
    ONNXLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "onnx"],
        exclude=[
            "cxx_only",
            "BlasOpt",
            "fusion",
            "inplace",
            "scan_save_mem_prealloc",
        ],
    ),
)

# Add to predefined modes
predefined_modes["ONNX"] = ONNX
```

### 3.4 Public API Exports

**File**: `pytensor/link/onnx/__init__.py`

```python
"""ONNX backend for PyTensor"""

from pytensor.link.onnx.linker import ONNXLinker
from pytensor.link.onnx.export import (
    export_onnx,
    export_function_onnx,
    compile_onnx,
)
from pytensor.link.onnx.dispatch import (
    onnx_funcify,
    onnx_typify,
    ONNX_OPSET_VERSION,
)

__all__ = [
    "ONNXLinker",
    "export_onnx",
    "export_function_onnx",
    "compile_onnx",
    "onnx_funcify",
    "onnx_typify",
    "ONNX_OPSET_VERSION",
]
```

### 3.5 Usage Examples

```python
# Example 1: Direct export from symbolic graph
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx

x = pt.matrix('x')
y = pt.matrix('y')
z = pt.dot(x, y)

export_onnx([x, y], z, 'matmul.onnx')

# Example 2: Export compiled function
import pytensor

x = pt.vector('x')
y = pt.sum(x ** 2)
fn = pytensor.function([x], y)

from pytensor.link.onnx import export_function_onnx
export_function_onnx(fn, 'sum_squares.onnx')

# Example 3: Compile with ONNX mode
from pytensor.link.onnx import compile_onnx

x = pt.vector('x')
y = pt.mean(x)
fn = compile_onnx([x], y)
result = fn([1, 2, 3, 4, 5])

# Example 4: Use ONNX mode string
fn = pytensor.function([x], y, mode='ONNX')
```

### 3.6 Export API Timeline

**Week 1:**
- Days 1-3: Core export functions
- Days 4-5: Mode integration and testing

**Dependencies**: Linker and dispatch system

---

## 4. Module Structure

### 4.1 Complete Directory Layout

```
pytensor/link/onnx/
├── __init__.py                  # Public API exports
├── linker.py                    # ONNXLinker class
├── export.py                    # export_onnx(), compile_onnx()
├── utils.py                     # Helper utilities
└── dispatch/
    ├── __init__.py              # Import all dispatch modules
    ├── basic.py                 # Core dispatch (onnx_funcify, onnx_typify)
    ├── elemwise.py              # Elemwise operations
    ├── shape.py                 # Shape operations
    ├── tensor_basic.py          # Tensor creation and joining
    ├── math.py                  # Reductions and math
    ├── nlinalg.py               # Linear algebra
    ├── slinalg.py               # Specialized linear algebra
    ├── blas.py                  # BLAS operations
    ├── subtensor.py             # Indexing/slicing
    ├── special.py               # Special functions
    ├── extra_ops.py             # Extra operations
    ├── sort.py                  # Sorting
    ├── control_flow.py          # IfElse, Scan
    └── pad.py                   # Padding

tests/link/onnx/
├── __init__.py
├── conftest.py                  # Pytest fixtures
├── test_basic.py                # Core functionality, compare_onnx_and_py
├── test_elemwise.py             # Element-wise operations
├── test_shape.py                # Shape operations
├── test_tensor_basic.py         # Tensor creation
├── test_math.py                 # Reductions
├── test_nlinalg.py              # Linear algebra
├── test_slinalg.py              # Specialized linalg
├── test_blas.py                 # BLAS
├── test_subtensor.py            # Indexing
├── test_special.py              # Special functions
├── test_extra_ops.py            # Extra ops
├── test_sort.py                 # Sorting
├── test_control_flow.py         # Control flow
├── test_export.py               # Export API
└── test_integration.py          # End-to-end tests
```

### 4.2 File Size Estimates

| File | Estimated LOC | Complexity |
|------|--------------|------------|
| `linker.py` | 200-300 | Medium |
| `export.py` | 150-200 | Low |
| `dispatch/basic.py` | 300-400 | High |
| `dispatch/elemwise.py` | 400-600 | Medium |
| `dispatch/shape.py` | 300-400 | High |
| `dispatch/tensor_basic.py` | 300-400 | Medium |
| `dispatch/math.py` | 200-300 | Low |
| `dispatch/nlinalg.py` | 400-500 | High |
| Each test file | 200-400 | Low-Medium |

**Total Backend Code**: ~3000-4000 LOC
**Total Test Code**: ~3000-4000 LOC

### 4.3 Module Organization Timeline

**Day 1: Directory Setup**
- Create directory structure
- Empty `__init__.py` files
- Basic imports

**Dependencies**: None (first task)

---

## 5. Testing Infrastructure

### 5.1 Core Test Utility

**File**: `tests/link/onnx/test_basic.py`

```python
"""Core testing utilities for ONNX backend"""

import numpy as np
import pytest
from functools import partial

# Import ONNX and skip tests if not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.onnx.linker import ONNXLinker
from pytensor.graph import RewriteDatabaseQuery


# Configure ONNX mode for testing
optimizer = RewriteDatabaseQuery(include=["onnx"], exclude=["cxx_only", "BlasOpt"])
onnx_mode = Mode(linker=ONNXLinker(), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_onnx_and_py(
    graph_inputs,
    graph_outputs,
    test_inputs,
    *,
    assert_fn=None,
    must_validate=True,
    onnx_mode=onnx_mode,
    py_mode=py_mode,
    opset_version=None,
):
    """Compare ONNX Runtime output and Python output for testing equality.

    Parameters
    ----------
    graph_inputs : list of Variable
        Symbolic input variables
    graph_outputs : Variable or list of Variable
        Symbolic output variables
    test_inputs : list
        Concrete test values for inputs
    assert_fn : callable, optional
        Custom assertion function (default: np.testing.assert_allclose with rtol=1e-4)
    must_validate : bool, optional
        Whether ONNX model must pass validation (default: True)
    onnx_mode : Mode, optional
        ONNX compilation mode
    py_mode : Mode, optional
        Python reference mode
    opset_version : int, optional
        ONNX opset version to test

    Returns
    -------
    onnx_fn : Function
        Compiled ONNX function
    onnx_res : array or list of arrays
        ONNX results

    Raises
    ------
    AssertionError
        If outputs don't match
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-6)

    # Validate inputs are root variables
    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables (no owner)")

    # Compile with ONNX backend
    pytensor_onnx_fn = pytensor.function(graph_inputs, graph_outputs, mode=onnx_mode)

    # Execute with ONNX Runtime
    onnx_res = pytensor_onnx_fn(*test_inputs)

    # Validate ONNX model if required
    if must_validate:
        onnx_model = pytensor_onnx_fn.maker.linker.onnx_model
        try:
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            pytest.fail(f"ONNX model validation failed: {e}")

    # Compile with Python backend (reference)
    pytensor_py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    # Compare results
    if isinstance(graph_outputs, (list, tuple)):
        assert len(onnx_res) == len(py_res), "Output count mismatch"
        for i, (o, p) in enumerate(zip(onnx_res, py_res, strict=True)):
            try:
                assert_fn(o, p)
            except AssertionError as e:
                raise AssertionError(f"Output {i} mismatch: {e}") from e
    else:
        assert_fn(onnx_res, py_res)

    return pytensor_onnx_fn, onnx_res


def get_onnx_node_types(fn):
    """Get list of ONNX node types in compiled function.

    Useful for verifying correct ONNX operators were used.

    Parameters
    ----------
    fn : Function
        Compiled PyTensor function with ONNX backend

    Returns
    -------
    list of str
        ONNX operator types
    """
    onnx_model = fn.maker.linker.onnx_model
    return [node.op_type for node in onnx_model.graph.node]


def get_onnx_node_by_type(fn, op_type):
    """Get ONNX node by operator type.

    Parameters
    ----------
    fn : Function
        Compiled function
    op_type : str
        ONNX operator type (e.g., "Conv", "MatMul")

    Returns
    -------
    onnx.NodeProto or None
        First matching node
    """
    onnx_model = fn.maker.linker.onnx_model
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            return node
    return None


# Module-level fixtures
@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    """Configure PyTensor for ONNX testing"""
    with pytensor.config.change_flags(cxx="", compute_test_value="ignore"):
        yield


@pytest.fixture
def rng():
    """Seeded random number generator"""
    return np.random.default_rng(42)
```

### 5.2 Test Example

```python
"""Test elemwise operations"""

import numpy as np
import pytest
from tests.link.onnx.test_basic import compare_onnx_and_py


def test_add():
    """Test addition operation"""
    import pytensor.tensor as pt

    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x + y

    x_val = np.array([1, 2, 3], dtype='float32')
    y_val = np.array([4, 5, 6], dtype='float32')

    fn, result = compare_onnx_and_py([x, y], z, [x_val, y_val])

    # Verify correct ONNX node was used
    from tests.link.onnx.test_basic import get_onnx_node_types
    assert "Add" in get_onnx_node_types(fn)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_sum(axis):
    """Test sum reduction with different axes"""
    import pytensor.tensor as pt

    x = pt.matrix('x', dtype='float32')
    y = pt.sum(x, axis=axis)

    x_val = np.arange(12, dtype='float32').reshape(3, 4)

    compare_onnx_and_py([x], y, [x_val])


@pytest.mark.parametrize("opset_version", [13, 15, 18])
def test_opset_compatibility(opset_version):
    """Test operation across different ONNX opsets"""
    import pytensor.tensor as pt
    from pytensor.compile.mode import Mode
    from pytensor.link.onnx.linker import ONNXLinker

    onnx_mode = Mode(linker=ONNXLinker(opset_version=opset_version), optimizer=None)

    x = pt.vector('x')
    y = pt.exp(x)

    x_val = np.array([1, 2, 3], dtype='float32')

    compare_onnx_and_py([x], y, [x_val], onnx_mode=onnx_mode)


def test_unsupported_op():
    """Test that unsupported operations raise appropriate errors"""
    import pytensor.tensor as pt
    from pytensor.link.onnx import export_onnx

    x = pt.vector('x')
    # Assume some op is not yet implemented
    y = pt.tensor.some_unimplemented_op(x)

    with pytest.raises(NotImplementedError, match="No ONNX conversion available"):
        export_onnx([x], y, '/tmp/test.onnx')
```

### 5.3 Conftest for Shared Fixtures

**File**: `tests/link/onnx/conftest.py`

```python
"""Shared pytest fixtures for ONNX backend tests"""

import numpy as np
import pytest
import pytensor


@pytest.fixture
def rng():
    """Seeded random number generator"""
    return np.random.default_rng(42)


@pytest.fixture
def float32_data(rng):
    """Common float32 test data"""
    return rng.normal(size=(3, 4)).astype('float32')


@pytest.fixture
def matrix_pair(rng):
    """Pair of compatible matrices for operations like dot"""
    A = rng.normal(size=(3, 4)).astype('float32')
    B = rng.normal(size=(4, 5)).astype('float32')
    return A, B


@pytest.fixture(scope="module", autouse=True)
def configure_pytensor():
    """Module-level PyTensor configuration"""
    with pytensor.config.change_flags(
        cxx="",
        compute_test_value="ignore",
        floatX="float32"
    ):
        yield
```

### 5.4 Testing Timeline

**Week 1: Core Utilities**
- Days 1-2: `test_basic.py` with `compare_onnx_and_py`
- Days 3-5: Basic operation tests

**Week 2: Comprehensive Coverage**
- Operation-specific test files
- Parameterized tests
- Error case tests

**Dependencies**: Linker and dispatch system

---

## 6. Build & CI Integration

### 6.1 Dependency Management

**File**: `pyproject.toml` (additions)

```toml
[project.optional-dependencies]
onnx = [
    "onnx>=1.12.0",
    "onnxruntime>=1.13.0",
]

[tool.pytest.ini_options]
markers = [
    "onnx: marks tests requiring ONNX backend (deselect with '-m \"not onnx\"')",
]
```

### 6.2 CI Workflow Addition

**File**: `.github/workflows/test.yml` (addition to matrix)

```yaml
# Add to test matrix
- install-onnx: 1
  os: "ubuntu-latest"
  python-version: "3.11"
  fast-compile: 0
  float32: 0
  part: "tests/link/onnx"

# Add installation step
- name: Install ONNX dependencies
  if: matrix.install-onnx == 1
  run: |
    python -m pip install onnx onnxruntime
```

### 6.3 Pre-commit Hooks

**File**: `.pre-commit-config.yaml` (if not exists)

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        files: ^pytensor/link/onnx/

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        files: ^pytensor/link/onnx/
        args: ['--max-line-length=100']
```

### 6.4 Build Timeline

**Days 1-2: Dependencies**
- Update `pyproject.toml`
- Test dependency installation

**Day 3: CI Integration**
- Add CI matrix entry
- Test CI pipeline

**Dependencies**: None

---

## 7. Documentation

### 7.1 API Documentation

**File**: `docs/library/onnx.rst` (new)

```rst
.. _onnx_backend:

ONNX Backend
============

PyTensor provides an ONNX backend that exports computation graphs to ONNX format for deployment.

Quick Start
-----------

Export a simple computation:

.. code-block:: python

    import pytensor.tensor as pt
    from pytensor.link.onnx import export_onnx

    x = pt.vector('x')
    y = pt.sum(x ** 2)

    export_onnx([x], y, 'model.onnx')

Supported Operations
--------------------

The ONNX backend currently supports:

**Tier 1 (Core Operations)**:
- Element-wise arithmetic: Add, Sub, Mul, Div, Neg, Abs
- Element-wise math: Exp, Log, Sqrt, Pow, Floor, Ceil, Round
- Min/Max operations

**Tier 2 (Shape Operations)**:
- Shape inspection: Shape, Reshape
- Dimension manipulation: Transpose, Squeeze, Unsqueeze
- Joining/splitting: Concatenate, Stack, Split
- Basic indexing: Slice

**Tier 3 (Reductions)**:
- Reductions: Sum, Prod, Max, Min, Mean
- Index operations: Argmax, Argmin
- Tensor creation: Zeros, Ones, Alloc, ARange

See the complete list in the :ref:`operations_roadmap`.

API Reference
-------------

.. autofunction:: pytensor.link.onnx.export_onnx
.. autofunction:: pytensor.link.onnx.compile_onnx
.. autofunction:: pytensor.link.onnx.export_function_onnx

.. autoclass:: pytensor.link.onnx.ONNXLinker
   :members:

Limitations
-----------

- No in-place operations (ONNX is immutable)
- Dynamic shapes require ONNX opset 11+
- Some linear algebra operations not in standard ONNX
- Control flow (Scan) has limitations

Examples
--------

Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytensor.tensor as pt
    from pytensor.link.onnx import export_onnx

    x = pt.matrix('x')
    y = pt.matrix('y')
    z = pt.dot(x, y)

    export_onnx([x, y], z, 'matmul.onnx')

Neural Network Layer
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytensor.tensor as pt
    from pytensor.link.onnx import export_onnx

    # Input
    x = pt.matrix('x')  # (batch, features)

    # Parameters
    W = pt.matrix('W')  # (features, hidden)
    b = pt.vector('b')  # (hidden,)

    # Linear + ReLU
    z = pt.dot(x, W) + b
    y = pt.maximum(z, 0)  # ReLU

    export_onnx([x, W, b], y, 'linear_relu.onnx')

Deployment
----------

Use ONNX Runtime for deployment:

.. code-block:: python

    import onnxruntime as ort
    import numpy as np

    # Load model
    session = ort.InferenceSession('model.onnx')

    # Run inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})
```

### 7.2 User Guide

**File**: `docs/tutorial/onnx_export.rst` (new)

```rst
Exporting Models to ONNX
=========================

This tutorial covers exporting PyTensor models to ONNX format.

Why Export to ONNX?
--------------------

ONNX (Open Neural Network Exchange) provides:

- **Cross-platform deployment**: Run on CPUs, GPUs, mobile, web
- **Optimized runtimes**: ONNX Runtime, TensorRT, OpenVINO
- **Hardware acceleration**: Specialized hardware support
- **Language interop**: Use models in C++, Java, JavaScript, etc.

Basic Export
------------

The simplest way to export:

.. code-block:: python

    import pytensor.tensor as pt
    from pytensor.link.onnx import export_onnx

    # Define computation
    x = pt.vector('x')
    y = (x - pt.mean(x)) / pt.std(x)  # Normalize

    # Export
    export_onnx([x], y, 'normalize.onnx')

Exporting Functions
-------------------

Export already-compiled PyTensor functions:

.. code-block:: python

    import pytensor
    import pytensor.tensor as pt
    from pytensor.link.onnx import export_function_onnx

    x = pt.matrix('x')
    y = pt.nnet.softmax(x)

    fn = pytensor.function([x], y)
    export_function_onnx(fn, 'softmax.onnx')

Multiple Outputs
----------------

Export graphs with multiple outputs:

.. code-block:: python

    x = pt.matrix('x')

    # Compute statistics
    mean = pt.mean(x, axis=0)
    std = pt.std(x, axis=0)
    minimum = pt.min(x, axis=0)
    maximum = pt.max(x, axis=0)

    export_onnx(
        [x],
        [mean, std, minimum, maximum],
        'statistics.onnx'
    )

Using Exported Models
---------------------

Load and run with ONNX Runtime:

.. code-block:: python

    import onnxruntime as ort
    import numpy as np

    # Load model
    session = ort.InferenceSession('model.onnx')

    # Inspect inputs/outputs
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} {inp.type}")

    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} {out.type}")

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run(
        [output_name],
        {input_name: input_data}
    )[0]

Troubleshooting
---------------

**NotImplementedError: No ONNX conversion for Op**

This operation is not yet supported. Check the supported operations list.

**ONNX validation error**

The generated ONNX model may be invalid. Common causes:

- Incompatible types (e.g., bool where float expected)
- Dynamic shapes not supported by operation
- Opset version too old

Try updating opset version:

.. code-block:: python

    export_onnx([x], y, 'model.onnx', opset_version=18)

**Runtime shape mismatch**

ONNX requires shape compatibility. Ensure input shapes match model expectations.
```

### 7.3 Documentation Timeline

**Week 1: API Documentation**
- Docstrings for all public functions
- API reference generation

**Week 2: User Guide**
- Tutorial with examples
- Troubleshooting section

**Dependencies**: Export API complete

---

## Implementation Checklist

### Foundation (Week 1)

#### Module Structure (Day 1)
- [ ] Create `pytensor/link/onnx/` directory
- [ ] Create `pytensor/link/onnx/dispatch/` directory
- [ ] Create `tests/link/onnx/` directory
- [ ] Add `__init__.py` files
- [ ] Update `pyproject.toml` with ONNX dependencies

#### Dispatch System (Days 2-5)
- [ ] Implement `onnx_funcify` singledispatch in `dispatch/basic.py`
- [ ] Implement `onnx_typify` singledispatch
- [ ] Implement `make_value_info` helper
- [ ] Add type mapping utilities
- [ ] Create `dispatch/__init__.py` with imports
- [ ] Write basic dispatch tests

### Core Infrastructure (Weeks 2-3)

#### Linker Implementation (Week 2)
- [ ] Create `ONNXLinker` class in `linker.py`
- [ ] Implement `fgraph_convert` method
- [ ] Implement FunctionGraph → ONNX conversion
- [ ] Add variable name management
- [ ] Add constant/initializer handling
- [ ] Implement ONNX Runtime wrapper
- [ ] Add model validation
- [ ] Write linker tests

#### Export API (Week 3, Days 1-3)
- [ ] Implement `export_onnx` function
- [ ] Implement `export_function_onnx` function
- [ ] Implement `compile_onnx` function
- [ ] Add ONNX Mode to `mode.py`
- [ ] Update `pytensor/link/onnx/__init__.py` with exports
- [ ] Write export API tests

#### Testing Infrastructure (Week 3, Days 4-5)
- [ ] Create `test_basic.py` with `compare_onnx_and_py`
- [ ] Add ONNX node inspection utilities
- [ ] Create `conftest.py` with fixtures
- [ ] Write integration tests
- [ ] Add parameterized test examples

### Polish & Integration (Weeks 4-6)

#### CI/CD (Week 4, Days 1-2)
- [ ] Update `.github/workflows/test.yml`
- [ ] Add ONNX test matrix entry
- [ ] Test CI pipeline
- [ ] Add pre-commit hooks

#### Documentation (Week 4-5)
- [ ] Write API documentation
- [ ] Write user guide with examples
- [ ] Add troubleshooting section
- [ ] Generate API reference docs
- [ ] Review and polish

#### Performance & Validation (Week 5-6)
- [ ] Add benchmarking utilities
- [ ] Compare ONNX Runtime vs Python performance
- [ ] Optimize hot paths
- [ ] Add comprehensive error messages
- [ ] Final code review

---

## Code References

### PyTensor Backend Architecture
- `pytensor/link/basic.py:144-717` - Linker base classes (Linker, JITLinker, PerformLinker)
- `pytensor/compile/mode.py:42-597` - Mode system and backend registration
- `pytensor/compile/function/__init__.py:95-348` - Function compilation API
- `pytensor/graph/fg.py:50-900` - FunctionGraph class
- `pytensor/graph/traversal.py` - Graph traversal utilities

### JAX Backend Reference
- `pytensor/link/jax/linker.py:9-127` - JAXLinker implementation
- `pytensor/link/jax/dispatch/basic.py:27-151` - JAX dispatch system
- `pytensor/link/jax/dispatch/elemwise.py:9-116` - Elemwise operation example
- `pytensor/link/jax/dispatch/__init__.py:1-24` - Dispatch module loading

### Other Backend Examples
- `pytensor/link/numba/linker.py:4-20` - NumbaLinker (simpler example)
- `pytensor/link/pytorch/linker.py:5-94` - PytorchLinker with compile control
- `pytensor/link/mlx/linker.py:4-70` - MLXLinker

### Testing Patterns
- `tests/link/jax/test_basic.py:36-96` - compare_jax_and_py utility
- `tests/link/jax/conftest.py` - Test fixtures
- `tests/link/jax/test_elemwise.py` - Parameterized tests
- `tests/link/jax/test_nlinalg.py` - Complex operation tests

### Graph Utilities
- `pytensor/link/utils.py:666-809` - fgraph_to_python utility
- `pytensor/link/utils.py:40-141` - Storage management
- `pytensor/graph/rewriting/basic.py` - Graph rewriting framework
- `pytensor/tensor/rewriting/` - Tensor-specific optimizations

---

## Related Research

**From thoughts/ directory**:
- `thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md` - Operations roadmap (companion document)
- `thoughts/shared/plans/onnx-backend-implementation.md` - Original demo-focused plan
- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - Backend architecture overview

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Foundation** | Week 1 | Module structure, dispatch system, basic tests |
| **Core Infrastructure** | Weeks 2-3 | Linker, export API, testing framework |
| **Polish & Integration** | Weeks 4-6 | CI/CD, documentation, performance optimization |
| **TOTAL** | **4-6 weeks** | Production-ready ONNX backend infrastructure |

**Critical Path**: Module Structure → Dispatch System → Linker → Export API

**Parallel Work Possible**:
- Documentation can be written alongside implementation
- Testing infrastructure can be built with linker
- CI/CD setup can happen early

---

## Success Criteria

### Foundation Complete
- ✅ Module structure created
- ✅ Basic dispatch system working
- ✅ Can register operation converters
- ✅ Basic tests pass

### Core Infrastructure Complete
- ✅ Linker converts FunctionGraph to ONNX ModelProto
- ✅ Export API generates valid `.onnx` files
- ✅ ONNX Runtime can execute exported models
- ✅ Tests compare ONNX vs Python outputs
- ✅ Type system fully integrated

### Production Ready
- ✅ CI/CD runs ONNX tests automatically
- ✅ Documentation covers all public APIs
- ✅ Error messages are clear and actionable
- ✅ Performance is comparable to Python reference
- ✅ Can export real PyTensor code

---

## Recommendations

### Start Here
1. **Day 1**: Create module structure and directories
2. **Days 2-5**: Build dispatch system with Tier 1 operations
3. **Week 2**: Implement linker with FunctionGraph conversion
4. **Week 3**: Add export API and testing utilities

### Parallel Tracks
- **Developer 1**: Linker + Export API
- **Developer 2**: Dispatch system + Operations
- **Developer 3**: Testing + Documentation

### Risks & Mitigation
1. **ONNX Runtime compatibility**: Test with multiple ONNX Runtime versions
2. **Type system complexity**: Reference JAX backend patterns closely
3. **Dynamic shapes**: Document limitations clearly, provide good errors
4. **Linear algebra gaps**: Use contrib ops or document as unsupported

---

## Conclusion

Building a production ONNX backend requires comprehensive infrastructure beyond just operation implementations. The 7 components in this roadmap (linker, dispatch, export API, module structure, testing, CI/CD, documentation) are the foundation that makes operation implementations useful.

**Timeline**: 4-6 weeks for complete infrastructure, can be built in parallel with operations from the operations roadmap.

**Next Steps**:
1. Review this roadmap with team
2. Start with module structure and dispatch system
3. Build linker and export API
4. Implement operations in tiers (see operations roadmap)
5. Iterate on testing and documentation

**Success depends on**:
- Following established PyTensor patterns (JAX backend as reference)
- Building incrementally (foundation → core → polish)
- Testing thoroughly at each stage
- Documenting as you build
