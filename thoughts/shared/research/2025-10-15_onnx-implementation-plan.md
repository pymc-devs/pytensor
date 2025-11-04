---
date: 2025-10-15T00:00:00Z
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pymc-devs/pytensor
topic: "ONNX Backend Implementation Plan - Concrete Steps"
tags: [implementation, plan, onnx, webassembly, backend, roadmap]
status: ready_to_implement
last_updated: 2025-10-15
last_updated_by: Claude
---

# ONNX Backend Implementation Plan

**Date**: 2025-10-15
**Status**: Ready to implement
**Target**: Basic ONNX export with WebAssembly demo

## Executive Summary

This document outlines the concrete implementation plan for adding ONNX export functionality to PyTensor, targeting **ONNX opset 18** with a focus on **basic operations first**. The goal is to enable exporting trained PyTensor models to run inference in the browser via WebAssembly.

**Key Decisions**:
- ✅ Target ONNX opset 18 (mature, good WASM support)
- ✅ Start with basic ops only (minimal viable backend)
- ✅ Integrate into PyTensor core (`pytensor/link/onnx/`)
- ✅ Convert shared variables to ONNX initializers (baked weights)
- ✅ Demo: Small neural network trained in PyTensor, inference in browser
- ✅ All training happens in PyTensor (browser only runs inference)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTensor Training                         │
│  1. Define model: x → Dense(128) → ReLU → Dense(10) → Softmax│
│  2. Train with gradient descent                              │
│  3. Compile inference function                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ export_onnx(f, "model.onnx")
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                   ONNX Export                                │
│  ONNXLinker.fgraph_convert(fgraph) → ONNX protobuf          │
│  - Convert ops to ONNX nodes                                 │
│  - Bake weights as initializers                              │
│  - Validate with onnx.checker                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ model.onnx file
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              Browser (WebAssembly)                           │
│  1. Load model with ONNX Runtime Web                         │
│  2. User provides input (e.g., image, vector)                │
│  3. Run inference: session.run(feeds)                        │
│  4. Display results                                          │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Minimal ONNX Export (Core Infrastructure)

**Goal**: Export simple PyTensor functions to valid ONNX files

### 1.1 File Structure

Create the following files in PyTensor core:

```
pytensor/link/onnx/
├── __init__.py                  # Public API: export_onnx()
├── linker.py                    # ONNXLinker class
└── dispatch/
    ├── __init__.py              # Re-exports
    └── basic.py                 # @singledispatch onnx_funcify, base conversions
```

### 1.2 Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
onnx = [
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",  # For validation/testing
]
```

### 1.3 Core Infrastructure Files

#### File: `pytensor/link/onnx/__init__.py`

```python
"""ONNX export functionality for PyTensor.

This module provides functionality to export PyTensor functions to ONNX format
for deployment in environments like WebAssembly, mobile, or edge devices.

Example:
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
from pytensor.link.onnx.linker import ONNXLinker

__all__ = ["export_onnx", "ONNXLinker"]
```

#### File: `pytensor/link/onnx/linker.py`

**Purpose**: Main linker class (not used for direct compilation, only for export)

```python
"""ONNX Linker for PyTensor.

Note: Unlike JAX/Numba/PyTorch linkers, ONNXLinker is not used for execution.
Instead, it's used exclusively for export to ONNX format.
"""

from pytensor.link.basic import JITLinker
from pytensor.link.onnx.dispatch.basic import onnx_funcify


class ONNXLinker(JITLinker):
    """Linker that converts PyTensor graphs to ONNX format.

    This linker is used for export only, not for execution.
    Use export_onnx() for the primary interface.
    """

    def fgraph_convert(self, fgraph, **kwargs):
        """Convert FunctionGraph to ONNX ModelProto.

        Parameters
        ----------
        fgraph : FunctionGraph
            The graph to convert
        **kwargs
            Additional arguments passed to onnx_funcify

        Returns
        -------
        onnx.ModelProto
            ONNX model representation
        """
        return onnx_funcify(fgraph, **kwargs)

    def jit_compile(self, fn, **kwargs):
        """Not implemented - ONNX export doesn't use JIT compilation.

        The exported ONNX model is compiled by ONNX Runtime at load time.
        """
        return fn

    def create_thunk_inputs(self, storage_map):
        """Not implemented - ONNX export doesn't create thunks."""
        raise NotImplementedError(
            "ONNXLinker is for export only. "
            "Use export_onnx() to export to ONNX format."
        )
```

#### File: `pytensor/link/onnx/dispatch/basic.py`

**Purpose**: Core dispatch system and FunctionGraph conversion

```python
"""Basic ONNX dispatch system.

This module provides the singledispatch-based conversion system for
converting PyTensor ops to ONNX nodes.
"""

from functools import singledispatch
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install onnx"
    )

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.type import Type


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
        Additional conversion parameters

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
        f"Node: {node}\n"
        f"This op is not yet supported for ONNX export. "
        f"Supported ops: Add, Mul, Sub, Div, Neg, Exp, Log, Sqrt, Dot, etc."
    )


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    opset_version: int = ONNX_OPSET_VERSION,
    model_name: str = "pytensor_model",
    **kwargs
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
    **kwargs
        Additional parameters

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    # Track converted nodes and value_info
    onnx_nodes: List[onnx.NodeProto] = []
    value_info: Dict[str, onnx.ValueInfoProto] = {}
    initializers: List[onnx.TensorProto] = []

    # Generate unique names for variables
    var_names: Dict[Variable, str] = {}
    name_counter = 0

    def get_var_name(var: Variable) -> str:
        """Get or create unique name for a variable."""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, 'name') and var.name:
                var_names[var] = var.name
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
                        np.asarray(inp.data),
                        name=name
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
            **kwargs
        )

        if onnx_node is not None:
            onnx_nodes.append(onnx_node)

    # Create inputs (only non-constant inputs)
    input_protos = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(
                make_value_info(inp, name)
            )

    # Create outputs
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(
            make_value_info(out, name)
        )

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
        graph,
        producer_name="PyTensor",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}")

    return model


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
        'float32': TensorProto.FLOAT,
        'float64': TensorProto.DOUBLE,
        'int32': TensorProto.INT32,
        'int64': TensorProto.INT64,
        'uint8': TensorProto.UINT8,
        'int8': TensorProto.INT8,
        'bool': TensorProto.BOOL,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (use symbolic dimensions if needed)
    if hasattr(var.type, 'shape'):
        shape = []
        for i, dim in enumerate(var.type.shape):
            if dim is None or dim < 0:
                # Dynamic dimension
                shape.append(f"dim_{i}")
            else:
                shape.append(dim)
    else:
        shape = None

    # Create tensor type
    tensor_type = helper.make_tensor_type_proto(
        elem_type=onnx_dtype,
        shape=shape
    )

    return helper.make_value_info(name, tensor_type)


@singledispatch
def onnx_typify(data, **kwargs):
    """Convert Python/NumPy data to ONNX tensor type.

    This is used for type inference during conversion.
    """
    # Default: return as-is
    return data
```

#### File: `pytensor/link/onnx/export.py`

**Purpose**: Main export function (public API)

```python
"""ONNX export API."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import onnx
except ImportError:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install onnx"
    )

from pytensor.compile.function import Function
from pytensor.link.onnx.dispatch.basic import onnx_funcify


def export_onnx(
    pytensor_function: Function,
    output_path: Union[str, Path],
    *,
    opset_version: int = 18,
    example_inputs: Optional[list] = None,
    model_name: str = "pytensor_model",
    **kwargs
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
    example_inputs : list, optional
        Example inputs for shape inference
        If provided, will be used to infer concrete shapes
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

    # If example inputs provided, we could do shape inference here
    # For now, we'll rely on the type information in the graph
    if example_inputs is not None:
        # TODO: Implement shape inference from example inputs
        pass

    # Convert to ONNX
    model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name=model_name,
        **kwargs
    )

    # Save to file
    output_path = Path(output_path)
    onnx.save(model, str(output_path))

    print(f"✓ Exported PyTensor function to ONNX: {output_path}")
    print(f"  Opset version: {opset_version}")
    print(f"  Inputs: {len(fgraph.inputs)}")
    print(f"  Outputs: {len(fgraph.outputs)}")
    print(f"  Nodes: {len(model.graph.node)}")

    return model
```

## Phase 2: Basic Op Conversions

**Goal**: Support fundamental operations for simple neural networks

### 2.1 Elemwise Operations

#### File: `pytensor/link/onnx/dispatch/elemwise.py`

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar import basic as scalar

try:
    from onnx import helper
except ImportError:
    raise ImportError("ONNX package required for export")


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
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}"
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
```

### 2.2 Matrix Operations

#### File: `pytensor/link/onnx/dispatch/nlinalg.py`

```python
"""ONNX conversion for linear algebra operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.blas import Dot
from pytensor.tensor.math import Dot as TensorDot, MatMul

try:
    from onnx import helper
except ImportError:
    raise ImportError("ONNX package required for export")


@onnx_funcify.register(Dot)
@onnx_funcify.register(MatMul)
def onnx_funcify_Dot(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot/MatMul to ONNX MatMul node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}"
    )

    return onnx_node
```

### 2.3 Activation Functions

#### File: `pytensor/link/onnx/dispatch/special.py`

```python
"""ONNX conversion for special/activation functions."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar.basic import Sigmoid
from pytensor.tensor.nnet import Softmax

try:
    from onnx import helper
except ImportError:
    raise ImportError("ONNX package required for export")


@onnx_funcify.register(Softmax)
def onnx_funcify_Softmax(op, node, var_names, get_var_name, **kwargs):
    """Convert Softmax to ONNX Softmax node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Get axis attribute
    axis = getattr(op, 'axis', -1)

    onnx_node = helper.make_node(
        "Softmax",
        inputs=input_names,
        outputs=output_names,
        axis=axis,
        name=f"Softmax_{output_names[0]}"
    )

    return onnx_node


# ReLU is typically an Elemwise(Maximum(x, 0))
# We'll handle it via pattern matching or a specific dispatch
```

## Phase 3: WebAssembly Demo

**Goal**: Complete end-to-end demo with trained model running in browser

### 3.1 Training Script (Python)

#### File: `examples/onnx_demo/train_model.py`

```python
"""Train a simple neural network and export to ONNX.

This demonstrates the complete workflow:
1. Define model in PyTensor
2. Train on sample data
3. Export to ONNX for browser inference
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx


def create_model():
    """Create a simple 2-layer neural network."""
    # Input
    x = pt.matrix('x', dtype='float32')  # Shape: (batch, 784)

    # Layer 1: Dense(128) + ReLU
    W1 = pt.shared(
        np.random.randn(784, 128).astype('float32') * 0.01,
        name='W1'
    )
    b1 = pt.shared(np.zeros(128, dtype='float32'), name='b1')
    h1 = pt.dot(x, W1) + b1
    h1_relu = pt.maximum(h1, 0)  # ReLU activation

    # Layer 2: Dense(10) + Softmax
    W2 = pt.shared(
        np.random.randn(128, 10).astype('float32') * 0.01,
        name='W2'
    )
    b2 = pt.shared(np.zeros(10, dtype='float32'), name='b2')
    y_logits = pt.dot(h1_relu, W2) + b2
    y_pred = pt.nnet.softmax(y_logits)

    return x, y_pred, [W1, b1, W2, b2]


def train_model():
    """Train the model (simplified for demo)."""
    print("Creating model...")
    x, y_pred, params = create_model()

    # For demo purposes, we'll just use random initialization
    # In practice, you'd train with actual data
    print("Model created (using random initialization for demo)")

    # Compile inference function
    print("Compiling inference function...")
    inference_fn = pytensor.function([x], y_pred)

    return inference_fn


def main():
    """Main training and export pipeline."""
    # Train model
    inference_fn = train_model()

    # Test inference
    print("\nTesting inference with random input...")
    test_input = np.random.randn(1, 784).astype('float32')
    test_output = inference_fn(test_input)
    print(f"Output shape: {test_output.shape}")
    print(f"Output (first 5): {test_output[0, :5]}")
    print(f"Sum of probabilities: {test_output.sum():.4f}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    export_onnx(
        inference_fn,
        "model.onnx",
        model_name="simple_nn",
        example_inputs=[test_input]
    )

    print("\n✓ Complete! Model exported to model.onnx")
    print("  Load it in the browser with ONNX Runtime Web")


if __name__ == "__main__":
    main()
```

### 3.2 Browser Demo

#### File: `examples/onnx_demo/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyTensor ONNX WebAssembly Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 5px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background: #e3f2fd;
        }
        .result {
            background: #f1f8e9;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .error {
            background: #ffebee;
            border-left-color: #f44336;
        }
        pre {
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PyTensor ONNX Demo</h1>
        <p>
            This demo shows a neural network trained in PyTensor,
            exported to ONNX, and running inference in your browser via WebAssembly.
        </p>

        <div class="status" id="status">
            Ready to load model...
        </div>

        <div>
            <button onclick="loadModel()" id="loadBtn">Load Model</button>
            <button onclick="runInference()" id="runBtn" disabled>Run Inference</button>
            <button onclick="runBenchmark()" id="benchBtn" disabled>Benchmark (100 runs)</button>
        </div>

        <div id="results"></div>

        <h2>About This Demo</h2>
        <ul>
            <li>Model: 2-layer neural network (784 → 128 → 10)</li>
            <li>Trained in: PyTensor (Python)</li>
            <li>Exported to: ONNX format</li>
            <li>Running on: ONNX Runtime WebAssembly</li>
            <li>Input: Random 784-dimensional vector</li>
            <li>Output: 10-class probability distribution</li>
        </ul>
    </div>

    <script>
        let session = null;

        function setStatus(message, isError = false) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = message;
            statusEl.className = isError ? 'status error' : 'status';
        }

        function addResult(content, isError = false) {
            const resultsEl = document.getElementById('results');
            const resultDiv = document.createElement('div');
            resultDiv.className = isError ? 'result error' : 'result';
            resultDiv.innerHTML = content;
            resultsEl.appendChild(resultDiv);
        }

        async function loadModel() {
            try {
                setStatus('Loading model...');
                document.getElementById('loadBtn').disabled = true;

                // Load ONNX model
                session = await ort.InferenceSession.create('model.onnx', {
                    executionProviders: ['wasm']
                });

                setStatus('✓ Model loaded successfully!');
                document.getElementById('runBtn').disabled = false;
                document.getElementById('benchBtn').disabled = false;

                addResult(`
                    <strong>Model Loaded</strong><br>
                    Inputs: ${session.inputNames.join(', ')}<br>
                    Outputs: ${session.outputNames.join(', ')}
                `);
            } catch (error) {
                setStatus('✗ Error loading model', true);
                addResult(`<strong>Error:</strong> ${error.message}`, true);
                document.getElementById('loadBtn').disabled = false;
            }
        }

        async function runInference() {
            try {
                setStatus('Running inference...');

                const startTime = performance.now();

                // Create random input (1 x 784)
                const inputData = new Float32Array(784);
                for (let i = 0; i < 784; i++) {
                    inputData[i] = Math.random() * 2 - 1; // Random [-1, 1]
                }

                // Create input tensor
                const feeds = {
                    [session.inputNames[0]]: new ort.Tensor('float32', inputData, [1, 784])
                };

                // Run inference
                const results = await session.run(feeds);

                const endTime = performance.now();
                const inferenceTime = (endTime - startTime).toFixed(2);

                // Get output
                const output = results[session.outputNames[0]];
                const probabilities = output.data;

                // Find top prediction
                let maxIdx = 0;
                let maxProb = probabilities[0];
                for (let i = 1; i < probabilities.length; i++) {
                    if (probabilities[i] > maxProb) {
                        maxProb = probabilities[i];
                        maxIdx = i;
                    }
                }

                setStatus(`✓ Inference complete in ${inferenceTime}ms`);

                // Format probabilities
                let probsHtml = '<pre>';
                for (let i = 0; i < probabilities.length; i++) {
                    const bar = '█'.repeat(Math.round(probabilities[i] * 50));
                    const highlight = i === maxIdx ? ' <-- Prediction' : '';
                    probsHtml += `Class ${i}: ${probabilities[i].toFixed(4)} ${bar}${highlight}\n`;
                }
                probsHtml += '</pre>';

                addResult(`
                    <strong>Inference Result</strong><br>
                    Time: ${inferenceTime}ms<br>
                    Predicted class: ${maxIdx} (confidence: ${(maxProb * 100).toFixed(1)}%)<br>
                    ${probsHtml}
                `);
            } catch (error) {
                setStatus('✗ Error during inference', true);
                addResult(`<strong>Error:</strong> ${error.message}`, true);
            }
        }

        async function runBenchmark() {
            try {
                setStatus('Running benchmark (100 inferences)...');
                document.getElementById('benchBtn').disabled = true;

                const inputData = new Float32Array(784);
                for (let i = 0; i < 784; i++) {
                    inputData[i] = Math.random() * 2 - 1;
                }

                const feeds = {
                    [session.inputNames[0]]: new ort.Tensor('float32', inputData, [1, 784])
                };

                // Warm-up
                await session.run(feeds);

                // Benchmark
                const numRuns = 100;
                const times = [];

                for (let i = 0; i < numRuns; i++) {
                    const start = performance.now();
                    await session.run(feeds);
                    const end = performance.now();
                    times.push(end - start);
                }

                // Calculate statistics
                const avgTime = times.reduce((a, b) => a + b) / times.length;
                const minTime = Math.min(...times);
                const maxTime = Math.max(...times);
                const medianTime = times.sort((a, b) => a - b)[Math.floor(times.length / 2)];

                setStatus('✓ Benchmark complete');
                document.getElementById('benchBtn').disabled = false;

                addResult(`
                    <strong>Benchmark Results (${numRuns} runs)</strong><br>
                    Average: ${avgTime.toFixed(2)}ms<br>
                    Median: ${medianTime.toFixed(2)}ms<br>
                    Min: ${minTime.toFixed(2)}ms<br>
                    Max: ${maxTime.toFixed(2)}ms<br>
                    Throughput: ${(1000 / avgTime).toFixed(1)} inferences/second
                `);
            } catch (error) {
                setStatus('✗ Error during benchmark', true);
                addResult(`<strong>Error:</strong> ${error.message}`, true);
                document.getElementById('benchBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
```

#### File: `examples/onnx_demo/README.md`

```markdown
# PyTensor ONNX WebAssembly Demo

This example demonstrates exporting a PyTensor model to ONNX and running it in the browser.

## Setup

1. Install dependencies:
```bash
pip install pytensor[onnx]
```

2. Train and export the model:
```bash
python train_model.py
```

This will create `model.onnx` in the current directory.

3. Serve the demo:
```bash
python -m http.server 8000
```

4. Open your browser to:
```
http://localhost:8000
```

## What's Happening

1. **Training (Python)**:
   - A 2-layer neural network is defined in PyTensor
   - Model parameters are initialized (random for demo)
   - The inference function is compiled
   - The model is exported to ONNX format

2. **Inference (Browser)**:
   - ONNX Runtime Web loads the .onnx file
   - JavaScript generates random input data
   - The model runs entirely in WebAssembly
   - Results are displayed in the browser

## Architecture

```
PyTensor Model
     ↓
[Export to ONNX]
     ↓
model.onnx
     ↓
[Load in Browser]
     ↓
ONNX Runtime WASM
     ↓
Inference Results
```

## Performance

Expected inference times:
- First run: 5-20ms (initialization)
- Subsequent runs: 1-5ms
- Throughput: ~200-1000 inferences/second

This is 3-10x slower than native CPU but still very fast for real-time applications.
```

## Testing Strategy

### Unit Tests

#### File: `tests/link/onnx/test_basic.py`

```python
"""Basic tests for ONNX export functionality."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx
import onnxruntime as ort

import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx


def test_export_simple_add():
    """Test exporting a simple addition."""
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = x + y

    f = pytensor.function([x, y], z)

    # Export
    model = export_onnx(f, "/tmp/test_add.onnx")

    # Validate
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model)

    # Test with ONNX Runtime
    session = ort.InferenceSession("/tmp/test_add.onnx")

    x_val = np.array([1, 2, 3], dtype='float32')
    y_val = np.array([4, 5, 6], dtype='float32')

    result = session.run(None, {'x': x_val, 'y': y_val})
    expected = x_val + y_val

    np.testing.assert_allclose(result[0], expected)


def test_export_multiple_ops():
    """Test exporting with multiple operations."""
    x = pt.vector('x', dtype='float32')
    y = pt.vector('y', dtype='float32')
    z = (x + y) * 2 - y

    f = pytensor.function([x, y], z)

    # Export
    model = export_onnx(f, "/tmp/test_multi_op.onnx")
    onnx.checker.check_model(model)

    # Test
    session = ort.InferenceSession("/tmp/test_multi_op.onnx")

    x_val = np.array([1, 2, 3], dtype='float32')
    y_val = np.array([4, 5, 6], dtype='float32')

    result = session.run(None, {'x': x_val, 'y': y_val})
    expected = (x_val + y_val) * 2 - y_val

    np.testing.assert_allclose(result[0], expected)


def test_export_matmul():
    """Test exporting matrix multiplication."""
    x = pt.matrix('x', dtype='float32')
    y = pt.matrix('y', dtype='float32')
    z = pt.dot(x, y)

    f = pytensor.function([x, y], z)

    # Export
    model = export_onnx(f, "/tmp/test_matmul.onnx")
    onnx.checker.check_model(model)

    # Test
    session = ort.InferenceSession("/tmp/test_matmul.onnx")

    x_val = np.random.randn(3, 4).astype('float32')
    y_val = np.random.randn(4, 5).astype('float32')

    result = session.run(None, {'x': x_val, 'y': y_val})
    expected = np.dot(x_val, y_val)

    np.testing.assert_allclose(result[0], expected, rtol=1e-5)
```

## Implementation Checklist

### Phase 1: Core Infrastructure ✓
- [ ] Create `pytensor/link/onnx/` directory structure
- [ ] Implement `ONNXLinker` class
- [ ] Implement `onnx_funcify` dispatcher
- [ ] Implement `export_onnx()` function
- [ ] Add ONNX to optional dependencies
- [ ] Write documentation

### Phase 2: Basic Ops ✓
- [ ] Elemwise operations (Add, Mul, Sub, Div, Neg)
- [ ] Basic math (Exp, Log, Sqrt, Pow, Abs)
- [ ] Matrix operations (Dot, MatMul)
- [ ] Activations (ReLU via Maximum, Sigmoid, Tanh, Softmax)
- [ ] Handle constants as initializers

### Phase 3: Demo ✓
- [ ] Create training script
- [ ] Create HTML demo page
- [ ] Add README with instructions
- [ ] Test in multiple browsers (Chrome, Firefox, Safari)

### Phase 4: Testing ✓
- [ ] Unit tests for basic ops
- [ ] Integration tests with ONNX Runtime
- [ ] Test shape inference
- [ ] Test error messages

### Phase 5: Documentation ✓
- [ ] API documentation
- [ ] Tutorial notebook
- [ ] Add to PyTensor docs
- [ ] List supported/unsupported ops

## Timeline Estimate

- **Phase 1** (Core Infrastructure): 2-3 days
- **Phase 2** (Basic Ops): 2-3 days
- **Phase 3** (Demo): 1-2 days
- **Phase 4** (Testing): 1-2 days
- **Phase 5** (Documentation): 1 day

**Total**: ~7-11 days for minimal viable implementation

## Future Enhancements

After the basic implementation:
1. Add more ops (Conv2D, Pooling, BatchNorm)
2. Implement shape inference from example inputs
3. Add graph optimizations (operator fusion)
4. Support for Scan → ONNX Loop conversion
5. Custom operators for unsupported ops
6. Quantization support

## Success Criteria

✅ A trained PyTensor model can be exported to ONNX
✅ The exported model runs in ONNX Runtime (Python)
✅ The exported model runs in the browser (WASM)
✅ Basic ops work correctly (validated against PyTensor)
✅ Clear error messages for unsupported ops
✅ Documentation and examples provided

---

**Ready to implement!** 🚀
