---
date: 2025-10-15T00:00:00Z
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pymc-devs/pytensor
topic: "Creating an ONNX backend for PyTensor to run in WebAssembly with browser demo"
tags: [research, codebase, onnx, webassembly, backend, linker, dispatch, graph-export]
status: complete
last_updated: 2025-10-15
last_updated_by: Claude
---

# Research: Creating an ONNX Backend for PyTensor to Run in WebAssembly

**Date**: 2025-10-15
**Researcher**: Claude
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: main
**Repository**: pymc-devs/pytensor

## Research Question

How can I create an ONNX backend for PyTensor and run it in WebAssembly, with the goal of running a sample graph in the browser with a demo app?

## Summary

PyTensor **does not currently have any ONNX export or backend functionality**, but it has a well-documented, modular backend architecture that would make adding ONNX support straightforward. The codebase contains:

1. **Multiple reference implementations** (JAX, PyTorch, Numba, MLX) showing the linker pattern
2. **A dispatch-based Op conversion system** using Python's `@singledispatch`
3. **Comprehensive graph representation** (FunctionGraph with Apply nodes)
4. **An existing design document** outlining ONNX backend architecture
5. **Example graphs and tutorials** showing how to create and execute computational graphs

To create an ONNX backend, you would:
- Create an `ONNXLinker` class that converts PyTensor's FunctionGraph to ONNX format
- Implement `onnx_funcify` dispatch to convert individual ops to ONNX nodes
- Export the ONNX model to a `.onnx` file
- Use ONNX Runtime with WebAssembly to execute in the browser
- Create a JavaScript/HTML demo app that loads and runs the model

## Detailed Findings

### 1. Current ONNX Status

**No ONNX implementation exists**. Comprehensive search found:
- ❌ No ONNX linker or dispatch system
- ❌ No ONNX export functionality (no `.onnx` file generation)
- ❌ No ONNX protobuf serialization
- ❌ No ONNX Runtime integration
- ❌ No ONNX-specific graph rewrites/optimizations

**However**, there exists a detailed planning document:
- [`thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md`](thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md)
- Contains complete architectural design for ONNX backend
- Discusses ONNX-specific challenges (static graphs, control flow, autodiff limitations)
- Proposes file structure and implementation strategy

### 2. Backend Architecture Pattern

#### Linker-Based Architecture

PyTensor uses a **Linker** abstraction where each backend implements a linker that converts the FunctionGraph to executable code.

**Base Linker Classes** ([`pytensor/link/basic.py:144-716`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/basic.py#L144-L716)):
- `Linker` - Abstract base with `make_thunk()` method
- `LocalLinker` - Base for per-node execution
- `PerformLinker` - Python implementation using `Op.perform()`
- `JITLinker` - Base for JIT-compiled backends (JAX, Numba, PyTorch)

**JITLinker Pattern** (lines 576-716):
```python
class JITLinker(LocalLinker):
    def fgraph_convert(self, fgraph, **kwargs):
        """Convert FunctionGraph to backend representation"""
        raise NotImplementedError

    def jit_compile(self, fn, **kwargs):
        """Apply JIT compilation"""
        raise NotImplementedError

    def create_thunk_inputs(self, storage_map):
        """Pre-process inputs"""
        raise NotImplementedError
```

#### Reference Implementation: JAX Backend

**JAXLinker** ([`pytensor/link/jax/linker.py:9-127`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/jax/linker.py#L9-L127)):

```python
class JAXLinker(JITLinker):
    def fgraph_convert(self, fgraph, **kwargs):
        # Convert entire graph to JAX implementation
        return jax_funcify(fgraph, **kwargs)

    def jit_compile(self, fn, **kwargs):
        # Apply JAX JIT compilation
        return jax.jit(fn, static_argnums=...)

    def create_thunk_inputs(self, storage_map):
        # Convert NumPy arrays to JAX arrays
        return [jax.numpy.asarray(v) for v in storage_map.values()]
```

**JAX Dispatch System** ([`pytensor/link/jax/dispatch/basic.py:43-62`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/jax/dispatch/basic.py#L43-L62)):

```python
@singledispatch
def jax_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a JAX compatible function from a PyTensor Op."""
    raise NotImplementedError(f"No JAX conversion for: {op}")

@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph, **kwargs):
    return fgraph_to_python(
        fgraph,
        jax_funcify,
        type_conversion_fn=jax_typify,
        **kwargs
    )
```

**Op Registration Example** ([`pytensor/link/jax/dispatch/elemwise.py:9-20`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/jax/dispatch/elemwise.py#L9-L20)):

```python
@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op
    base_fn = jax_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        return base_fn(*jnp.asarray(inputs))

    return elemwise_fn
```

#### Other Backend References

**PyTorch Backend** ([`pytensor/link/pytorch/linker.py:5-94`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/pytorch/linker.py#L5-L94))
- Uses `torch.compile()` for JIT compilation
- Dispatch in `pytensor/link/pytorch/dispatch/*.py`

**Numba Backend** ([`pytensor/link/numba/linker.py:4-20`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/numba/linker.py#L4-L20))
- Uses `numba.njit()` for compilation
- Dispatch in `pytensor/link/numba/dispatch/*.py`

**MLX Backend** (Apple Silicon) - [`pytensor/link/mlx/linker.py`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/mlx/linker.py)

#### Backend Registration

**Mode Registration** ([`pytensor/compile/mode.py:464-531`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/compile/mode.py#L464-L531)):

```python
predefined_linkers = {
    "py": PerformLinker(),
    "c": CLinker(),
    "jax": JAXLinker(),
    "pytorch": PytorchLinker(),
    "numba": NumbaLinker(),
    "mlx": MLXLinker(),
}

# Modes combine linker + optimizer
JAX = Mode(
    JAXLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "jax"],
        exclude=["cxx_only", "BlasOpt", ...]
    ),
)

predefined_modes = {
    "JAX": JAX,
    "NUMBA": NUMBA,
    "PYTORCH": PYTORCH,
    ...
}
```

### 3. Graph Representation

#### Core Data Structures

**Variable** ([`pytensor/graph/basic.py:350-683`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/graph/basic.py#L350-L683)):
- Represents data nodes in the graph
- Has `type`, `owner` (Apply node that created it), `index`, `name`

**Apply** ([`pytensor/graph/basic.py:113-348`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/graph/basic.py#L113-L348)):
- Represents operation application
- Has `op`, `inputs` (list of Variables), `outputs` (list of Variables)

**FunctionGraph** ([`pytensor/graph/fg.py:50-927`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/graph/fg.py#L50-L927)):
- Container for complete computational subgraph
- Maintains `inputs`, `outputs`, `apply_nodes`, `variables`
- Has `clients` dict for bidirectional traversal
- Supports graph manipulation: `replace()`, `import_var()`, `import_node()`, `remove_node()`

#### Graph Traversal

**Traversal Utilities** ([`pytensor/graph/traversal.py:40-708`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/graph/traversal.py#L40-L708)):
- `walk()` - Generic BFS/DFS walker
- `ancestors()` - Collect ancestor variables
- `toposort()` - Topological sort for execution order
- `graph_inputs()` - Find root inputs
- `applys_between()` - Get Apply nodes in subgraph

#### Graph Conversion Utility

**fgraph_to_python()** ([`pytensor/link/utils.py:666-808`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/link/utils.py#L666-L808)):

```python
def fgraph_to_python(
    fgraph: FunctionGraph,
    op_conversion_fn: Callable,  # e.g., jax_funcify, onnx_funcify
    *,
    type_conversion_fn: Callable = lambda x: x,
    order: list[Apply] | None = None,
    storage_map: Optional[StorageMapType] = None,
    **kwargs,
) -> Callable:
    """Convert a FunctionGraph into a regular Python function.

    This is the core conversion function used by all JIT backends.
    """
```

This function:
1. Topologically sorts the graph
2. Converts each Apply node via `op_conversion_fn`
3. Creates a Python function that executes the converted ops

### 4. Op System

#### Op Base Class

**Op Interface** ([`pytensor/graph/op.py:137-621`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/graph/op.py#L137-L621)):

```python
class Op:
    def make_node(self, *inputs) -> Apply:
        """Create Apply node representing operation application"""
        raise NotImplementedError

    def perform(self, node, inputs, output_storage):
        """Execute computation with numeric inputs"""
        raise NotImplementedError

    def grad(self, inputs, output_grads):
        """Compute symbolic gradients"""
        raise NotImplementedError

    def make_thunk(self, node, storage_map, ...):
        """Create zero-argument callable for execution"""
        # Default implementation wraps perform()
```

#### Example Ops

**Scalar Add** ([`pytensor/scalar/basic.py:1943-1982`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/scalar/basic.py#L1943-L1982)):

```python
class Add(ScalarOp):
    def impl(self, *inputs):
        return sum(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        return f"{outputs[0]} = {' + '.join(inputs)};"

    def grad(self, inputs, output_grads):
        return [gz for _ in inputs]  # ∂/∂xᵢ(x₁+x₂) = 1
```

**Tensor Elemwise** ([`pytensor/tensor/elemwise.py:301-1136`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/tensor/elemwise.py#L301-L1136)):
- Wraps scalar ops and broadcasts to tensors
- Handles inplace operations
- Uses NumPy ufuncs for execution

**Matrix Multiply (Gemm)** ([`pytensor/tensor/blas.py:800-1113`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/tensor/blas.py#L800-L1113)):
- Computes Z = alpha*X*Y + beta*Z
- Generates optimized BLAS C code
- Supports inplace operations

### 5. Compilation Flow

**High-Level Process**:

1. **User creates graph**: `z = pt.add(x, y)`
2. **Function compilation**: `f = pt.function([x, y], z, mode="JAX")`
3. **FunctionMaker** ([`pytensor/compile/function/types.py:1510-1639`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/compile/function/types.py#L1510-L1639)):
   - Creates FunctionGraph from inputs/outputs
   - Applies graph optimizations (rewrites)
   - Assigns linker based on mode
4. **Linker converts graph**: `JAXLinker.fgraph_convert(fgraph)`
5. **JIT compilation**: `JAXLinker.jit_compile(fn)`
6. **Execution**: User calls `f(1.0, 2.0)`

**Entry Points**:
- `pytensor.function()` → [`pytensor/compile/function/__init__.py:95-348`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/compile/function/__init__.py#L95-L348)
- Delegates to `pfunc()` → [`pytensor/compile/function/pfunc.py:358-476`](https://github.com/pymc-devs/pytensor/blob/c58f10beb2aa5e5238f1420107e3bc1103e87c31/pytensor/compile/function/pfunc.py#L358-L476)
- Creates `FunctionMaker` → compiles → returns callable `Function`

### 6. Example Graphs and Demos

#### Tutorial Files

**Basic Examples**:
- [`doc/tutorial/adding_solution_1.py`](doc/tutorial/adding_solution_1.py) - Vector addition/arithmetic
- [`doc/tutorial/profiling_example.py`](doc/tutorial/profiling_example.py) - Function compilation
- [`doc/tutorial/modes_solution_1.py`](doc/tutorial/modes_solution_1.py) - Logistic regression with shared variables
- [`doc/tutorial/loop_solution_1.py`](doc/tutorial/loop_solution_1.py) - Scan/loop examples

**Documentation**:
- [`doc/tutorial/index.rst`](doc/tutorial/index.rst) - Tutorial index
- [`doc/tutorial/adding.rst`](doc/tutorial/adding.rst) - Baby steps in algebra
- [`doc/introduction.rst`](doc/introduction.rst) - Main introduction
- [`README.rst`](README.rst) - Quick start examples

#### Jupyter Notebooks

- [`doc/gallery/introduction/pytensor_intro.ipynb`](doc/gallery/introduction/pytensor_intro.ipynb) - Interactive introduction
- [`doc/gallery/scan/scan_tutorial.ipynb`](doc/gallery/scan/scan_tutorial.ipynb) - Scan operations
- [`doc/gallery/autodiff/vector_jacobian_product.ipynb`](doc/gallery/autodiff/vector_jacobian_product.ipynb) - Automatic differentiation

#### Test Patterns

**Matrix Operations**:
- [`tests/tensor/test_blas.py`](tests/tensor/test_blas.py) - dot, gemm, gemv patterns
  - `test_dot_vv`, `test_dot_vm`, `test_dot_mv` for vector/matrix ops
  - `test_batched_dot` for batched operations

**Basic Operations**:
- [`tests/test_gradient.py`](tests/test_gradient.py) - Gradient computation
- [`tests/tensor/test_elemwise.py`](tests/tensor/test_elemwise.py) - Elementwise ops
- [`tests/tensor/test_basic.py`](tests/tensor/test_basic.py) - Basic tensor operations

**Backend Tests**:
- [`tests/link/jax/test_basic.py`](tests/link/jax/test_basic.py) - JAX backend examples
- [`tests/link/numba/test_basic.py`](tests/link/numba/test_basic.py) - Numba backend examples
- [`tests/link/pytorch/test_basic.py`](tests/link/pytorch/test_basic.py) - PyTorch backend examples

#### Simple Example Code

```python
import pytensor
import pytensor.tensor as pt

# Create variables
a = pt.vector('a')
b = pt.vector('b')

# Build graph
out = a ** 2 + b ** 2 + 2 * a * b

# Compile function
f = pytensor.function([a, b], out)

# Execute
result = f([1, 2], [4, 5])  # [25.  49.]
```

## Code References

### Backend Implementation Files

- **Linker base classes**: `pytensor/link/basic.py:144-716`
- **JAX linker**: `pytensor/link/jax/linker.py:9-127`
- **JAX dispatch**: `pytensor/link/jax/dispatch/basic.py:43-62`
- **PyTorch linker**: `pytensor/link/pytorch/linker.py:5-94`
- **Numba linker**: `pytensor/link/numba/linker.py:4-20`
- **Mode registration**: `pytensor/compile/mode.py:42-531`

### Graph Representation Files

- **Variable and Apply**: `pytensor/graph/basic.py:113-683`
- **FunctionGraph**: `pytensor/graph/fg.py:50-927`
- **Graph traversal**: `pytensor/graph/traversal.py:40-708`
- **Graph to Python**: `pytensor/link/utils.py:666-808`

### Op Implementation Files

- **Op base class**: `pytensor/graph/op.py:137-621`
- **Scalar ops**: `pytensor/scalar/basic.py:1943-2100`
- **Tensor elemwise**: `pytensor/tensor/elemwise.py:301-1136`
- **BLAS ops**: `pytensor/tensor/blas.py:800-1113`
- **C backend**: `pytensor/link/c/op.py:35-649`
- **JAX ops**: `pytensor/link/jax/ops.py:16-537`

### Compilation Flow Files

- **Function entry point**: `pytensor/compile/function/__init__.py:95-348`
- **pfunc**: `pytensor/compile/function/pfunc.py:358-476`
- **FunctionMaker**: `pytensor/compile/function/types.py:1510-1639`
- **Graph rewriting**: `pytensor/graph/rewriting/basic.py:61-331`

## Architecture Insights

### Key Design Patterns

1. **Linker Pattern**: Each backend implements a Linker that converts FunctionGraph to executable code
2. **Dispatch Pattern**: Using Python's `@singledispatch`, each backend registers converters for Op types
3. **Graph as IR**: FunctionGraph serves as the intermediate representation between user code and backend execution
4. **Storage Indirection**: All data passed through single-element lists for mutability across thunks
5. **Feature System**: FunctionGraph has extensible features for tracking inplace operations, debugging, etc.

### Backend Architecture

```
User Graph → FunctionGraph → Linker.fgraph_convert() → Backend IR → JIT Compile → Executable
                    ↓
              Graph Rewriting (Optimizations)
```

**For ONNX**:
```
PyTensor FunctionGraph → onnx_funcify(graph) → ONNX protobuf → .onnx file
                                                                      ↓
                                                      ONNX Runtime (WebAssembly) → Browser Execution
```

### Module Structure for New Backend

Following the established pattern, an ONNX backend would have:

```
pytensor/link/onnx/
├── __init__.py
├── linker.py              # ONNXLinker(JITLinker)
└── dispatch/
    ├── __init__.py
    ├── basic.py           # @singledispatch onnx_funcify, onnx_typify
    ├── elemwise.py        # @onnx_funcify.register(Elemwise)
    ├── tensor_basic.py    # @onnx_funcify.register(Reshape, Transpose, ...)
    ├── math.py            # @onnx_funcify.register(Exp, Log, ...)
    └── nlinalg.py         # @onnx_funcify.register(MatMul, Dot, ...)
```

### ONNX-Specific Challenges

From the existing design document:

1. **Static Graphs**: ONNX requires static shapes - need to handle dynamic shapes at export time
2. **Control Flow**: ONNX has limited control flow support (no general recursion)
3. **Random Operations**: ONNX has no standard RNG - may need to pre-compute or handle specially
4. **Autodiff**: ONNX has limited gradient support - compute gradients in PyTensor before export
5. **Opset Versions**: Need to target specific ONNX opset version for compatibility

## Historical Context (from thoughts/)

### Related Research

- [`thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md`](thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md) - Complete architectural design for ONNX/XLA backends
  - Detailed implementation plan
  - File structure proposals
  - Discussion of challenges and solutions
  - Example code for ONNXLinker and onnx_funcify

This document contains the architectural blueprint for implementing the ONNX backend.

## Implementation Roadmap

### Phase 1: Basic ONNX Linker

1. **Create ONNXLinker class** (`pytensor/link/onnx/linker.py`):
   ```python
   class ONNXLinker(JITLinker):
       def fgraph_convert(self, fgraph, **kwargs):
           # Convert FunctionGraph to ONNX ModelProto
           return onnx_funcify(fgraph, **kwargs)

       def jit_compile(self, fn, **kwargs):
           # Optional: wrap with ONNX Runtime
           return fn  # Or onnxruntime.InferenceSession(fn)
   ```

2. **Create onnx_funcify dispatcher** (`pytensor/link/onnx/dispatch/basic.py`):
   ```python
   import onnx
   from functools import singledispatch

   @singledispatch
   def onnx_funcify(op, node=None, **kwargs):
       raise NotImplementedError(f"No ONNX conversion for: {op}")

   @onnx_funcify.register(FunctionGraph)
   def onnx_funcify_FunctionGraph(fgraph, **kwargs):
       # Convert to ONNX ModelProto
       graph = onnx.helper.make_graph(
           nodes=...,  # Convert Apply nodes
           inputs=...,  # Convert input Variables
           outputs=...,  # Convert output Variables
           initializers=...,  # Convert constants
       )
       model = onnx.helper.make_model(graph)
       return model
   ```

3. **Implement basic op conversions** (elemwise, math, tensor ops):
   ```python
   @onnx_funcify.register(Elemwise)
   def onnx_funcify_Elemwise(op, node, **kwargs):
       # Map PyTensor scalar op to ONNX op type
       scalar_op_to_onnx = {
           scalar.add: "Add",
           scalar.mul: "Mul",
           scalar.sub: "Sub",
           # ...
       }
       onnx_op_type = scalar_op_to_onnx[type(op.scalar_op)]
       return onnx.helper.make_node(
           onnx_op_type,
           inputs=[...],
           outputs=[...]
       )
   ```

### Phase 2: ONNX Export Functionality

1. **Add export method** to save `.onnx` files:
   ```python
   def export_onnx(pytensor_function, output_path):
       """Export PyTensor function to ONNX format."""
       fgraph = pytensor_function.fgraph
       model = onnx_funcify(fgraph)
       onnx.save(model, output_path)
   ```

2. **Handle shape inference** and type conversion
3. **Add validation** via `onnx.checker.check_model()`

### Phase 3: WebAssembly Integration

1. **Install ONNX Runtime Web**:
   ```bash
   npm install onnxruntime-web
   ```

2. **Create JavaScript loader**:
   ```javascript
   import * as ort from 'onnxruntime-web';

   async function runModel(modelPath, inputs) {
       const session = await ort.InferenceSession.create(modelPath);
       const feeds = { input: new ort.Tensor('float32', inputs, [2]) };
       const results = await session.run(feeds);
       return results.output.data;
   }
   ```

3. **Create HTML demo app**:
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
   </head>
   <body>
       <h1>PyTensor ONNX Demo</h1>
       <input id="input1" type="number" value="1.0">
       <input id="input2" type="number" value="2.0">
       <button onclick="runModel()">Compute</button>
       <div id="result"></div>

       <script>
           async function runModel() {
               const session = await ort.InferenceSession.create('model.onnx');
               const input1 = parseFloat(document.getElementById('input1').value);
               const input2 = parseFloat(document.getElementById('input2').value);

               const feeds = {
                   'input1': new ort.Tensor('float32', [input1], [1]),
                   'input2': new ort.Tensor('float32', [input2], [1])
               };

               const results = await session.run(feeds);
               document.getElementById('result').innerText =
                   'Result: ' + results.output.data[0];
           }
       </script>
   </body>
   </html>
   ```

### Phase 4: Testing and Optimization

1. **Create test suite** following existing backend patterns:
   - `tests/link/onnx/test_basic.py` - Basic ops
   - `tests/link/onnx/test_elemwise.py` - Elementwise operations
   - `tests/link/onnx/test_nlinalg.py` - Linear algebra

2. **Add ONNX-specific rewrites** for optimization:
   - Fuse operations where possible
   - Optimize for ONNX Runtime execution
   - Handle unsupported ops (fallback strategies)

3. **Register mode** in `pytensor/compile/mode.py`:
   ```python
   ONNX = Mode(
       ONNXLinker(),
       RewriteDatabaseQuery(
           include=["fast_run", "onnx"],
           exclude=["cxx_only", "inplace", ...]
       ),
   )

   predefined_modes["ONNX"] = ONNX
   ```

### Example End-to-End Workflow

```python
# Python side - Create and export model
import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx

# Create simple graph
x = pt.scalar('x')
y = pt.scalar('y')
z = x + y * 2

# Compile with ONNX mode
f = pytensor.function([x, y], z, mode="ONNX")

# Export to ONNX file
export_onnx(f, "demo_model.onnx")
```

```javascript
// JavaScript side - Load and run in browser
async function demo() {
    const session = await ort.InferenceSession.create('demo_model.onnx');

    const feeds = {
        'x': new ort.Tensor('float32', [1.0], [1]),
        'y': new ort.Tensor('float32', [2.0], [1])
    };

    const results = await session.run(feeds);
    console.log('Result:', results.z.data[0]);  // Should be 5.0
}
```

## Answered Implementation Questions

*See [`thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md`](thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md) for full details.*

### 1. Shape Inference

**Question**: How to handle dynamic shapes in PyTensor graphs when exporting to ONNX?

**Answer**: **Use shape annotations at compile time**
- Provide `example_inputs` when exporting to infer concrete shapes
- Leverage PyTensor's existing `Op.infer_shape()` method
- Support dynamic dimensions with symbolic names (e.g., `['batch_size', 784]`)
- Use `dynamic_axes` parameter to mark truly dynamic dimensions

```python
# Recommended approach
export_onnx(f, "model.onnx",
            example_inputs=[np.zeros((32, 784)), np.zeros((784, 10))],
            dynamic_axes={'x': [0], 'output': [0]})  # First dim is dynamic
```

### 2. Unsupported Ops

**Question**: Which PyTensor ops don't have ONNX equivalents?

**Answer**: **~150+ ops (>50%) lack direct ONNX support**

**Categories with NO/LIMITED ONNX support**:
- ❌ **Special functions** (~50 ops): Gamma, Bessel, Beta, Hypergeometric families
- ❌ **Sparse operations** (100% unsupported): All ~40 sparse ops
- ❌ **Advanced linear algebra**: Cholesky, QR, LU, SVD, Eig, matrix solvers
- ❌ **Most probability distributions**: Beta, Gamma, Exponential, Poisson, etc.
- ❌ **Complex numbers**: Limited support
- ❌ **Fourier transforms**: FFT/IFFT operations

**Categories with GOOD ONNX support**:
- ✅ Basic arithmetic, math, trigonometry
- ✅ Neural network ops (Conv, BatchNorm, Softmax, ReLU)
- ✅ Reductions and tensor manipulation
- ✅ Matrix multiply (MatMul, Gemm)

**Mitigation strategies**:
1. Implement as custom ONNX operators (requires C++)
2. Pre-compute unsupported ops in Python, pass as inputs
3. Approximate special functions with polynomials
4. Raise clear, informative errors for unsupported ops
5. Auto-convert sparse to dense with warnings

### 3. Gradient Computation

**Question**: Should gradients be computed in PyTensor or use ONNX's gradient support?

**Answer**: **Compute gradients in PyTensor before export (RECOMMENDED)**

**Reasons**:
- ✅ Guaranteed compatibility with all PyTensor ops
- ✅ ONNX Runtime WASM may not support training/gradients (inference-focused)
- ✅ Full control over gradient computation and optimizations
- ✅ Consistent behavior across Python and browser
- ✅ Export forward + backward pass as single graph

```python
# Recommended: Include gradients in exported graph
x = pt.matrix('x')
w = pt.vector('w')
loss = ((pt.dot(x, w) - y) ** 2).mean()

# Compute gradient in PyTensor
grad_w = pt.grad(loss, w)

# Export function with gradients included
f = pt.function([x, y, w], [loss, grad_w])
export_onnx(f, "model_with_gradients.onnx")
```

**When to consider ONNX gradients**: Only if using ONNX Runtime's training mode on server/desktop (not WASM) with basic ops only.

### 4. RNG Operations

**Question**: How to handle random number generation?

**Answer**: **Pre-compute random values in JavaScript with fixed seeds (RECOMMENDED for WASM)**

**Approach**: Don't use RandomVariable ops in exported graph
- Generate random values in JavaScript with seedable RNG library
- Pass random values as inputs to the ONNX model
- Ensures reproducibility and works reliably in WASM

```javascript
// Use seedrandom library for deterministic random numbers
import seedrandom from 'seedrandom';
const rng = seedrandom('my-fixed-seed');

function generateRandomNormal(size, mean = 0, std = 1) {
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        // Box-Muller transform
        const u1 = rng(), u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        values[i] = mean + std * z;
    }
    return values;
}

// Pass as input to ONNX model
const feeds = { 'random_input': new ort.Tensor('float32', generateRandomNormal(100), [100]) };
```

**Alternative**: Use ONNX's `RandomNormal`/`RandomUniform` with fixed seeds, but note ONNX Runtime may not guarantee determinism across platforms.

### 5. Control Flow

**Question**: How to handle Scan ops and conditional operations?

**Answer**: Use multiple strategies depending on the case

**PyTensor Scan** is more flexible than **ONNX Loop**, requiring careful conversion:

**Strategy 1: Loop Unrolling (RECOMMENDED for small fixed-length loops)**
```python
# Convert Scan to explicit sequential operations
# Only works for fixed-length sequences
# Simple and reliable, but graph becomes large for long sequences
```

**Strategy 2: Replace with ONNX Built-ins (BEST when possible)**
```python
# Replace cumulative sum Scan with ONNX CumSum operator
# Replace reductions with ReduceSum, ReduceMean, etc.
```

**Strategy 3: Convert to ONNX Loop (for dynamic loops)**
- Complex to implement - ONNX Loop semantics differ from Scan
- Create separate GraphProto for loop body
- Specify loop carried dependencies explicitly

**Strategy 4: Raise Error for Unsupported Scans**
- For complex Scans that can't be easily converted
- Provide clear error messages with suggestions

**IfElse**: Direct mapping to ONNX `If` operator (straightforward)
- Create `then_branch` and `else_branch` subgraphs
- Both branches must have same output types

**Recommendation for WASM demo**:
1. Avoid Scan if possible - use built-in reductions
2. If needed: use fixed-length sequences and unroll, or replace with ONNX built-ins
3. IfElse: Convert to ONNX If (straightforward)

### 6. Performance

**Question**: What's the performance overhead of ONNX Runtime WASM vs native?

**Answer**: **Expect 3-10x slowdown vs native, acceptable for demos**

**Performance Comparison**:

| Backend | Platform | Typical Speed | Notes |
|---------|----------|---------------|-------|
| Native CPU | Server/Desktop | 1.0x (baseline) | Full SIMD, multi-threading |
| Native GPU | Server/Desktop | 10-100x | For large models |
| ONNX RT Native | Server/Desktop | 0.8-1.0x | Very close to native |
| ONNX RT WASM | Browser | **0.1-0.5x** | **3-10x slower** |
| JavaScript | Browser | 0.01-0.1x | Very slow |

**Concrete Measurements**:
- **Small models** (MobileNet): 30-50 ms vs 10 ms native (3-7x slower)
- **Medium models** (ResNet-50): 150-300 ms vs 50 ms native (3-6x slower)
- **Large models** (BERT): 500-1000 ms vs 100 ms native (5-10x slower)

**Why WASM is slower**:
1. Limited SIMD (128-bit vs native 512-bit AVX)
2. Memory constraints and copying overhead
3. Threading limitations (SharedArrayBuffer required)
4. JIT compilation overhead
5. Garbage collection pauses

**Optimization strategies**:
1. **Model quantization**: Reduce to int8 (4x smaller, 2-3x faster)
2. **Graph optimization**: Enable all ONNX Runtime optimizations
3. **Use WebGPU**: 2-5x faster than WASM CPU (when available)
4. **Batch processing**: Amortize overhead across multiple inferences
5. **Web Workers**: Offload to background thread

**Realistic expectations for demo**:
- Simple computation (z = x + y * 2): ~1-5 ms - excellent
- Small neural network (10 layers, 1M params): ~30-50 ms - acceptable
- Large model (BERT, GPT): ~500-1000 ms - may feel slow

**Recommendation**:
- Start with small models for demos
- Measure performance early in target browsers
- Document that it's a proof-of-concept, not production
- Design for future WebGPU support to improve performance

**Bottom line**: WASM will be slower but for demos and small models, this is acceptable. Users understand browser limitations.

## Related Research

- Previous ONNX backend design: [`thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md`](thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md)

## Next Steps

1. **Start with minimal implementation**:
   - ONNXLinker class
   - Basic onnx_funcify for simple ops (Add, Mul, etc.)
   - Export function to save `.onnx` files

2. **Create simple demo**:
   - PyTensor graph: `z = x + y`
   - Export to ONNX
   - Load in browser with ONNX Runtime Web
   - Display result in HTML

3. **Expand op coverage**:
   - Elementwise ops
   - Matrix operations
   - Activation functions
   - Gradients

4. **Optimize and test**:
   - Add comprehensive tests
   - Benchmark performance
   - Handle edge cases
   - Document usage

The architecture is well-documented and the path forward is clear. The existing backend implementations (especially JAX and PyTorch) provide excellent templates to follow.
