---
date: 2025-10-14T00:00:00-00:00
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pytensor
topic: "How to Support Another Backend: ONNX or XLA"
tags: [research, codebase, backend, architecture, linker, dispatch, onnx, xla]
status: complete
last_updated: 2025-10-14
last_updated_by: Claude
---

# Research: How to Support Another Backend: ONNX or XLA

**Date**: 2025-10-14
**Researcher**: Claude
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: main
**Repository**: pytensor

## Research Question

What if I want to support another backend, like ONNX or XLA, in PyTensor?

## Summary

PyTensor uses a **Linker-based architecture** to support multiple backends (Python, C, JAX, Numba, PyTorch, MLX). Adding a new backend like ONNX or XLA requires:

1. **Creating a Linker subclass** (preferably `JITLinker` for JIT-compiled backends)
2. **Implementing a dispatch system** using `@singledispatch` to convert PyTensor `Op`s to backend-specific implementations
3. **Registering the linker** with PyTensor's compilation Mode system
4. **Optionally adding backend-specific graph rewrites** for optimization

The architecture is highly modular. JAX and Numba backends provide excellent templates, with JAX having the most complete implementation (21 dispatch files, 163+ tests) and Numba having the most extensive (32 dispatch files, 155+ tests, full LAPACK support).

## Detailed Findings

### 1. Backend Architecture Overview

**Core Pattern: Linker + Dispatch**

All backends follow the same fundamental pattern:
- **Linker**: Orchestrates the conversion and compilation of PyTensor FunctionGraphs
- **Dispatch System**: Converts individual PyTensor Ops to backend-specific implementations using `@singledispatch`
- **Mode Integration**: Registers the linker with PyTensor's compilation system

**File Structure Template**:
```
pytensor/link/<backend>/
├── __init__.py              # Exports linker
├── linker.py                # Main linker class
└── dispatch/
    ├── __init__.py          # Imports all dispatch modules
    ├── basic.py             # Core dispatch functions (funcify/typify)
    ├── elemwise.py          # Elemwise operations
    ├── tensor_basic.py      # Basic tensor operations
    ├── math.py              # Math operations
    ├── nlinalg.py           # Numerical linear algebra
    ├── random.py            # Random number generation
    ├── scan.py              # Scan (loop) operations
    └── ...                  # More specialized modules
```

### 2. Linker Hierarchy and Interface

**Base Classes** (`pytensor/link/basic.py`):

```
Linker (ABC) - line 144
├── LocalLinker - line 231
│   └── PerformLinker - line 276
│       └── JITLinker (ABC) - line 576  ← Recommended for new backends
│           ├── JAXLinker
│           ├── NumbaLinker
│           ├── PytorchLinker
│           └── MLXLinker
└── WrapLinker - line 399
```

**JITLinker Abstract Interface** (`pytensor/link/basic.py:576-717`):

Three required methods:
1. **`fgraph_convert()`** (line 585): Convert FunctionGraph to JIT-able function
2. **`jit_compile()`** (line 605): Apply JIT compilation
3. **`create_thunk_inputs()`** (line 591): Pre-process inputs

Two optional override methods:
- **`input_filter()`** (line 608): Filter input data before processing
- **`output_filter()`** (line 612): Filter output data after computation

### 3. Existing Backend Implementations

#### JAX Backend (Most Complete)

**Linker**: `pytensor/link/jax/linker.py:9-127`
- Handles RNG state conversion (Generator → JAX PRNGKey)
- Identifies scalar shape inputs for static compilation
- Uses `jax.jit()` with `static_argnums` for optimization

**Dispatch**: 21 files, 2359+ lines
- `basic.py`: Core dispatch (`jax_funcify`, `jax_typify`)
- `elemwise.py`, `scalar.py`: Element-wise operations
- `tensor_basic.py`: Basic tensor ops (Alloc, Join, ARange, Eye, etc.)
- `random.py`: Random variables with nested dispatch for distributions
- `scan.py`: Complex control flow (line 9-202)
- `blas.py`, `nlinalg.py`, `slinalg.py`: Linear algebra
- `subtensor.py`: Indexing/slicing
- `shape.py`: Shape operations (includes `JAXShapeTuple` for concrete shapes)
- Plus: `math.py`, `einsum.py`, `blockwise.py`, `extra_ops.py`, `pad.py`, `sort.py`, `sparse.py`

**Special Features**:
- `JAXOp` class (`pytensor/link/jax/ops.py:16-196`): Wraps JAX functions as PyTensor Ops
- `wrap_jax` decorator (line 198-348): High-level API for JAX → PyTensor conversion
- JAX-specific rewrites (`pytensor/tensor/rewriting/jax.py`):
  - Boolean indexing transformations
  - Shape parameter as tuple conversion

**Tests**: 20 files, 163+ tests

#### Numba Backend (Most Extensive)

**Linker**: `pytensor/link/numba/linker.py:4-20`
- Minimal implementation (12 lines)
- Uses `numba_njit` wrapper with configuration

**Dispatch**: 32 files, 8570+ lines
- `basic.py`: Core dispatch with **fallback to object mode** for unsupported ops (line 284-330)
- LAPACK support: 18 files in `dispatch/linalg/` subdirectory
  - Cholesky, LU, QR decompositions
  - Linear solvers (general, symmetric, triangular, tridiagonal, etc.)
  - Direct LAPACK bindings
- Custom vectorization framework (`elemwise.py:265`)
- Code generation for reductions (`create_multiaxis_reducer`, line 122)
- Cython function wrapping for scipy.special (`scalar.py:64-74`)

**Special Features**:
- **Graceful degradation**: Falls back to `Op.perform()` in object mode when no specialized implementation exists
- **Configuration**: `numba__cache`, `numba__fastmath` flags
- **Type system**: `get_numba_type()` (line 97-139) with sparse matrix support

**Tests**: 17 files, 155+ tests

#### Other Backends

**MLX Backend** (`pytensor/link/mlx/`): 9 dispatch files, 58+ tests
- Apple Silicon focus
- Similar structure to JAX

**PyTorch Backend** (`pytensor/link/pytorch/`): 13 dispatch files, 51+ tests
- Advanced linker with `gen_functors` registry (line 14-26)
- Wrapper class to handle `torch.compile` closure issues (line 40-85)
- Input/output conversion via `pytorch_typify`

**C Backend** (`pytensor/link/c/`): 11 files
- Default/legacy backend
- Generates and compiles C code
- Used by default in FAST_RUN mode (with CVM)

### 4. Dispatch Mechanism: Singledispatch Pattern

All backends use Python's `functools.singledispatch` for extensible Op conversion.

**JAX Example** (`pytensor/link/jax/dispatch/basic.py`):

```python
@singledispatch
def jax_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a JAX compatible function from a PyTensor Op."""
    raise NotImplementedError(f"No JAX conversion for Op: {op}")

@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph, **kwargs):
    return fgraph_to_python(
        fgraph,
        jax_funcify,           # Recursive dispatch
        type_conversion_fn=jax_typify,
        **kwargs
    )

@jax_funcify.register(IfElse)
def jax_funcify_IfElse(op, **kwargs):
    def ifelse(cond, *args):
        return jax.lax.cond(cond, lambda _: args[:n_outs],
                           lambda _: args[n_outs:], operand=None)
    return ifelse
```

**Numba Example** (`pytensor/link/numba/dispatch/basic.py`):

```python
@singledispatch
def numba_funcify(op, node=None, storage_map=None, **kwargs):
    """Generate a numba function for a given op."""
    # Fallback to object mode
    return generate_fallback_impl(op, node, storage_map, **kwargs)

@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(fgraph, **kwargs):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        type_conversion_fn=numba_typify,
        **kwargs
    )
```

**Key Pattern**: `fgraph_to_python()` utility (`pytensor/link/utils.py:666-808`) is used by all JIT backends to convert FunctionGraphs to Python source code.

### 5. Backend Registration and Mode System

**Linker Registration** (`pytensor/compile/mode.py:42-62`):

```python
predefined_linkers = {
    "py": PerformLinker(),
    "c": CLinker(),
    "jax": JAXLinker(),
    "numba": NumbaLinker(),
    "pytorch": PytorchLinker(),
    "mlx": MLXLinker(),
}

def register_linker(name, linker):
    """Add a Linker which can be referred to by name in Mode."""
    if name in predefined_linkers:
        raise ValueError(f"Linker name already taken: {name}")
    predefined_linkers[name] = linker
```

**Mode Creation** (lines 452-531):

```python
# JAX Mode
JAX = Mode(
    JAXLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "jax"],
        exclude=["cxx_only", "BlasOpt", "fusion", "inplace",
                "scan_save_mem_prealloc"]
    )
)

# Numba Mode
NUMBA = Mode(
    NumbaLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "numba"],
        exclude=["cxx_only", "BlasOpt", "local_careduce_fusion",
                "scan_save_mem_prealloc"]
    )
)
```

**Usage**:
```python
import pytensor
import pytensor.tensor as pt

x = pt.vector('x')
y = pt.sum(x ** 2)

# Use specific backend
f = pytensor.function([x], y, mode='JAX')
# or
f = pytensor.function([x], y, mode=pytensor.compile.mode.JAX)
```

### 6. Complete Implementation Checklist for ONNX/XLA

#### Step 1: Create Linker Class

**File**: `pytensor/link/onnx/linker.py`

```python
from pytensor.link.basic import JITLinker

class ONNXLinker(JITLinker):
    """A Linker that compiles PyTensor graphs to ONNX."""

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.onnx.dispatch import onnx_funcify
        return onnx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            **kwargs
        )

    def jit_compile(self, fn):
        import onnxruntime as ort
        # Convert Python function to ONNX graph
        # Create InferenceSession
        # Return wrapper function
        pass

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
```

#### Step 2: Create Dispatch System

**File**: `pytensor/link/onnx/dispatch/__init__.py`

```python
# Import core dispatchers
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import all dispatch specializations to register them
import pytensor.link.onnx.dispatch.elemwise
import pytensor.link.onnx.dispatch.tensor_basic
import pytensor.link.onnx.dispatch.math
import pytensor.link.onnx.dispatch.nlinalg
# ... more modules
```

**File**: `pytensor/link/onnx/dispatch/basic.py`

```python
from functools import singledispatch
from pytensor.graph.fg import FunctionGraph
from pytensor.link.utils import fgraph_to_python
import numpy as np

@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert PyTensor types to ONNX-compatible types."""
    if dtype is None:
        return data
    return np.array(data, dtype=dtype)

@singledispatch
def onnx_funcify(op, node=None, storage_map=None, **kwargs):
    """Create ONNX-compatible function from PyTensor Op."""
    raise NotImplementedError(
        f"No ONNX conversion for the given Op: {op}"
    )

@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(fgraph, node=None,
                                fgraph_name="onnx_funcified_fgraph",
                                **kwargs):
    return fgraph_to_python(
        fgraph,
        onnx_funcify,
        type_conversion_fn=onnx_typify,
        fgraph_name=fgraph_name,
        **kwargs
    )
```

#### Step 3: Implement Op Dispatches

**File**: `pytensor/link/onnx/dispatch/elemwise.py`

```python
from pytensor.tensor.elemwise import Elemwise, CAReduce, DimShuffle
from pytensor.link.onnx.dispatch.basic import onnx_funcify

@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, **kwargs):
    """Convert Elemwise operations to ONNX."""
    scalar_op = op.scalar_op
    # Get ONNX equivalent operation
    # Map PyTensor scalar op to ONNX node type
    # Return function that applies operation
    pass

@onnx_funcify.register(CAReduce)
def onnx_funcify_CAReduce(op, **kwargs):
    """Convert reduction operations to ONNX."""
    # Map to ReduceSum, ReduceMax, etc.
    pass

@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(op, **kwargs):
    """Convert DimShuffle to ONNX Transpose."""
    pass
```

**File**: `pytensor/link/onnx/dispatch/tensor_basic.py`

```python
from pytensor.tensor.basic import Alloc, Join, Split, Eye
from pytensor.link.onnx.dispatch.basic import onnx_funcify

@onnx_funcify.register(Alloc)
def onnx_funcify_Alloc(op, node, **kwargs):
    """Map to ONNX ConstantOfShape or Expand."""
    pass

@onnx_funcify.register(Join)
def onnx_funcify_Join(op, **kwargs):
    """Map to ONNX Concat."""
    pass
```

#### Step 4: Register with Mode System

**Modify**: `pytensor/compile/mode.py`

```python
# Add import at top
from pytensor.link.onnx.linker import ONNXLinker

# Add to predefined_linkers (around line 51)
predefined_linkers = {
    # ... existing linkers
    "onnx": ONNXLinker(),
}

# Create ONNX mode (around line 522)
ONNX = Mode(
    ONNXLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "onnx"],
        exclude=["cxx_only", "BlasOpt", "fusion", "inplace"]
    )
)

# Add to predefined_modes (around line 533)
predefined_modes = {
    # ... existing modes
    "ONNX": ONNX,
}
```

#### Step 5: Add Backend-Specific Rewrites (Optional)

**File**: `pytensor/tensor/rewriting/onnx.py`

```python
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB

# Create ONNX optimization database
optdb = EquilibriumDB()

@node_rewriter([SomeOp])
def onnx_specific_rewrite(fgraph, node):
    """Transform graph for ONNX compatibility."""
    # Example: Replace unsupported ops with ONNX-compatible alternatives
    pass

# Register rewrite with "onnx" tag
optdb.register(
    "onnx_specific_rewrite",
    dfs_rewriter(onnx_specific_rewrite),
    "onnx",
    position=100
)
```

#### Step 6: Add Tests

**File**: `tests/link/onnx/test_basic.py`

```python
import pytest
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode

def test_onnx_basic_ops():
    """Test basic operations with ONNX backend."""
    x = pt.vector('x')
    y = x + 1

    f = pytensor.function([x], y, mode='ONNX')
    result = f([1.0, 2.0, 3.0])
    expected = np.array([2.0, 3.0, 4.0])

    np.testing.assert_allclose(result, expected)

def test_onnx_elemwise():
    """Test elemwise operations."""
    # Add tests for Elemwise ops
    pass
```

### 7. Key Utilities and Helper Functions

**`fgraph_to_python()`** (`pytensor/link/utils.py:666-808`):
- Core function used by all JIT backends
- Converts FunctionGraph to executable Python source code
- Handles topological sorting, constant propagation, storage management
- Takes `op_conversion_fn` (e.g., `jax_funcify`) as parameter

**`compile_function_src()`** (`pytensor/link/utils.py:580-601`):
- Compiles dynamically generated Python code
- Creates temporary file for debugging
- Returns callable with `__source__` attribute

**`unique_name_generator()`** (`pytensor/link/utils.py:630-663`):
- Generates unique variable names for generated code
- Prevents naming conflicts in generated functions

## Code References

### Core Backend Infrastructure
- `pytensor/link/basic.py:144-229` - `Linker` base class
- `pytensor/link/basic.py:576-717` - `JITLinker` abstract base class
- `pytensor/link/utils.py:666-808` - `fgraph_to_python()` converter
- `pytensor/compile/mode.py:42-62` - Linker registration
- `pytensor/compile/mode.py:288-328` - Mode class
- `pytensor/compile/mode.py:452-531` - Predefined modes

### JAX Backend (Template)
- `pytensor/link/jax/linker.py:9-127` - JAXLinker implementation
- `pytensor/link/jax/dispatch/basic.py:43-151` - Core dispatch
- `pytensor/link/jax/dispatch/elemwise.py:9-69` - Elemwise operations
- `pytensor/link/jax/dispatch/random.py:83-128` - Random variables
- `pytensor/link/jax/dispatch/scan.py:9-202` - Scan operation
- `pytensor/link/jax/ops.py:16-196` - JAXOp wrapper class
- `pytensor/tensor/rewriting/jax.py` - JAX-specific rewrites

### Numba Backend (Template with Fallback)
- `pytensor/link/numba/linker.py:4-20` - NumbaLinker implementation
- `pytensor/link/numba/dispatch/basic.py:333-389` - Core dispatch
- `pytensor/link/numba/dispatch/basic.py:284-330` - Fallback mechanism
- `pytensor/link/numba/dispatch/basic.py:97-139` - Type system
- `pytensor/link/numba/dispatch/elemwise.py:265-340` - Elemwise with custom vectorization
- `pytensor/link/numba/dispatch/elemwise.py:122-244` - Reduction code generator

### PyTorch Backend (Advanced Features)
- `pytensor/link/pytorch/linker.py:5-94` - PytorchLinker with functor registry
- `pytensor/link/pytorch/dispatch/basic.py` - Core dispatch

### Utilities
- `pytensor/link/utils.py:580-601` - `compile_function_src()`
- `pytensor/link/utils.py:630-663` - `unique_name_generator()`

## Architecture Insights

### Design Patterns

1. **Single Dispatch Pattern**: All backends use `@singledispatch` for extensible Op conversion
   - Allows registration of new Ops without modifying core code
   - Enables multiple backends to coexist without conflicts

2. **Template Method Pattern**: `JITLinker` defines compilation template
   - Subclasses fill in backend-specific steps
   - Consistent pipeline across all JIT backends

3. **Strategy Pattern**: Different conversion strategies based on Op properties
   - Constants vs runtime values
   - Scalars vs arrays
   - Static vs dynamic shapes

4. **Factory Pattern**: `*_funcify()` returns closures that capture Op configuration
   - Generated functions are lightweight and efficient
   - Deferred evaluation until actual compilation

5. **Fallback Pattern** (Numba): Graceful degradation to Python's `Op.perform` via object mode
   - Ensures all ops work, even without specialized implementations
   - Provides path for incremental backend development

### Key Architectural Decisions

1. **Storage Map Contract**: Variables → single-element lists
   - Enables in-place updates
   - Supports lazy evaluation (compute_map tracking)
   - Allows sharing storage between operations

2. **Separate Type Conversion**: `*_typify()` functions for input/output transformations
   - Decouples type handling from operation implementation
   - Enables backend-specific type requirements

3. **Graph-Level Optimization**: Rewrites tagged by backend
   - Backends can register optimizations without modifying ops
   - Conditional optimization based on mode

4. **JIT Compilation Pipeline**: Three-stage process
   - `fgraph_convert()`: Op-level translation
   - `jit_compile()`: Backend-specific compilation
   - `create_thunk_inputs()`: Input preparation

5. **Lazy Backend Loading**: Dispatch modules imported on first use
   - Reduces import time
   - Allows missing optional dependencies
   - Backend registration happens at import time

### Comparison: JAX vs Numba Design

| Aspect | JAX | Numba |
|--------|-----|-------|
| **Error Handling** | Raises `NotImplementedError` immediately | Falls back to object mode with warning |
| **Type System** | Simple (`jax_typify` for arrays) | Complex (`get_numba_type` with layouts, sparse support) |
| **Elemwise** | Relies on JAX auto-vectorization | Custom `_vectorized` framework with pattern encoding |
| **Reductions** | Uses `jax.lax.reduce` | Generates nested loops via `create_multiaxis_reducer` |
| **RNG** | Functional (PRNGKey), stateless | Stateful (Generator) |
| **Special Features** | `JAXOp` wrapper for arbitrary JAX code | Direct LAPACK bindings for linear algebra |
| **Code Generation** | Minimal (mostly direct mappings) | Extensive (loop generation, code templates) |
| **Flexibility** | Strict (must implement all ops) | Flexible (fallback allows incremental development) |

### Extension Points

1. **New Op Support**: Register via `@{backend}_funcify.register(OpClass)`
2. **New Type Support**: Register via `@{backend}_typify.register(TypeClass)`
3. **New Rewrites**: Use `@node_rewriter` with backend tag
4. **New Mode**: Call `register_mode(name, mode)`
5. **New Linker**: Call `register_linker(name, linker)`

### Minimal vs Full Implementation

**Minimal Backend** (like XLA might be):
- Linker with 3 required methods (~50 lines)
- Dispatch system with `funcify`/`typify` (~30 lines)
- 5-10 dispatch files for common ops (~500-1000 lines)
- Registration in mode.py (~10 lines)
- **Total**: ~600-1100 lines to get started

**Full Backend** (like JAX):
- 21 dispatch files
- 2359+ lines of dispatch code
- Custom ops and wrappers
- Backend-specific rewrites
- 20 test files with 163+ tests
- **Total**: ~3000+ lines for production readiness

## Historical Context (from thoughts/)

### Related Research

**`thoughts/shared/research/2025-10-14_06-44-01_jaxop-optimization-opportunities.md`**

This document provides detailed insights into JAX backend architecture:

1. **Backend Integration Patterns**:
   - Dispatch pattern for existing PyTensor Ops
   - Wrapper pattern (JAXOp) for arbitrary backend functions

2. **JAXOp Architecture** (lines 16-196 in `pytensor/link/jax/ops.py`):
   - Wraps JAX functions as PyTensor Ops
   - Automatic differentiation using `jax.vjp()`
   - Creates separate JAXOp instances for gradient operations

3. **Blockwise Vectorization** (lines 155+ in `pytensor/tensor/blockwise.py`):
   - Generic vectorization using NumPy gufunc signatures
   - Backend-specific dispatch in `pytensor/link/jax/dispatch/blockwise.py`
   - Used extensively for linear algebra operations

4. **Compilation Infrastructure**:
   - Rewrite system in `pytensor/tensor/rewriting/jax.py`
   - Shape inference via `ShapeFeature` and `infer_shape` protocol
   - Optimization opportunities for `value_and_grad` pattern

5. **Key Insight**: Two approaches for backend integration:
   - For existing Ops: Create dispatch handlers in `pytensor/link/<backend>/dispatch/*.py`
   - For custom functions: Create wrapper Op (like JAXOp)

## Open Questions

1. **ONNX-Specific Considerations**:
   - How to handle ONNX's static graph requirement vs PyTensor's dynamic graphs?
   - Best approach for control flow (If, Scan) → ONNX control flow operators?
   - Should we target ONNX opset 17+ for better operator coverage?

2. **XLA-Specific Considerations**:
   - XLA has overlap with JAX (JAX uses XLA as backend) → Should we create a direct XLA backend or leverage JAX?
   - How to handle XLA's HLO (High-Level Operations) vs PyTensor's Ops?
   - Device placement strategy (CPU/GPU/TPU)?

3. **Performance Optimization**:
   - What rewrites are most critical for ONNX/XLA performance?
   - Should we support operator fusion at the PyTensor level or rely on backend optimizers?
   - Caching strategy for compiled graphs?

4. **Testing Strategy**:
   - Should we test against ONNX Runtime or other backends?
   - How to handle operator coverage gaps (ops that don't have ONNX equivalents)?
   - Performance benchmarking framework?

5. **Deployment Considerations**:
   - Export mechanism for ONNX models (serialize FunctionGraph → .onnx file)?
   - Version compatibility (ONNX opset versions, XLA versions)?
   - Integration with model serving frameworks (TensorFlow Serving, TorchServe, Triton)?

6. **Gradient Computation**:
   - ONNX has limited autodiff support → Should we compute gradients in PyTensor then export?
   - XLA has good autodiff support → Can we leverage it directly?

7. **Random Number Generation**:
   - ONNX has no standard RNG → How to handle random ops?
   - XLA has RNG support → How does it compare to JAX's PRNGKey approach?

8. **Sparse Tensors**:
   - ONNX has experimental sparse tensor support → Worth implementing?
   - XLA sparse tensor support status?

## Next Steps

For implementing ONNX backend:
1. Start with minimal linker + dispatch for ~20 common ops
2. Test with simple models (linear regression, small MLPs)
3. Add ONNX export functionality (serialize to .onnx file)
4. Expand operator coverage based on real use cases
5. Add rewrites for ONNX-specific optimizations
6. Performance benchmarking vs other backends

For implementing XLA backend:
1. Evaluate relationship with existing JAX backend
2. If pursuing direct XLA: Start with HLO translation layer
3. Focus on control flow (While, Cond) and custom calls
4. Leverage XLA's compiler optimizations
5. Add device placement API
6. Test on TPU hardware if available
