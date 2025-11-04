---
date: 2025-11-04T05:44:21-06:00
researcher: Claude (Sonnet 4.5)
git_commit: b556aec588e2f55a347e5e30ed955d3a611f8a20
branch: onnx-backend
repository: clsandoval/pytensor-workshop-demo
topic: "Dev Environment Setup and Testing Strategy for ONNX Backend"
tags: [research, codebase, onnx, backend, dev-environment, testing, uv]
status: complete
last_updated: 2025-11-04
last_updated_by: Claude (Sonnet 4.5)
---

# Research: Dev Environment Setup and Testing Strategy for ONNX Backend

**Date**: 2025-11-04T05:44:21-06:00
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: b556aec588e2f55a347e5e30ed955d3a611f8a20
**Branch**: onnx-backend
**Repository**: clsandoval/pytensor-workshop-demo

## Research Question

How should I install the development environment using uv to run tests for adding ONNX as a PyTensor backend?

## Summary

The PyTensor project supports both **uv** (for local development) and **micromamba** (for CI/CD). A `uv.lock` file already exists in the repository, making uv the recommended tool for local development. The project follows a consistent backend architecture pattern where all backends (JAX, Numba, MLX, PyTorch) extend `JITLinker` and use Python's `singledispatch` pattern for operation registration. Extensive ONNX research and planning documents already exist in the `thoughts/` directory, providing a production roadmap and implementation strategy.

## Detailed Findings

### 1. Development Environment Setup with uv

#### Current State
- **uv version**: 0.9.5 (installed at `/snap/bin/uv`)
- **uv.lock**: Present in repository root (157KB, 3945 lines)
- **Python version**: Requires `>=3.11, <3.14`
- **Project configuration**: `pyproject.toml` with standard setuptools build

#### Installation Steps

**Option 1: Using uv (Recommended for Local Development)**

```bash
# 1. Clone the repository (if not already done)
git clone git@github.com:clsandoval/pytensor-workshop-demo.git
cd pytensor-workshop-demo

# 2. Create and activate virtual environment with uv
uv venv

# 3. Install development dependencies
uv sync --all-extras

# 4. Install pytensor in editable mode (if not already done by sync)
uv pip install -e .

# 5. Install test dependencies explicitly
uv pip install pytest pytest-cov pytest-benchmark pytest-mock pre-commit

# 6. Install optional backend dependencies (for testing patterns)
uv pip install jax jaxlib numba

# 7. Verify installation
uv run python -c "import pytensor; print(pytensor.__version__)"
uv run python -c "import pytensor; print(pytensor.config)"
```

**Option 2: Using micromamba (For CI-Matching Environment)**

```bash
# As documented in .github/copilot-instructions.md:67-69
micromamba create -n pytensor-test -f environment.yml
micromamba run -n pytensor-test python -c 'import pytensor; print(pytensor.__version__)'
```

#### Running Tests with uv

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/link/jax/test_basic.py -v

# Run tests with coverage
uv run pytest tests/ --cov=pytensor --cov-report=html

# Run backend-specific tests
uv run pytest tests/link/jax/ -v
uv run pytest tests/link/numba/ -v

# Run with benchmark support
uv run pytest tests/link/numba/test_blockwise.py::test_blockwise_benchmark -v

# Include slow tests
uv run pytest tests/ --runslow
```

#### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit checks manually
uv run pre-commit run --all-files
```

### 2. Backend Architecture Overview

#### Directory Structure

All backends follow this consistent pattern:

```
pytensor/link/
├── __init__.py              # Exports backend linkers
├── basic.py                 # Base JITLinker class
├── utils.py                 # fgraph_to_python() core translation
├── jax/                     # JAX backend
│   ├── __init__.py         # Exports JAXLinker
│   ├── linker.py           # JAXLinker implementation
│   ├── ops.py              # JAXOp wrapper class
│   └── dispatch/           # Operation implementations
│       ├── __init__.py
│       ├── basic.py        # jax_funcify singledispatch
│       ├── elemwise.py     # Element-wise operations
│       ├── math.py         # Math operations
│       ├── blas.py         # BLAS operations
│       ├── blockwise.py    # Vectorized operations
│       ├── random.py       # Random operations
│       └── ...             # 17 dispatch modules total
├── numba/                   # Numba backend
│   ├── linker.py           # NumbaLinker implementation
│   └── dispatch/           # 15+ dispatch modules
│       ├── linalg/         # Extensive linear algebra
│       │   ├── decomposition/
│       │   └── solve/
│       └── ...
├── mlx/                     # MLX backend (Apple Silicon)
├── pytorch/                 # PyTorch backend
└── c/                       # Native C backend
```

**Key Files Referenced**:
- `pytensor/link/basic.py:576-717` - JITLinker base class
- `pytensor/link/jax/linker.py:9-127` - JAXLinker implementation
- `pytensor/link/jax/dispatch/basic.py:27-151` - jax_funcify dispatcher
- `pytensor/link/utils.py:666-765` - fgraph_to_python() graph compiler

#### Three-Layer Architecture

**Layer 1: Linker** (Framework-specific compilation)
- Extends `JITLinker` base class
- Implements: `fgraph_convert()`, `jit_compile()`, `create_thunk_inputs()`
- Example: `JAXLinker`, `NumbaLinker`, `MLXLinker`

**Layer 2: Dispatch** (Operation translation)
- Uses Python's `@singledispatch` decorator
- Maps PyTensor Ops to backend functions
- Example: `@jax_funcify.register(Elemwise)`

**Layer 3: Graph Compilation** (Generic traversal)
- `fgraph_to_python()` walks computation graph
- Calls dispatcher for each Op
- Generates executable Python function

#### Execution Flow

```python
1. User calls pytensor.function([x], [y], mode="JAX")
2. JAXLinker.make_all() orchestrates compilation
3. JAXLinker.fgraph_convert() calls jax_funcify(fgraph)
4. fgraph_to_python() walks graph topologically
   For each node:
   - Calls @jax_funcify.register(OpType) dispatcher
   - Gets backend-specific function
   - Generates Python assignment statement
5. Returns generated Python function
6. JAXLinker.jit_compile() applies jax.jit()
7. Wrapped in thunk with storage handling
```

### 3. Backend Test Patterns

Tests for backends follow established patterns in `tests/link/`:

#### Pattern 1: Comparison Testing (Primary Pattern)

**Location**: `tests/link/jax/test_basic.py:36-96`

```python
def compare_jax_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
):
    """Compare Python and JAX backend outputs for correctness."""

    # Compile with backend
    pytensor_jax_fn = function(graph_inputs, graph_outputs, mode=jax_mode)
    jax_res = pytensor_jax_fn(*test_inputs)

    # Verify backend-specific output type
    assert isinstance(jax_res, jax.Array)

    # Compile with Python mode for reference
    pytensor_py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    # Compare results
    assert_fn(jax_res, py_res)
    return pytensor_jax_fn, jax_res
```

**Usage**:
```python
def test_jax_operation():
    x = dscalar("x")
    y = x + 1
    compare_jax_and_py([x], [y], [np.array(2.0)])
```

#### Pattern 2: Mode Configuration

**Location**: `tests/link/jax/test_basic.py:22-33`

```python
@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    with config.change_flags(cxx="", compute_test_value="ignore"):
        yield

jax = pytest.importorskip("jax")

# Backend-specific mode
optimizer = RewriteDatabaseQuery(include=["jax"], exclude=JAX._optimizer.exclude)
jax_mode = Mode(linker=JAXLinker(), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)
```

#### Pattern 3: Parametrized Testing

**Location**: `tests/link/numba/test_elemwise.py:34-124`

```python
@pytest.mark.parametrize(
    "inputs, input_vals, output_fn",
    [
        ([pt.vector()], [rng.uniform(size=100)], lambda x: pt.gammaln(x)),
        ([pt.vector()], [rng.standard_normal(100)], lambda x: pt.sigmoid(x)),
        # ... more test cases
    ],
    ids=["gammaln", "sigmoid", ...],
)
def test_Elemwise(inputs, input_vals, output_fn):
    outputs = output_fn(*inputs)
    compare_numba_and_py(inputs, outputs, input_vals)
```

#### Test Organization

```
tests/link/
├── jax/
│   ├── test_basic.py          # Core comparison functions + basic tests
│   ├── test_elemwise.py       # Element-wise operations
│   ├── test_math.py           # Math operations
│   ├── test_blas.py           # BLAS operations
│   ├── test_wrap_jax.py       # JAXOp wrapper tests
│   └── signal/
│       └── test_conv.py       # Convolution operations
├── numba/
│   ├── test_basic.py          # Numba comparison + object mode testing
│   ├── test_elemwise.py       # Parametrized elemwise tests
│   ├── test_performance.py   # Performance benchmarks
│   └── linalg/solve/
│       └── test_tridiagonal.py
└── mlx/
    └── test_basic.py
```

#### Pytest Configuration

**Location**: `pyproject.toml:119-122`

```toml
[tool.pytest.ini_options]
addopts = "--durations=50 --doctest-modules --ignore=pytensor/link"
testpaths = ["pytensor/", "tests/"]
xfail_strict = true
```

### 4. Existing ONNX Research

The `thoughts/` directory contains **19 ONNX-related documents**:

#### Research Documents (13 files)

1. **Production Roadmap** (Most Recent)
   - `thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md`
   - Comprehensive roadmap for production-ready ONNX backend

2. **Implementation Strategy**
   - `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md`
   - Core implementation plan and architecture decisions

3. **Coverage Analysis**
   - `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md`
   - Detailed analysis of operation coverage

4. **Gap Analysis**
   - `thoughts/shared/research/2025-10-15_00-05-01_onnx-cnn-gap-analysis.md`
   - CNN operations gap analysis for ONNX backend
   - `thoughts/shared/research/2025-10-15_updated-yolo11n-onnx-gaps.md`
   - YOLO11n-specific gaps and blockers

5. **Backend Guides**
   - `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md`
   - General guide for adding new backends

6. **Special Features**
   - `thoughts/shared/research/2025-10-15_onnx-backend-webassembly.md`
   - WebAssembly support research
   - `thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md`
   - Q&A document addressing common questions

#### Implementation Plans (6 files)

1. **Main Plan**
   - `thoughts/shared/plans/onnx-backend-implementation.md`
   - Core implementation roadmap

2. **TDD Plans**
   - `thoughts/shared/plans/onnx-tier1-blockers-tdd.md` - Critical path items
   - `thoughts/shared/plans/onnx-tier2-correctness-tdd.md` - Correctness improvements
   - `thoughts/shared/plans/onnx-conv2d-tdd.md` - Conv2D specific

3. **Quality & Testing**
   - `thoughts/shared/plans/onnx-backend-coverage-and-quality-improvements.md`
   - `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`
   - Property-based testing with Hypothesis

### 5. Step-by-Step Guide for ONNX Backend

Based on the established backend patterns, here's the recommended approach:

#### Phase 1: Environment Setup

```bash
# 1. Ensure uv is installed
uv --version  # Should be 0.9.5 or later

# 2. Set up development environment
cd /home/clsandoval/cs/pytensor-workshop-demo
uv sync --all-extras

# 3. Install ONNX dependencies
uv pip install onnx onnxruntime numpy

# 4. Verify current tests pass
uv run pytest tests/link/jax/test_basic.py -v  # Check baseline
```

#### Phase 2: Create ONNX Backend Structure

```bash
# Create directory structure (if not exists)
mkdir -p pytensor/link/onnx/dispatch
mkdir -p tests/link/onnx
```

#### Phase 3: Implement Core Components

**File 1**: `pytensor/link/onnx/linker.py`

```python
from pytensor.link.basic import JITLinker

class ONNXLinker(JITLinker):
    """A Linker that converts PyTensor graphs to ONNX models."""

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.onnx.dispatch import onnx_funcify
        return onnx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            **kwargs
        )

    def jit_compile(self, fn):
        # ONNX uses InferenceSession, not JIT compilation
        # Return function that creates ONNX session on first call
        return fn

    def create_thunk_inputs(self, storage_map):
        thunk_inputs = []
        for inp in self.fgraph.inputs:
            sinput = storage_map[inp]
            thunk_inputs.append(sinput)
        return thunk_inputs
```

**File 2**: `pytensor/link/onnx/dispatch/basic.py`

```python
from functools import singledispatch
import onnx
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.link.utils import fgraph_to_python

@singledispatch
def onnx_funcify(op, node=None, storage_map=None, **kwargs):
    """Create ONNX-compatible function from PyTensor Op."""
    raise NotImplementedError(f"No ONNX conversion for Op: {op}")

@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(fgraph, **kwargs):
    return fgraph_to_python(
        fgraph,
        onnx_funcify,
        type_conversion_fn=onnx_typify,
        fgraph_name="onnx_funcified_fgraph",
        **kwargs,
    )

@singledispatch
def onnx_typify(data, **kwargs):
    """Convert data to ONNX-compatible format."""
    import numpy as np
    return np.asarray(data)
```

**File 3**: `pytensor/link/onnx/__init__.py`

```python
from pytensor.link.onnx.linker import ONNXLinker

__all__ = ["ONNXLinker"]
```

#### Phase 4: Implement Operation Dispatchers

**File 4**: `pytensor/link/onnx/dispatch/elemwise.py`

```python
from pytensor.tensor.elemwise import Elemwise
from pytensor.link.onnx.dispatch.basic import onnx_funcify

@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op
    base_fn = onnx_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        return base_fn(*inputs)
    return elemwise_fn
```

**File 5**: `pytensor/link/onnx/dispatch/math.py`

```python
from pytensor.tensor.math import Dot, Add, Mul
from pytensor.link.onnx.dispatch.basic import onnx_funcify
import onnxruntime as ort

@onnx_funcify.register(Add)
def onnx_funcify_Add(op, **kwargs):
    def add(x, y):
        # TODO: Generate ONNX Add node
        return x + y
    return add

@onnx_funcify.register(Dot)
def onnx_funcify_Dot(op, **kwargs):
    def dot(x, y):
        # TODO: Generate ONNX MatMul node
        return x @ y
    return dot
```

#### Phase 5: Create Test Suite

**File 6**: `tests/link/onnx/test_basic.py`

```python
import pytest
import numpy as np
from pytensor import config, function
from pytensor.compile.mode import Mode
from pytensor.scalar import ScalarType
from pytensor.tensor import dscalar, vector, matrix
from pytensor.link.onnx.linker import ONNXLinker

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

# Configure modes
onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
py_mode = Mode(linker="py", optimizer=None)

def compare_onnx_and_py(
    graph_inputs,
    graph_outputs,
    test_inputs,
    *,
    assert_fn=None,
):
    """Compare ONNX and Python backend outputs."""
    if assert_fn is None:
        assert_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-4)

    # Compile with ONNX backend
    pytensor_onnx_fn = function(graph_inputs, graph_outputs, mode=onnx_mode)
    onnx_res = pytensor_onnx_fn(*test_inputs)

    # Compile with Python mode
    pytensor_py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    # Compare
    if isinstance(graph_outputs, (list, tuple)):
        for o, p in zip(onnx_res, py_res):
            assert_fn(o, p)
    else:
        assert_fn(onnx_res, py_res)

    return pytensor_onnx_fn, onnx_res

def test_onnx_scalar_add():
    """Test basic scalar addition."""
    a = dscalar("a")
    b = dscalar("b")
    c = a + b

    compare_onnx_and_py(
        [a, b],
        [c],
        [np.array(2.0, dtype=config.floatX), np.array(3.0, dtype=config.floatX)]
    )

def test_onnx_vector_operations():
    """Test vector operations."""
    x = vector("x")
    y = x * 2 + 1

    compare_onnx_and_py(
        [x],
        [y],
        [np.array([1.0, 2.0, 3.0], dtype=config.floatX)]
    )
```

**File 7**: `tests/link/onnx/__init__.py` (empty file)

#### Phase 6: Run Tests

```bash
# Run ONNX backend tests
uv run pytest tests/link/onnx/test_basic.py -v

# Run with verbose output for debugging
uv run pytest tests/link/onnx/test_basic.py -vv -s

# Run with coverage
uv run pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term-missing
```

#### Phase 7: Iterate on Operations

Follow the dispatch registration pattern for each operation category:

1. **Elemwise**: `dispatch/elemwise.py`
2. **Math**: `dispatch/math.py`
3. **BLAS**: `dispatch/blas.py`
4. **Blockwise**: `dispatch/blockwise.py`
5. **Random**: `dispatch/random.py`
6. **Shape**: `dispatch/shape.py`
7. **Subtensor**: `dispatch/subtensor.py`
8. **Linear Algebra**: `dispatch/nlinalg.py`, `dispatch/slinalg.py`

For each operation:
```python
@onnx_funcify.register(OpClass)
def onnx_funcify_OpClass(op, node, **kwargs):
    def implementation(*inputs):
        # Convert to ONNX node/operation
        return result
    return implementation
```

## Code References

### Backend Architecture
- `pytensor/link/basic.py:576-717` - JITLinker base class definition
- `pytensor/link/jax/linker.py:9-127` - JAXLinker implementation example
- `pytensor/link/jax/dispatch/basic.py:27-151` - jax_funcify dispatcher pattern
- `pytensor/link/utils.py:666-765` - fgraph_to_python() core compiler
- `pytensor/link/jax/ops.py:16-196` - JAXOp wrapper with VJP gradients

### Test Patterns
- `tests/link/jax/test_basic.py:36-96` - compare_jax_and_py() comparison function
- `tests/link/jax/test_basic.py:22-33` - Backend mode configuration
- `tests/link/numba/test_elemwise.py:34-124` - Parametrized test pattern
- `tests/link/numba/test_basic.py:172-256` - Numba object mode testing
- `tests/link/jax/test_wrap_jax.py` - Custom operator wrapper tests

### Project Configuration
- `pyproject.toml:119-122` - Pytest configuration
- `pyproject.toml:48-82` - Project dependencies and optional extras

## Architecture Insights

### Backend Design Principles

1. **Separation of Concerns**
   - Linker handles compilation pipeline
   - Dispatcher handles operation translation
   - Graph compiler provides generic traversal

2. **Singledispatch Pattern**
   - Type-based dispatch using `@register(OpClass)`
   - Composable: ops can dispatch to other ops
   - Extensible: new ops just register implementations

3. **Test-First Development**
   - Comparison testing validates correctness
   - Backend mode fixtures isolate testing
   - Parametrized tests cover edge cases

4. **Type Conversion**
   - `typify` functions handle framework-specific types
   - Storage map manages value lifetimes
   - Containers provide type filtering

### Key Challenges for ONNX

1. **Static Graphs**: ONNX uses static computation graphs, unlike JAX/PyTorch
2. **Type Inference**: ONNX requires explicit shape/dtype information
3. **Execution Model**: ONNX uses InferenceSession, not JIT compilation
4. **Operation Coverage**: ONNX has different operation set than NumPy/JAX
5. **Gradient Computation**: Need to handle both forward and backward pass

## Historical Context (from thoughts/)

### Production Roadmap
- `thoughts/shared/research/2025-11-04_11-34-58_onnx-backend-production-roadmap.md`
  - Comprehensive roadmap for ONNX backend production deployment
  - Defines tier 1 (critical) and tier 2 (correctness) priorities

### Implementation Strategy
- `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md`
  - Core architectural decisions
  - Operation prioritization strategy

### Gap Analysis
- `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md`
  - Detailed coverage of operations needed
  - Identifies missing implementations

- `thoughts/shared/research/2025-10-15_00-05-01_onnx-cnn-gap-analysis.md`
  - CNN-specific operations analysis
  - Conv2D, MaxPool, BatchNorm patterns

### Testing Strategy
- `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`
  - Property-based testing approach using Hypothesis
  - Automated test generation strategy

## Related Research

- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - General backend addition guide
- `thoughts/shared/research/2025-10-15_onnx-backend-webassembly.md` - WebAssembly deployment strategy
- `thoughts/shared/research/2025-10-15_07-28-53_gpu-training-support.md` - GPU training architecture
- `thoughts/shared/plans/onnx-conv2d-tdd.md` - Conv2D TDD plan

## Quick Start Commands

### Setup Development Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
cd /home/clsandoval/cs/pytensor-workshop-demo
uv sync --all-extras

# Install ONNX dependencies
uv pip install onnx onnxruntime

# Verify installation
uv run python -c "import pytensor; import onnx; print('OK')"
```

### Run Backend Tests

```bash
# Run specific backend tests
uv run pytest tests/link/jax/ -v          # JAX backend tests
uv run pytest tests/link/numba/ -v        # Numba backend tests

# Run ONNX tests (once implemented)
uv run pytest tests/link/onnx/ -v

# Run with coverage
uv run pytest tests/link/onnx/ --cov=pytensor.link.onnx
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/onnx-backend-implementation

# 2. Make changes to pytensor/link/onnx/

# 3. Run tests
uv run pytest tests/link/onnx/ -v

# 4. Run pre-commit checks
uv run pre-commit run --all-files

# 5. Commit changes
git add .
git commit -m "Add ONNX backend dispatcher for elemwise ops"
```

## Next Steps

1. **Immediate Actions**
   - Review existing ONNX research documents in `thoughts/`
   - Set up development environment with `uv sync`
   - Run existing backend tests to understand patterns

2. **Implementation Priorities** (from ONNX roadmap)
   - **Tier 1**: Critical path operations (Add, Mul, Dot, Conv2D)
   - **Tier 2**: Correctness improvements (proper shape inference)
   - **Tier 3**: Advanced features (gradient computation, optimization)

3. **Testing Strategy**
   - Start with comparison tests (ONNX vs Python mode)
   - Add parametrized tests for edge cases
   - Consider property-based testing with Hypothesis

4. **Documentation**
   - Document ONNX-specific limitations
   - Create operation support matrix
   - Write integration examples

## Open Questions

1. **ONNX Graph Construction**: Should we build the ONNX graph incrementally or all at once?
2. **Gradient Support**: How should we handle automatic differentiation in ONNX?
3. **Dynamic Shapes**: How to handle PyTensor's dynamic shapes in ONNX's static graph?
4. **Optimization**: Should we apply ONNX Runtime optimizations or rely on PyTensor's optimizer?
5. **Backend Selection**: Should ONNX backend support multiple execution providers (CPU, CUDA, TensorRT)?
