# Compilation Pipeline

How `pytensor.function()` transforms a symbolic graph into an executable function. After reading this you can: understand the three-stage pipeline (FunctionGraph construction → graph rewrites → linking), choose the right compilation mode (`FAST_RUN`, `JAX`, `NUMBA`, `PYTORCH`, `MLX`, `FAST_COMPILE`, `DebugMode`), leverage automatic rewrites (constant folding, BLAS specialization, numerical stability transforms like `log1p`), configure PyTensor via environment variables or `.pytensorrc`, and understand how each backend (C, JAX, Numba, PyTorch, MLX) generates code through `singledispatch`.

## Contents
- Three-stage pipeline overview
- Stage 1: FunctionGraph construction
- Stage 2: Graph rewrites (optimization)
- Stage 3: Linking (code generation)
- Compilation modes
- Key automatic rewrites
- Configuration options

## Three-Stage Pipeline

Triggered by `pytensor.function(inputs, outputs, mode=..., updates=...)`:

```
Source graph → FunctionGraph → Graph Rewrites → Linking → Compiled function
```

## Stage 1: FunctionGraph Construction

The subgraph from user-specified inputs/outputs is wrapped in a `FunctionGraph`:

- Registers callback hooks for graph modifications
- `FunctionGraph.replace()` for variable substitution
- Attaches `Feature` objects (e.g., immutability constraints on inputs)

## Stage 2: Graph Rewrites

A rewriter from the compilation Mode is applied via `GraphRewriter.rewrite()`. The rewrite system uses **`optdb`** (a `SequenceDB`) that orders rewrites:

| Order | Phase | Description |
|:---:|---|---|
| 0 | **merge1** | Initial deduplication of identical subgraphs |
| 1 | **canonicalize** | Replace esoteric ops with elementary equivalents; fold constants; canonical ordering |
| 2 | **specialize** | Replace general ops with optimized special cases |
| 49 | **merge2** | Mid-stage deduplication |
| 49.5 | **add_destroy_handler** | Marker — inplace ops must register at position >= 50 |
| 100 | **merge3** | Final deduplication |

Both `canonicalize` and `specialize` use `EquilibriumGraphRewriter` — iteratively applies rewrites until fixed-point.

## Key Automatic Rewrites (FAST_RUN)

| Category | Examples |
|---|---|
| Algebraic simplification | `b * (a/b)` → `a`, `pow(x,0)` → `1`, `pow(x,1)` → `x` |
| Numerical stability | `log(1+x)` → `log1p(x)`, `log(softmax(x))` → `log_softmax(x)` |
| BLAS specialization | `dot(matrix, matrix)` → `dot22` (GEMM call) |
| Elemwise fusion | Chain of elementwise ops fused into single pass |
| Constant folding | All-constant expressions pre-computed at compile time |
| Inplace operations | Intermediate buffers reused where safe |
| Shape optimization | When only `.shape` is needed, skip actual computation |
| Merge | Duplicate subgraphs computed once |

## Stage 3: Linking (Code Generation)

The linker produces:
1. A **thunk** — no-argument callable that performs the computation
2. **Input/output containers** — typed wrappers for data

The linker calls `FunctionGraph.toposort()` for execution ordering. Compiled C libraries are cached in `~/.pytensor/compiledir_xxx` with a lock mechanism to prevent parallel compilation conflicts.

## Compilation Modes

| Mode | Optimizer | Linker | Use Case |
|---|---|---|---|
| `FAST_RUN` (default) | Full optimization | C + Python fallback | Production |
| `FAST_COMPILE` | Minimal optimization | Python only | Quick testing |
| `JAX` | JAX-specific rewrites | JAX transpilation | GPU/TPU execution |
| `NUMBA` | Numba-specific rewrites | Numba JIT | Fast CPU without C |
| `PYTORCH` | PyTorch-specific rewrites | PyTorch transpilation | PyTorch ecosystem |
| `MLX` | MLX-specific rewrites | MLX dispatch | Apple Silicon |
| `DebugMode` | Full + validation | Debug linker | Development/debugging |
| `NanGuardMode` | Full optimization | C + NaN checking | NaN hunting |

### Usage

```python
f = pytensor.function([x], y)                      # Default FAST_RUN
f = pytensor.function([x], y, mode="JAX")           # JAX backend
f = pytensor.function([x], y, mode="FAST_COMPILE")  # Quick compile
f = pytensor.function([x], y, mode="PYTORCH")       # PyTorch backend
f = pytensor.function([x], y, mode="MLX")           # MLX (Apple Silicon)
f = pytensor.function([x], y, mode="DebugMode")     # Full validation
f = pytensor.function([x], y, mode="NanGuardMode")  # NaN hunting
```

## Backend-Specific Behavior

### JAX Backend
- Excludes fusion/inplace rewrites (JAX handles these internally)
- Requires static shapes for JIT compilation
- Transpiles via `jax_funcify()` singledispatch

### Numba Backend
- Transpiles via `numba_funcify()` singledispatch
- Near-complete coverage for PyMC workloads
- Fast CPU without C compilation overhead

### MLX Backend
- Targets Apple Silicon (M-series chips)
- Transpiles via `mlx_funcify()` singledispatch
- Covers math, elemwise, shape, subtensor, linalg, blockwise, signal

### C Backend
- Generates C code per Op
- Compiled libraries cached on disk
- Leverages BLAS/LAPACK for matrix operations

## Configuration

### Environment Variables

```bash
export PYTENSOR_FLAGS='device=cpu,floatX=float64,optimizer=fast_run'
```

### .pytensorrc File

```ini
[global]
device = cpu
floatX = float64

[mode]
optimizer = fast_run
```

### Programmatic

```python
pytensor.config.floatX = "float64"

# Temporary change
with pytensor.config.change_flags(mode="FAST_COMPILE"):
    f = pytensor.function([x], y)
```

## Module Layout

| Module | Purpose |
|---|---|
| `pytensor/graph/` | Core graph framework: Variable, Apply, Op, Type, FunctionGraph |
| `pytensor/tensor/` | Symbolic NumPy: tensor types, elementwise ops, linalg, random |
| `pytensor/scalar/` | Symbolic scalar types underlying elementwise ops |
| `pytensor/compile/` | Compilation pipeline: `pytensor.function()`, modes |
| `pytensor/link/` | Backend linkers: `link/c/`, `link/jax/`, `link/numba/`, `link/pytorch/`, `link/mlx/` |
| `pytensor/scan/` | The Scan Op — loops/recurrence |
| `pytensor/xtensor/` | Named-dimension tensors (xarray-like, experimental) |
| `pytensor/sparse/` | Symbolic sparse matrices |
| `pytensor/printing/` | Graph pretty-printing |

## External Docs

| Topic | URL |
|---|---|
| Pipeline | https://pytensor.readthedocs.io/en/latest/extending/pipeline.html |
| Graph Rewriting | https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html |
| Modes | https://pytensor.readthedocs.io/en/latest/library/compile/mode.html |
| Optimizations | https://pytensor.readthedocs.io/en/latest/optimizations.html |
| Graph Structures | https://pytensor.readthedocs.io/en/latest/extending/graphstructures.html |
| Configuration | https://pytensor.readthedocs.io/en/latest/tutorial/modes.html |
