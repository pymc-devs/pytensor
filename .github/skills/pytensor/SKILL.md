---
name: pytensor
description: Use when writing PyTensor code, building computational graphs, using pytensor.scan, computing gradients with pytensor.grad, creating custom Ops, compiling to JAX/Numba/C/MLX backends, using named dimensions with xtensor, integrating PyTensor with PyMC models, or debugging PyTensor graphs. Covers tensor creation, automatic differentiation, scan loops, graph manipulation with clone_replace, shared variables, compilation modes, named-dimension xtensors, and PyMC distribution architecture.
---

# PyTensor Programming Guide

PyTensor is a symbolic tensor computation library that builds a **static computational graph** which can be inspected, rewritten, and compiled to C, JAX, Numba, PyTorch, or MLX backends.

## Quick Start

```python
import pytensor
import pytensor.tensor as pt
import numpy as np

x = pt.dvector("x")
y = pt.sum(x ** 2)
gy = pytensor.grad(y, x)
f = pytensor.function([x], [y, gy])

val, grad = f([1.0, 2.0, 3.0])
# val = 14.0, grad = [2.0, 4.0, 6.0]
```

## Core Concepts

The graph is a **bipartite DAG** of four abstractions: **Type** (dtype, shape constraints — primary: `TensorType`), **Variable** (data nodes — `TensorVariable`, `TensorConstant`, `TensorSharedVariable`), **Op** (computation — `make_node()`, `perform()`, `grad()`), and **Apply** (Op applied to specific inputs producing outputs).

```python
y = x ** 2 + 3 * x
y.owner              # Apply node
y.owner.op           # Elemwise{add}
y.owner.inputs       # [x**2, 3*x]
```

Key imports: `import pytensor.tensor as pt` for tensor ops, `import pytensor` for `grad`, `function`, `shared`, `scan`.

For full API (creation, math, shapes, linalg, random, shared variables): See [reference/api-reference.md](reference/api-reference.md)

## Automatic Differentiation

Symbolic reverse-mode AD. First argument must be **scalar**:

```python
gy = pytensor.grad(y, x)                          # single
gW, gb = pytensor.grad(loss, [W, b])              # multiple
grad = pytensor.grad(pt.sum(vector_expr), x)       # reduce non-scalar first
```

Higher-order: `jacobian`, `Rop` (JVP), `Lop` (VJP), `hessian_vector_product` in `pytensor.gradient`.

For full AD reference: See [reference/api-reference.md](reference/api-reference.md)

## Scan — Loops and Recurrence

`pytensor.scan()` expresses loops in the static graph. The `fn` receives arguments in strict order: **sequences → recurrent output taps (oldest first) → non-sequences**.

```python
results, updates = pytensor.scan(
    fn=lambda x_t, y_tm1: y_tm1 + x_t,
    sequences=[x],
    outputs_info=[np.float64(0)]
)
```

**Always pass `updates` to `pytensor.function`** (required for RNG state). Prefer vectorized ops when no recurrence exists.

For RNN, Kalman, AR, power iteration, while loops, scan_checkpoints: See [reference/scan-patterns.md](reference/scan-patterns.md)

## Compilation

```python
f = pytensor.function([x], y)                      # Default FAST_RUN (C)
f = pytensor.function([x], y, mode="JAX")           # GPU/TPU
f = pytensor.function([x], y, mode="NUMBA")         # Numba JIT CPU
f = pytensor.function([x], y, mode="PYTORCH")       # PyTorch interop
f = pytensor.function([x], y, mode="MLX")           # Apple Silicon
f = pytensor.function([x], y, mode="FAST_COMPILE")  # Quick testing
```

Shared variable updates: `pytensor.function([x], loss, updates=[(W, W - lr * gW)])`.

For pipeline details, graph rewrites, backend internals, config: See [reference/compilation.md](reference/compilation.md)

## Graph Manipulation

Swap variables in the graph without re-building — a unique power of PyTensor:

```python
from pytensor.graph.replace import clone_replace

y_new = clone_replace(y, replace={x: z + 1})
```

## XTensor — Named Dimensions (Experimental)

`pytensor.xtensor` provides **xarray-like named dimensions**. Operations use dim names instead of axis indices; broadcasting/alignment is automatic. XOps are lowered to tensor ops before compilation, so all backends work.

```python
import pytensor.xtensor as px

x = px.xtensor("x", dims=("time", "feature"), shape=(100, 3))
x.sum(dim="time")                 # reduce by name
x.isel(time=0)                    # index by name
x.transpose("feature", "time")    # reorder by name
x.values                          # → plain TensorVariable
```

For full xtensor API (creation, reductions, dot, indexing, stack, linalg, random, lowering): See [reference/xtensor.md](reference/xtensor.md)

## PyMC Integration

Inside `pm.Model()`, every distribution creates a symbolic `TensorVariable`. Use **coords/dims** for labeled InferenceData and **nutpie** as default sampler:

```python
import pymc as pm

with pm.Model(coords={"features": feature_names}) as model:
    beta = pm.Normal("beta", 0, 1, dims="features")
    mu = pt.dot(X, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    idata = pm.sample(nuts_sampler="nutpie", random_seed=42)
```

**Critical gotcha**: Use `pt.switch(cond, a, b)`, not Python `if/else` — Python conditionals evaluate once at graph construction, not per sample.

For distributions, compile_logp, transforms, CustomDist, hierarchical patterns, time series, debugging: See [reference/pymc-integration.md](reference/pymc-integration.md)

## Debugging

```python
y.dprint()                                          # graph structure
pytensor.dprint(compiled_fn)                        # optimized graph
from pytensor.printing import Print
x_debug = Print("x value")(x)                      # trace values inside compiled fns
f = pytensor.function([x], y, mode="DebugMode")    # full validation
f = pytensor.function([x], y, mode="NanGuardMode") # NaN hunting
```

For MonitorMode, test values, profiling, d3viz, function_dump: See [reference/debugging.md](reference/debugging.md)

## Best Practices

| Practice | Why |
|---|---|
| Prefer vectorized ops over `scan` | Scan has compile+runtime overhead |
| Use `pt.specify_shape()` | Better shape inference enables more rewrites |
| Use `borrow=True` for large shared variables | Avoids memory copies |
| Always pass `updates` from scan | Required for RNG state management |
| Use `pytensor.function()` not `.eval()` | `.eval()` doesn't update RNG state |
| Use `pt.as_tensor_variable()` for Python inputs | Ensures proper type wrapping |
| Use `pt.switch`, not Python `if/else` | Python conditionals evaluate at graph-build time |

## Key Differences from PyTorch/TensorFlow

| Feature | PyTensor | PyTorch/TF |
|---|---|---|
| Graph type | Static (built then compiled) | Eager by default |
| AD mechanism | Symbolic (graph-to-graph) | Tape-based (runtime) |
| `clone_replace` | First-class variable substitution | No equivalent |
| Primary use | Probabilistic programming IR | Deep learning |
| Backends | C, JAX, Numba, PyTorch, MLX | CUDA/CPU, XLA |

## Reference Documentation

- **Full API** (creation, math, shapes, linalg, random, AD): See [reference/api-reference.md](reference/api-reference.md)
- **XTensor / Named Dimensions** (creation, reductions, indexing, dot, lowering): See [reference/xtensor.md](reference/xtensor.md)
- **Scan patterns** (RNN, Kalman, AR, power iteration, checkpoints): See [reference/scan-patterns.md](reference/scan-patterns.md)
- **PyMC integration** (distributions, transforms, CustomDist, hierarchical, time series): See [reference/pymc-integration.md](reference/pymc-integration.md)
- **Debugging** (DebugMode, NanGuardMode, MonitorMode, profiling): See [reference/debugging.md](reference/debugging.md)
- **Custom Ops and backends** (creating Ops, gradients, backend dispatch): See [reference/custom-ops-and-backends.md](reference/custom-ops-and-backends.md)
- **Compilation pipeline** (FunctionGraph, rewrite phases, linkers, modes): See [reference/compilation.md](reference/compilation.md)

## External Documentation

| Topic | URL |
|---|---|
| PyTensor Docs | https://pytensor.readthedocs.io/en/latest/ |
| GitHub | https://github.com/pymc-devs/pytensor |
| PyMC + PyTensor | https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html |
| Scan Tutorial | https://pytensor.readthedocs.io/en/latest/gallery/scan/scan_tutorial.html |
| Graph Rewriting | https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html |
