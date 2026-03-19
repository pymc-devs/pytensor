# PyTensor Agent Instructions

## Overview

**PyTensor**: Python library for defining, optimizing, and evaluating mathematical expressions with multi-dimensional arrays. Focus on hackable graph analysis and manipulation. Supports C, JAX, and Numba compilation backends. ~27MB, 492 Python files, Python support as per numpy NEP 29, uses NumPy, SciPy, pytest.

## PyTensor Design Principles

Graph manipulation in Python, graph evaluation out of Python.
Emulate NumPy user-facing API as much as possible.

### API Differences from NumPy

1. **Lazy evaluation**: Expressions are symbolic until `pytensor.function()` compiles or `.eval()` evaluates
2. **Pure semantics**: `new_x = x[idx].set(y)` instead of `x[idx] = y`
3. **Immutable/hashable**: PyTensor variables are hashable. `a == b` tests identity (`a is b`), not elementwise equality.
4. **Static shapes**: Broadcasting requires static shape of 1. Valid: `pt.add(pt.vector("x", shape=(1,)), pt.vector("y"))`. Invalid: `pt.add(pt.vector("x", shape=(None,)), pt.vector("y"))` with x.shape=1.
5. **Static rank and type**. PyTensor functions accepts variables with a specific dtype and number of dimensions. Length of each dimension can be static or dynamic.

## Code Style

**Uses pre-commit with ruff**. Code should pass pre-commit before being committed.

**Performance**
* Could should be performant
* Avoid expensive work in hot loops
* Avoid redundant checks. Let errors raise naturally
* In contrast, silent errors should be prevented

**Comments**: Should be used sparingly, only for complex logic

**Testing**: Should be succinct
 - Prefer `tests.unittest_tools.assert_equal_computations` over numerical evaluation
 - Test multiple inputs on one compiled function vs multiple compilations
 - Minimize test conditions. Be smart, not fearful
 - Integrate with similar existing tests

## Repository Structure

### Root
- `.github/` (workflows),
- `doc/` (docs)
- `pyproject.toml` (config),
- `setup.py` (Cython build),
- `conftest.py` (pytest config),
- `environment.yml` (conda env)

### Source (`pytensor/`)
- `configdefaults.py`: Config system (floatX, mode)
- `gradient.py`: Auto-differentiation
- `compile/`: Function compilation
- `graph/`: IR and optimization (`graph/rewriting/`)
- `link/`: Backends (`c/`, `jax/`, `numba/`, `mlx/`, `pytorch/`)
- `tensor/`: Tensor ops (largest module, subdirs: `random/`, `rewriting/`, `conv/`)
- `scalar/`: Scalar ops
- `scan/`: Loop operations (`scan_perform.pyx` Cython)
- `sparse/`: Sparse tensors
- `xtensor/` Tensor Ops with dimensions (lowers to Tensor ops)

### Tests (`tests/`)
Mirrors source structure. `unittest_tools.py` has testing utilities.

## Testing & Building

```bash
python -m pytest tests/                        # All tests
python -m pytest tests/test_updates.py -v      # Single file
python -m pytest tests/ --runslow              # Include slow tests
```

Tests are run with `config.mode = "FAST_COMPILE"`. If needed:
- Cast numerical values `test_value.astype(symbolic_var.type.dtype)`
- Use custom function mode `get_default_mode().excluding("fusion")` or skip tests in `FAST_COMPILE` if they are not directly relevant to the mode.

Alternative backends (JAX, PyTorch, MLX) are optional. Use `pytest.importorskip` to fail gracefully.

### MyPy

```bash
python ./scripts/run_mypy.py --verbose
```
**PyTensor incompatible with strict mypy**. Type-hints are for users/developers not to appease mypy. Liberal `type: ignore[rule]` and file exclusions are acceptable.

### Documentation

```bash
python -m sphinx -b html ./doc ./html  # Build docs (2-3 min)
```
**Never commit `html` directory**.


## Debugging

### Inspecting graphs

Use `pytensor.dprint` to inspect graphs. It works on both raw variables (before optimization) and compiled functions (after optimization):

```python
pytensor.dprint(y, print_type=True)                          # Before optimization
pytensor.dprint(f, print_type=True, print_memory_map=True)   # After optimization
```

`print_type=True` shows the type and shape of each variable. `print_memory_map=True` shows memory allocation labels, useful for spotting whether intermediates share memory.

### Rewriting without compiling

Use `rewrite_graph` to apply rewrites to a graph without the full `pytensor.function` compilation:

```python
from pytensor.graph.rewriting.utils import rewrite_graph
y_opt = rewrite_graph(y, include=("canonicalize", "specialize"))
pytensor.dprint(y_opt, print_type=True)
```

### Inspecting rewrites

Use `optimizer_verbose=True` to see which rewrites are applied during compilation:

```python
with pytensor.config.change_flags(optimizer_verbose=True):
    f = pytensor.function([x], y)
```

This prints each rewrite that fires, showing what it replaced and with what.


## CI/CD Pipeline

### Workflows (`.github/workflows/`)
1. **test.yml**: Main suite - Several Python versions, 7 test parts + backend jobs (jax, torch)
2. **mypy.yml**: Type checking

CI runs tests under three modes: Default (`"NUMBA"`), `"CVM"`, and `"FAST_COMPILE"`. Tests must pass in all three.


## Known Gotchas

- **Numba scalar outputs**: Numba-compiled scalar functions return Python `float`/`int`, not NumPy scalars. Keep this in mind when writing tests that check output types.
