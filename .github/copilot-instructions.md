# PyTensor Copilot Instructions

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

**Uses pre-commit with ruff**

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

## Critical: Environment & Commands

**ALWAYS use micromamba environment**: PyTensor is pre-installed as editable in `.github/workflows/copilot-setup-steps.yml`. 

All commands MUST use: `micromamba run -n pytensor-test <command>`

Example: `micromamba run -n pytensor-test python -c 'import pytensor; print(pytensor.__version__)'`

## Testing & Building

### Running Tests (ALWAYS use micromamba)

```bash
micromamba run -n pytensor-test python -m pytest tests/              # All tests
micromamba run -n pytensor-test python -m pytest tests/test_updates.py -v  # Single file
micromamba run -n pytensor-test python -m pytest tests/ --runslow    # Include slow tests
```

Tests are run with `config.floatX == "float32"` and `config.mode = "FAST_COMPILE"`. If needed:
- Cast numerical values `test_value.astype(symbolic_var.type.dtype)` 
- Use custom function mode `get_default_mode().excluding("fusion")` or skip tests in `FAST_COMPILE` if they are not directly relevant to the mode.

Alternative backends (JAX, NUMBA, ...) are optional. Use `pytest.importorskip` to fail gracefully.

### Pre-commit

```bash
micromamba run -n pytensor-test pre-commit
```

### MyPy

```bash
micromamba run -n pytensor-test python ./scripts/run_mypy.py --verbose
```
**PyTensor incompatible with strict mypy**. Type-hints are for users/developers not to appease mypy. Liberal `type: ignore[rule]` and file exclusions are acceptable.

### Documentation

```bash
micromamba run -n pytensor-test python -m sphinx -b html ./doc ./html  # Build docs (2-3 min)
```
**Never commit `html` directory**.


## CI/CD Pipeline

### Workflows (`.github/workflows/`)
1. **test.yml**: Main suite - Several Python versions, fast-compile (0/1), float32 (0/1), 7 test parts + backend jobs (numba, jax, torch)
2. **mypy.yml**: Type checking
3. **copilot-setup-steps.yml**: Environment setup


## Trust These Instructions
These instructions are comprehensive and tested. Only search for additional information if:
1. Instructions are incomplete for your specific task
2. Instructions are found to be incorrect
3. You need deeper understanding of an implementation detail
 
 For most coding tasks, these instructions provide everything needed to build, test, and validate changes efficiently.
