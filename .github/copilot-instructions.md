# PyTensor Copilot Instructions

## Repository Overview

PyTensor is a Python library for defining, optimizing, and efficiently evaluating mathematical expressions involving multi-dimensional arrays. It is the computational backend for PyMC and provides graph-based symbolic computation with compilation to C, JAX, and Numba backends.

**Repository Stats:**
- Size: ~21MB source code
- Languages: Python (primary), C, Cython (.pyx)
- Test Files: 217 test files across multiple modules
- Python Version: 3.10-3.13 (requires >=3.10, <3.14)

## Environment Setup

### Prerequisites
**ALWAYS** use conda/mamba for environment management. pip-only installations may fail or be incomplete.

### Initial Setup (Required Steps)
```bash
# Clone and navigate to repository
git clone git@github.com:<your-username>/pytensor.git
cd pytensor

# Create environment - use the existing environment.yml
conda env create -f environment.yml
conda activate pytensor-dev

# Install pre-commit hooks (REQUIRED before committing)
pre-commit install

# Install package in editable mode
pip install -e .
```

**Critical:** Always activate `pytensor-dev` environment before running any commands. Many dependencies (especially BLAS libraries like MKL) are only available via conda.

### Verification
After setup, verify the installation:
```bash
python -c "import pytensor; print(pytensor.__version__)"
python -c "import pytensor; print(pytensor.config.__str__(print_doc=False))"
# Verify Cython extension compiled:
python -c "from pytensor.scan import scan_perform; print(scan_perform.get_version())"
```

## Build Process

### Compilation
PyTensor includes a Cython extension that must be compiled:
- **Source:** `pytensor/scan/scan_perform.pyx`
- **Compilation:** Happens automatically during `pip install -e .`
- **Output:** `pytensor/scan/scan_perform.so` (or `.pyd` on Windows)

The C compilation uses numpy headers and compiles to native code. If you modify `scan_perform.pyx`, reinstall with `pip install -e .` to recompile.

### Pre-commit Hooks
Pre-commit runs automatically on `git commit`. It includes:
- **ruff**: Code linting and formatting (auto-fixes most issues)
- **debug-statements**: Checks for leftover debug code (excludes specific files like breakpoint.py)
- **check-merge-conflict**: Ensures no merge conflict markers
- **sphinx-lint**: Documentation linting

To run manually:
```bash
pre-commit run --all-files
```

## Testing

### Quick Test
```bash
pytest
```

### Comprehensive Testing
Tests are organized by module. Run specific test suites:
```bash
# Run specific module tests
pytest tests/tensor/
pytest tests/scan/
pytest tests/link/numba/  # Requires numba
pytest tests/link/jax/    # Requires jax

# Run with slow tests (some tests marked @pytest.mark.slow)
pytest --runslow

# Run doctests
pytest --doctest-modules pytensor --ignore=pytensor/misc/check_duplicate_key.py --ignore=pytensor/link --ignore=pytensor/ipython.py
```

### Test Configuration
- **conftest.py**: Sets `PYTENSOR_FLAGS` environment variables for testing
- **pyproject.toml**: pytest configuration with test discovery paths and options
- **Duration reporting**: Tests show the 50 slowest tests by default

### Running Tests Like CI
The CI test matrix is extensive. Key test patterns:
```bash
# Standard test configuration
export PYTENSOR_FLAGS=warn__ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,gcc__cxxflags=-pipe
pytest -r A --verbose --runslow --durations=50 --cov=pytensor/ tests/

# Fast compile mode (faster feedback, less optimization)
export PYTENSOR_FLAGS=mode=FAST_COMPILE
pytest tests/

# Float32 mode (tests precision handling)
export PYTENSOR_FLAGS=floatX=float32
pytest tests/
```

**Important**: CI tests take 5-15 minutes per job. Run locally first on relevant test subsets.

## Linting and Type Checking

### Ruff (Primary Linter)
```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Configuration in `pyproject.toml`:
- Line length: 88 characters
- Import sorting with 2 lines after imports
- Various rule sets enabled (F, E, W, UP, RUF, PERF, etc.)

### MyPy (Type Checking)
```bash
python scripts/run_mypy.py --verbose
```

MyPy is configured to allow known failing files (listed in `scripts/mypy-failing.txt`). The script compares current results against the known failures. If new files fail, the CI will fail. If known failing files now pass, update `mypy-failing.txt`.

## Documentation

### Building Documentation
```bash
# Build HTML documentation
python -m sphinx -b html ./doc ./html

# View locally
cd html
python -m http.server
# Navigate to http://localhost:8000
```

**Do NOT commit the `html` directory** - documentation is built automatically by ReadTheDocs.

### Documentation Dependencies
The main `environment.yml` includes doc dependencies, or use:
```bash
conda env create -f doc/environment.yml
conda activate pytensor-docs
```

## Project Structure

### Core Directories
```
pytensor/
├── compile/          # Compilation and function creation
├── graph/            # Graph data structures and optimization framework
├── link/             # Backends (C, JAX, Numba, PyTorch)
│   ├── c/           # C backend (default, most mature)
│   ├── jax/         # JAX backend
│   ├── numba/       # Numba backend
│   └── pytorch/     # PyTorch backend
├── scalar/           # Scalar operations and types
├── scan/             # Scan operation (includes Cython extension)
├── tensor/           # Tensor operations (largest module)
│   ├── rewriting/   # Tensor graph optimizations
│   └── *.py         # Core tensor ops (basic.py, elemwise.py, etc.)
├── sparse/           # Sparse matrix support
└── xtensor/          # xarray integration

tests/               # Mirrors pytensor/ structure
├── tensor/          # Largest test suite
├── scan/
├── link/
└── ...
```

### Key Files
- **pyproject.toml**: Project metadata, dependencies, tool configurations
- **setup.py**: Build configuration (Cython extension)
- **conftest.py**: pytest configuration and fixtures
- **.pre-commit-config.yaml**: Pre-commit hook configuration
- **environment.yml**: Conda environment specification

### Configuration Files
- **pyproject.toml**: ruff, mypy, pytest, coverage configuration
- **codecov.yml**: Code coverage requirements (70-100% range)
- **.readthedocs.yaml**: ReadTheDocs build configuration

## CI/CD Workflows

### Main Test Workflow (`.github/workflows/test.yml`)
Runs on every PR and push to main:
1. **Style Check**: Runs pre-commit on Python 3.10 and 3.13
2. **Test Matrix**: Tests multiple configurations:
   - OS: Ubuntu (primary), macOS (limited)
   - Python: 3.10, 3.13
   - NumPy: ~=1.26.0, >=2.0
   - Modes: fast-compile (0/1), float32 (0/1)
   - Backends: Default, Numba, JAX, PyTorch
3. **Benchmarks**: Performance regression testing
4. **Coverage Upload**: Uploads to Codecov

**Time estimates:**
- Style checks: 1-2 minutes
- Individual test jobs: 5-15 minutes
- Full matrix: 30-60 minutes

### MyPy Workflow (`.github/workflows/mypy.yml`)
Runs type checking on every PR:
```bash
python scripts/run_mypy.py --verbose
```

### PyPI Workflow (`.github/workflows/pypi.yml`)
Builds wheels for distribution. Runs on:
- Pushes to main (if build files changed)
- Pull requests (if build files changed)
- Releases (always)

Includes Cython compilation and tests sdist/wheel installation.

## Common Pitfalls and Solutions

### Installation Issues
**Problem**: `pip install -e .` fails with missing dependencies
**Solution**: Use conda environment - many dependencies (MKL, compilers) require conda

**Problem**: Cython extension doesn't compile
**Solution**: Ensure `cython` and `numpy` are installed first, then run `pip install -e .`

### Test Failures
**Problem**: Tests fail with "BLAS flags are empty"
**Solution**: Install via conda to get MKL: `micromamba install mkl mkl-service`

**Problem**: Import errors for JAX/Numba in tests
**Solution**: These are optional dependencies. Tests use `pytest.importorskip()` to skip gracefully. Install if needed: `pip install jax jaxlib` or `pip install numba`

### Pre-commit Issues
**Problem**: Pre-commit hook fails on commit
**Solution**: Run `pre-commit run --all-files` to see and fix issues. Most ruff issues auto-fix with `--fix`.

**Problem**: Debug statements rejected by pre-commit
**Solution**: Only specific files are allowed to have debug statements (see `.pre-commit-config.yaml`). Remove or use allowed files.

### Common Test Patterns
- Use `pytensor.function()` to compile and test ops
- Use `pytensor.gradient.verify_grad()` to test gradients (from `tests.unittest_tools`)
- Compare outputs to NumPy reference implementations
- Tests are organized by module - find similar tests in the appropriate `tests/<module>/` directory

## Making Changes

### Before Starting
1. Check existing issues/PRs to avoid duplication
2. Create an issue to discuss significant changes
3. Ensure pre-commit is installed: `pre-commit install`

### During Development
1. **Write tests first** - PyTensor has extensive test coverage
2. **Test incrementally** - Run relevant test subset frequently
3. **Check types** - Add type hints where appropriate
4. **Document** - Follow NumPy docstring standard (see `doc/dev_start_guide.rst`)

### Testing Your Changes
```bash
# Test the specific module you changed
pytest tests/tensor/test_basic.py -v

# Test with various flags
PYTENSOR_FLAGS=mode=FAST_COMPILE pytest tests/tensor/test_basic.py

# Run pre-commit before committing
pre-commit run --all-files

# For gradient-dependent changes, verify gradients pass
# Use verify_grad() from tests.unittest_tools in your tests
```

### Before Submitting PR
1. **Run relevant tests locally** - Don't rely solely on CI
2. **Check pre-commit passes** - It will run on CI anyway
3. **Write descriptive commit messages** - Follow conventional commits
4. **Update documentation** - If adding features or changing APIs
5. **Reference issues** - Link related issues in PR description

## Quality Standards

### Code Quality
- **Test coverage**: Aim for complete coverage of new code
- **Type hints**: Add where appropriate (especially public APIs)
- **Docstrings**: Required for public functions/classes (NumPy format)
- **Pre-commit**: Must pass all hooks

### Commit Standards
- Descriptive commit messages (see https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
- Logical commits (no "fix previous commit" commits in final PR)
- Reference related issues

### PR Requirements
- Tests for new functionality
- No unnecessary test removals/modifications
- Pre-commit hooks pass
- CI passes (all tests, style, mypy)
- Relevant documentation updates

## Trust These Instructions

These instructions have been validated against the actual repository structure, workflows, and documentation. When in doubt:
1. Follow these instructions first
2. Check the official docs: https://pytensor.readthedocs.io/en/latest/dev_start_guide.html
3. Look at recent PRs for examples
4. Only search/explore if instructions are unclear or incomplete

The environment setup, test commands, and CI/CD patterns described here are verified and current as of the repository state. Following them will minimize build failures and rejected PRs.
