# PyTensor Import Testing Results

## Overview

This document summarizes the results of testing different strategies for importing `pytensor` with the `pytensor-test` micromamba environment.

## Test Environment

- **System:** GitHub Actions Runner
- **Micromamba version:** Available at `/home/runner/micromamba-bin/micromamba`
- **Environment name:** `pytensor-test`
- **Environment path:** `/home/runner/micromamba/envs/pytensor-test`
- **PyTensor version:** `0+untagged.2.g092cfbf`
- **PyTensor location:** `/home/runner/work/pytensor/pytensor/pytensor/__init__.py`

## Test Strategies and Results

### ❌ Strategy 1: Direct Python Import (No Activation)

**Command:**
```bash
python -c 'import pytensor'
```

**Result:** FAILED

**Error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Analysis:** The system Python (`/usr/bin/python`) does not have numpy or pytensor installed. The conda environment must be used.

---

### ❌ Strategy 2: Micromamba Activate

**Command:**
```bash
micromamba activate pytensor-test
python -c 'import pytensor'
```

**Result:** FAILED

**Error:**
```
'micromamba' is running as a subprocess and can't modify the parent shell.
Thus you must initialize your shell before using activate and deactivate.
```

**Analysis:** Direct activation doesn't work in a subprocess without shell initialization. The `activate` command needs to modify shell environment variables, which cannot be done when running as a subprocess.

---

### ✅ Strategy 2b: Micromamba Run (RECOMMENDED)

**Command:**
```bash
micromamba run -n pytensor-test python -c 'import pytensor'
```

**Result:** SUCCESS ✅

**Output:**
```
PyTensor version: 0+untagged.2.g092cfbf
Successfully imported from: /home/runner/work/pytensor/pytensor/pytensor/__init__.py
```

**Analysis:** This is the **RECOMMENDED** approach for CI/CD and automated workflows. The `micromamba run` command:
- Does not require shell initialization
- Works reliably in non-interactive environments
- Is the standard way to run commands in conda/micromamba environments
- Has a clean, simple syntax

---

### ❌ Strategy 3: Mamba Activate

**Command:**
```bash
mamba activate pytensor-test
python -c 'import pytensor'
```

**Result:** FAILED

**Error:**
```
bash: mamba: command not found
```

**Analysis:** The `mamba` command is not installed on this system. Only `micromamba` is available.

---

### ✅ Strategy 4: Shell Hook + Activate

**Command:**
```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate pytensor-test
python -c 'import pytensor'
```

**Result:** SUCCESS ✅

**Analysis:** This approach works by first initializing the shell with the micromamba hook, then activating the environment. While functional, it's more complex than using `micromamba run` and is primarily useful for interactive shell sessions.

---

## Functional Testing

To verify that pytensor works correctly beyond just importing, we tested basic functionality:

```python
import pytensor
import pytensor.tensor as pt
import numpy as np

# Test symbolic operations
x = pt.dscalar('x')
y = pt.dscalar('y')
z = x + y
f = pytensor.function([x, y], z)
result = f(2.0, 3.0)  # Returns 5.0 ✓

# Test vector operations
a = pt.dvector('a')
b = pt.dvector('b')
c = a * b
g = pytensor.function([a, b], c)
result = g([1., 2., 3.], [4., 5., 6.])  # Returns [4., 10., 18.] ✓
```

**Result:** All functional tests passed! ✅

---

## Recommendations

### For CI/CD and GitHub Actions

Use the `micromamba run` command:

```yaml
- name: Run tests
  run: |
    micromamba run -n pytensor-test python -m pytest tests/
```

Or for multiple commands:

```yaml
- name: Run tests
  run: |
    micromamba run -n pytensor-test python -c "import pytensor; print(pytensor.__version__)"
    micromamba run -n pytensor-test python -m pytest tests/
```

### For Interactive Development

Initialize the shell first, then activate:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate pytensor-test
python -c 'import pytensor'
```

### For Documentation

Update any documentation or workflow files to use:
```bash
micromamba run -n pytensor-test <command>
```

---

## Verification of `post-cleanup: 'none'` Setting

**Status:** ✅ CONFIRMED WORKING

The `pytensor-test` environment is available and functional after the micromamba setup step, confirming that the `post-cleanup: 'none'` setting is working correctly and preventing the environment from being destroyed prematurely.

---

## Summary

| Strategy | Command | Status | Recommended For |
|----------|---------|--------|-----------------|
| Direct import | `python -c 'import pytensor'` | ❌ Failed | N/A |
| Micromamba activate | `micromamba activate ...` | ❌ Failed | N/A |
| **Micromamba run** | `micromamba run -n pytensor-test ...` | ✅ **SUCCESS** | **CI/CD, Automation** |
| Mamba activate | `mamba activate ...` | ❌ Failed | N/A |
| Shell hook + activate | `eval "$(...)"; micromamba activate ...` | ✅ Success | Interactive shells |

**Conclusion:** Use `micromamba run -n pytensor-test <command>` for all automated workflows and CI/CD pipelines.
