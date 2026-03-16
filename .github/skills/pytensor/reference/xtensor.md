# XTensor — Named Dimensions Reference

Comprehensive reference for `pytensor.xtensor`, providing xarray-like named dimensions for symbolic tensors. After reading this you can: create `XTensorVariable`s with named dims, perform reductions/transposes/indexing by dimension name, use `isel` for labeled indexing, contract dimensions with `dot`, stack/unstack dimensions, use XTensor random variables, convert between `TensorVariable` and `XTensorVariable`, and understand how XOps are lowered to standard tensor ops before compilation.

> **Warning**: The xtensor module is experimental. Importing it triggers a warning.

## Contents
- Type system (`XTensorType`, `XTensorVariable`, `XTensorConstant`)
- Creation functions
- Properties and accessors
- Elementwise math and casting
- Reductions by dimension name
- Transpose and shape manipulation
- Indexing and functional updates (`isel`, `set`, `inc`)
- Dot product with dimension contraction
- Rename dimensions
- Stack / unstack
- Linear algebra
- Random variables
- Converting between Tensor and XTensor
- Lowering: how XOps compile
- Gradient support

## Type System

### XTensorType

`XTensorType` extends PyTensor's type system with named dimensions:

```python
from pytensor.xtensor.type import XTensorType

t = XTensorType(dtype="float64", dims=("batch", "feature"), shape=(32, 10))
t.dtype   # "float64"
t.dims    # ("batch", "feature")
t.shape   # (32, 10)
t.ndim    # 2
```

Properties `(__props__)`: `("dtype", "shape", "dims")`. Dims must be unique strings. Shape `None` entries mean unknown size (same as `TensorType`).

### XTensorVariable

The symbolic variable type. Inherits from `Variable` and provides an xarray-like API. Created by calling an `XTensorType` instance or via constructor functions.

### XTensorConstant

Constant with named dims. Shape info is extracted from data.

### XTensorSharedVariable

Persistent named-dimension variable across function calls.

## Creation

```python
import pytensor.xtensor as px
import numpy as np

# Symbolic variable — dims is required, keyword-only
x = px.xtensor("x", dims=("time", "feature"), shape=(100, 3), dtype="float64")
x = px.xtensor("x", dims=("time", "feature"))  # shape defaults to (None, None)

# From constant data
c = px.xtensor_constant(np.array([[1, 2], [3, 4]]), dims=("a", "b"))

# From existing TensorVariable (dims required for non-scalars)
from pytensor.xtensor.type import as_xtensor
x_named = as_xtensor(tensor_var, dims=("batch", "channel"))

# Scalars convert without dims
s = as_xtensor(scalar_var)

# Shared variables with dims
from pytensor.xtensor.type import xtensor_shared
W = xtensor_shared(np.random.randn(3, 4), dims=("input", "output"), name="W")

# xarray DataArrays convert automatically (when xarray is installed)
import xarray as xr
c = px.xtensor_constant(xr.DataArray(np.zeros((2, 3)), dims=("a", "b")))
```

### xtensor() Signature

```python
px.xtensor(
    name=None,      # optional variable name
    *,
    dims,           # required: tuple of str
    shape=None,     # optional: tuple of int|None (defaults to all None)
    dtype="floatX", # defaults to pytensor.config.floatX
)
```

## Properties and Accessors

```python
x.dims        # ("time", "feature") — dimension names
x.sizes       # {"time": TensorVariable, "feature": TensorVariable}
x.values      # underlying TensorVariable (strips dim info)
x.ndim        # 2
x.dtype       # "float64"
x.shape       # tuple of TensorVariables
x.size        # product of shape elements
x.T           # full reverse transpose
```

**Important**: `.values` returns a plain `TensorVariable` — this is how you cross from the xtensor world back to the tensor world.

## Elementwise Math and Casting

All standard arithmetic operators work on `XTensorVariable` with automatic dim alignment:

```python
z = x + y       # aligns by dim name, broadcasts mismatched dims
z = x * 2
z = x ** 2
z = x / y
z = -x
z = abs(x)
```

Math functions via `px.math`:

```python
import pytensor.xtensor.math as pxm

z = pxm.exp(x)
z = pxm.log(x)
z = pxm.sqrt(x)
z = pxm.sigmoid(x)
z = pxm.tanh(x)
z = pxm.softmax(x, axis=-1)
z = pxm.abs(x)
z = pxm.clip(x, 0, 1)     # also x.clip(0, 1)
```

Casting:

```python
z = x.astype("float32")
z = pxm.cast(x, "int64")
```

## Reductions by Dimension Name

All reductions accept `dim` as string, tuple of strings, `None` (all dims), or `...` (all dims):

```python
x.sum(dim="time")                # reduce "time", keep "feature"
x.sum(dim=("time", "feature"))   # reduce both
x.sum(dim=None)                  # reduce all (default)
x.sum()                          # same as dim=None

x.mean(dim="time")
x.var(dim="time", ddof=1)        # variance with degrees of freedom
x.std(dim="feature", ddof=0)     # standard deviation
x.max(dim="time")
x.min(dim="time")
x.prod(dim="time")

# Boolean reductions (cast non-bool to bool via neq(x, 0))
x.all(dim="time")
x.any(dim="time")

# Cumulative reductions (preserve all dims, accumulate along specified dim)
x.cumsum(dim="time")
x.cumprod(dim="time")

# Discrete difference
x.diff(dim="time", n=1)
```

### Module-level reduction functions

```python
from pytensor.xtensor.reduction import sum, mean, var, std, max, min, prod, all, any, cumsum, cumprod

z = sum(x, dim="time")
```

## Transpose and Shape Manipulation

### Transpose

```python
x.transpose("feature", "time")       # reorder dims by name
x.transpose(..., "time")             # ellipsis for remaining dims first
x.T                                   # full reverse transpose
```

`missing_dims` parameter controls behavior for non-existent dims: `"raise"` (default), `"warn"`, `"ignore"`.

### Stack and Unstack

Combine multiple dims into one, or split one dim into multiple:

```python
# Stack: merge "time" and "feature" into a single "flat" dim
stacked = x.stack({"flat": ["time", "feature"]})

# Unstack: split "flat" back into original dims
unstacked = stacked.unstack({"flat": x.sizes})
```

### Squeeze and Expand Dims

```python
# Remove size-1 dims
x.squeeze(dim="singleton_dim")
x.squeeze(dim=None)    # all statically-known size-1 dims
x.squeeze(axis=0)      # by axis index

# Add new dims
x.expand_dims(dim="batch")          # size 1 at front
x.expand_dims(dim={"batch": 32})    # specific size
x.expand_dims(dim=["a", "b"])       # multiple size-1 dims
```

### Broadcast

```python
from pytensor.xtensor.shape import broadcast

a_bc, b_bc = broadcast(a, b)                   # align and broadcast
a_bc, b_bc = broadcast(a, b, exclude=("time",)) # exclude dims from broadcast
```

### Fill-like Constructors

```python
from pytensor.xtensor.shape import zeros_like, ones_like, full_like

z = zeros_like(x)
o = ones_like(x)
f = full_like(x, fill_value=3.14)
```

## Indexing and Functional Updates

### Positional Indexing

```python
x[0, :]              # index first dim positionally
x[:10, 1:3]          # slicing
x[idx_tensor]        # tensor indexing
```

### Named Indexing with isel

```python
x.isel(time=0)                                  # single index by name
x.isel({"time": slice(0, 10), "feature": 1})    # dict form
x.isel(time=idx_tensor)                         # tensor index by name
x.isel(time=0, feature=slice(1, 3))             # multiple dims
```

`missing_dims` parameter: `"raise"` (default), `"warn"`, `"ignore"`.

### Functional Updates

```python
# Set: replace indexed values
x.isel(time=0).set(new_values)
x[:, 0].set(new_values)

# Increment: add to indexed values
x.isel(time=0).inc(delta)
x[:, 0].inc(delta)
```

`.set()` and `.inc()` can only be called on the output of an index/isel operation.

### Head, Tail, Thin

```python
x.head(time=5)         # first 5 along "time"
x.head(3)              # first 3 along all dims
x.tail(feature=2)      # last 2 along "feature"
x.thin(time=3)         # every 3rd element along "time"
```

## Dot Product with Dimension Contraction

`dot` performs generalized tensor contraction by dimension name:

```python
from pytensor.xtensor import dot

x = px.xtensor("x", dims=("a", "b"))
y = px.xtensor("y", dims=("b", "c"))

# Contract shared dims (default: dim=None contracts all shared dims)
z = dot(x, y)             # contracts "b" → result dims ("a", "c")
z = x.dot(y)              # method form

# Contract specific dims
z = dot(x, y, dim="b")
z = dot(x, y, dim=("b",))

# Contract ALL dims (scalar result)
z = dot(x, y, dim=...)
```

Lowers to `einsum` internally.

## Rename Dimensions

```python
x_renamed = x.rename({"time": "step", "feature": "channel"})
x_renamed = x.rename(time="step")     # kwargs form
x_copy = x.copy(name="x_copy")        # identity op with new name
```

## Linear Algebra

Via `XBlockwise` wrappers that handle dimension alignment:

```python
from pytensor.xtensor import linalg

L = linalg.cholesky(cov_matrix)   # Cholesky decomposition
x = linalg.solve(A, b)            # solve Ax = b
```

## Random Variables

XTensor wraps PyTensor's random variable ops with named dimensions:

```python
from pytensor.xtensor import random as pxr

z = pxr.normal(loc=0, scale=1, dims=("batch", "feature"), size=(32, 10))
z = pxr.bernoulli(p=0.5, dims=("trial",), size=(100,))
z = pxr.uniform(low=0, high=1, dims=("a", "b"), size=(3, 4))
```

Available distributions: `bernoulli`, `beta`, `betabinom`, `binomial`, `categorical`, `cauchy`, `dirichlet`, `exponential`, `gamma`, `gengamma`, `geometric`, `gumbel`, `halfcauchy`, `halfnormal`, `hypergeometric`, `integers`, `invgamma`, `laplace`, `logistic`, `lognormal`, `multinomial`, `multivariate_normal`, `negative_binomial`, `normal`, `pareto`, `poisson`, `t`, `triangular`, `truncexpon`, `uniform`, `vonmises`, `wald`, `weibull`, `standard_normal`, `chisquare`, `rayleigh`.

## Converting Between Tensor and XTensor

```python
# XTensor → Tensor (strips dim info, returns TensorVariable)
tensor_var = x.values

# Tensor → XTensor (requires dims for non-scalars)
from pytensor.xtensor.type import as_xtensor
x_named = as_xtensor(tensor_var, dims=("a", "b"))

# Explicit cast ops
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
t = tensor_from_xtensor(x)
x = xtensor_from_tensor(t, dims=("a", "b"))
```

**Important**: Automatic conversion of `XTensorVariable` to `TensorVariable` is blocked by default to prevent subtle bugs. Use `.values` or pass `allow_xtensor_conversion=True` to `as_tensor_variable`.

## How XOps Are Lowered (Compilation)

XOps are **never executed directly**. A rewrite pass (`lower_xtensor_db`) runs at position 0.09 (before shape optimization) and lowers all XOps to standard tensor operations:

| XTensor Op | Lowers To |
|---|---|
| `XElemwise` | `Elemwise` + `dimshuffle` for dim alignment |
| `XBlockwise` | `Blockwise` |
| `XRV` | Core `RandomVariable` op |
| `XReduce` | `Sum`, `Prod`, `Max`, `Min`, `All`, `Any` via `CAReduce` |
| `XCumReduce` | `CumOp` |
| `Dot` | `einsum` |
| `Transpose` | `tensor.transpose` |
| `Stack` | `moveaxis` + `reshape` |
| `UnStack` | `reshape` + `moveaxis` |
| `Concat` | `concat_with_broadcast` |
| `Squeeze` | `tensor.squeeze` |
| `ExpandDims` | `expand_dims` / `broadcast_to` |
| `Broadcast` | `broadcast_arrays` |
| `Index` | `Subtensor` / advanced indexing |
| `IndexUpdate` | `inc_subtensor` |
| `TensorFromXTensor` | Identity (data passthrough) |
| `XTensorFromTensor` | Identity (data passthrough) |
| `Rename` | Identity (data passthrough) |

After lowering, the resulting graph uses standard `TensorType` and compiles to any backend (C, JAX, Numba, PyTorch, MLX).

## Gradient Support

Gradients flow through the cast ops between XTensor and Tensor:

- `TensorFromXTensor.L_op`: wraps gradient back into `xtensor_from_tensor(g, dims=...)`
- `XTensorFromTensor.L_op`: unwraps gradient via `tensor_from_xtensor(g)`
- `Rename.L_op`: renames gradient dims back to original names

Other XOps are lowered before gradient computation, so standard tensor gradient rules apply.

## External Docs

| Topic | URL |
|---|---|
| XArray DataArray API | https://docs.xarray.dev/en/latest/api.html#dataarray |
| PyTensor GitHub | https://github.com/pymc-devs/pytensor |
