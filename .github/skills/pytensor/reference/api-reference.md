# PyTensor API Reference

Complete reference for PyTensor's tensor operations — the symbolic equivalents of NumPy. After reading this you can: create typed symbolic variables (scalars through 7D tensors), perform elementwise and matrix math, apply reductions (sum, mean, max), manipulate shapes with `dimshuffle`/reshape/concatenate, do functional indexing with `set_subtensor`/`inc_subtensor`, use broadcasting, call linear algebra routines (Cholesky, SVD, solve), manage persistent state with shared variables, generate random samples via `RandomStream`, and compute full Jacobians, Hessians, and JVP/VJP products. For named-dimension tensors (xarray-like), see [xtensor.md](xtensor.md).

## Contents
- Tensor creation (typed constructors, generic, shared, filled)
- Math operations (elementwise, nonlinear, matrix)
- Reductions and aggregation
- Comparisons and conditionals
- Shape manipulation and dimshuffle
- Indexing and functional updates
- Broadcasting rules
- Linear algebra
- Shared variables (read, write, updates)
- Random variables and RandomStream

## Tensor Creation

### Typed Constructors

Prefix convention: `b`=int8, `w`=int16, `i`=int32, `l`=int64, `f`=float32, `d`=float64, `c`=complex64, `z`=complex128.

```python
import pytensor.tensor as pt

# float64
s = pt.dscalar("s")       # scalar (0-d)
v = pt.dvector("v")       # vector (1-d)
m = pt.dmatrix("m")       # matrix (2-d)
t = pt.dtensor3("t")      # 3-d (up to dtensor7)

# float32
s = pt.fscalar("s")
m = pt.fmatrix("m")

# int32
i = pt.iscalar("i")
iv = pt.ivector("iv")

# Plural constructors
a, b, c = pt.dmatrices("a", "b", "c")
x1, x2, x3 = pt.dscalars(3)
```

### Generic Constructor

```python
x = pt.tensor(dtype="float64", shape=(4, 3), name="x")
x_partial = pt.tensor(dtype="float64", shape=(None, 3), name="x")  # any rows, 3 cols
x_any = pt.tensor(dtype="float64", shape=(None, None), name="x")   # any 2D
```

`None` means any size, integer means fixed size.

### TensorType Directly

```python
from pytensor.tensor.type import TensorType

T = TensorType(dtype="float64", shape=(None, None))   # any 2D float64
T_fixed = TensorType(dtype="float64", shape=(2, 3))   # exactly 2x3
T_partial = TensorType(dtype="float64", shape=(2, None))  # 2 rows, any cols
```

### From Python/NumPy

```python
import numpy as np
c = pt.as_tensor_variable(np.array([1, 2, 3]))
```

### Filled Tensors

```python
pt.zeros((3, 4))
pt.ones((3, 4))
pt.eye(3)                  # 3x3 identity
pt.zeros_like(x)
pt.ones_like(x)
pt.arange(10)              # [0, 1, ..., 9]
pt.alloc(value, *shape)    # N-D filled with value
```

## Math Operations

### Elementwise Arithmetic

```python
# Standard operators: +, -, *, /, //, **, %
pt.abs(x)
pt.neg(x)
pt.exp(x)
pt.log(x)
pt.sqrt(x)
```

### Nonlinear Activations

```python
pt.tanh(x)
pt.sigmoid(x)              # 1 / (1 + exp(-x))
pt.special.softmax(x)     # softmax lives in pt.special
pt.clip(x, min_val, max_val)
```

### Matrix Operations

```python
pt.dot(x, y)               # matrix multiply (also x @ y)
pt.outer(v1, v2)           # outer product
pt.tensordot(x, y, axes=2)
```

## Reductions

```python
pt.sum(x, axis=None, keepdims=False)
pt.prod(x, axis=None)
pt.mean(x, axis=None)
pt.var(x, axis=None)
pt.std(x, axis=None)
pt.max(x, axis=None)
pt.min(x, axis=None)
pt.argmax(x, axis=None)
pt.argmin(x, axis=None)
pt.all(x, axis=None)       # logical AND
pt.any(x, axis=None)       # logical OR
```

## Comparisons and Conditionals

Comparisons return `bool` dtype:

```python
pt.lt(a, b)    # <
pt.le(a, b)    # <=
pt.gt(a, b)    # >
pt.ge(a, b)    # >=
pt.eq(a, b)    # ==
pt.neq(a, b)   # !=
pt.isnan(a)
pt.isinf(a)
pt.isclose(a, b, rtol=1e-5, atol=1e-8)

# Elementwise if-else
pt.switch(cond, if_true, if_false)
pt.where(cond, if_true, if_false)  # alias
```

## Shape Manipulation

```python
x.reshape((2, 3))
x.flatten()
x.T                        # transpose
x.transpose(2, 0, 1)       # permute axes
x.swapaxes(0, 1)
x.squeeze(axis)            # remove size-1 dimensions
```

### dimshuffle — Dimension Reordering

PyTensor's equivalent to NumPy's `np.newaxis` + transpose combined:

```python
v = pt.dvector("v")        # shape (N,)
v.dimshuffle('x', 0)       # (N,) → (1, N) — add leading dim
v.dimshuffle(0, 'x')       # (N,) → (N, 1) — add trailing dim

x = pt.dmatrix("x")        # shape (M, N)
x.dimshuffle(0, 'x', 1)    # (M, N) → (M, 1, N) — insert dim
x.dimshuffle(1, 0)          # (M, N) → (N, M) — transpose
```

`'x'` inserts a broadcastable (size-1) dimension. Integers reference existing dims.

### Joining and Shape Utilities

```python
pt.concatenate([a, b], axis=0)
pt.stack([a, b], axis=0)
pt.shape(x)                    # symbolic shape as lvector
pt.shape_padleft(x, n_ones=1)  # pad left with 1s
pt.shape_padright(x)           # pad right with 1s
pt.specify_shape(x, (100, 50)) # assert shape (helps optimizer)
pt.tile(x, reps)
pt.roll(x, shift, axis)
```

## Indexing and Functional Updates

Basic indexing is NumPy-compatible:

```python
x[0, 0]           # single element
x[0]              # first row
x[:, 0]           # first column
x[1:3, :]         # slice
x[idx]            # gather by index vector
```

PyTensor is functional — no in-place assignment. Use these for updates:

```python
from pytensor.tensor.subtensor import set_subtensor, inc_subtensor

x_new = set_subtensor(x[0], new_values)    # replace row 0
x_inc = inc_subtensor(x[0], delta_values)   # add to row 0

# Method shortcuts
x_new = x[0].set(new_values)
x_inc = x[0].inc(delta_values)
```

## Broadcasting

Follows NumPy broadcasting rules. Use `dimshuffle` to control broadcast axes:

```python
m = pt.dmatrix("m")    # (M, N)
v = pt.dvector("v")    # (N,)
r = m + v              # broadcasts v across rows automatically

v2 = pt.dvector("v2")  # (M,)
r2 = m + v2.dimshuffle(0, 'x')  # broadcasts v2 across columns
```

When dimensions don't match, PyTensor expands to the left by padding with broadcastable (size-1) dimensions.

## Linear Algebra

```python
import pytensor.tensor.nlinalg as nla
import pytensor.tensor.slinalg as sla

nla.det(x)              # determinant
nla.matrix_inverse(x)   # inverse
nla.eig(x)              # eigendecomposition
nla.eigh(x)             # symmetric eigendecomposition
nla.svd(x)              # SVD
nla.trace(x)            # trace
sla.cholesky(x)         # Cholesky decomposition
sla.solve(A, b)         # solve Ax = b
pt.diag(x)              # extract/create diagonal
```

## Shared Variables

Persistent values across function calls with in-place update support:

```python
import pytensor

W = pytensor.shared(np.random.randn(100, 50).astype("float64"), name="W")
b = pytensor.shared(np.zeros(50, dtype="float64"), name="b")

# Read
W.get_value()                  # returns a copy
W.get_value(borrow=True)       # returns reference (no copy)

# Write
W.set_value(new_array)
W.set_value(new_array, borrow=True)

# Use in function with update rules
x = pt.dmatrix("x")
lr = pt.dscalar("lr")
gW = pytensor.grad(loss, W)
train = pytensor.function([x, lr], loss, updates=[(W, W - lr * gW)])
```

## Random Variables

### RandomStream (Recommended)

```python
from pytensor.tensor.random.utils import RandomStream

srng = RandomStream(seed=42)
rv_uniform = srng.uniform(low=0, high=1, size=(2, 3))
rv_normal = srng.normal(loc=0, scale=1, size=(5,))

f = pytensor.function([], rv_normal)
print(f())  # different values each call (state auto-updates)

srng.seed(123)  # reseed all RVs from this stream
```

Available distributions: `normal`, `uniform`, `bernoulli`, `binomial`, `poisson`, `exponential`, `gamma`, `beta`, `dirichlet`, `categorical`, `multinomial`, `multivariate_normal`, etc.

### Direct Usage (Less Recommended)

```python
x = pt.random.normal(loc=0, scale=1, size=(3, 3))
# Warning: .eval() doesn't update RNG state
```

### Custom RandomVariable

```python
from pytensor.tensor.random.op import RandomVariable

class MyDistRV(RandomVariable):
    name = "my_dist"
    signature = "(),()->()"
    dtype = "floatX"

    @classmethod
    def rng_fn(cls, rng, param1, param2, size):
        return rng.normal(loc=param1, scale=param2, size=size)

my_dist = MyDistRV()
```

## Automatic Differentiation — Full Reference

### pytensor.grad()

```python
# Scalar cost required
x = pt.dscalar("x")
y = x ** 2 + 3 * x + 1
gy = pytensor.grad(y, x)         # symbolic: 2x + 3

# Multiple inputs
gW, gb = pytensor.grad(loss, [W, b])
```

### Op-Level Gradient Methods

| Method | Purpose |
|---|---|
| `Op.grad(inputs, output_gradients)` | Chain-rule terms |
| `Op.L_op(inputs, outputs, output_gradients)` | VJP (reverse mode) |
| `Op.R_op(inputs, eval_points)` | JVP (forward mode) |

### Jacobian

```python
from pytensor.gradient import jacobian

x = pt.dvector("x")
y = x ** 2
J = jacobian(y, x)                         # uses scan internally
J_vec = jacobian(y, x, vectorize=True)     # faster, more memory
```

Efficient vectorized Jacobian:

```python
from pytensor.gradient import Lop
from pytensor.graph import vectorize_graph

row_cotangent = pt.dvector("row_cotangent")
J_row = Lop(y, x, row_cotangent)
J = vectorize_graph(J_row, replace={row_cotangent: pt.eye(x.size)})
```

### Hessian

```python
x = pt.dvector("x")
cost = pt.sum(x ** 3)
gy = pytensor.grad(cost, x)
H, updates = pytensor.scan(
    lambda i, gy, x: pytensor.grad(gy[i], x),
    sequences=pt.arange(gy.shape[0]),
    non_sequences=[gy, x]
)
```

### R-op and L-op (JVP/VJP without full Jacobian)

```python
from pytensor.gradient import Rop, Lop, hessian_vector_product

Jv = Rop(y, x, v)     # J·v (forward mode)
vJ = Lop(y, x, v)     # vᵀ·J (reverse mode)
Hv = hessian_vector_product(cost, x, v)
```

### Gradient Through Scan

Gradients propagate automatically:

```python
results, updates = pytensor.scan(
    fn=lambda x_t, acc: acc * x_t,
    sequences=[x],
    outputs_info=[pt.as_tensor_variable(np.float64(1.0))]
)
loss = results[-1]
grad = pytensor.grad(loss, x)  # works automatically
```

## External Docs

| Topic | URL |
|---|---|
| Tensor Basic API | https://pytensor.readthedocs.io/en/latest/library/tensor/basic.html |
| Gradients Tutorial | https://pytensor.readthedocs.io/en/latest/tutorial/gradients.html |
| Broadcasting | https://pytensor.readthedocs.io/en/latest/tutorial/broadcasting.html |
| Shape Info | https://pytensor.readthedocs.io/en/latest/tutorial/shape_info.html |
| Random Variables | https://pytensor.readthedocs.io/en/latest/library/tensor/random/index.html |
| slinalg | https://pytensor.readthedocs.io/en/latest/library/tensor/slinalg.html |
| nlinalg | https://pytensor.readthedocs.io/en/latest/library/tensor/nlinalg.html |
