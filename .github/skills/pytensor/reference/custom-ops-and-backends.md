# Custom Ops and Backend Support

How to extend PyTensor with new operations and make them work across backends. After reading this you can: create a custom `Op` with `make_node()`/`perform()`, implement symbolic gradients via `grad()`/`L_op()`/`R_op()`, verify gradient correctness with `verify_grad()`, register backend-specific implementations for JAX (`jax_funcify`), Numba (`numba_funcify`), PyTorch (`pytorch_funcify`), and MLX (`mlx_funcify`) via singledispatch, write custom graph rewrites using `node_rewriter`/`PatternNodeRewriter`/`KanrenRelationSub`, and understand the current maturity of each backend.

## Contents
- Creating a custom Op
- Gradient methods (grad, L_op, R_op)
- Gradient verification
- Backend dispatch (JAX, Numba, PyTorch, MLX)
- Custom graph rewrites
- Backend status and limitations

## Creating a Custom Op

Every Op must implement at minimum `make_node()` and `perform()`:

```python
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

class DoubleOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        output_storage[0][0] = x * 2

    def grad(self, inputs, output_gradients):
        return [output_gradients[0] * 2]

double = DoubleOp()
x = pt.dscalar("x")
y = double(x)
```

### __props__

Tuple of attribute names that uniquely identify this Op instance. Used for Op equality comparison and hashing. If the Op has no configuration parameters, use `()`.

```python
class ScaleOp(Op):
    __props__ = ("scale_factor",)

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0] * self.scale_factor
```

### infer_shape

Optional but improves optimizer performance:

```python
def infer_shape(self, fgraph, node, input_shapes):
    return [input_shapes[0]]  # output shape = input shape
```

## Gradient Methods

### Op.grad(inputs, output_gradients)

The primary gradient method. Returns list of gradient expressions (one per input):

```python
def grad(self, inputs, output_gradients):
    (x,) = inputs
    (gz,) = output_gradients
    return [gz * 2]  # d(2x)/dx = 2, chain rule: gz * 2
```

### Op.L_op(inputs, outputs, output_gradients)

Vector-Jacobian product (VJP). More general than `grad` — works for multi-output Ops:

```python
def L_op(self, inputs, outputs, output_gradients):
    (x,) = inputs
    (gz,) = output_gradients
    return [gz * self.jacobian_transpose(x)]
```

### Op.R_op(inputs, eval_points)

Jacobian-vector product (JVP). Used for forward-mode AD:

```python
def R_op(self, inputs, eval_points):
    (x,) = inputs
    (ev,) = eval_points
    if ev is None:
        return [None]
    return [ev * 2]
```

## Gradient Verification

```python
from pytensor.gradient import verify_grad
import numpy as np

rng = np.random.default_rng(42)
verify_grad(double, [rng.standard_normal(5)], rng=rng)
```

Compares analytic (symbolic) gradient against numeric (finite differences). Raises exception if discrepancy detected.

## Backend Dispatch

Each backend uses `functools.singledispatch`. Register a function that converts a PyTensor Op into the equivalent backend operation.

### JAX Backend

```python
from pytensor.link.jax.dispatch import jax_funcify

@jax_funcify.register(DoubleOp)
def jax_funcify_DoubleOp(op, **kwargs):
    def double_jax(*inputs):
        return inputs[0] * 2
    return double_jax
```

### Numba Backend

```python
from pytensor.link.numba.dispatch import numba_funcify

@numba_funcify.register(DoubleOp)
def numba_funcify_DoubleOp(op, node, **kwargs):
    import numba

    @numba.njit
    def double_numba(x):
        return x * 2
    return double_numba
```

### PyTorch Backend

```python
from pytensor.link.pytorch.dispatch import pytorch_funcify

@pytorch_funcify.register(DoubleOp)
def pytorch_funcify_DoubleOp(op, **kwargs):
    def double_pytorch(*inputs):
        return inputs[0] * 2
    return double_pytorch
```

### MLX Backend

```python
from pytensor.link.mlx.dispatch import mlx_funcify

@mlx_funcify.register(DoubleOp)
def mlx_funcify_DoubleOp(op, **kwargs):
    import mlx.core as mx

    def double_mlx(*inputs):
        return inputs[0] * 2
    return double_mlx
```

## Custom Graph Rewrites

Register custom rewrites to transform the graph before compilation:

```python
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.compile.optdb import optdb

@node_rewriter([MyOp])
def my_custom_rewrite(fgraph, node):
    if some_condition(node):
        return [replacement_var]
    return False

optdb.register('my_rewrite', my_custom_rewrite, 'fast_run', position=1.5)
```

### Rewriter Classes

| Class | Description |
|---|---|
| `NodeRewriter` | Operates on individual Apply nodes |
| `GraphRewriter` | Applies NodeRewriters across entire FunctionGraph |
| `EquilibriumGraphRewriter` | Applies repeatedly until fixed-point |
| `PatternNodeRewriter` | Pattern-match and replace |
| `SubstitutionNodeRewriter` | Replace one Op with another |
| `RemovalNodeRewriter` | Remove an Op (output = input) |
| `KanrenRelationSub` | miniKanren relational programming for bidirectional rewrites |

### PatternNodeRewriter Example

```python
from pytensor.graph.rewriting.basic import PatternNodeRewriter

# Replace MyOp(x, 0) with x
rewrite = PatternNodeRewriter(
    (my_op, 'x', 0),   # pattern
    'x',                 # replacement
    name='remove_zero'
)
```

## Backend Status (as of early 2026)

### C Backend (Default)
- Most mature, inherited from Theano
- Fastest for many CPU operations (BLAS, LAPACK)
- Compiled libraries cached in `~/.pytensor/compiledir_xxx`

### JAX Backend
- Production-ready for most PyMC models
- **Requires static shapes** for JIT
- Used via `pm.sample(nuts_sampler="numpyro")` or `"blackjax"`

### Numba Backend
- Near-complete for PyMC workloads
- Used via `nutpie` for Rust-based NUTS
- Some `Advanced(Inc)Subtensor` variants missing

### PyTorch Backend
- Work-in-progress
- Has: elemwise, math, shape, subtensor, blockwise, BLAS, sort, nlinalg, QR
- Missing: **Scan**, **Cholesky**, **Solve**, RandomVariable ops
- Most PyMC models cannot run on PyTorch backend yet

### MLX Backend
- Targets Apple Silicon (M-series chips)
- Has: elemwise, math, shape, subtensor, blockwise, slinalg, nlinalg, sort, signal, extra_ops
- Missing: **Scan**, RandomVariable ops
- Transpiles via `mlx_funcify()` singledispatch

## External Docs

| Topic | URL |
|---|---|
| Creating an Op | https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html |
| JAX/Numba/PyTorch Backend | https://pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html |
| Graph Rewriting | https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html |
| Graph Rewrites Gallery | https://pytensor.readthedocs.io/en/latest/gallery/rewrites/graph_rewrites.html |
| PyTorch Backend Issue | https://github.com/pymc-devs/pytensor/issues/821 |
