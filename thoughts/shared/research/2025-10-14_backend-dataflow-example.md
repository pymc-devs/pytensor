---
date: 2025-10-14T00:00:00-00:00
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pytensor
topic: "Backend Implementation: Dataflow Example"
tags: [research, backend, dataflow, execution, jax, compilation]
status: complete
last_updated: 2025-10-14
last_updated_by: Claude
---

# Backend Implementation: Complete Dataflow Example

**Date**: 2025-10-14
**Researcher**: Claude
**Repository**: pytensor

## Overview

This document traces the complete dataflow of a simple PyTensor operation through the JAX backend, from user code to execution. We'll use the example: `y = pt.sum(x ** 2)`.

## Example: Sum of Squares with JAX Backend

### Step 1: User Code

```python
import pytensor
import pytensor.tensor as pt
import numpy as np

# Define symbolic variables
x = pt.vector('x', dtype='float32')

# Build computation graph
y = pt.sum(x ** 2)

# Compile with JAX backend
f = pytensor.function([x], y, mode='JAX')

# Execute
result = f(np.array([1.0, 2.0, 3.0], dtype='float32'))
print(result)  # Output: 14.0
```

---

## Stage 1: Graph Construction

### What Happens During Graph Building

```python
y = pt.sum(x ** 2)
```

**PyTensor Operations Created:**

1. **`x ** 2`** creates:
   - Op: `Elemwise(Pow)` with scalar_op = `Pow()`
   - Inputs: `[x, Constant(2)]`
   - Output: `TensorVariable` (call it `x_squared`)

2. **`pt.sum(...)`** creates:
   - Op: `CAReduce(Add)` with scalar_op = `Add()`
   - Inputs: `[x_squared]`
   - Output: `TensorVariable` (call it `y`)

**Resulting FunctionGraph Structure:**

```
Input: x [TensorType(float32, (?,))]
  ↓
Node 1: Elemwise(Pow)
  inputs: [x, Constant(2)]
  output: x_squared [TensorType(float32, (?,))]
  ↓
Node 2: CAReduce(Add, axis=None)
  inputs: [x_squared]
  output: y [TensorType(float32, ())]
```

**Key Data Structures:**

```python
# FunctionGraph.inputs
[x]  # List of input Variables

# FunctionGraph.outputs
[y]  # List of output Variables

# FunctionGraph.apply_nodes (topological order)
[
    Apply(op=Elemwise(Pow), inputs=[x, Constant(2)], outputs=[x_squared]),
    Apply(op=CAReduce(Add), inputs=[x_squared], outputs=[y])
]
```

---

## Stage 2: Compilation (`pytensor.function([x], y, mode='JAX')`)

### Step 2.1: Mode Initialization

**File**: `pytensor/compile/mode.py:477-492`

```python
JAX = Mode(
    JAXLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "jax"],
        exclude=["cxx_only", "BlasOpt", "fusion", "inplace", ...]
    )
)
```

**What happens:**
1. `JAXLinker()` instance created
2. Optimizer query configured with JAX-specific tags

### Step 2.2: Graph Optimization

**Optimizer applies rewrites tagged with "fast_run" + "jax":**

```python
# Example rewrites applied:
# - Canonicalization: (x ** 2) stays as is
# - Constant folding: None needed here
# - JAX-specific: shape_parameter_as_tuple (not applicable here)
```

**Graph remains:**
```
x → Elemwise(Pow) → x_squared → CAReduce(Add) → y
```

### Step 2.3: Linker Compilation

**Entry Point**: `JAXLinker.make_all()`
**File**: `pytensor/link/basic.py:683-707` (inherited from `JITLinker`)

```python
def make_all(self, profiler=None, input_storage=None, output_storage=None):
    # 1. Create input/output storage
    input_storage = [[None] for _ in self.fgraph.inputs]   # [[None]]
    output_storage = [[None] for _ in self.fgraph.outputs] # [[None]]

    # 2. Build storage_map (Variable → storage cell)
    storage_map = {
        x: input_storage[0],      # x → [None]
        y: output_storage[0]      # y → [None]
    }

    # 3. Convert FunctionGraph to JIT-able function
    compute_fn = self.fgraph_convert(
        self.fgraph,
        order=self.schedule(self.fgraph),  # Topological order of nodes
        input_storage=input_storage,
        output_storage=output_storage,
        storage_map=storage_map
    )

    # 4. JIT compile
    jitted_fn = self.jit_compile(compute_fn)

    # 5. Create thunk
    thunk = self.create_jitable_thunk(
        compute_fn=jitted_fn,
        input_storage=input_storage,
        output_storage=output_storage,
        storage_map=storage_map
    )

    return (thunk, input_storage, output_storage)
```

---

## Stage 3: JAXLinker.fgraph_convert()

**File**: `pytensor/link/jax/linker.py:18-93`

### Step 3.1: RNG Handling (not applicable here)

```python
# Lines 23-72: Handle RandomType shared variables
# Our example has no random variables, so this is skipped
```

### Step 3.2: Scalar Shape Detection (not applicable here)

```python
# Lines 76-89: Identify scalar inputs used only in JAXShapeTuple
# Our example has no shape operations, so scalar_shape_inputs = []
```

### Step 3.3: Call jax_funcify()

**File**: `pytensor/link/jax/linker.py:91-92`

```python
return jax_funcify(
    self.fgraph,
    input_storage=input_storage,
    storage_map=storage_map,
    **kwargs
)
```

**This triggers**: `@jax_funcify.register(FunctionGraph)`

---

## Stage 4: jax_funcify(FunctionGraph)

**File**: `pytensor/link/jax/dispatch/basic.py:49-62`

```python
@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph, node=None,
                                fgraph_name="jax_funcified_fgraph",
                                **kwargs):
    return fgraph_to_python(
        fgraph,
        jax_funcify,  # Op conversion function
        type_conversion_fn=jax_typify,
        fgraph_name=fgraph_name,
        **kwargs
    )
```

**This calls**: `fgraph_to_python()` utility

---

## Stage 5: fgraph_to_python() - Code Generation

**File**: `pytensor/link/utils.py:666-808`

### Step 5.1: Topological Sort

```python
# Line 720-721
nodes = fgraph.toposort()
# Result: [Apply(Elemwise(Pow)), Apply(CAReduce(Add))]
```

### Step 5.2: Generate Unique Names

```python
# Line 733-734
unique_names = unique_name_generator(
    [fgraph_name] + [str(v) for v in fgraph.variables]
)

# Generated names:
# x → "x"
# Constant(2) → "_constant_2"
# x_squared → "_x_squared"
# y → "_y"
```

### Step 5.3: Process Each Node

#### Node 1: Elemwise(Pow)

```python
# Line 736-746: Convert Op
op = node.op  # Elemwise(Pow)
node_inputs = [x, Constant(2)]
node_outputs = [x_squared]

# Call jax_funcify for Elemwise
elemwise_fn = jax_funcify(
    Elemwise(Pow),
    node=node,
    **kwargs
)
```

**Triggers**: `@jax_funcify.register(Elemwise)`
**File**: `pytensor/link/jax/dispatch/elemwise.py:9-20`

```python
@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op  # Pow()

    # Convert scalar op to JAX function
    base_fn = jax_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        # Runtime broadcast check
        Elemwise._check_runtime_broadcast(node, tuple(map(jnp.asarray, inputs)))
        return base_fn(*inputs)

    return elemwise_fn
```

**Nested call**: `jax_funcify(Pow())`
**File**: `pytensor/link/jax/dispatch/scalar.py:78-118`

```python
@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op, node, **kwargs):
    # For Pow, nfunc_spec = ("power", 2)
    func_name = op.nfunc_spec[0]  # "power"
    jax_func = getattr(jnp, func_name)  # jnp.power

    return jax_func
```

**Result**: `elemwise_fn` is a closure that calls `jnp.power` with broadcast checking.

#### Node 2: CAReduce(Add)

```python
# Call jax_funcify for CAReduce
careduce_fn = jax_funcify(
    CAReduce(Add, axis=None),
    node=node,
    **kwargs
)
```

**Triggers**: `@jax_funcify.register(CAReduce)`
**File**: `pytensor/link/jax/dispatch/elemwise.py:23-69`

```python
@jax_funcify.register(CAReduce)
def jax_funcify_CAReduce(op, **kwargs):
    axis = op.axis  # None (reduce all axes)
    scalar_op = op.scalar_op  # Add()

    # Add has nfunc_spec = ("add", 2)
    # Look up JAX function
    jax_op = getattr(jnp, "add")  # jnp.add

    # Map to reduction
    # For Add → sum
    acc_dtype = node.outputs[0].type.dtype  # float32

    def careduce(x):
        if axis is None:
            axes_to_reduce = list(range(x.ndim))
        else:
            axes_to_reduce = axis

        # Use jnp.sum for Add reduction
        return jnp.sum(x, axis=axes_to_reduce).astype(acc_dtype)

    return careduce
```

**Result**: `careduce_fn` is a closure that calls `jnp.sum`.

### Step 5.4: Generate Python Source Code

**File**: `pytensor/link/utils.py:761-799`

```python
# Build function body
func_body = []

# Node 1: Elemwise(Pow)
func_body.append("_x_squared = elemwise_pow_fn(x, _constant_2)")

# Node 2: CAReduce(Add)
func_body.append("_y = careduce_add_fn(_x_squared)")

# Return statement
func_body.append("return _y")

# Complete function
func_src = f"""
def jax_funcified_fgraph(x):
    {chr(10).join(func_body)}
"""
```

**Generated Source Code:**

```python
def jax_funcified_fgraph(x):
    _constant_2 = jnp.array(2, dtype='int64')
    _x_squared = elemwise_pow_fn(x, _constant_2)
    _y = careduce_add_fn(_x_squared)
    return _y
```

**Where**:
- `elemwise_pow_fn` is the closure from `jax_funcify_Elemwise`
- `careduce_add_fn` is the closure from `jax_funcify_CAReduce`

### Step 5.5: Compile Python Source

**File**: `pytensor/link/utils.py:804-806`

```python
# Compile generated source
exec_globals = {
    'jnp': jax.numpy,
    'elemwise_pow_fn': elemwise_pow_fn,
    'careduce_add_fn': careduce_add_fn,
}

exec(compile(func_src, '<string>', 'exec'), exec_globals)
jax_funcified_fgraph = exec_globals['jax_funcified_fgraph']

return jax_funcified_fgraph
```

**Result**: Callable Python function that uses JAX operations.

---

## Stage 6: JAXLinker.jit_compile()

**File**: `pytensor/link/jax/linker.py:95-113`

```python
def jit_compile(self, fn):
    import jax

    # No scalar shape inputs in our example
    jit_fn = jax.jit(fn, static_argnums=[])

    return jit_fn
```

**What happens**:
1. `jax.jit()` traces the function
2. Converts JAX operations to XLA HLO (High-Level Operations)
3. XLA compiles HLO to optimized machine code
4. Returns JIT-compiled function

**JAX Tracing Example:**

```python
# When jax.jit first traces with input shape (3,)
x_traced = jax.ShapedArray((3,), dtype='float32')
_constant_2 = jnp.array(2)
_x_squared = jnp.power(x_traced, _constant_2)  # ShapedArray((3,), float32)
_y = jnp.sum(_x_squared)                        # ShapedArray((), float32)
# JAX records operations and compiles to XLA
```

---

## Stage 7: Create Thunk

**File**: `pytensor/link/basic.py:616-681` (JITLinker.create_jitable_thunk)

```python
def create_jitable_thunk(self, compute_fn, input_storage,
                          output_storage, storage_map):
    # Prepare thunk inputs
    thunk_inputs = self.create_thunk_inputs(storage_map)
    # For our example: [input_storage[0]] → [[None]]

    # Create thunk
    def thunk():
        # Get input values from storage
        inputs = [inp[0] for inp in thunk_inputs]  # [input_storage[0][0]]

        # Filter inputs
        filtered_inputs = [self.input_filter(inp) for inp in inputs]

        # Execute JIT-compiled function
        outputs = compute_fn(*filtered_inputs)

        # Store outputs
        output_storage[0][0] = outputs

    return thunk
```

**JAXLinker.create_thunk_inputs():**
**File**: `pytensor/link/jax/linker.py:115-126`

```python
def create_thunk_inputs(self, storage_map):
    from pytensor.link.jax.dispatch import jax_typify

    thunk_inputs = []
    for n in self.fgraph.inputs:  # [x]
        sinput = storage_map[n]   # input_storage[0]

        # Convert Generator to JAX PRNGKey if needed (not applicable here)
        if isinstance(sinput[0], Generator):
            sinput[0] = jax_typify(sinput[0])

        thunk_inputs.append(sinput)

    return thunk_inputs  # [[None]]
```

---

## Stage 8: Function Execution

### User Calls: `f(np.array([1.0, 2.0, 3.0]))`

**Function Wrapper** (created by `pytensor.function`):

```python
# Simplified version of what pytensor.function creates
class Function:
    def __init__(self, thunk, input_storage, output_storage):
        self.thunk = thunk
        self.input_storage = input_storage
        self.output_storage = output_storage

    def __call__(self, *args):
        # Store input values
        for storage, value in zip(self.input_storage, args):
            storage[0] = value

        # Execute thunk
        self.thunk()

        # Return output values
        return self.output_storage[0][0]
```

### Execution Flow:

**Step 1**: Store input
```python
input_storage[0][0] = np.array([1.0, 2.0, 3.0], dtype='float32')
```

**Step 2**: Execute thunk
```python
thunk()
  ↓
inputs = [np.array([1.0, 2.0, 3.0])]
  ↓
outputs = jitted_fn(*inputs)
  ↓
# JAX executes compiled XLA code:
_constant_2 = jnp.array(2)
_x_squared = jnp.power([1.0, 2.0, 3.0], 2)  # [1.0, 4.0, 9.0]
_y = jnp.sum([1.0, 4.0, 9.0])               # 14.0
  ↓
output_storage[0][0] = 14.0
```

**Step 3**: Return output
```python
return output_storage[0][0]  # 14.0
```

---

## Complete Dataflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ USER CODE                                                        │
│ f = pytensor.function([x], pt.sum(x**2), mode='JAX')           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ GRAPH CONSTRUCTION                                               │
│ x → Elemwise(Pow) → x_squared → CAReduce(Add) → y              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODE INITIALIZATION                                              │
│ JAXLinker() + RewriteDatabaseQuery(include=["fast_run", "jax"]) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ GRAPH OPTIMIZATION                                               │
│ Apply rewrites: canonicalize, constant folding, JAX-specific    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ JAXLinker.make_all()                                             │
│ 1. Create storage: input_storage=[[None]], output_storage=[[]]  │
│ 2. Call fgraph_convert()                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ JAXLinker.fgraph_convert()                                       │
│ 1. Handle RNG (skip)                                             │
│ 2. Detect scalar shapes (skip)                                   │
│ 3. Call jax_funcify(fgraph)                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ jax_funcify(FunctionGraph)                                       │
│ → fgraph_to_python(fgraph, jax_funcify, jax_typify)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ fgraph_to_python() - CODE GENERATION                             │
│                                                                  │
│ For Node 1: Elemwise(Pow)                                        │
│   ├─ jax_funcify(Elemwise) → elemwise_pow_fn                    │
│   └─ jax_funcify(Pow) → jnp.power                               │
│                                                                  │
│ For Node 2: CAReduce(Add)                                        │
│   ├─ jax_funcify(CAReduce) → careduce_add_fn                    │
│   └─ Maps to jnp.sum                                             │
│                                                                  │
│ Generated Python Source:                                         │
│   def jax_funcified_fgraph(x):                                   │
│       _constant_2 = jnp.array(2)                                 │
│       _x_squared = elemwise_pow_fn(x, _constant_2)              │
│       _y = careduce_add_fn(_x_squared)                           │
│       return _y                                                  │
│                                                                  │
│ Compile source → Return callable function                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ JAXLinker.jit_compile()                                          │
│ jitted_fn = jax.jit(jax_funcified_fgraph)                       │
│                                                                  │
│ JAX traces function:                                             │
│   x (ShapedArray) → jnp.power → jnp.sum → scalar               │
│                                                                  │
│ XLA compiles to optimized machine code                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ create_jitable_thunk()                                           │
│ def thunk():                                                     │
│     inputs = [input_storage[0][0]]                               │
│     outputs = jitted_fn(*inputs)                                 │
│     output_storage[0][0] = outputs                               │
│                                                                  │
│ Return (thunk, input_storage, output_storage)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXECUTION: f([1.0, 2.0, 3.0])                                   │
│                                                                  │
│ 1. input_storage[0][0] = [1.0, 2.0, 3.0]                        │
│                                                                  │
│ 2. thunk()                                                       │
│    ├─ inputs = [[1.0, 2.0, 3.0]]                                │
│    ├─ jitted_fn executes XLA code:                               │
│    │   └─ [1,2,3]² = [1,4,9] → sum = 14.0                       │
│    └─ output_storage[0][0] = 14.0                                │
│                                                                  │
│ 3. Return output_storage[0][0] = 14.0                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures Throughout

### Storage Map
```python
storage_map = {
    x: [None],           # Will hold input array
    Constant(2): [2],    # Constant value
    x_squared: [None],   # Intermediate (not used with JIT)
    y: [None]            # Will hold output scalar
}
```

**Note**: With JIT backends, intermediate values are managed by the JIT compiler (JAX/XLA), not stored in `storage_map`.

### Input/Output Storage
```python
# Before execution
input_storage = [[None]]
output_storage = [[None]]

# During execution (f([1.0, 2.0, 3.0]))
input_storage = [[np.array([1.0, 2.0, 3.0])]]
output_storage = [[None]]  # Still None until thunk runs

# After thunk execution
input_storage = [[np.array([1.0, 2.0, 3.0])]]
output_storage = [[14.0]]
```

---

## Comparison: Numba Backend Dataflow

The Numba backend follows a similar pattern with key differences:

### Different at Stage 5.2: Numba Dispatch

**File**: `pytensor/link/numba/dispatch/elemwise.py:265-340`

```python
@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):
    # Numba uses custom vectorization framework
    scalar_op_fn = numba_funcify(op.scalar_op, node=scalar_node)

    # Encode broadcasting patterns
    input_bc_patterns = encode_patterns(node.inputs)
    output_bc_patterns = encode_patterns(node.outputs)

    def elemwise_wrapper(*inputs):
        return _vectorized(
            scalar_op_fn,
            input_bc_patterns,
            output_bc_patterns,
            output_dtypes,
            inplace_pattern,
            constant_inputs,
            inputs,
            core_output_shapes,
            size
        )

    return elemwise_wrapper
```

**Key Difference**: Numba generates explicit loops, JAX uses auto-vectorization.

### Different at Stage 6: Numba JIT

**File**: `pytensor/link/numba/linker.py:12-16`

```python
def jit_compile(self, fn):
    from pytensor.link.numba.dispatch.basic import numba_njit

    jitted_fn = numba_njit(
        fn,
        no_cpython_wrapper=False,
        no_cfunc_wrapper=False
    )
    return jitted_fn
```

**Key Difference**:
- Numba compiles to LLVM IR → native code
- JAX compiles to XLA HLO → native code
- Numba can fall back to Python object mode
- JAX requires all ops to be traceable

---

## Execution Timeline

For `f([1.0, 2.0, 3.0])`:

```
Time (ms)  | Stage                          | What Happens
-----------|--------------------------------|----------------------------------
0.000      | User calls f()                 | Entry into Function.__call__
0.001      | Store input                    | input_storage[0][0] = array
0.002      | Call thunk                     | Enter thunk()
0.003      | Input filtering                | Apply input_filter if any
0.004      | Execute JIT function (1st run) | JAX traces and compiles
           |                                | - Tracing: 10-50ms
           |                                | - XLA compilation: 100-500ms
0.600      | XLA execution                  | Run compiled code on device
0.601      | Store output                   | output_storage[0][0] = 14.0
0.602      | Return                         | Return output value
-----------|--------------------------------|----------------------------------
           | Subsequent calls               | Cached JIT, ~0.1ms
```

**First call is slow** (JIT compilation overhead)
**Subsequent calls are fast** (cached compiled code)

---

## Memory Flow

```
Input Array (NumPy)
[1.0, 2.0, 3.0] (CPU memory)
        ↓
JAX converts to DeviceArray
[1.0, 2.0, 3.0] (GPU/CPU via XLA)
        ↓
XLA executes on device
        ↓ jnp.power
[1.0, 4.0, 9.0] (GPU/CPU)
        ↓ jnp.sum
[14.0] (GPU/CPU)
        ↓
Convert back to NumPy
14.0 (CPU memory)
        ↓
Store in output_storage
output_storage[0][0] = 14.0
```

**Note**: JAX may keep data on GPU for performance. Conversion back to NumPy only happens when returning to Python.

---

## Key Takeaways

1. **Dispatch is Recursive**: `jax_funcify(FunctionGraph)` → `jax_funcify(Elemwise)` → `jax_funcify(Pow)`

2. **Code Generation**: `fgraph_to_python()` generates Python source that chains operations

3. **JIT Compilation**: Backend-specific (JAX uses XLA, Numba uses LLVM)

4. **Storage Contract**: Single-element lists `[value]` for all variables

5. **First Call Overhead**: JIT compilation happens on first execution, cached for subsequent calls

6. **Modularity**: Each component is independent:
   - Linker orchestrates
   - Dispatch converts ops
   - Utils generate code
   - JIT compilers optimize

7. **Extensibility**: Add new ops by registering `@{backend}_funcify.register(NewOp)`

---

## Relevant Code Paths

### Compilation Path
1. `pytensor/compile/function/__init__.py` - `function()` entry point
2. `pytensor/compile/mode.py:477-492` - JAX Mode definition
3. `pytensor/link/basic.py:683-707` - `JITLinker.make_all()`
4. `pytensor/link/jax/linker.py:18-113` - JAX-specific conversion/compilation
5. `pytensor/link/jax/dispatch/basic.py:49-62` - FunctionGraph dispatch
6. `pytensor/link/utils.py:666-808` - Code generation
7. `pytensor/link/jax/dispatch/elemwise.py` - Elemwise/CAReduce dispatch
8. `pytensor/link/jax/dispatch/scalar.py` - Scalar op dispatch

### Execution Path
1. `pytensor/compile/function/types.py` - Function wrapper
2. `pytensor/link/basic.py:616-681` - Thunk creation
3. JAX XLA runtime - Actual execution

---

## Summary

The backend implementation follows a clear pipeline:

1. **Graph** → (optimization) → **Optimized Graph**
2. **Optimized Graph** → (dispatch) → **Backend Functions**
3. **Backend Functions** → (code gen) → **Python Source**
4. **Python Source** → (compile) → **Executable Function**
5. **Executable Function** → (JIT) → **Compiled Code**
6. **Compiled Code** → (thunk) → **Callable**

Each backend customizes steps 2-5, while steps 1 and 6 are shared infrastructure.
