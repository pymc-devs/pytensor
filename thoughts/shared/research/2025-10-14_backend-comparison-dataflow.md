---
date: 2025-10-14T00:00:00-00:00
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pytensor
topic: "Backend Comparison: Complete Dataflow Examples"
tags: [research, backend, comparison, dataflow, jax, numba, c, python, performlinker, compilation]
status: complete
last_updated: 2025-10-14
last_updated_by: Claude
---

# Backend Comparison: Complete Dataflow Examples

**Date**: 2025-10-14
**Researcher**: Claude
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: main
**Repository**: pytensor

## Research Question

How do different PyTensor backends handle the same computation? Provide detailed dataflow examples for `y = pt.sum(x ** 2)` across all available backends.

## Summary

PyTensor supports **6 backends** with fundamentally different execution strategies:

1. **Python (PerformLinker)** - Direct `perform()` method calls, no compilation
2. **C (CLinker)** - Generates and compiles C++ code to shared library
3. **Numba (NumbaLinker)** - JIT compilation via LLVM
4. **JAX (JAXLinker)** - JIT compilation via XLA
5. **PyTorch (PytorchLinker)** - PyTorch tensor operations
6. **MLX (MLXLinker)** - Apple Silicon acceleration

This document provides detailed dataflow examples for the **first 4 backends** using the operation `y = pt.sum(x ** 2)`.

---

## Available Backends in PyTensor

### Backend Locations

| Backend | Linker File | Dispatch Directory | Lines of Code (est.) |
|---------|-------------|-------------------|---------------------|
| **Python** | `link/basic.py:276` (PerformLinker) | N/A (uses Op.perform()) | ~120 |
| **C** | `link/c/basic.py:546` (CLinker) | N/A (Ops provide c_code()) | ~2000+ |
| **JAX** | `link/jax/linker.py:9` (JAXLinker) | `link/jax/dispatch/` (17+ modules) | ~2500+ |
| **Numba** | `link/numba/linker.py:4` (NumbaLinker) | `link/numba/dispatch/` (20+ modules) | ~3000+ |
| **PyTorch** | `link/pytorch/linker.py:5` (PytorchLinker) | `link/pytorch/dispatch/` (12+ modules) | ~1500+ |
| **MLX** | `link/mlx/linker.py:4` (MLXLinker) | `link/mlx/dispatch/` (10+ modules) | ~1200+ |

### Mode Definitions (`compile/mode.py:524-531`)

```python
predefined_linkers = {
    "py": PerformLinker(),
    "c": CLinker(),
    "c|py": OpWiseCLinker(),
    "vm": VMLinker(use_cloop=False),
    "cvm": VMLinker(use_cloop=True),
    "jax": JAXLinker(),
    "numba": NumbaLinker(),
    "pytorch": PytorchLinker(),
}

# Predefined modes
FAST_COMPILE  # Uses 'vm' (Python VM)
FAST_RUN      # Uses 'cvm' (C-accelerated VM)
NUMBA         # Uses NumbaLinker
JAX           # Uses JAXLinker
PYTORCH       # Uses PytorchLinker
MLX           # Uses MLXLinker
```

---

## Example Operation: `y = pt.sum(x ** 2)`

### User Code (Common to All Backends)

```python
import pytensor
import pytensor.tensor as pt
import numpy as np

# Define symbolic variables
x = pt.vector('x', dtype='float32')

# Build computation graph
y = pt.sum(x ** 2)

# Graph structure is identical for all backends:
# x (input) → Elemwise(Pow, [x, 2]) → x_squared → CAReduce(Add) → y (output)
```

### Graph Structure (All Backends)

```
FunctionGraph:
  Inputs: [x: TensorType(float32, (?,))]

  Node 0: Apply(Elemwise(Pow), inputs=[x, Constant(2)], outputs=[x_squared])
  Node 1: Apply(CAReduce(Add, axis=None), inputs=[x_squared], outputs=[y])

  Outputs: [y: TensorType(float32, ())]
```

---

## Backend 1: Python (PerformLinker)

### Compilation: `f = pytensor.function([x], y, mode='FAST_COMPILE')`

#### Stage 1: Graph Optimization
- Minimal optimizations (canonicalization only)
- Graph remains: `x → Elemwise(Pow) → x_squared → CAReduce(Add) → y`

#### Stage 2: PerformLinker.make_all() (`link/basic.py:319-396`)

**Storage Creation:**
```python
storage_map = {
    x: [None],           # Input storage
    Constant(2): [2],    # Constant data
    x_squared: [None],   # Intermediate storage
    y: [None]            # Output storage
}

compute_map = {
    x: [True],           # Inputs already "computed"
    Constant(2): [True],
    x_squared: [False],  # Needs computation
    y: [False]
}

input_storage = [[None]]    # Reference to storage_map[x]
output_storage = [[None]]   # Reference to storage_map[y]
```

**Thunk Creation (lines 337-347):**
```python
thunks = []

# Thunk 1: Elemwise(Pow)
thunk1 = Elemwise(Pow).make_py_thunk(
    node=node0,
    storage_map=storage_map,
    compute_map=compute_map,
    no_recycling=[]
)
thunk1.inputs = [storage_map[x], storage_map[Constant(2)]]
thunk1.outputs = [storage_map[x_squared]]

# Thunk 2: CAReduce(Add)
thunk2 = CAReduce(Add).make_py_thunk(
    node=node1,
    storage_map=storage_map,
    compute_map=compute_map,
    no_recycling=[]
)
thunk2.inputs = [storage_map[x_squared]]
thunk2.outputs = [storage_map[y]]
```

**Streamline Function (line 375):**
```python
def streamline_f():
    # Clear no-recycling storage
    for x in no_recycling:
        x[0] = None

    try:
        # Execute thunk 1
        thunk1()
        # GC: Clear storage for temps no longer needed

        # Execute thunk 2
        thunk2()
    except Exception:
        raise_with_op(fgraph, node, thunk)
```

#### Stage 3: Execution - `f(np.array([1.0, 2.0, 3.0]))`

**Step 1: User provides input**
```python
input_storage[0][0] = np.array([1.0, 2.0, 3.0], dtype='float32')
# Now storage_map[x][0] = array([1.0, 2.0, 3.0])
```

**Step 2: Call streamline_f()**

**Thunk 1 Execution:**
```python
# thunk1() is a closure:
def thunk1():
    inputs = [storage_map[x][0], storage_map[Constant(2)][0]]
    # inputs = [array([1.0, 2.0, 3.0]), 2]

    # Call Elemwise(Pow).perform()
    Elemwise(Pow).perform(node0, inputs, [storage_map[x_squared]])
```

**Inside Elemwise(Pow).perform() (`tensor/elemwise.py:662-729`):**
```python
def perform(self, node, inputs, output_storage):
    # inputs = [array([1.0, 2.0, 3.0]), 2]
    # self.ufunc = np.power (created from scalar.pow)

    result = np.power(inputs[0], inputs[1])
    # result = array([1.0, 4.0, 9.0])

    output_storage[0][0] = result
    # storage_map[x_squared][0] = array([1.0, 4.0, 9.0])

    compute_map[x_squared][0] = True
```

**Thunk 2 Execution:**
```python
def thunk2():
    inputs = [storage_map[x_squared][0]]
    # inputs = [array([1.0, 4.0, 9.0])]

    # Call CAReduce(Add).perform()
    CAReduce(Add).perform(node1, inputs, [storage_map[y]])
```

**Inside CAReduce(Add).perform() (`tensor/elemwise.py:1745-1773`):**
```python
def perform(self, node, inputs, output_storage):
    # inputs = [array([1.0, 4.0, 9.0])]
    input = inputs[0]

    if self.axis is None:
        result = np.sum(input)  # Sum all elements
    else:
        result = np.sum(input, axis=self.axis)

    # result = 14.0
    output_storage[0][0] = result.astype(node.outputs[0].dtype)
    # storage_map[y][0] = np.float32(14.0)

    compute_map[y][0] = True
```

**Step 3: Return result**
```python
return output_storage[0][0]  # 14.0
```

### Key Characteristics: Python Backend

- **No Compilation**: Pure Python execution
- **Per-Node Thunks**: One thunk per Apply node
- **Direct NumPy Calls**: Delegates to `np.power` and `np.sum`
- **Storage Cells**: Single-element lists `[value]` for communication
- **Python Overhead**: Function call per operation
- **Easy Debugging**: Can set breakpoints in `perform()` methods
- **Slowest**: Python loop + function call overhead

**Execution Time (first call)**: ~0.01ms (no compilation)
**Execution Time (subsequent)**: ~0.01ms (no caching benefit)

---

## Backend 2: C (CLinker)

### Compilation: `f = pytensor.function([x], y, mode='FAST_RUN')`

#### Stage 1: Graph Optimization
- Applies extensive optimizations (inplace, fusion, etc.)
- For this simple example, graph likely stays the same

#### Stage 2: CLinker.make_thunk() (`link/c/basic.py:1142-1191`)

**Code Generation Process:**

**Step 1: Fetch Variables (`link/c/basic.py:576-640`)**
```python
inputs = [x]
outputs = [y]
orphans = [Constant(2)]  # Constants not from inputs
temps = [x_squared]      # Intermediate results
```

**Step 2: Generate C Code (`link/c/basic.py:641-890`)**

For each variable, generates `CodeBlock` instances:

**Variable: x (Input)**
```c
// In struct init:
PyObject* storage_V1;  // Input storage

// In run():
PyArrayObject* V1;
py_V1 = PyList_GET_ITEM(storage_V1, 0);  // Extract from Python list
V1 = (PyArrayObject*)(py_V1);
// Validate type, shape, etc.
```

**Variable: x_squared (Temp)**
```c
// In struct (reused across calls):
PyArrayObject* V2;  // Temp storage in struct

// In run():
if (V2 == NULL || !PyArray_ISCONTIGUOUS(V2) || ...) {
    // Allocate new array
    V2 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT32, 0);
}
```

**Variable: y (Output)**
```c
// In struct init:
PyObject* storage_V3;  // Output storage

// In run():
PyArrayObject* V3;
// Allocate scalar array
V3 = (PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_FLOAT32, 0);

// After computation: sync back to Python
PyList_SET_ITEM(storage_V3, 0, (PyObject*)V3);
```

**Step 3: Generate Op Code**

**Node 0: Elemwise(Pow) (`tensor/elemwise.py:753-987`)**

Elemwise generates nested loops:
```c
// Op class Elemwise
{
    npy_float32* V1_ptr = (npy_float32*)PyArray_DATA(V1);
    npy_float32* V2_ptr = (npy_float32*)PyArray_DATA(V2);
    npy_intp V1_n = PyArray_DIM(V1, 0);

    // Loop over array
    for (npy_intp i = 0; i < V1_n; i++) {
        // Call scalar pow operation
        V2_ptr[i] = pow(V1_ptr[i], 2.0f);
    }
}
```

**Node 1: CAReduce(Add) (`tensor/elemwise.py:1422-1580`)**

CAReduce generates reduction loop:
```c
// Op class CAReduce
{
    npy_float32* V2_ptr = (npy_float32*)PyArray_DATA(V2);
    npy_intp V2_n = PyArray_DIM(V2, 0);
    npy_float32* V3_ptr = (npy_float32*)PyArray_DATA(V3);

    // Initialize accumulator
    npy_float32 acc = 0.0f;

    // Reduction loop
    for (npy_intp i = 0; i < V2_n; i++) {
        acc = acc + V2_ptr[i];
    }

    // Store result
    *V3_ptr = acc;
}
```

**Step 4: Struct Assembly (`link/c/basic.py:186-326`)**

```cpp
struct __struct_compiled_op_c58f10be {
    PyObject* __ERROR;
    PyObject* storage_V1;  // Input storage
    PyObject* storage_V3;  // Output storage
    PyArrayObject* V2;     // Temp array (reused)

    __struct_compiled_op_c58f10be() {
        memset(this, 0, sizeof(*this));
    }

    int init(PyObject* __ERROR, PyObject* storage_V1, PyObject* storage_V3) {
        this->__ERROR = __ERROR;
        this->storage_V1 = storage_V1;
        this->storage_V3 = storage_V3;
        Py_XINCREF(storage_V1);
        Py_XINCREF(storage_V3);
        return 0;
    }

    void cleanup(void) {
        Py_XDECREF(storage_V1);
        Py_XDECREF(storage_V3);
        Py_XDECREF(V2);
    }

    int run(void) {
        int __failure = 0;
        PyArrayObject* V1 = NULL;
        PyArrayObject* V3 = NULL;

        {  // V1 extract block
            PyObject* py_V1 = PyList_GET_ITEM(storage_V1, 0);
            V1 = (PyArrayObject*)(py_V1);

            {  // V2 allocation block
                if (V2 == NULL) {
                    npy_intp dims[1] = {PyArray_DIM(V1, 0)};
                    V2 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT32, 0);
                }

                {  // Elemwise(Pow) operation
                    npy_float32* V1_ptr = (npy_float32*)PyArray_DATA(V1);
                    npy_float32* V2_ptr = (npy_float32*)PyArray_DATA(V2);
                    npy_intp n = PyArray_DIM(V1, 0);

                    for (npy_intp i = 0; i < n; i++) {
                        V2_ptr[i] = pow(V1_ptr[i], 2.0f);
                    }

                    {  // V3 allocation block
                        V3 = (PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_FLOAT32, 0);

                        {  // CAReduce(Add) operation
                            npy_float32* V2_ptr = (npy_float32*)PyArray_DATA(V2);
                            npy_float32* V3_ptr = (npy_float32*)PyArray_DATA(V3);
                            npy_intp n = PyArray_DIM(V2, 0);

                            npy_float32 acc = 0.0f;
                            for (npy_intp i = 0; i < n; i++) {
                                acc = acc + V2_ptr[i];
                            }
                            *V3_ptr = acc;

                            {  // V3 sync block
                                PyList_SET_ITEM(storage_V3, 0, (PyObject*)V3);
                                Py_INCREF(V3);
                            }
                        }
                    }
                }
            }
        }

        return __failure;
    }
};
```

#### Stage 3: Compilation (`link/c/cmodule.py:2501-2690`)

**Compile Command:**
```bash
g++ -shared -g -O3 -fno-math-errno \
    -march=native -ffast-math \
    -I/path/to/python/include \
    -I/path/to/numpy/include \
    -fvisibility=hidden \
    -o /tmp/pytensor_cache/compiledir_XXXX/mod.so \
    /tmp/pytensor_cache/compiledir_XXXX/mod.cpp
```

**Cache Key:** Hash of source + compilation flags + NumPy ABI version
**Cache Location:** `~/.pytensor/compiledir_*/`

#### Stage 4: Dynamic Loading (`link/c/cmodule.py:2685-2690`)

```python
# Load compiled shared library
module = dlimport('/tmp/pytensor_cache/.../mod.so')

# Get instantiation function
instantiate = module.instantiate

# Create struct instance
cthunk_capsule = instantiate(error_storage, storage_V1, storage_V3)
```

#### Stage 5: Thunk Wrapper (`link/c/basic.py:1693-1767`)

```python
class _CThunk:
    def __init__(self, cthunk, ...):
        from pytensor.link.c.cutils import run_cthunk
        self.run_cthunk = run_cthunk  # C extension function
        self.cthunk = cthunk  # PyCapsule

    def __call__(self):
        failure = self.run_cthunk(self.cthunk)
        if failure:
            # Extract and raise error
            raise exception
```

#### Stage 6: Execution - `f(np.array([1.0, 2.0, 3.0]))`

**Step 1: Store input**
```python
storage_V1[0] = np.array([1.0, 2.0, 3.0], dtype='float32')
```

**Step 2: Call thunk**
```python
thunk()  # _CThunk.__call__
  ↓
run_cthunk(cthunk_capsule)  # C function
  ↓
struct_ptr = PyCapsule_GetContext(cthunk_capsule)
executor_fn = PyCapsule_GetPointer(cthunk_capsule)
  ↓
return executor_fn(struct_ptr)
  ↓
return struct_ptr->run()
```

**Step 3: Inside struct->run() (native C code)**
```c
// Extract V1 from storage: [1.0, 2.0, 3.0]
// Allocate V2 if needed
// Loop 1: Elemwise(Pow)
//   V2[0] = pow(1.0, 2) = 1.0
//   V2[1] = pow(2.0, 2) = 4.0
//   V2[2] = pow(3.0, 2) = 9.0
// Allocate V3
// Loop 2: CAReduce(Add)
//   acc = 0.0 + 1.0 = 1.0
//   acc = 1.0 + 4.0 = 5.0
//   acc = 5.0 + 9.0 = 14.0
//   V3[0] = 14.0
// Sync V3 back to storage
return 0;  // Success
```

**Step 4: Return result**
```python
return storage_V3[0]  # 14.0
```

### Key Characteristics: C Backend

- **Ahead-of-Time Compilation**: Compiles to native code before execution
- **Single Struct**: Entire graph in one C++ struct
- **Explicit Loops**: Hand-written C loops for operations
- **Direct Memory Access**: Pointer arithmetic on NumPy arrays
- **Caching**: Compiled code reused across sessions
- **Fast CPU Execution**: Optimized with `-O3`, `-march=native`
- **Compilation Overhead**: First call requires gcc compilation (~500ms-2s)

**Execution Time (first call)**: ~1000ms (includes compilation)
**Execution Time (subsequent, cached)**: ~0.001ms

---

## Backend 3: Numba (NumbaLinker)

### Compilation: `f = pytensor.function([x], y, mode='NUMBA')`

#### Stage 1: Graph Optimization
- Applies Numba-compatible optimizations
- Graph: `x → Elemwise(Pow) → x_squared → CAReduce(Add) → y`

#### Stage 2: NumbaLinker.make_all() (`link/basic.py:514-547`)

Inherits from `JITLinker`, which creates a single thunk for entire graph.

**Step 1: fgraph_convert() (`link/numba/linker.py:7-10`)**

Calls `numba_funcify(fgraph)` → `fgraph_to_python()`:

**Dispatch for Node 0: Elemwise(Pow)**

Triggers `@numba_funcify.register(Elemwise)` (`link/numba/dispatch/elemwise.py:265-340`):

```python
@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op  # Pow()

    # Get scalar function
    scalar_op_fn = numba_funcify(scalar_op, node=scalar_node)
    # Returns: numba-compiled version of pow()

    # Wrap for vectorization
    core_op_fn = store_core_outputs(scalar_op_fn, nin=2, nout=1)

    # Encode broadcast patterns
    input_bc_patterns = encode_patterns([x, Constant(2)])
    output_bc_patterns = encode_patterns([x_squared])

    def elemwise_wrapper(*inputs):
        return _vectorized(
            core_op_fn,
            input_bc_patterns,
            output_bc_patterns,
            output_dtypes=[np.float32],
            inplace_pattern=None,
            constant_inputs={1: 2},  # Constant(2)
            inputs,
            core_output_shapes=(),
            size=None
        )

    return elemwise_wrapper
```

**_vectorized() is a Numba Intrinsic (`link/numba/dispatch/vectorize_codegen.py:74-274`)**

At compile time, generates LLVM IR:

```python
@numba.extending.intrinsic
def _vectorized(typingctx, core_op_fn, input_bc_patterns, ...):
    # Type inference phase (lines 99-196)
    def typer(core_op_fn, input_bc_patterns, ...):
        # Decode patterns from pickled literals
        # Determine core input types
        # Return signature
        return ret_type(*arg_types)

    # Code generation phase (lines 200-273)
    def codegen(context, builder, sig, args):
        # Step 1: compute_itershape() - broadcast shapes
        # Step 2: make_outputs() - allocate output arrays
        # Step 3: make_loop_call() - generate nested loops

        # Generated LLVM IR (pseudo-code):
        iter_shape = compute_itershape(inputs)  # (3,)
        outputs = make_outputs(iter_shape)      # Allocate array

        # Nested loop generation:
        for i in range(iter_shape[0]):  # i = 0, 1, 2
            # Load inputs (with broadcasting)
            inp0_val = input0_ptr[i]    # x[i]
            inp1_val = 2                 # Constant

            # Call scalar op
            out_val = core_op_fn(inp0_val, inp1_val)

            # Store output
            output0_ptr[i] = out_val

        return outputs[0]

    return sig, codegen
```

**Dispatch for Node 1: CAReduce(Add)**

Triggers `@numba_funcify.register(CAReduce)` (`link/numba/dispatch/elemwise.py:343-410`):

```python
@numba_funcify.register(CAReduce)
def numba_funcify_CAReduce(op, **kwargs):
    scalar_op = op.scalar_op  # Add()
    axis = op.axis            # None (reduce all)

    # Get scalar function
    scalar_op_fn = numba_funcify(scalar_op)

    def careduce(x):
        if axis is None:
            axes_to_reduce = tuple(range(x.ndim))
        else:
            axes_to_reduce = axis

        # Use reduce_using_scalar for custom reduction
        return reduce_using_scalar(x, scalar_op_fn, axes_to_reduce, dtype)

    return careduce
```

**reduce_using_scalar() (`link/numba/dispatch/elemwise.py:205-262`)**

Generates reduction loop:

```python
@numba.extending.overload(reduce_using_scalar)
def reduce_using_scalar_impl(x, scalar_fn, axes, dtype):
    def reduce_impl(x, scalar_fn, axes, dtype):
        # Allocate output (scalar in this case)
        out = np.empty((), dtype=dtype)

        # Initialize accumulator
        acc = scalar_fn.identity  # 0 for Add

        # Flatten to 1D and reduce
        for i in range(x.size):
            val = x.flat[i]
            acc = scalar_fn(acc, val)

        out[()] = acc
        return out

    return reduce_impl
```

**fgraph_to_python() Result:**

Generates Python source:
```python
def numba_funcified_fgraph(x):
    _constant_2 = 2
    _x_squared = elemwise_pow_wrapper(x, _constant_2)
    _y = careduce_add_fn(_x_squared)
    return _y
```

Compiles and returns callable function.

#### Stage 3: jit_compile() (`link/numba/linker.py:12-16`)

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

**numba_njit() (`link/numba/dispatch/basic.py:53-87`)**

```python
@numba.njit(
    cache=config.numba__cache,  # Cache compiled code
    fastmath=config.numba__fastmath,  # LLVM fast-math flags
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True
)
def numba_funcified_fgraph(x):
    # ... (as above)
```

**Numba Compilation Pipeline:**

1. **Type Inference**: Infers types from first call: `x: float32[:]`
2. **Lowering**: Python bytecode → Numba IR
3. **Optimization**: Numba-level optimizations
4. **LLVM Generation**: Numba IR → LLVM IR
   - The `_vectorized` intrinsic directly generates LLVM loop IR
   - Optimizes with fast-math flags: `-ffast-math`, `-march=native`
5. **LLVM Optimization**: LLVM optimization passes (auto-vectorization, loop unrolling)
6. **Machine Code**: LLVM → native code

#### Stage 4: Create Thunk (`link/basic.py:616-681`)

```python
def thunk():
    # Extract inputs from storage
    inputs = [input_storage[0][0]]  # [array([1.0, 2.0, 3.0])]

    # Call JIT-compiled function
    outputs = jitted_fn(*inputs)

    # Store outputs
    output_storage[0][0] = outputs
```

#### Stage 5: Execution - `f(np.array([1.0, 2.0, 3.0]))`

**Step 1: Store input**
```python
input_storage[0][0] = np.array([1.0, 2.0, 3.0], dtype='float32')
```

**Step 2: Call thunk (first time)**

```python
thunk()
  ↓
jitted_fn(array([1.0, 2.0, 3.0]))
  ↓
# Numba compiles on first call
# Type inference: x is float32[:]
# Generates LLVM IR
# Compiles to machine code
  ↓
# Execute compiled code
```

**Step 3: Inside compiled Numba function (LLVM → native code)**

**Elemwise(Pow) - _vectorized intrinsic:**
```llvm
; LLVM IR (simplified)
define float* @elemwise_pow(float* %x, i64 %n) {
entry:
  %output = call @allocate_array(i64 %n, i32 4)  ; Allocate float32 array
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %x_ptr = getelementptr float, float* %x, i64 %i
  %x_val = load float, float* %x_ptr
  %out_val = call @powf(float %x_val, float 2.0)
  %out_ptr = getelementptr float, float* %output, i64 %i
  store float %out_val, float* %out_ptr
  %i.next = add i64 %i, 1
  %cond = icmp ult i64 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret float* %output
}

; With auto-vectorization (AVX2):
; Processes 8 floats at once with SIMD instructions
```

**CAReduce(Add) - reduce_using_scalar:**
```llvm
; LLVM IR (simplified)
define float @reduce_sum(float* %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %acc = phi float [0.0, %entry], [%acc.next, %loop]
  %x_ptr = getelementptr float, float* %x, i64 %i
  %x_val = load float, float* %x_ptr
  %acc.next = fadd float %acc, %x_val
  %i.next = add i64 %i, 1
  %cond = icmp ult i64 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret float %acc
}

; With auto-vectorization:
; Horizontal sum reduction with SIMD
```

**Concrete Execution:**
```
Input: [1.0, 2.0, 3.0]
  ↓ elemwise_pow (SIMD optimized)
[1.0, 4.0, 9.0]
  ↓ reduce_sum (SIMD optimized)
14.0
```

**Step 4: Return result**
```python
output_storage[0][0] = 14.0
return 14.0
```

### Key Characteristics: Numba Backend

- **JIT Compilation**: Compiles on first call
- **LLVM Backend**: Generates LLVM IR → native code
- **Custom Vectorization**: Explicit loop generation via intrinsics
- **Auto-Vectorization**: LLVM can apply SIMD optimizations
- **Type-Specific**: Compiles separate version for each type signature
- **Caching**: Can cache compiled code in `__pycache__`
- **Pure CPU**: No GPU support (without CUDA target)

**Execution Time (first call)**: ~100-500ms (JIT compilation)
**Execution Time (subsequent, cached)**: ~0.002ms

---

## Backend 4: JAX (JAXLinker)

### Compilation: `f = pytensor.function([x], y, mode='JAX')`

#### Stage 1: Graph Optimization
- Applies JAX-compatible optimizations
- Excludes: C++-only, BLAS, fusion, inplace
- Includes: fast_run, jax
- Graph: `x → Elemwise(Pow) → x_squared → CAReduce(Add) → y`

#### Stage 2: JAXLinker.make_all() (`link/basic.py:514-547`)

Inherits from `JITLinker`.

**Step 1: fgraph_convert() (`link/jax/linker.py:18-93`)**

**RNG Handling (lines 23-72):**
- Not applicable (no random variables in our example)

**Scalar Shape Detection (lines 76-89):**
- Not applicable (no shape operations)

Calls `jax_funcify(fgraph)`:

#### Stage 3: jax_funcify(FunctionGraph) (`link/jax/dispatch/basic.py:49-62`)

```python
@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph, **kwargs):
    return fgraph_to_python(
        fgraph,
        jax_funcify,           # Op conversion function
        type_conversion_fn=jax_typify,
        **kwargs
    )
```

**fgraph_to_python() Process:**

**Dispatch for Node 0: Elemwise(Pow)**

Triggers `@jax_funcify.register(Elemwise)` (`link/jax/dispatch/elemwise.py:9-20`):

```python
@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op  # Pow()

    # Get JAX function for scalar op
    base_fn = jax_funcify(scalar_op, node=node, **kwargs)
    # Returns: jnp.power

    def elemwise_fn(*inputs):
        # Runtime broadcast check
        Elemwise._check_runtime_broadcast(node, tuple(map(jnp.asarray, inputs)))
        return base_fn(*inputs)

    return elemwise_fn
```

**Nested dispatch: jax_funcify(Pow())**

Triggers `@jax_funcify.register(ScalarOp)` (`link/jax/dispatch/scalar.py:78-118`):

```python
@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op, node, **kwargs):
    # Pow has nfunc_spec = ("power", 2)
    func_name = op.nfunc_spec[0]  # "power"
    jax_func = getattr(jnp, func_name)  # jnp.power

    return jax_func
```

**Dispatch for Node 1: CAReduce(Add)**

Triggers `@jax_funcify.register(CAReduce)` (`link/jax/dispatch/elemwise.py:23-69`):

```python
@jax_funcify.register(CAReduce)
def jax_funcify_CAReduce(op, **kwargs):
    axis = op.axis  # None
    scalar_op = op.scalar_op  # Add()

    # Add → jnp.add
    # Use sum for Add reduction
    acc_dtype = node.outputs[0].type.dtype  # float32

    def careduce(x):
        if axis is None:
            axes_to_reduce = list(range(x.ndim))  # [0]
        else:
            axes_to_reduce = axis

        return jnp.sum(x, axis=axes_to_reduce).astype(acc_dtype)

    return careduce
```

**fgraph_to_python() Result:**

Generates Python source:
```python
def jax_funcified_fgraph(x):
    _constant_2 = jnp.array(2, dtype='int64')
    _x_squared = elemwise_pow_fn(x, _constant_2)
    _y = careduce_add_fn(_x_squared)
    return _y
```

Where:
- `elemwise_pow_fn` is the closure from `jax_funcify_Elemwise` (calls `jnp.power`)
- `careduce_add_fn` is the closure from `jax_funcify_CAReduce` (calls `jnp.sum`)

Compiles and returns callable function.

#### Stage 4: jit_compile() (`link/jax/linker.py:95-113`)

```python
def jit_compile(self, fn):
    import jax

    # No scalar shape inputs in our example
    jit_fn = jax.jit(fn, static_argnums=[])

    return jit_fn
```

**jax.jit() Process:**

JAX's `jit` performs **tracing** and **XLA compilation**:

1. **Tracing**: Executes function with abstract values
2. **JAXPR Generation**: Creates JAX expression (functional IR)
3. **XLA Lowering**: JAXPR → XLA HLO (High-Level Operations)
4. **XLA Compilation**: HLO → optimized machine code
5. **Caching**: Compiled code cached by input shapes/types

#### Stage 5: Create Thunk (`link/basic.py:616-681`)

```python
def thunk():
    # Extract inputs from storage
    inputs = [input_storage[0][0]]  # [array([1.0, 2.0, 3.0])]

    # Apply input filter (no filtering for JAX)
    filtered_inputs = inputs

    # Call JIT-compiled function
    outputs = jitted_fn(*filtered_inputs)

    # Store outputs
    output_storage[0][0] = outputs
```

#### Stage 6: Execution - `f(np.array([1.0, 2.0, 3.0]))`

**Step 1: Store input**
```python
input_storage[0][0] = np.array([1.0, 2.0, 3.0], dtype='float32')
```

**Step 2: Call thunk (first time)**

```python
thunk()
  ↓
jitted_fn(array([1.0, 2.0, 3.0]))
  ↓
# JAX tracing phase
```

**Step 3: JAX Tracing**

```python
# JAX traces with abstract shapes
x_traced = jax.ShapedArray((3,), dtype='float32')
_constant_2 = jnp.array(2)

# Trace elemwise_pow_fn
_x_squared = jnp.power(x_traced, _constant_2)
# Records: power operation, inputs: (float32[3], int32[]), output: float32[3]

# Trace careduce_add_fn
_y = jnp.sum(_x_squared, axis=[0])
# Records: reduce_sum operation, input: float32[3], output: float32[]

# Build JAXPR (functional IR)
```

**Generated JAXPR (simplified):**
```python
{ lambda ; a:f32[3].
  let b:i32[] = constant 2
      c:f32[3] = pow a b
      d:f32[] = reduce_sum[axes=(0,)] c
  in (d,) }
```

**Step 4: XLA Lowering (JAXPR → HLO)**

```
HLO module {
  ENTRY main {
    %x = f32[3] parameter(0)
    %const = f32[] constant(2)
    %const_broadcast = f32[3] broadcast(%const)
    %pow = f32[3] power(%x, %const_broadcast)
    %init = f32[] constant(0)
    %sum = f32[] reduce(%pow, %init), dimensions={0}, to_apply=add
    ROOT %result = (f32[]) tuple(%sum)
  }

  add {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add = f32[] add(%lhs, %rhs)
  }
}
```

**Step 5: XLA Compilation (HLO → Machine Code)**

XLA applies optimizations:
- **Fusion**: Combines pow + sum into single kernel
- **Vectorization**: Uses SIMD instructions (AVX, AVX2, AVX-512)
- **Layout Optimization**: Optimal memory access patterns
- **Target-Specific**: Can target CPU, GPU, TPU

**Compiled Kernel (pseudo-assembly for CPU):**
```asm
; Fused pow + sum kernel (AVX2 SIMD)
vmovups ymm0, [x]           ; Load 8 floats (may process in chunks)
vbroadcastss ymm1, [2.0]    ; Broadcast constant 2
vmulps ymm0, ymm0, ymm0     ; Square (x * x, faster than pow for ^2)
vhaddps ymm0, ymm0, ymm0    ; Horizontal add (partial sums)
vhaddps ymm0, ymm0, ymm0    ; Continue reduction
; ... final scalar sum
```

**Step 6: Execute Compiled Code**

```
Input: np.array([1.0, 2.0, 3.0])
  ↓ JAX converts to DeviceArray
jax.DeviceArray([1.0, 2.0, 3.0])
  ↓ Execute XLA compiled kernel (fused pow+sum)
jax.DeviceArray(14.0)
  ↓ Convert back to NumPy
np.float32(14.0)
```

**Step 7: Return result**
```python
output_storage[0][0] = 14.0
return 14.0
```

### Key Characteristics: JAX Backend

- **JIT Compilation**: Compiles on first call via tracing
- **XLA Backend**: Generates XLA HLO → optimized code
- **Functional**: Immutable arrays, pure functions
- **Auto-Fusion**: XLA automatically fuses operations
- **Auto-Differentiation**: Built-in grad support
- **Multi-Backend**: CPU, GPU, TPU support
- **Transformations**: jit, grad, vmap, pmap, etc.

**Execution Time (first call)**: ~100-1000ms (XLA compilation)
**Execution Time (subsequent, cached)**: ~0.001ms (CPU), faster on GPU

---

## Comparative Summary

### Compilation Strategy

| Backend | Strategy | When Compiles | Output |
|---------|----------|---------------|--------|
| **Python** | No compilation | N/A | Python bytecode |
| **C** | Ahead-of-time | On first use or cache miss | GCC-compiled `.so` |
| **Numba** | JIT (LLVM) | On first call | LLVM-compiled machine code |
| **JAX** | JIT (XLA) | On first call | XLA-compiled machine code |

### Execution Model

| Backend | Thunks | Fusion | Memory Model |
|---------|--------|--------|--------------|
| **Python** | One per node | None | Storage cells (list[1]) |
| **C** | Single struct | Manual (in Ops) | Direct pointers |
| **Numba** | Single function | Automatic (LLVM) | Direct arrays |
| **JAX** | Single function | Automatic (XLA) | Functional (immutable) |

### Optimization Level

| Backend | Loop Optimization | Vectorization | Parallelization |
|---------|-------------------|---------------|-----------------|
| **Python** | None (NumPy internal) | NumPy's BLAS | NumPy's threading |
| **C** | `-O3` gcc flags | Manual + gcc auto-vec | OpenMP (optional) |
| **Numba** | LLVM passes | LLVM auto-vec | `parallel=True` |
| **JAX** | XLA fusion | XLA auto-vec | GPU/TPU automatic |

### Performance Characteristics

For `y = sum(x**2)` with `x = [1.0, 2.0, 3.0]`:

| Backend | First Call | Cached Call | Memory Overhead | Best For |
|---------|-----------|-------------|-----------------|----------|
| **Python** | ~0.01ms | ~0.01ms | Low | Debugging |
| **C** | ~1000ms | ~0.001ms | Medium (shared lib) | CPU-heavy |
| **Numba** | ~200ms | ~0.002ms | Low (cached) | General purpose |
| **JAX** | ~500ms | ~0.001ms | Medium (XLA buffers) | GPU/research |

### Code Generation Examples

For `Elemwise(Pow)`:

**Python:**
```python
def perform(self, node, inputs, outputs):
    outputs[0][0] = np.power(inputs[0], inputs[1])
```

**C:**
```c
for (i = 0; i < n; i++) {
    output_ptr[i] = pow(input0_ptr[i], input1_ptr[i]);
}
```

**Numba:**
```python
# _vectorized intrinsic generates LLVM IR
@numba.extending.intrinsic
def _vectorized(...):
    # → LLVM loop + auto-vectorization
```

**JAX:**
```python
def elemwise_fn(*inputs):
    return jnp.power(*inputs)  # XLA handles everything
```

### Key Architectural Differences

#### Python (PerformLinker)
- **Philosophy**: Simplicity and debuggability
- **Abstraction**: High (Python/NumPy)
- **Control**: Low (delegates to NumPy)
- **Flexibility**: High (easy to modify)

#### C (CLinker)
- **Philosophy**: Maximum CPU performance
- **Abstraction**: Low (direct C code)
- **Control**: High (explicit loops, memory)
- **Flexibility**: Low (requires C code)

#### Numba (NumbaLinker)
- **Philosophy**: Python convenience + native speed
- **Abstraction**: Medium (Python → LLVM)
- **Control**: Medium (LLVM optimizations)
- **Flexibility**: High (pure Python)

#### JAX (JAXLinker)
- **Philosophy**: Functional, composable, differentiable
- **Abstraction**: High (pure functions)
- **Control**: Low (XLA handles everything)
- **Flexibility**: Medium (functional constraints)

---

## When to Use Each Backend

### Python (PerformLinker)
**Use when:**
- Debugging graph construction
- Developing new Ops
- Quick prototyping
- Small computations where overhead doesn't matter

**Avoid when:**
- Performance critical
- Large arrays
- Production code

### C (CLinker)
**Use when:**
- Maximum CPU performance needed
- Production deployments on CPU
- Long-running processes (amortize compilation cost)
- Custom C implementations available

**Avoid when:**
- Rapid development/iteration
- GPU acceleration needed
- Compilation time is critical

### Numba (NumbaLinker)
**Use when:**
- Need good CPU performance without C code
- Rapid development
- Custom ops in pure Python
- Caching is important

**Avoid when:**
- Need GPU acceleration
- Complex BLAS operations
- Extremely large graphs

### JAX (JAXLinker)
**Use when:**
- GPU/TPU acceleration available
- Need automatic differentiation
- Research/experimentation
- Want functional programming model
- Need transformations (vmap, pmap)

**Avoid when:**
- CPU-only environment
- In-place operations critical
- Need mutable state

---

## Related Research

- `thoughts/shared/research/2025-10-14_backend-dataflow-example.md` - JAX backend detailed dataflow

## Code References

### Backend Implementations
- `pytensor/link/basic.py:276` - PerformLinker
- `pytensor/link/c/basic.py:546` - CLinker
- `pytensor/link/numba/linker.py:4` - NumbaLinker
- `pytensor/link/jax/linker.py:9` - JAXLinker

### Dispatch Systems
- `pytensor/link/jax/dispatch/basic.py:49` - jax_funcify(FunctionGraph)
- `pytensor/link/numba/dispatch/basic.py:333` - numba_funcify(FunctionGraph)

### Code Generation
- `pytensor/link/utils.py:666` - fgraph_to_python()
- `pytensor/link/c/basic.py:641` - CLinker.code_gen()

### Compilation
- `pytensor/link/c/cmodule.py:2501` - GCC_compiler.compile_str()
- `pytensor/link/numba/dispatch/basic.py:53` - numba_njit()
- `pytensor/link/jax/linker.py:95` - JAXLinker.jit_compile()

---

## Conclusion

PyTensor's multi-backend architecture provides flexibility to choose the right tool for each use case:

- **Python** for development and debugging
- **C** for maximum CPU performance
- **Numba** for balanced performance and ease of use
- **JAX** for GPU acceleration and automatic differentiation

All backends share the same graph representation and optimization infrastructure, with backend-specific compilation in the final stage. This separation of concerns makes PyTensor a powerful framework for array computations across different hardware and performance requirements.
