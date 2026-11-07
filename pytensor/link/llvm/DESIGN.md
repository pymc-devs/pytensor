# LLVM backend: design notes

Status: proof of concept on branch `llvm-backend`. This document describes what
exists now, why it is built this way, and where it is not clean yet. The last
section ("Rough edges") is the honest list of things that need rethinking before
this could be more than a prototype.

## Motivation

The numba backend goes typed graph → untyped Python source → numba type inference
→ LLVM. We already know every type, shape and dtype when we build that source,
then throw the information away and ask numba to rediscover it. That costs compile
time (seconds per function, cold) and control: we cannot decide about buffers,
layouts or what gets compiled together.

This backend generates LLVM IR directly from the typed graph, driven from a Rust
library. That gives us:

- **Bufferization**: a planner decides once, per function, where every
  intermediate lives, which buffers are reused, what is written in place, and
  which outputs are views.
- **Explicit compilation units**: small kernels compiled in parallel and cached
  on disk individually; small runs of nodes merged into one function
  ("segments").
- **Precompiled kernels**: ops where codegen buys nothing (matmul, sort,
  indexing, shape ops, scan's loop driver) are ordinary Rust functions with the
  same ABI as generated kernels.
- **A C API** for callers like nutpie that call the function from many threads
  with no Python involved.

## Overview

```
PyTensor FunctionGraph (after rewrites in mode "LLVM")
   │  pytensor/link/llvm/dispatch.py: one converter per Op (singledispatch)
   ▼
GraphBuilder (Rust, exposed with pyo3)        ── unsupported op ──► numba cfunc
   │  ir.rs: Graph { vars, nodes, inputs, outputs, constants }       (Foreign node)
   ▼
Program::new (runtime.rs)
   │  layout pass → output modes → storage/liveness plan → segments (segment.rs)
   │  → list of Steps, each with a KernelSpec or a native function + Params
   ▼
get_kernels (compile.rs): KernelSpecs → LLVM IR (codegen.rs) → object files
   │  parallel (rayon), disk cache, loaded into one process-wide ORC LLJIT (jit.rs)
   ▼
CompiledFunction (py.rs) / PtFunction (capi.rs)
   └─ execute(CallState): bind inputs, infer shapes, run steps, hand out outputs
```

Code map (`rust/pytensor-llvm/src`):

| file | contents |
|---|---|
| `abi.rs` | kernel ABI structs: `PtArray`, `PtBuffer`, `PtRuntime`, `PtCtx`, `Params` |
| `dtype.rs` | supported dtypes and upcasting |
| `ir.rs` | graph IR and the `Op` enum |
| `builder.rs` | `GraphBuilder`, the Python-facing way to construct an `ir::Graph` |
| `spec.rs` | `Layout`, `KernelSpec` (what codegen gets), `CODEGEN_VERSION` |
| `codegen.rs` | LLVM IR for elemwise, reduce, fused and segment kernels |
| `compile.rs` | options, target setup, vector library, object cache |
| `jit.rs` | the LLJIT singleton |
| `native.rs` | precompiled Rust kernels and scalar math functions |
| `segment.rs` | forming segments; `whole_graph_segment` for scan steps |
| `runtime.rs` | the planner (`Program`) and executor (`CallState`) |
| `scan.rs` | the Scan driver kernel |
| `capi.rs`, `include/pytensor_llvm.h` | external C API |
| `py.rs` | Python module `pytensor_llvm` |

Python side (`pytensor/link/llvm`): `linker.py` (`LLVMLinker`), `dispatch.py`
(converters), `numba_fallback.py`. `pytensor/compile/mode.py` registers the
`"llvm"` linker and the `"LLVM"` mode.

## Python side

`LLVMLinker` is a `JITLinker`. `fgraph_convert` turns the fgraph into a
`GraphBuilder` and calls `builder.compile(...)`, which returns a
`CompiledFunction`. Linker options: `fastmath`, `opt_level`, `cache`, `dump_ir`,
`segments`, `segment_max_elems` (default 64), `numba_fallback` (default on).

The `LLVM` mode is `fast_run` plus the `LLVM_REWRITES` from `linker.py`:

- the FusedElemwise rewrites (`fuse_indexed_into_elemwise`, `fuse_elemwise`,
  `wrap_reduced_gather_in_elemwise`);
- `scan_save_mem_no_prealloc`, the save-memory scan rewrite for JIT linkers.

The linker declares these rewrites incompatible:

- `cxx_only` and `BlasOpt`: there are no C ops;
- `local_careduce_fusion`;
- `scan_reduce_trace_prealloc`;
- `inplace`: the planner does its own buffer reuse.

`GraphConversion` maps PyTensor variables to builder ids. It has three entry
points:

- `var()` creates a builder variable or constant on first use.
- `convert_node()` calls the registered converter. On `NotImplementedError` it
  falls back to numba, unless that is disabled.
- `inline()` converts an inner graph (OpFromGraph, or a FusedElemwise that the
  native spec can't express) in place. It uses its own id scope, because merged
  nodes can share one inner graph.

Converters are thin: they translate op parameters into builder calls. All
decisions about memory, layout and kernels are made in Rust.

### Numba fallback

For an unsupported op, `numba_fallback_kernel(node)` builds a numba `cfunc` with
the kernel signature `int32(ctx, args, params)`:

- **Inputs** become numpy views built with numba intrinsics that read
  `PtArray`s.
- **Outputs** are allocated with `ctx->rt->alloc`, so the result is a
  kernel-allocated `PtBuffer`.
- **Exceptions** are caught and turned into a nonzero return value.

The cfunc is cached through PyTensor's numba cache, keyed by a sha256 of the
node signature plus `FALLBACK_VERSION`. In the graph it is a `Foreign{addr}`
node, and a Python reference keeps the cfunc alive.

## Kernel ABI

Every kernel, generated or native, has the same signature:

```c
int32_t kernel(PtCtx *ctx, PtArray *args, const void *params);

typedef struct { char *data; int64_t *shape; int64_t *strides; /* bytes */
                 PtBuffer *owner; int64_t ndim; } PtArray;     /* 40 bytes */
typedef struct { uint32_t version; alloc; buffer_incref; buffer_decref;
                 scratch; set_error; } PtRuntime;
typedef struct { const PtRuntime *rt; ... } PtCtx;
```

- **Arguments.** `args` holds inputs followed by outputs. `params` is a typed
  blob (`Params`, stored as `Vec<u64>`), owned by the step, that carries static
  information such as axes, index descriptions or a pointer to scan data.
- **Return value.** 0 means success. Nonzero means failure, with a message set
  through `rt->set_error`.
- **Strides are arbitrary.** Kernels must handle them, but may have fast paths
  for layouts the planner proves statically.
- **Every output has one of three modes, fixed when the function is planned:**
  - *provided*: the planner passes memory with known shape and strides, and the
    kernel writes into it;
  - *kernel-allocated*: the kernel calls `rt->alloc`, which returns a
    refcounted `PtBuffer`, and fills in data, shape, strides and owner. This is
    needed whenever the output shape is only known inside the kernel, such as a
    while-scan or a fallback op;
  - *declared view*: the output aliases input `i`. The kernel only sets data,
    shape and strides; the driver sets the owner.
- **Memory.** Buffers are freed when their refcount drops to zero. There is no
  arena or pool on purpose: if the planner can't prove reuse, memory is freed
  and allocated again.
- **Scratch.** `rt->scratch` is a per-call-state scratch area for kernels that
  need temporary space.

## Planner (`Program::new`)

The planner runs once per function, over the graph in topological order:

1. **Static layouts.** Each variable gets a `Layout` of `C`, `F` or `Any`.
   Inputs are `Any`. Constants and anything the runtime allocates are `C`.
   Transpose flips C and F; elemwise follows its inputs. Layouts don't depend
   on storage, so this pass comes first.
2. **Output modes** per node (`out_mode`): provided, kernel-allocated, or view
   of input `i`. Some of these depend on layout. For example, `Reshape` is a
   view only if its input is statically C-contiguous, and a copy otherwise.
3. **Storage.** Each variable gets one of:
   - `Input(k)`, `Const(k)` or `View{root}`;
   - `Slot(i)`: a reusable buffer;
   - `KernelAlloc`;
   - `Take`: written in place into a dying kernel-allocated input, taking over
     its buffer.

   Liveness analysis assigns slots and reuses a slot once its last reader has
   run. Slots persist in the `CallState` across calls, so steady-state calls
   don't allocate. An in-place candidate is chosen per output when an input of
   the same shape and dtype dies at the node.
4. **Segments.** See below.
5. **Kernel specs.** One `KernelSpec` per generated step. `get_kernels`
   deduplicates them by hash.

**Execution** (`execute`):

- Inputs are bound with their actual strides.
- Shapes are inferred lazily, per step and in Rust.
- Slots are resized only when needed.
- Kernels run in order.
- Outputs are either written into caller-provided memory or handed out with
  their owner. Python gets zero-copy numpy arrays whose base is a capsule that
  holds the `PtBuffer` reference.

## Code generation

`codegen.rs` emits one LLVM function per `KernelSpec`, plus a thin wrapper with
the ABI signature.

- **Elemwise.** An n-ary scalar expression tree over broadcast inputs. There is
  a fast path for inputs with contiguous (or broadcast) strides, selected at
  runtime, and a general strided loop nest. Loop order follows the output
  layout.
- **Reduce** (CAReduce). The loop order follows the input layout. The inner
  loop has a contiguous version, and accumulators live in registers.
- **Fused** (PyTensor's `FusedElemwise`). One loop nest that combines:
  - plain elementwise outputs;
  - indexed reads (gathers) and indexed writes (scatter/inc);
  - reductions, with several accumulators.

  The outer levels are generic. The innermost level has a contiguous version
  and a strided version, chosen at runtime. A failed bounds check on an index
  returns an error.
- **Segment.** A run of small nodes compiled as one function: the member
  kernels are `alwaysinline`, and intermediates live in stack temporaries.
- **Scalar math.** Scalar ops use LLVM intrinsics or libm, and declare libmvec
  vector variants through `vector-function-abi-variant` attributes. That lets
  the loop vectorizer use `exp`, `log1p`, `tanh` and so on, even where LLVM 18's
  built-in table lacks them. The libm declarations are `memory(none)`. A few
  functions (erf, gammaln, digamma, ...) are Rust functions exported to the JIT
  as absolute symbols.
- **Fast math.** By default floating-point code uses `arcp contract afn reassoc
  nsz`. It deliberately doesn't use `nnan`/`ninf`, so NaN and inf still
  propagate.

### Native kernels (`native.rs`)

- **Copies and views:** copy, dimshuffle, assert/check views, reshape (view or
  copy).
- **Creation and shapes:** arange, alloc, alloc_empty, shape, shape_i,
  make_vector.
- **matmul:** uses `matrixmultiply`, with gemv and dot fast paths.
- **sort.**
- **Indexing:**
  - subtensor and inc/set_subtensor, with runtime index resolution;
  - take and inc_take (advanced indexing on one axis), with 1-d fast paths;
  - join.
- **scan:** the Scan driver.

## Segments

Many graphs contain long runs of tiny ops: scalars, shapes, 3-element vectors.
A separate kernel call for each costs more than the work itself. `form_segments`
groups consecutive nodes that are statically small (at most `segment_max_elems`
elements) and can be expressed as inlined members. Each group compiles to one
function. The group's external inputs and outputs go through the normal ABI;
everything else lives on the stack.

`whole_graph_segment(graph)` turns an entire graph into one segment if possible.
Scan uses it for its step function.

## Compilation and caching

- **Parallel and cached.** Kernels compile in parallel. Each object file is
  cached on disk under `PTLLVM_CACHE_DIR` or a default cache directory. The key
  covers the spec hash, `CODEGEN_VERSION`, LLVM version, CPU and features, opt
  level and vector library.
- **Loading.** Cached objects go straight into the LLJIT; there is no IR step.
  A kernel already linked in this process is reused by name.
- **Typical compile times:**
  - 0.15–0.4 s cold for the benchmark models (numba: 2–5 s);
  - 0.02–0.1 s for scans, including their gradients.

## Scan

Scan is the op that usually breaks new backends, so it gets its own section.

The driver is the native kernel `ptl_scan`. Its params hold a pointer to a
`ScanData`, which carries the scan's info (taps, counts, `as_while`), the
compiled step and the inner output types.

**Step execution** comes in two forms:

- **Fast path.** If the whole inner graph fits into one segment (static shapes,
  at most 1024 elements), the step is a single generated function. Per step, the
  driver passes views of the sequence rows and tap rows, the step's constants,
  and preallocated temporaries for the outputs. One step costs about 40 ns plus
  the step's own work, and the loop allocates nothing.
- **General path.** Otherwise the inner graph becomes its own `Program`, with a
  `CallState` owned by the driver. Inner programs can contain anything,
  including nested scans and numba fallbacks.

**Buffers.**

- The outer state buffers (mit_mot, mit_sot, sit_sot) are copied into
  C-contiguous buffers the driver owns.
- Tap rows are addressed as `(i + lookback + tap) % len`, so the circular
  buffers produced by the save-memory rewrites work unchanged.
- nit_sot buffers are allocated before the first step if their shape is static,
  and after it otherwise.
- Untraced sit_sot state is kept as a single value. It is reallocated if its
  shape changes.
- While-loops stop when the condition output is true.

**Post-processing** follows PyTensor's Python implementation: rotate circular
buffers, truncate after a while-loop, and zero any unused tail.

**Gradients** are scans with mit_mot inputs and outputs. Their tap layout is
the same as mit_sot's.

## External C API

`include/pytensor_llvm.h`. From Python, `pytensor_llvm.c_api_table()` gives the
address of a `PtFunctionApi` table, and `compiled.c_function()` gives a new
`PtFunction` reference. Consumers link against nothing and go through the table:

- `function_incref` / `function_decref`;
- `n_inputs` / `n_outputs` and `input_type` / `output_type` (dtype and static
  shape);
- `state_new` / `state_free`, one state per thread;
- `call(state, inputs, outputs)`, which writes each output either into
  caller-provided memory or into a runtime-allocated buffer the caller releases;
- `last_error`;
- `buffer_decref`.

A `PtFunction` is immutable and can be shared across threads. `call` doesn't
touch Python. The fixed per-call overhead from C is about 1–2 µs for small
functions.

## Testing and benchmarks (current numbers)

**Tests**

- `tests/link/llvm`: 182 tests. They compare against the Python backend and
  cover the ABI and runtime, the C API (with a small C harness), the numba
  fallback, indexing, fused kernels and scan.
- PyTensor's `tests/scan` under `mode=LLVM`: 223 passed, 20 failed.
  - Almost all of the failures use random generator state, which isn't
    supported.
  - Four check the scan in-place rewrite, which this mode excludes.
  - One pickles a compiled function.

**Rough timings** (N = 100k, compared with numba):

| | LLVM | numba |
|---|---|---|
| regression logp+grad | 1.1–1.8 ms | 2.8 ms |
| elemwise-heavy model | 11–13 ms | 52–57 ms |
| small hierarchical model | 167–190 µs | 144–173 µs |
| scan, 1000 steps, forward | 38 µs | 197 µs |
| gradient of that scan | 139 µs | 625 µs |

## Rough edges

These are known problems, roughly ordered by how much they shape the design.

### The kernel/planner contract around layouts is not thought through

- **Two places decide layouts.** The planner fixes a static `Layout` per
  variable. Kernels are also handed actual strides at runtime and pick
  contiguous or strided code paths themselves. So which fast paths actually run
  can only be seen at runtime.
- **The rules are a hand-written `match` over ops in `runtime.rs`.** Wrong
  `Any`s cost speed (copies, strided loops). A wrong `C` would be a correctness
  bug, and nothing checks it.
- **Inputs are always `Any`.** Every kernel that reads an input needs a runtime
  check. A layout in PyTensor's `TensorType` would let us state the layout
  instead. It probably belongs there anyway.
- **"View or copy" depends on the static layout.** For example `Reshape`, and
  in-place candidates. A dynamic decision ("view if contiguous at runtime")
  would need a fourth output mode, "maybe view", with ownership decided per
  call. That isn't designed.
- **Output layouts are fixed.** Provided outputs whose strides don't match the
  planner's choice go through an extra copy, and the kernel can't adapt.

### Op knowledge is scattered

Adding an op means touching all of these:

- `ir.rs` (the `Op` variant);
- `builder.rs` (the constructor and validation);
- `runtime.rs` (`out_mode`, layout rules, shape inference, `op_name`, the
  storage arm);
- `native.rs` or `codegen.rs`;
- `spec.rs` (if generated);
- `dispatch.py`.

These should be one trait per op ("output mode", "layout transfer", "infer
shape", "emit or native fn"), or at least one module per op.

### Shape inference lives in Rust and is ad hoc

- `infer_shapes` reimplements per-op shape rules that PyTensor already has, and
  it runs lazily per step on every call.
- Static shapes come from PyTensor's types, but nothing else is propagated
  statically. For example, a static shape behind a dynamic `Reshape` is lost.
- Segments, and the scan fast path, require fully static shapes.

A cleaner split would be to infer shapes symbolically once, ideally as a
PyTensor shape graph compiled into the program, and only evaluate it per call.

### Kernels that allocate vs. the planner

- Kernel-allocated outputs (`KernelAlloc`) are outside the planner's slot reuse.
  The only reuse is taking over a dying buffer in place.
- Scan always copies its outer state buffers, even when they die at the scan
  node. PyTensor's scan `inplace` rewrite is excluded, and nothing replaces it.

### Scan

- **Fast-path threshold.** The 1024-element limit for the fast path is
  arbitrary. A step just above it falls back to the general path, which costs a
  full `execute` per step (shape inference, slot bookkeeping).
- **Loop not compiled.** The loop itself is Rust calling a function pointer.
  For tiny steps, compiling the whole loop in LLVM would allow cross-step
  optimization and save the per-step call.
- **Untraced state reallocation.** Untraced sit_sot reallocates whenever its
  shape changes, which can be every step for growing state.
- **No random generators.** RNG state in scans is unsupported, as are random
  variables in general.

### Error handling

- Errors are strings set through `rt->set_error`, with no error codes.
- The Python layer turns messages containing "out of bounds" into `IndexError`.
  That is a substring hack; the ABI should carry an error kind.
- `SpecifyShape` is an identity view: there is no runtime shape check yet.
- Integer `//` and `%` by zero return 0, like numpy. `INT_MIN // -1` is still
  undefined behaviour in the generated code.

### Numba fallback

- The intrinsics hard-code ABI offsets: `RUNTIME_ALLOC_OFFSET = 8`, and the
  `PtArray` field layout. A change in `abi.rs` must be mirrored by hand, and
  only the ABI version guards it.
- Each fallback node is a separate numba compile, so a cold compile with many
  fallbacks is as slow as numba.
- Only tensor and scalar variables are supported. Ops on other types (random
  generators, sparse, typed lists) can't fall back either.

### Coverage

- **dtypes.** float64, float32, int8/16/32/64, uint8 and bool. Not
  uint16/32/64, float16 or complex.
- **Ops.** Many are missing: linalg beyond Dot, cumsum and cumop, most of
  `tensor.extra_ops`, random variables, sparse.
- **Dot.** Uses `matrixmultiply` rather than a system BLAS. That is fine for
  the small and medium sizes in typical models, but slower for large GEMMs.

### Lifetime and process-level state

- **JIT code is never freed.** There is one process-wide LLJIT, and code stays
  loaded until exit. Long-running processes that compile many distinct
  functions grow. Resource trackers per `Program` would fix this.
- **Cache maintenance is manual.** Invalidation relies on bumping
  `CODEGEN_VERSION` and `FALLBACK_VERSION`, and nothing prunes the cache
  directory.
- **No pickling.** `CompiledFunction` can't be pickled. It could keep the
  builder graph and recompile from the cache.

### Build

- The extension links dynamically against system LLVM 18
  (`llvm18-1-prefer-dynamic`), and is built with `maturin --auditwheel skip`.
  That isn't a distributable wheel.
- Shipping needs one of:
  - a statically linked LLVM;
  - LLVM from a wheel or conda;
  - building against the LLVM that numba's llvmlite already ships.
- The Rust crate lives inside the PyTensor repo with its own build script, and
  is not part of PyTensor's packaging.

## Possible next steps

1. Put memory layout into `TensorType`, and redefine the planner/kernel contract
   around it (static guarantee vs. runtime check, "maybe view" outputs).
2. Restructure ops behind one trait or module each.
3. Compile shape graphs instead of interpreting shape rules in Rust.
4. Planner-aware in-place for kernel-allocated outputs, including scan state.
5. Error kinds in the ABI.
6. A distributable build.
7. More ops, driven by what real PyMC models hit; the foreign (numba fallback)
   steps listed by `CompiledFunction.info()` show which.
