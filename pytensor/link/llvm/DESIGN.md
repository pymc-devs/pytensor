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
- `scan_save_mem_no_prealloc`, the save-memory scan rewrite for JIT linkers;
- `llvm_scalar_adv_index_to_basic`, which rewrites `x[g, 0]` into
  `x[:, 0][g]`, and likewise for inc/set. It runs before fusion, so the
  remaining single-array gather can use the native take and the FusedElemwise
  rewrites. It lives in `tensor/rewriting/subtensor.py` and is opt-in by name.

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
- **Scratch.** `rt->scratch(ctx, n)` hands out 64-byte-aligned temporary memory
  that is valid until the kernel returns, for example faer's work space or a
  contiguous copy of a strided vector.
  - It comes from a bump arena owned by the context, one per call state.
  - A request that doesn't fit gets its own overflow block, so earlier pointers
    stay valid.
  - When the kernel returns, the arena regrows to the peak demand, so from the
    second call on, scratch requests don't allocate.
  - Drivers that call kernels repeatedly with the same context (the Scan fast
    path) use `scratch_mark` / `scratch_release` around each call.

## Planner (`Program::new`)

The planner runs once per function, over the graph in topological order:

1. **Static layouts.** Each variable gets a `Layout` of `C`, `F` or `Any`.
   Inputs are `Any`. Constants and anything the runtime allocates are `C`.
   Transpose flips C and F; elemwise follows its inputs. Layouts don't depend
   on storage, so this pass comes first.
2. **Output modes** per node (`out_mode`): provided, kernel-allocated, or view
   of input `i`. Some of these depend on layout or static shapes:
   - `Reshape` is a view only if its input is statically C-contiguous, and a
     copy otherwise;
   - `Alloc` / `AllocEmpty` are provided (a planner slot, no malloc per call)
     when the output shape is static, and kernel-allocated otherwise.

   Kernels are told their mode through their params; they never infer it from
   the pointers. A stale data pointer left over from the previous call caused a
   use-after-free once.
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
  a fast path for contiguous (or broadcast) strides, selected at runtime, and a
  general strided loop nest. Loop order follows the output layout. Inputs
  without a static layout, and outputs that aren't planner slots (slices inside
  segments), are checked before the fast path is taken. Which dims are length 1
  comes from the output operand, so broadcast copies (`Alloc`) work.
- **Reduce** (CAReduce). The loop order follows the input layout. The inner
  loop has a contiguous version, and accumulators live in registers.
- **Fused** (PyTensor's `FusedElemwise`). One loop nest that combines:
  - plain elementwise outputs;
  - indexed reads (gathers) and indexed writes (scatter/inc);
  - reductions, with several accumulators.

  The outer levels are generic. In the innermost level:
  - inputs that are one value for the whole loop are loaded once, before it;
  - each index array is loaded once per element and checked with a single
    unsigned compare against the smallest length it indexes; negative or
    out-of-range values take a slow path that wraps and checks per array;
  - there is a contiguous version, using element indexing for loop strides and
    for gathers/scatters, and a byte-strided version, chosen at runtime.

  A failed bounds check returns an error.
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

### Native kernels (`native.rs`, `linalg.rs`)

- **Copies and views:** copy, dimshuffle, assert/check views, reshape (view or
  copy), diagonal (`ExtractDiag`, a view).
- **Creation and shapes:** arange, alloc, alloc_empty, shape, shape_i,
  make_vector.
- **Linear algebra with faer, sequential (`Par::Seq`):**
  - matmul (`Dot`);
  - Cholesky, with LAPACK potrf semantics: the other triangle is zero, and the
    result is NaN if the matrix isn't positive definite;
  - triangular solve, with trtrs semantics: NaN for a zero diagonal.

  faer's work space comes from the context scratch.

  One case bypasses faer: matrix-vector products with contiguous short rows,
  like `X @ beta` with X row-major. faer does poorly there, so a hand-written
  loop handles it: 4 rows at a time, 4 lanes along each row. It is
  multiversioned, choosing an AVX2+FMA copy at runtime, because the crate itself
  is built for baseline x86-64.
- **Cumulative ops:** cumsum and cumprod along an axis.
- **sort.**
- **Indexing:**
  - subtensor and inc/set_subtensor, with runtime index resolution;
  - take and inc_take (advanced indexing on one axis), with 1-d fast paths;
  - join.
- **scan:** the Scan driver.

### Ops lowered in the converter

Some ops are written as a small PyTensor graph over supported ops, then
converted in place of the node (`_lower` in `dispatch.py`):

- **`Split`** becomes basic slices. With constant split sizes those slices are
  static, so they can live in segments.
- **Multi-index advanced indexing** `x[i, j, ...]` covers several integer index
  arrays on consecutive axes, with full slices elsewhere. The case of one array
  plus scalars is rewritten earlier, as described above. Reads and
  inc/set both become:
  1. a flat index, with numpy's negative wrapping and bounds check;
  2. a reshape of the indexed axes into one;
  3. the 1-d take / inc_take kernels.

  This covers the packed triangle of a Cholesky factor (`chol[rows, cols]`) and
  `ab[g, 0]`.

## Segments

Many graphs contain long runs of tiny ops: scalars, shapes, 3-element vectors.
A separate kernel call for each costs more than the work itself. `form_segments`
groups consecutive nodes that are statically small (at most `segment_max_elems`
elements) and can be expressed as inlined members. Each group compiles to one
function. The group's external inputs and outputs go through the normal ABI;
everything else lives on the stack.

Segment members, besides Elemwise, Reduce, Shape_i, MakeVector and Assert:
- **Basic slices and integer indices** with constant bounds become views
  (`SegLoc::Slice`), resolved statically with Python semantics. The base may be
  large (the joined parameter vector), as long as the result is small.
- **`Join`** becomes copies into slices of the output.
- **`Alloc`** becomes a broadcast copy.
- **`IncSubtensor` / `SetSubtensor`** with static indices become a copy of `x`,
  then `+=` or `=` into a slice of the copy.
- **`FusedElemwise`** without indexed writes.

With these, small nutpie-style functions become one kernel. Those functions
slice one joined vector, compute, and join the gradients. eight schools runs
in 0.12 µs (1 kernel, plus a view for the scalar output).

`whole_graph_segment(graph)` turns an entire graph into one segment if possible.
Scan uses it for its step function.

## Compilation and caching

- **Parallel and cached.** Kernels compile in parallel. Each object file is
  cached on disk under `PTLLVM_CACHE_DIR` or a default cache directory. The key
  covers the spec hash, `CODEGEN_VERSION`, LLVM version, CPU and features, opt
  level and vector library.
- **Loading.** Cached objects go straight into the LLJIT; there is no IR step.
  A kernel already linked in this process is reused by name.
- **Target CPU.** The host CPU by default. `PTLLVM_CPU=haswell` (for example)
  targets another CPU, which is useful for running under valgrind, since
  valgrind doesn't support AVX-512.
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

## Safety

Generated kernels do no bounds checks, apart from indexed reads and writes, and
Rust kernels work on raw pointers. So the backend's memory safety rests on a few
invariants. This section lists them, who establishes each one, and who relies on
it. Code that relies on one says so in a `// SAFETY:` comment. Clippy's
`undocumented_unsafe_blocks` is on, and `unsafe_op_in_unsafe_fn` is denied
outside the three kernel modules.

### 1. Arrays from outside are validated once, at the boundary

Established by `bind` (inputs) and `check_provided` (caller memory for outputs),
both via `validate_array`. The checks are:
- ndim and static shapes;
- non-negative lengths;
- a size that fits `isize`;
- non-null data for non-empty arrays;
- data and strides aligned to the itemsize;
- an address range that doesn't overflow;
- provided outputs don't overlap any input or each other.

The C API checks each array's `ndim` and its shape/stride pointers before
building any slice from them. The Python path copies byte-swapped or misaligned
arrays to native, aligned ones, the same way it converts non-arrays.

Not checkable, so stated as preconditions in `pytensor_llvm.h`:
- In the C API, the dtype must be the one `input_type` reports (`PtArray` has no
  dtype field).
- The memory an array describes must exist.
- A `PtCallState` is used by one thread at a time.

Relied on by everything after: from here on, an argument array is "valid".

### 2. Sizes are computed with checked arithmetic

Established by `checked_nbytes`. Every allocation sized from a shape goes
through it:
- planner slots;
- the alloc, alloc_empty, arange and reshape kernels;
- Scan buffers.

Reshape's element count is checked too. A shape whose element count overflows
`i64` is an error, never a small buffer that a large loop then writes past.

Relied on by kernels that loop over an output's shape.

### 3. The kernel contract

Established by the planner (`Program::new`) and the executor (`execute`). Relied
on by every kernel: generated, Rust or numba. When a kernel is called:

- **Arguments.** Every argument is a valid array (in the sense of point 1) of the
  dtype and ndim the planner recorded. Its shape has been inferred and checked
  for consistency (Elemwise broadcasting, Dot's inner dimension, ...).
- **Provided outputs** are sized from that shape, and their strides follow the
  planned layout.
- **Kernel-allocated outputs** must be allocated by the kernel. The kernel learns
  this from its params, never by inspecting the output's pointer. Guessing from
  the pointer caused a use-after-free once.
- **Declared views** point into input `i`.
- **Aliasing.** Outputs don't overlap inputs, except where the planner chose
  in-place: the same slot, or `Take` of a dying kernel-allocated buffer. For
  in-place Elemwise the aliasing is exact (same pointer and strides), which
  elementwise code handles.
- **Shape and stride storage.** Each variable has its own shape and stride
  storage, so shape slices of different arguments never alias.
- **Params** are the step's own and live as long as the Program.
- **Context.** `ctx` is the call state's context, used by this kernel alone for
  the call. Its scratch memory is valid until the kernel returns.

This is the contract `PTLLVM_CHECKED=1` verifies at runtime, without trusting the
planner. Before each kernel, every argument must lie inside the memory that backs
it (its slot, owned buffer, constant or caller array), and no output may overlap
an input unless the planner chose in-place. Outputs are poisoned with 0xFF (NaN
for floats), so unwritten elements show up. After the call, kernel-allocated
outputs and views must lie inside their buffers. The tests, the model corpus and
PyTensor's scan tests all pass in checked mode, and a deliberately broken kernel
is caught.

### 4. Constants are never written

Established by the in-place selection in `Program::new`: only slot-backed or
kernel-allocated buffers that die at the node qualify, and no other input of the
node may share their root. Graph outputs that are constants are copied.

Relied on by `unsafe impl Sync for ConstData` (a Program is shared across
threads) and by reusing one Program across calls.

### 5. Buffer ownership

Every `PtBuffer` reference has exactly one holder:
- `CallState::owners[v]` while the call runs;
- a kernel output's `owner` field;
- an `OutArray` handed to the caller;
- a numpy capsule;
- an `OwnedBuf`.

`execute` releases everything left on errors and panics (`release_all`). Scan
holds its buffers as `OwnedBuf`, so an error or panic releases them. Releasing
is a decref; at refcount 0 the buffer is freed, with no pool.

### 6. Panics become errors

Rust kernels use the `"C-unwind"` ABI, so a panic (a bug, or running out of
memory) unwinds to the single `catch_unwind` in `execute`. That catch releases
the call's buffers and returns an error; the C API has one more catch at its
entry point.

The runtime table (`rt->alloc`, ...) stays `"C"`, because generated and numba
frames can't be unwound: a panic there aborts. Generated code calls no Rust
code that can panic.

### 7. Threads

- A `Program` is immutable after construction and shared freely: across Python
  threads, and across C callers through the `PtFunction` refcount.
- Each call needs its own `CallState`. Python keeps a pool, so concurrent calls
  run in parallel and the GIL is never released while a lock is held. C callers
  create one state per thread.
- The JIT is a process-wide singleton behind a mutex. JIT code is never freed, so
  kernel addresses stay valid.

### Not covered

- **Generated IR is trusted** to match its `KernelSpec`. That is a codegen
  correctness question, tested against the Python backend and exercised by the
  checked mode, not something types can enforce.
- **The numba fallback** reads `PtArray` and `PtRuntime` through hard-coded
  offsets.
- **Uninitialized memory.** `AllocEmpty` returns it, as numpy's `empty` does.

## Profiling with perf

Set `PYTENSOR_ENABLE_PERF=1` before the first function is compiled (it is read once per
process). Then:

- **Frame pointers.** Generated functions keep a frame pointer
  (`"frame-pointer"="all"`), so `perf record -g` can walk the stack through
  them. The setting is part of the object cache key, so profiling objects and
  normal objects are cached separately.
- **Perf map.** Every JIT-linked function is appended to
  `/tmp/perf-<pid>.map`, which `perf report` reads to name samples in JIT
  memory. Entry points are listed as `ptk_<hash> [<kernel description>]`; the
  hash matches the names in `CompiledFunction.plan()`.

The map holds only names and address ranges. `perf annotate` can't show the
instructions of JIT code; that would need jitdump. Numba fallback kernels are
not listed.

Stacks currently end at `Program::execute`, because the Rust crate is built
without frame pointers. Building with `RUSTFLAGS="-C force-frame-pointers=yes"`
fixes that.

## Testing and benchmarks (current numbers)

**Tests**

- `tests/link/llvm`: 235 tests. They compare against the Python backend and
  cover:
  - the ABI and runtime, and the C API (with a small C harness);
  - the numba fallback;
  - indexing, segments, fused kernels and scan;
  - faer linalg, and the lowered and added ops.

  The main files also run clean under valgrind:
  `PTLLVM_CPU=haswell PYTENSOR_FLAGS=cxx= PYTHONMALLOC=malloc valgrind ...`.
- PyTensor's `tests/scan` under `mode=LLVM`: 223 passed, 20 failed.
  - Almost all of the failures use random generator state, which isn't
    supported.
  - Four check the scan in-place rewrite, which this mode excludes.
  - One pickles a compiled function.

**Model corpus.** 20 PyMC models (synthetic data), compiled with nutpie's own
`_make_functions`, and timed per logp+gradient evaluation from C against
numba's jitted function:
- all 20 run with no numba fallback;
- cold compile takes 10.6 s in total, against 210 s for numba;
- run time is a geometric mean of 1.0× numba: faster on 8 models, slower on 8;
- logp and gradient agree with numba to 2e-9.

Still slower: radon (the fused gather/scatter loop and per-step overhead),
ordered_logistic and categorical_softmax (the lowered multi-index paths and
lots of small steps), and stochastic_volatility.

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
- **Implicit assumptions in generated kernels.** Elemwise assumed its outputs
  had the planner's layout. That was only true until segments started writing
  into slices, and it produced wrong results until the runtime check was added.
  The same kind of unstated assumption probably exists elsewhere. Each
  `KernelSpec` should say what it assumes about every operand.

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
- Segments, and the scan fast path, require fully static shapes. Static slice
  bounds in segments rely on index constants being visible to the planner
  (`ConstInts`); indices computed at runtime keep the slice out of the segment.

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
  That is a substring hack; the ABI should carry an error kind. Size errors come
  back as RuntimeError, where numpy raises ValueError.
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
- **Ops.** Many are missing:
  - argmax/argsort;
  - linalg beyond Dot, Cholesky and triangular solves (det, solve, eigh, ...);
  - Blockwise (batched) linalg;
  - most of `tensor.extra_ops`;
  - random variables;
  - sparse;
  - boolean-mask and non-consecutive advanced indexing.
- **Lowered ops aren't rewritten.** The small graphs that replace Split and
  multi-index indexing are converted as built, without PyTensor's rewrites or
  fusion. For example, multi-index reads pay for a separate bounds-check
  reduction instead of a fused one.
- **Dense linalg is sequential.** faer runs single-threaded (`Par::Seq`), which
  is right for samplers running one chain per thread, but slow for one large
  problem.
- **One hand-written SIMD kernel.** Only the short-row matvec is
  multiversioned. Other Rust kernels are built for baseline x86-64 and don't
  use AVX.

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
   steps listed by `CompiledFunction.info()` show which. The model corpus
   (benchmark scripts outside the repo for now) should become a checked-in
   benchmark.
