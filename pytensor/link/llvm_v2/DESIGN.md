# The LLVM backend, second design

The graph stays in PyTensor. Code is generated from Python: a dispatch function per Op
emits LLVM IR through a small wrapper around LLVM (`pytensor_llvm.ir`), using the real
node — its types, constants, `view_map`/`destroy_map`, `infer_shape`. Rust provides the
wrapper, the compiler (optimization, machine code, JIT, object cache), the runtime
(interpreter, workspace, buffers, C API) and precompiled kernels for the hard ops.

How much is compiled together is decided in the graph: an `OpFromGraph` with
`unit=UnitKind.COMPILED` becomes one function, `UnitKind.INTERPRETED` runs node by
node, and the `llvm_units` rewrite groups small nodes into compiled units.

## 1. Values: one LLVM representation per PyTensor type

A representation carries only what the type doesn't fix; static information becomes
constants in the code.

| PyTensor type | LLVM representation |
|---|---|
| `ScalarType(dtype)` | the scalar by value: `double`, `float`, `i64`, `i32`, `i16`, `i8`; `i1` for bool |
| `TensorType`, ndim 0 | `ptr` (address of the element) |
| `TensorType`, ndim ≥ 1 | `{ ptr data, [d x i64] lengths, [s x i64] strides }` |
| `NoneTypeT` | no argument |

* `d` is the number of axes whose length isn't static, in axis order.
* Strides are in bytes. `s` is the number of axes whose stride the value's layout
  (section 1a) doesn't fix; the others are computed in the callee.
* In memory, bool is a byte (`i8`); by value it is `i1`.
* Other types (slices, random generators, ...) have no representation yet; nodes using
  them can't be generated.

## 1a. Layouts

A `Layout` (layout.py) is what is known before running about a value's strides. Each
axis is `PACKED`, a static number of bytes, or `DYNAMIC`, and an axis order says how
axes nest in memory. Packed axes are the innermost block of that order: a packed axis's
stride is the itemsize times the lengths of the axes inside it. Axes of static length 1
don't matter and are always packed. C-contiguous is "all packed, identity order";
`x.T` of a C array is `[P0, P1]`, `x[:, :k]` is `[?, P0]`, `s[:, ::2]` of a static
`(4, 5, 6)` array is `[240, 96, P0]`.

Layouts are claims the code relies on (packed strides aren't passed, they're computed),
so they come only from how a value is produced:

* new memory has the layout the op's `llvm_outputs` declares (default C-contiguous;
  Elemwise follows its inputs); Rust kernels and Scan allocate C-contiguous, graph
  outputs the planner provides are C-contiguous, constants are stored C-contiguous;
* a view's layout follows from its input's, as the op's `llvm_outputs` declares it
  (DimShuffle, Subtensor, Split, Reshape, SpecifyShape, ...; helpers in layout.py); a
  view nothing is declared for is DYNAMIC;
* numba fall-backs and graph inputs are DYNAMIC. With `input_layout="c"` the inputs are
  C-contiguous: Python calls copy other arrays, the C API rejects them.

The planner is authoritative: it propagates layouts while it picks each step, so they
reflect what actually runs (a numba fall-back gives DYNAMIC outputs). It specializes
every generated step for its inputs' layouts, and compiled units do the same for the
values inside them. Scan's step program gets the layouts of what the Scan kernel
passes it (rows of the sequences, rows of its own C buffers, the non-sequences).
`program.dump()` shows the layouts a step is specialized for, e.g.
`Sum{axis=0} {[P0, P1]}`.

`LayoutFeature` gives the same inference on a FunctionGraph, for rewrites and
inspection (`fgraph.llvm_layout(var)`), like the DestroyHandler does for views. It
predicts what the planner will do; replacing a node's input invalidates everything
downstream (a weaker layout can make derived facts wrong, not just stale).

**In generated code.** An `Array` carries its `Layout` (unpacked from the signature,
kept through views, also inside compiled units where views are read back from their
slots). The shared loop builder uses it:

* loops follow memory order: the order most operands share (`loop_order`);
* a flat loop when every operand is packed in that order;
* the layouts decide this at compile time; only operands whose layout leaves it open are
  checked at run time, and then there is one extra variant (flat or nest), not one per
  combination: an elementwise op with six C inputs and one unknown has one check.

So far Elemwise, FusedElemwise (the flat loop) and CAReduce use it: a reduction loops
in its input's memory order and accumulates the innermost memory axis in a register
(summing `x.T` over axis 0 reads rows contiguously), and a full reduction of an array
contiguous in any order is one flat loop. FusedElemwise's loop nest and the Rust kernels
don't follow the layout yet.

Code adapts to the layout, not the other way around: nothing copies an array to give
it a better layout (a strided pass is usually cheaper than a copy and a contiguous
pass; where a copy pays, e.g. data reused many times, the kernel decides at run time
with the real sizes).

`check_layouts=True` (or `PYTENSOR_LLVM_CHECK_LAYOUTS=1`) makes every generated step
verify the layouts it was specialized for, to test the inference.

## 2. Generated functions

The LLVM function of a node (or of a compiled unit) has one argument per input and
output:

```llvm
define internal i32 @f(ptr %ctx, ptr %env, ptr %ws, <inputs>, <outputs>)
```

* `ctx`: the runtime context (errors, allocation, dynamic workspace).
* `env`: the function's closure: large constants and handles of called kernels.
* `ws`: the static workspace, provided by the caller (see 4).
* Inputs: by value, in their representation.
* Outputs: by mode (see 3).
* Returns 0 on success, nonzero on failure with the message set on `ctx`.

Generated functions call each other with this signature. The output slots are allocas
in the caller and disappear after inlining.

## 3. Outputs

How each output comes to be is declared by the implementation that produces it: for
generated code by one dispatch per op, `llvm_outputs(op, node, in_layouts) ->
[Output(repr, mode, view_of)]`, the mode and the layout (section 1a) from the input
layouts; for a Rust kernel by the kernel itself (`Kernel.outputs`, in Rust next to the
kernel); a numba fall-back allocates with unknown layout. The planner picks the
implementation first, then asks it.

The default `llvm_outputs` follows the types and `view_map`: C-contiguous new memory,
views of unknown layout (always true, so always safe). Ops register their own to
describe their views or to choose their outputs' layout (Elemwise uses the memory
order most of its inputs share). Declarations are binding:

* provided: whoever provides the memory provides it in the declared layout. The
  planner's arena memory can have any packed order. A graph output must be
  C-contiguous; one declared otherwise is written to an arena temporary and copied. A
  compiled unit does the same with its own outputs;
* allocated: `SlotOutput.allocate` and temporaries allocate in the declared layout;
* view: `SlotOutput.set_view` checks, when generating the code, that the view array's
  layout implies the declared one (view emitters attach layouts to their arrays).

| mode | argument | the caller | the callee |
|---|---|---|---|
| provided | the representation, like an input | owns the memory, chooses lengths and strides | writes the elements |
| allocated | `ptr` to an output slot | zero-initializes the slot | allocates, fills the slot (one new reference) |
| view of input *i* | `ptr` to an output slot | zero-initializes the slot; sets the owner to input *i*'s | fills data, lengths, strides |
| in place of input *i* | none | passes input *i* as overwritable | writes into input *i* |
| scalar | `ptr` to the value | | stores the value |

An output slot is `{ ptr data, [ndim x i64] lengths, [ndim x i64] strides, ptr owner }`
(all lengths: the caller needs them). Representations don't carry buffer owners, so for
views the caller sets the owner from the input the view lies in.

**Layout.** Nothing is contiguous unless the signature says so (section 1a). Every
output's layout is part of the signature, as declared (above); a provided output can
be declared with any strides (e.g. to write into a slice of a larger array), but the
default is C-contiguous. Code reading an array checks contiguity at run time only where
its layout doesn't settle it.

## 4. Workspace

A function declares how many bytes of static workspace it needs. The caller passes that
much memory (64-byte aligned) as `ws`; the interpreter takes it from its arena, so
temporaries are reused by liveness like any other value. Static temporaries are fixed
offsets into `ws`.

Dynamically sized temporaries come from the context:

* `ptl_ws_alloc(ctx, bytes) -> ptr`: 64-byte aligned, valid until the caller ends the
  call (or releases a mark);
* `ptl_ws_mark(ctx) -> mark`, `ptl_ws_release(ctx, mark)`: reuse inside loops.

## 5. Errors

On failure a function returns nonzero after setting the message
(`ptl_fail(ctx, msg)`). It does not clean up: workspace memory belongs to the caller,
and the caller releases every non-null owner in the output slots. A function therefore
only has to fill a slot's owner as soon as it allocates.

## 6. The C ABI

The interpreter, the kernel trampoline, Rust kernels and the C API use

```c
int32_t f_abi(PtCtx *ctx, PtArray *args /* inputs, then outputs */, const void *env, void *ws);
```

with `PtArray { data, shape, strides, owner, ndim }`. A wrapper is generated from the
LLVM function when needed: it unpacks each `PtArray` into its representation (loading
only dynamic lengths and non-fixed strides), calls the function (inlined), and copies
output slots back into their `PtArray`s, also on failure.

`ScalarType` values never cross the C ABI: functions with scalar inputs or outputs have
no wrapper. The interpreter only ever holds tensors; every scalar variable lives inside a
compiled unit (conversion inserts 0-d tensors at unit boundaries where needed).

**Contract.** The caller guarantees dtypes, static lengths, layouts the type promises,
valid memory, and that provided outputs don't overlap inputs. The function checks what
types don't promise (equal dynamic lengths for broadcasting, index bounds). With
`PTLLVM_CHECKED=1` the wrapper verifies the contract on every call.

## 7. `pytensor_llvm.ir`

A small LLVM wrapper (`rust/pytensor-llvm/src/ir`) on top of `llvm-sys`:

* `Module` owns an LLVM context; `Function`, `Value` and `Builder` refer to it.
* Every builder method checks its operands (types, module, terminated blocks) before
  calling LLVM, and raises a Python exception otherwise: with assertions off, LLVM's
  behavior on malformed IR is undefined.
* Control flow comes as `with b.loop(n) as i:`, `with b.if_(cond):`, and closures for
  values that flow out: `b.fold(n, init, body)`, `b.if_else(cond, then, else_)`.
* Things that need care are Rust methods: libm calls with vector variants,
  intrinsics, runtime calls.
* `Module.compile()` verifies, optimizes, generates machine code and adds it to the JIT;
  objects are cached by a hash of the IR text and the target.

## 8. Programs

`program.py` plans a FunctionGraph into steps for the Rust interpreter
(`rust/pytensor-llvm/src/rt`): OpFromGraphs without a unit are inlined, compiled ones
become units, interpreted ones nested programs; nodes touching ScalarType values are
grouped into units; FusedElemwise stays one node when its pattern is supported. Each
node then runs a Rust kernel (`kernels.py`), generated code (`codegen/`), or numba.
Python assigns arena offsets by liveness (static temporaries and step workspaces) and
release points; all generated modules compile in one parallel batch.

`program.dump()` prints a program: every value with its type and storage (input,
constant, arena offset, fresh output, allocated by its step, view of another value),
then every step with its arguments, workspace and releases.

Scan is a Rust kernel that runs its step function, a nested program, once per step;
the interpreter itself has no control flow.

`ScalarLoop` (inside Elemwise or FusedElemwise) is generated like Composite: the loop
is a `b.fold(n, init, body, until=...)` whose body is the inner scalar graph emitted
inline, so LLVM sees one loop with no calls. A while loop stops after the first step
whose until flag is set and returns the flag as its last output.

## 9. Calling from C

`program.c_function()` returns a new `PtFunction*` reference and
`pytensor_llvm.rt.c_api_table()` (also `program.c_api_table()`) the matching
`PtFunctionApi`; the header is `pytensor_llvm.c_header()`, the same as for the first
backend, with the same contract (per-thread `PtCallState`, provided or runtime-allocated
outputs, `last_error`, no GIL). The handle holds the program and the objects that own
its machine code, so it outlives the Python function. Memory provided for an output is
written directly by the producing step when the output has a static shape and the
caller's strides are C-contiguous; otherwise the result is copied there.

## 10. Profiling

`program.set_profiling(True)` times every step (two clock reads per step), in nested
programs too (a step that is a program, Scan's step program). `program.profile()`
returns rows `(depth, description, calls, seconds)`, `program.profile_report()` a table,
slowest first; `reset_profile()` clears the counters. Counters are atomics on the
program, so calls from several threads (or through the C API) add up.

With `PYTENSOR_ENABLE_PERF=1` generated functions keep frame pointers (part of the cache
key) and every linked object is written to `/tmp/perf-<pid>.map`. The planner labels
each module with its step (`ptl:<step description>`), and functions appear as
`ptl:<step>::<function> [<hash prefix>]`, e.g.
`ptl:FusedElemwise{Exp, reduce[add@(0,)]}::abi [39ddfa22]`.

## 11. Relation to the first backend

The first backend (mode `"LLVM"`, `pytensor/link/llvm`) still exists and shares the
JIT, the precompiled kernels (`native.rs`, `linalg.rs`) and the perf support with this
one. The earlier Rust-side graph (`src/v2`) has been removed.
