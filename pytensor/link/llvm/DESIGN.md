# The LLVM backend

The graph stays in PyTensor. Code is generated from Python: a dispatch function per Op
emits LLVM IR through a small wrapper around LLVM (`pytensor_llvm.ir`), using the real
node — its types, constants, `view_map`/`destroy_map`, `infer_shape`. Rust provides the
wrapper, the compiler (optimization, machine code, JIT, object cache), the runtime
(interpreter, memory, buffers, C API) and Rust kernels for the hard ops.

How much is compiled together is decided in the graph: an `OpFromGraph` with
`unit=UnitKind.COMPILED` becomes one function, `UnitKind.INTERPRETED` runs node by
node, and the `llvm_units` rewrite groups small nodes into compiled units.

## Glossary

*Compile time*

* **graph, node, variable**: PyTensor's.
* **planner** (`program.py`): turns a graph into a program. It names values by **value
  ids** (vids) and schedules **items**: a **node** (`NodeItem`), a **unit** (several
  nodes generated as one kernel, `UnitItem`) or a **subprogram** (`SubprogramItem`).
* **emitter**: the code generation of one op (`llvm_emit`), working with a `Codegen`
  (the builder and the parameters of the kernel being generated).
* **node function**: the LLVM function generated for one node, inlined into kernels.

*Run time*

* **program**: what the interpreter runs: a list of **tasks**. Each task calls one
  **kernel** with its inputs and outputs.
* **kernel**: anything with the kernel ABI (section 6): a **generated kernel** (LLVM: a
  node or a unit), a **Rust kernel**, a **numba kernel**, or a **subprogram** (a nested
  program). Each has a **config**: its read-only data, made when planning (a Rust
  kernel's parameters, a subprogram's `Program`, a generated kernel's link table).
* **link table**: a generated kernel's config: the addresses of the kernels it calls
  (each followed by its config) and of its array constants (section 2a).
* **session**: everything one caller (e.g. one sampler chain) needs to run a compiled
  function: the memory of its calls and its context (C: `PtSession`).
* **context** (`PtCtx`): a session's errors, scratch memory and random state; everything
  a call runs uses it.
* **runtime**: the process-wide part: the runtime functions, the JIT, the thread pool.

*Memory*

* **arena**: a program's static memory, laid out by the planner.
* **frame**: a task's fixed-size memory for one call, in the arena (section 4).
* **scratch**: dynamically sized memory from the context, released by marks.
* **buffer**: refcounted memory (`PtBuffer`), owned through `owner` pointers.
* **output slot**: the struct an allocated or view output is returned through.

*Scan*

* **iteration**: one run of the **loop body** (Scan's inner graph). PyTensor's own
  names (`n_steps`, sit-sot states, ...) keep their meaning.

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
  (section 1a) doesn't fix; the others are computed in the called function.
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

The planner is authoritative: it propagates layouts while it picks each kernel, so they
reflect what actually runs (a numba fall-back gives DYNAMIC outputs). It specializes
every generated kernel for its inputs' layouts, and compiled units do the same for the
values inside them. A Scan's loop body gets the layouts of what the loop passes it (rows of
the sequences, rows of its own C buffers, the non-sequences).
`program.dump()` shows the layouts a task's kernel is specialized for, e.g.
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

`check_layouts=True` (or `PYTENSOR_LLVM_CHECK_LAYOUTS=1`) makes every generated kernel
verify the layouts it was specialized for, to test the inference.

## 2. Generated functions

The LLVM function of a node (or of a compiled unit) has one argument per input and
output:

```llvm
define internal i32 @f(ptr %ctx, ptr %config, ptr %frame, <inputs>, <outputs>)
```

* `ctx`: the context (section 5a): runtime functions, errors, scratch memory, the
  random state.
* `config`: the kernel's link table (section 2a): the kernels it calls and its array
  constants.
* `frame`: the kernel's frame, provided by the caller (see 4).
* Inputs: by value, in their representation.
* Outputs: by mode (see 3).
* Returns 0 on success, nonzero on failure with the message set on `ctx`.

Generated functions call each other with this signature. The output slots are allocas
in the caller and disappear after inlining.

### 2a. Calls between kernels

Every kernel has the kernel ABI the interpreter calls (section 6), and generated code
calls Rust kernels, numba kernels and subprograms the same way (`codegen/call.py`): it
packs `PtArray`s, calls the kernel, and copies what it wrote for allocated and view
outputs (owner included) into the node's output slots. A node implemented by such a
kernel gets a node function like any other (`build_node_function(..., kernel=...)`),
so units and generated loops can contain every kind of node. The called kernel's
scratch is released after the call; an error returns its code (the kernel set the
message).

Addresses aren't part of the IR, which must not depend on the process (the compile
cache is keyed on it). A generated kernel's **link table** (`LinkTable`) holds them as
pointer-sized entries: two per called kernel (its address, then its config), one per
array constant (the address of its contents, a C-contiguous copy kept alive with the
table). The IR loads entry `k` from `config`; the link table is the generated kernel's
config.

Inside generated code the planner decides per node (`Planner.kernel_for`, given the
input layouts): generate it if we can (inlined: preferred; tried in a scratch module,
so a failure leaves nothing behind), else its Rust kernel, else a numba kernel.
Interpreted OpFromGraphs are subprograms when they are tasks of their own, and inlined
inside generated code. A unit releases the buffers the kernels it calls allocate for
values inside it when it returns (also on errors).

### Aliasing

The memory a kernel writes (its provided or allocated outputs) never overlaps its inputs
or its other outputs: the `inplace` rewrites are excluded, the arena never places an
output over a value live during the task, temporaries in a frame aren't reused within a
function, and the C API requires (and checks) that provided outputs don't overlap the
inputs or each other. Views are never written.

Node functions tell LLVM: `Builder.alias_domain(name)` starts a scope domain and a TBAA
root for the function, and `Builder.mark_array(ptr, scope, noalias)` marks the data
pointer of each input (`i<pos>`) and of each written output (`o<j>`, not aliasing the
inputs and other outputs). Pointers derived from a marked one (GEPs, captures into
parallel bodies) inherit the marks; loads and stores through them get `!alias.scope`,
`!noalias` and a `!tbaa` tag of the accessed type. Accesses through anything else
(slots, shapes, pointers through phis or selects) get none, which is always safe.

Domains and TBAA roots are per node function, so claims never cross nodes: after
inlining into a unit, memory may be reused for values of other dtypes (tags of different
roots may alias). This lets LLVM hoist loads of index arrays out of loops that store
into outputs (gathers, scatters, fused indexed reads), and drop run-time overlap checks
before vectorized loops.

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

| mode | argument | the caller | the called function |
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

## 4. Frames and scratch

A generated kernel declares how many bytes of frame it needs: fixed-size memory for
one call. The caller passes that much memory (64-byte aligned) as `frame`; the
interpreter takes it from its arena, so frames are reused by liveness like any other
value. Temporaries of static size are fixed offsets into the frame.

Dynamically sized temporaries come from the context's scratch memory:

* `ptl_rt_scratch(ctx, bytes) -> ptr`: 64-byte aligned, valid until the task ends (or
  an earlier mark is released);
* `ptl_rt_scratch_mark(ctx) -> mark`, `ptl_rt_scratch_release(ctx, mark)`: reuse inside
  loops. The interpreter marks around every task; generated code around every call and
  every Scan iteration, Rust kernels around subprograms, so nested calls never release
  their caller's scratch.

The frame is the planned, static part; scratch the dynamic part.

## 5. Errors

On failure a function returns nonzero after setting the message
(`ptl_rt_set_error(ctx, code, msg)`). It does not clean up: the frame belongs to the caller,
and the caller releases every non-null owner in the output slots. A function therefore
only has to fill a slot's owner as soon as it allocates.

## 5a. Runtime, session, context

Three levels of state:

* **The runtime**, one per process (`abi::runtime()`): the table of runtime functions
  and the process-wide state behind it (the JIT; later e.g. device contexts). The
  functions are defined once (`runtime_functions!`): the C table kernels reach through
  `ctx->rt` (numba fall-backs, older kernels) and the JIT symbols `ptl_rt_<name>`
  generated code links against come from the same list.
* **The session**, one per function per caller, owned by the caller: the C API's
  `PtSession` (e.g. one per sampler chain), one per compiled Python function (calls
  from several threads take turns). It holds the program's memory for a call
  (`CallMemory`) and the context.
* **The context** (`PtCtx`), part of the session: errors, scratch memory and the
  random state. Everything a call runs uses it, subprograms included (they have their
  own memory but their caller's context).

**Random numbers.** A session has one random state (Xoshiro256++), advanced by
every draw, in order: a seeded state gives reproducible draws (`program.seed(n)` in
Python, `api->seed(state, n)` in C); unseeded states start from process entropy.
PyTensor's random generator values are tokens without data here (like `NoneTypeT`): no
argument, no value, a generator output is passed back as the generator it came from,
and two random variables never share draws. The runtime functions `ptl_rt_rng_*` draw
uniform, normal (also many at once, into scratch memory), gamma, beta, poisson and
binomial numbers (`b.rng(ir.Rng.X, ctx, args)`); random variables built from uniform or
normal numbers draw them all at once and transform them in a vectorizable loop
(codegen/random.py). Draws differ from numpy's.

## 5b. Parallel loops and work estimates

`work.py`: `llvm_work(op, node) -> Work` estimates a node's work as a number of items
(lengths, some known only at run time) times a cost per item, in units of one simple
float operation (`scalar_cost`: an add is 1, an `exp` 20, `gammaln` 60; Composite: its
inner graph). Rough on purpose: it separates microseconds from milliseconds.

Generated loops may run in parallel. The builder's `with b.parallel(ctx, n, grain) as
r:` puts the code of the block into an outlined body function, called per chunk
`[r.start, r.end)` by the runtime's `parallel_for` on the runtime's rayon pool (the
process-wide `Runtime`; `pytensor_llvm.rt.set_threads`, `PYTENSOR_LLVM_THREADS`).
Values of the enclosing function used inside are captured automatically (stored in an
environment, loaded in the body), and the enclosing `ctx` stands for the chunk's own
context there: emitters don't change. The loop builder (`for_each`, CAReduce's loops)
uses a region for its flat loop or its outermost loop when the cost estimate says it's
worth it (statically when the shapes are; otherwise the runtime runs a single chunk
directly), with chunks of about `CHUNK_COST`. So far: Elemwise, copies and partial
reductions over a non-reduced outermost axis; FusedElemwise, full reductions and the
Rust kernels don't run in parallel yet.

Chunks only depend on `n` and `grain`; each runs with a child context (its own scratch,
errors: the first failing chunk's is reported; a random state derived from one draw of
the caller's and the chunk index). Results, random draws included, are therefore the
same whatever the number of threads, and whether a session allows parallelism:
`program.set_parallel(False)`, `api->set_parallel(state, 0)` (C API version 3) run the
chunks one after the other, e.g. for samplers already running a chain per core.

## 6. The kernel ABI

Every kernel (generated, Rust, subprogram; numba kernels without `frame`) is called as

```c
int32_t kernel(PtCtx *ctx, PtArray *args /* inputs, then outputs */, const void *config, void *frame);
```

with `PtArray { data, shape, strides, owner, ndim }`. A wrapper is generated from the
LLVM function when needed: it unpacks each `PtArray` into its representation (loading
only dynamic lengths and non-fixed strides), calls the function (inlined), and copies
output slots back into their `PtArray`s, also on failure.

`ScalarType` values never cross the kernel ABI: functions with scalar inputs or outputs have
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

`program.py` plans a FunctionGraph into a program, a list of tasks for the Rust
interpreter (`rust/pytensor-llvm/src/rt`): OpFromGraphs without a unit are inlined,
compiled ones become units, interpreted ones subprograms; nodes touching ScalarType
values are grouped into units; FusedElemwise stays one node when its pattern is
supported. Each node's task then runs a Rust kernel (`kernels.py`), a generated kernel
(`codegen/`), or a numba kernel. Python assigns arena offsets by liveness (static
temporaries and frames) and release points; all generated modules compile in one
parallel batch.

`program.dump()` prints a program: every value with its type and storage (input,
constant, arena offset, fresh output, allocated by its task, view of another value),
then every task with its arguments, frame and releases.

Scan is generated (`codegen/scan.py`): one kernel holds the loop, the buffers and the
loop body, whose items are emitted like a unit's (nodes we generate inlined, the
others called). The buffer conventions are PyTensor's: traced
outputs (mit-mot, mit-sot, sit-sot) are circular buffers whose first rows hold the
initial taps, read at `(i + lookback + tap) % len`; after the loop short buffers are
rotated, while-loops truncated, unused rows zeroed. Body outputs are produced straight
into their rows when the row can't be one the same iteration reads (a static buffer
length decides), else copied there; nit-sot buffers whose rows have no static shape are
allocated by the first iteration; untraced states are read from their output slot in
every iteration and reallocated when an iteration changes their shape. Buffers of
called kernels and scratch are released after every iteration. The interpreter itself
has no control flow.

The Rust Scan kernel (`rt/scan.rs`), running the loop body as a subprogram, remains for
`generate_scan=False` and as the fallback if generating fails.

`ScalarLoop` (inside Elemwise or FusedElemwise) is generated like Composite: the loop
is a `b.fold(n, init, body, until=...)` whose body is the inner scalar graph emitted
inline, so LLVM sees one loop with no calls. A while loop stops after the first
iteration whose until flag is set and returns the flag as its last output.

## 9. Calling from C

`program.c_function()` returns a new `PtFunction*` reference and
`pytensor_llvm.rt.c_api_table()` (also `program.c_api_table()`) the matching
`PtFunctionApi`; the header is `pytensor_llvm.c_header()`, the same as for the first
backend, with the same contract (a `PtSession` per caller, provided or runtime-allocated
outputs, `last_error`, no GIL; `seed(session, n)` since API version 2). The handle holds the program and the objects that own
its machine code, so it outlives the Python function. Memory provided for an output is
written directly by the producing task when the output has a static shape and the
caller's strides are C-contiguous; otherwise the result is copied there.

## 10. Profiling

`program.set_profiling(True)` times every task (two clock reads per task), in
subprograms too (a task that runs a subprogram, the Scan kernel's loop body). `program.profile()`
returns rows `(depth, description, calls, seconds)`, `program.profile_report()` a table,
slowest first; `reset_profile()` clears the counters. Counters are atomics on the
program, so calls from several threads (or through the C API) add up.

With `PYTENSOR_ENABLE_PERF=1` generated functions keep frame pointers (part of the cache
key) and every linked object is written to `/tmp/perf-<pid>.map`. The planner labels
each module with its task (`ptl:<task description>`), and functions appear as
`ptl:<task>::<function> [<hash prefix>]`, e.g.
`ptl:FusedElemwise{Exp, reduce[add@(0,)]}::abi [39ddfa22]`.

## 11. History

An earlier version of this backend described the graph to Rust as an op specification
and generated and interpreted the code there, so the supported ops were fixed in Rust.
This design replaced it: ops are registered in Python, Rust only provides the compiler,
the interpreter and the Rust kernels. The earlier version is in the git history.
