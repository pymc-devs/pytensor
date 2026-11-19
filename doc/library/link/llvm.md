(libdoc_link_llvm)=
# The LLVM backend

The graph stays in PyTensor. Code is generated from Python: a dispatch function per Op
emits LLVM IR through a small wrapper around LLVM (`pytensor_llvm.ir`), using the real
node — its types, constants, `view_map`/`destroy_map`, `infer_shape`. Rust provides the
wrapper, the compiler (optimization, machine code, JIT, object cache), the runtime
(interpreter, memory, buffers, C API) and Rust kernels for the hard ops.

{doc}`/extending/creating_an_llvm_op` shows how to implement an op.

How much is compiled together is decided in the graph: an `OpFromGraph` with
`execution=Execution.COMPILED` becomes a unit (one kernel), one with
`Execution.INTERPRETED` a subprogram (run node by node), and the `llvm_units` rewrite
groups small nodes into units.

## Glossary

*Compile time*

* **graph, node, variable**: PyTensor's.
* **planner** (`program.py`): turns a graph into a program. It names values by **value
  ids** (vids) and schedules **items**: a **node** (`NodeItem`), a **unit** (several
  nodes generated as one kernel, `UnitItem`) or a **subprogram** (`SubprogramItem`).
* **emitter**: the code generation of one op (`llvm_emit`), working with a `Codegen`
  (the builder and the parameters of the kernel being generated).
* **node function**: the LLVM function generated for one node, inlined into kernels.
* **kernel wrapper**: the function with the kernel ABI generated around a node or unit
  function (its symbol is `kernel`).
* **output mode** (`OutputMode`): how an output comes to be (section 3): provided,
  returned (new memory, or a view of an input the op's `view_map` names) or in place;
  in generated code also scalar or absent.
* **stages** of planning (section 8): flatten, resolve, tasks, compile, memory.

*Run time*

* **program**: what the interpreter runs: a list of **tasks**. Each task calls one
  **kernel** with its inputs and outputs.
* **kernel**: anything with the kernel ABI (section 6): a **generated kernel** (LLVM: a
  node or a unit), a **Rust kernel**, a **numba kernel**, or a **subprogram** (a nested
  program). Each has a **config**: its read-only data, made when planning (a Rust
  kernel's parameters, a subprogram's `Program`, a generated kernel's link table).
* **link table**: a generated kernel's config: the addresses of the kernels it calls
  (each followed by its config) and of its array constants (section 2a).
* **session**: everything one caller (e.g. one sampler chain) needs to run a program:
  the memory of its calls and its context (C: `PtSession`).
* **function** (`PtFunction`): what the C API calls a program, together with the
  objects that keep its machine code alive. In Python it is always a program.
* **context** (`PtCtx`): a session's errors, scratch memory and random state; everything
  a call runs uses it.
* **runtime** (`runtime.rs`): the process-wide part: the runtime functions (`ctx->rt`,
  `ptl_rt_*`), the JIT, the thread pool.
* **interpreter** (`interp/`, Python `pytensor_llvm.interp`): runs programs.

*Memory*

* **arena**: a program's static memory, laid out by the planner.
* **temporary**: memory a kernel needs only while it runs (section 4): stack memory if
  it is small and of static size, else scratch.
* **scratch**: temporary memory from the context, freed piece by piece by generated
  code, else released by marks (at the latest when the task ends).
* **buffer**: refcounted memory (`PtBuffer`), owned through `owner` pointers.
* **output slot**: the struct a returned output comes back through.

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

* new memory has the layout the op's `llvm_layout` declares (default C-contiguous;
  Elemwise follows its inputs); Rust kernels and Scan allocate C-contiguous, graph
  outputs the planner provides are C-contiguous, constants are stored C-contiguous;
* a view's layout follows from its input's, as the op's `llvm_layout` declares it
  (DimShuffle, Subtensor, Split, Reshape, SpecifyShape, ...; helpers in layout.py); a
  view nothing is declared for is DYNAMIC;
* an output that overwrites an input (section 3) has that input's layout;
* numba kernels and graph inputs are DYNAMIC. With `input_layout="c"` the inputs are
  C-contiguous: Python calls copy other arrays, the C API rejects them.

The planner is authoritative: it propagates layouts while it picks each kernel, so they
reflect what actually runs (a numba kernel gives DYNAMIC outputs). It specializes
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
define internal i32 @f(ptr %ctx, ptr %config, <inputs>, <outputs>)
```

* `ctx`: the context (section 5a): runtime functions, errors, scratch memory, the
  random state.
* `config`: the kernel's link table (section 2a): the kernels it calls and its array
  constants.
* Inputs: by value, in their representation.
* Outputs: by mode (see 3).
* Returns 0 on success, nonzero on failure with the message set on `ctx`.

Generated functions call each other with this signature. The output slots are allocas
in the caller and disappear after inlining.

### 2a. Calls between kernels

Every kernel has the kernel ABI the interpreter calls (section 6), and generated code
calls Rust kernels, numba kernels and subprograms the same way (`codegen/call.py`): it
packs `PtArray`s (with the number of axes and the dtype of each), calls the kernel, and copies what it wrote for allocated and view
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

Inside generated code the planner decides per node (`Resolver.inside`, given the
input layouts): generate it if we can (inlined: preferred; tried in a scratch module,
so a failure leaves nothing behind), else its Rust kernel, else a numba kernel.
Interpreted OpFromGraphs are subprograms when they are tasks of their own, and inlined
inside generated code. A unit releases the buffers the kernels it calls allocate for
values inside it when it returns (also on errors).

### Aliasing

Which memory may overlap is the op's statement, not the implementation's:

* `view_map[o]` lists the inputs output `o` *may* be a view of (one, or several:
  `{0: [0, 1]}`). An output with no entry is new memory, distinct from everything the
  kernel is given.
* `destroy_map[o]` lists the inputs the op invalidates. By itself it says nothing about
  the output; an implementation is free to ignore it and write new memory.

So the memory a kernel writes never overlaps what it reads, with one exception it is
told about: an output that overwrites an input (section 3). The arena never places an
output over a value live during the task, temporaries are distinct memory (stack
slots, or scratch, whose pointers are declared `noalias`), and the C API requires (and
checks) that provided outputs don't overlap the inputs or each other. Memory reached
through a view is only written by a task that overwrites it.

Node functions tell LLVM exactly that: `Builder.alias_domain(name)` starts a scope
domain and a TBAA root for the function, and `Builder.mark_array(ptr, scope, noalias)`
marks the data pointer of each input (`i<pos>`) and of each written output (`o<j>`).
A written output doesn't alias the inputs or the other outputs; an input that is
overwritten carries its output's scope instead of one of its own, and inputs of one
node may overlap each other freely (they are only read). Pointers derived from a
marked one (GEPs, captures into parallel bodies) inherit the marks; loads and stores
through them get `!alias.scope`, `!noalias` and a `!tbaa` tag of the accessed type.
Accesses through anything else (slots, shapes, pointers through phis or selects) get
none, which is always safe.

Domains and TBAA roots are per node function, so claims never cross nodes: after
inlining into a unit, memory may be reused for values of other dtypes (tags of different
roots may alias). This lets LLVM hoist loads of index arrays out of loops that store
into outputs (gathers, scatters, fused indexed reads), and drop run-time overlap checks
before vectorized loops.

## 3. Outputs

How an output comes to be follows from the op and the types, not from the emitter
(`node_outputs` in `codegen/function.py`):

| the op says | the type says | mode |
|---|---|---|
| new memory (no `view_map` entry) | static shape | provided |
| new memory | shape known at run time only | returned, aliases nothing |
| may be a view (`view_map[o] = [i, ...]`) | | returned, may alias inputs `i, ...` |
| destroys input *i*, and the planner grants it | | in place |
| | a ScalarType | scalar |
| | no data (a random generator) | absent |

The one thing an op's implementation declares is the layout of each output, by a
dispatch per op: `llvm_layout(op, node, in_layouts) -> [Layout | None]` (section 1a;
`None`: the default, C-contiguous for new memory and DYNAMIC for something that may be
a view). It describes the output whichever way the emitter produces it. Rust kernels
say the same about theirs in Rust, next to the kernel (`Kernel.outputs`); a numba
kernel returns new memory of unknown layout. Declarations are binding:

* provided: whoever provides the memory provides it in the declared layout. The
  planner's arena memory can have any packed order. A graph output must be
  C-contiguous; one declared otherwise is written to an arena temporary and copied. A
  compiled unit does the same with its own outputs;
* returned: `allocate` gives memory in the declared layout, and `set_view` checks,
  when generating the code, that the view's layout implies the declared one;
* in place: the output has the input's layout.

| mode | argument | the caller | the called function |
|---|---|---|---|
| provided | the representation, like an input | owns the memory, chooses lengths and strides | writes the elements |
| returned | `ptr` to an output slot | zero-initializes the slot | allocates and fills the slot (one new reference), or fills data, lengths and strides of a view of an input the op names |
| in place | none: the output is the input | makes sure nothing else reads the input | writes the input's elements |
| scalar | `ptr` to the value | | stores the value |
| absent | none | | (a value without data: a random generator) |

An output slot is `{ ptr data, [ndim x i64] lengths, [ndim x i64] strides, ptr owner }`
(all lengths: the caller needs them). Representations don't carry buffer owners. For
a returned output that may alias exactly one input, the wrapper sets a null owner to
that input's. One that may alias several comes back without an owner: the interpreter
keeps all the candidates alive as long as the output, and copies it if it leaves the
program.

**What emitters do.** Each output the emitter gets has the same three methods, so an
emitter doesn't need to know the mode:

* `allocate(shape)`: memory for the result (the provided memory, new memory, or,
  in place, the input's memory still holding the input's values);
* `overwrite(cg, x)`: memory holding the values of input `x`, for results computed by
  modifying `x`: `x`'s own memory in place, else `allocate` and a copy;
* `set_view(array)`: the output is this view of an input (`x.view(...)`). Only where
  the `view_map` allows that input, and checked when the code is generated.
  `set_output(cg, out, array)` does this if the output may be a view of the array's
  input and copies otherwise.

Mistakes that are cheap to see are errors when the code is generated (a view of an
input the `view_map` doesn't name, `overwrite` with another array than the destroyed
input, a layout that doesn't hold, an output never produced). Beyond that
implementations are trusted: an emitter that writes to an input it wasn't given as an
output is a bug nothing catches.

**In place.** An emitter that can write its result over an input says so:
`llvm_inplace(op, node) -> {output: input}`, for pairs in the op's `destroy_map`. The
default is `{}`: ignore `destroy_map` and produce new memory, which is always correct.
Elemwise (every element is read before it is written), IncSubtensor and
AdvancedIncSubtensor opt in. The planner grants a pair (`Planner.grant_inplace`) when
nothing else can see the memory:

* the node is one of the graph being planned, and that graph has a DestroyHandler
  (PyTensor then guarantees the destroyed variable isn't a protected input or constant
  and isn't read afterwards, views included);
* the value is not a constant, appears once among the node's inputs, and doesn't stand
  for several variables (an inlined OpFromGraph that hands an input on makes two
  variables one value; PyTensor doesn't know);
* it shares no memory with the node's other inputs. The planner tracks, per value, the
  values whose memory it may lie in (from the tasks' alias declarations);
* if the output is a graph output and the memory is the planner's, it is C-contiguous.

Nodes inside units, in inlined inner graphs, in Scan bodies and Blockwise cores never
work in place. A numba kernel does what PyTensor's numba implementation does, which
overwrites destroyed inputs; where the conditions above don't hold the planner gives it
a copy to overwrite (`Planner.keep_destroyed`), and inside generated code it always
gets copies (`CalledKernel.destroys`).

For the interpreter an in-place output is not a new value: the task's output id is
one of its inputs' (`plan/memory.py` maps it to the value it overwrites, which then
lives until the last reader of either).

**Layout.** Nothing is contiguous unless the signature says so (section 1a). Every
output's layout is part of the signature, as declared (above); a provided output can
be declared with any strides (e.g. to write into a slice of a larger array), but the
default is C-contiguous. Code reading an array checks contiguity at run time only where
its layout doesn't settle it.

## 4. Temporary memory

A kernel has no memory of its own besides its outputs. What it needs while it runs is a
temporary (`Codegen.temp_memory`, `Codegen.temporary` for an array):

* **Stack memory**, for a static size of at most 1 KiB (`STACK_TEMPORARY_BYTES`; at most
  16 KiB per generated function): an `alloca` in the entry block, 64-byte aligned. LLVM
  sees all of it and often keeps it in registers or removes it. Node functions are
  always inlined, so these are slots of the kernel's own stack frame. Inside a
  parallel region the memory is the chunk's own.
* **Scratch memory** of the context, for everything else:
  * `ptl_rt_scratch(ctx, bytes) -> ptr`: 64-byte aligned, or null with the error set.
    Declared like `malloc`: a `noalias` result, `allockind("alloc,uninitialized")`,
    `allocsize(1)`, and touching no memory LLVM sees (`memory(inaccessiblemem:
    readwrite)`).
  * `ptl_rt_scratch_free(ctx, p)`: gives `p` back (anything that isn't a live piece of
    scratch, null included, is ignored). It may touch `p`'s memory (`memory(argmem:
    readwrite, inaccessiblemem: readwrite)`), so nothing using `p` moves past it.
  * `ptl_rt_scratch_mark(ctx) -> mark`, `ptl_rt_scratch_release(ctx, mark)`: release
    everything taken since the mark. The interpreter marks around every task (this
    also frees what an error left behind), generated code around every call of
    another kernel, Rust kernels around subprograms, so nested calls never release
    their caller's scratch.

  Scratch is an arena of the context that keeps its largest size: after the first call
  a request is a bump of a pointer. Pieces may be freed in any order; the arena moves
  back once the pieces on top are all free, never below an outstanding mark.

Generated code frees every piece it takes, so LLVM sees where each one ends (a release
would free memory behind its back: it can't tell which pieces a mark covers). A
`Codegen.scratch_scope` frees what was taken inside it at its end: one around every
function body, every Blockwise and Scan iteration, and every branch that takes scratch.
Scratch of a static size is taken in the entry block of the function being generated
(of a parallel body: per chunk), like stack memory, and freed when the function ends:
a loop asking for it in every iteration gets the same memory, with no calls in the
loop. Other scratch is held only until its scope ends. On errors, nothing is freed: the
task's mark releases it.

Node functions are inlined where they are called, entry block included, so their own
static scratch (above the stack limit) is taken per call: inside a Blockwise or Scan
loop, per iteration. The values passed between the nodes of a loop body are the loop
function's own temporaries, taken once.

Memory that is still used after the function that asked for it returned is never stack
memory: the output of a node inside a unit, when it is a temporary of the unit, is
scratch (`escapes=True`). The node writes the piece into the output slot's owner (null
for a view), and the unit frees it at the end of its own scope.

There is no memory planned per task. An earlier design passed every kernel a *frame*,
a fixed-size block in the program's arena, for its static temporaries. It bought
little over the stack for small temporaries and nothing over scratch for large ones,
and cost an argument in the kernel ABI, a second ABI for numba kernels and a kind of
interval in the planner.

## 5. Errors

On failure a function returns nonzero after setting the message
(`ptl_rt_set_error(ctx, code, msg)`). It does not clean up: scratch belongs to the caller,
and the caller releases every non-null owner in the output slots. A function therefore
only has to fill a slot's owner as soon as it allocates.

## 5a. Runtime, session, context

Three levels of state:

* **The runtime**, one per process (`runtime::runtime()`): the table of runtime functions
  and the process-wide state behind it (the JIT; later e.g. device contexts). The
  functions are defined once (`runtime_functions!`): the C table kernels reach through
  `ctx->rt` (numba kernels, Rust kernels) and the JIT symbols `ptl_rt_<name>`
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
Python, `api->seed(session, n)` in C); unseeded states start from process entropy.
PyTensor's random generator values are tokens without data here (like `NoneTypeT`): no
argument, no value, a generator output is passed back as the generator it came from,
and two random variables never share draws. The runtime functions `ptl_rt_rng_*` draw
uniform, normal (also many at once, into scratch memory), gamma, beta, poisson and
binomial numbers (`b.rng(ir.Rng.X, ctx, args)`); random variables built from uniform or
normal numbers draw them all at once and transform them in a vectorizable loop
(codegen/random.py). Draws differ from numpy's.

## 5b. Parallel loops and work estimates

`work.py`: `llvm_work(op, node) -> Work` estimates a node's work as a number of elements
(lengths, some known only at run time) times a cost per element, in units of one simple
float operation (`scalar_cost`: an add is 1, an `exp` 20, `gammaln` 60; Composite: its
inner graph). Rough on purpose: it separates microseconds from milliseconds.

Generated loops may run in parallel. The builder's `with b.parallel(ctx, n, grain) as
r:` puts the code of the block into an outlined body function, called per chunk
`[r.start, r.end)` by the runtime's `parallel_for` on the runtime's rayon pool (the
process-wide `Runtime`; `pytensor_llvm.set_threads`, `PYTENSOR_LLVM_THREADS`).
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
`program.set_parallel(False)`, `api->set_parallel(session, 0)` (C API version 3) run the
chunks one after the other, e.g. for samplers already running a chain per core.

## 6. The kernel ABI

Every kernel (generated, Rust, numba, subprogram) is called as

```c
typedef struct {
    char *data;
    int64_t *shape, *strides;   /* ndim entries each; strides in bytes */
    PtBuffer *owner;
    int32_t ndim;
    int32_t dtype;              /* a dtype code */
} PtArray;                      /* 40 bytes */

int32_t kernel(PtCtx *ctx, PtArray *args /* inputs, then outputs */,
               int32_t n_inputs, int32_t n_outputs, const void *config);
```

A wrapper is generated from the LLVM function when needed: it unpacks each `PtArray`
into its representation (loading only dynamic lengths and non-fixed strides), calls the
function (inlined), and copies output slots back into their `PtArray`s, also on
failure.

`ScalarType` values never cross the kernel ABI: functions with scalar inputs or outputs have
no wrapper. The interpreter only ever holds tensors; every scalar variable lives inside a
compiled unit (conversion inserts 0-d tensors at unit boundaries where needed).

**What a call says about itself.** The caller fills in `ndim`, `dtype` and the `shape`
and `strides` pointers of every argument, outputs included, and passes the number of
inputs and outputs. Kernels never write these. An output has `data` if the caller
provides its memory, and a NULL `data` if the kernel is to return it (allocated, or a
view). So a kernel can tell whether it is called with the arguments it was made for.
Generated kernels know their arguments from their signature and don't look (with
`check_layouts`, or `PYTENSOR_LLVM_CHECK_LAYOUTS=1`, the wrapper does, on every call:
the counts, every `ndim` and `dtype`, and the layouts the signature claims). Rust
kernels always look.

**Contract.** The caller guarantees what can't be looked at: valid memory for what
the arguments describe, the layouts the types promise, and that provided outputs
overlap no input and no other output (unless the kernel works in place, which only
generated kernels do). The function checks what types don't promise (equal dynamic
lengths for broadcasting, index bounds).

**In Rust**, a `PtArray` is only the layout (`src/abi.rs`): it has no methods. Code
works with the types of `src/array.rs`, each of which has an invariant, an `unsafe`
constructor that takes the invariant on trust (or checks the part of it that can be
checked), and safe methods:

| type | what it is | made by |
|---|---|---|
| `PtArrayRef<'a>` | an array to read: valid, aligned memory of its dtype that nothing writes during `'a` | `from_raw`; `from_raw_checked` / `checked` for arrays from outside (the C API, numpy) |
| `PtArrayMut<'a>` | an array to write, that nothing else uses during `'a` | `from_raw_checked`; an `OutSlot` |
| `OutSlot<'a>` | an output still to produce: `provided()` memory, `alloc(ctx, shape)`, `set(owned_array)` or a view | `from_raw` |
| `OwnedArray` | an array in a buffer we hold a reference to (`OwnedBuf`); writable while nobody else holds the buffer | safe constructors (`zeros`, `copy_of`, `in_buffer`, which checks that the array lies in the buffer) |

The constructor calls are where something is taken on trust, and there are few of
them: the kernel entry (`kernel::entry`, for the arguments of the call), the C API and
the numpy boundary (arrays from outside, checked), and the interpreter (the values it
keeps for a call, which it knows from the plan). Buffers are `OwnedBuf`s (one
reference each, released when dropped); raw `PtBuffer*`s only exist in `PtArray`s and
at the C and numpy boundaries.

**Rust kernels** are written with `src/kernel.rs`: a kernel is a type holding its
parameters that implements `Kernel` (`arity`, and `run(ctx, args)`), and `entry::<K>`
is its function with the ABI above, with a `K` as its config. Writing one needs no
`unsafe`. `Args` hands out the inputs as `PtArrayRef`s and the outputs as `OutSlot`s,
and typed arrays on top (`Array<T>`, `ArrayMut<T>`: also as faer matrices, or lane by
lane along an axis). Asking for input `i` as an array of `T` checks its dtype; asking
for an output checks its dtype, number of axes, and that it is provided or to be
returned as the kernel's mode says; `entry` checks the number of arguments. A kernel
called with other arguments than it was made for fails with an error. Scan
(`interp/scan.rs`) and subprograms (`impl Kernel for Program`) are kernels like the
others. One kernel keeps raw pointers of its own: the hand-vectorized matrix-vector
loop in `linalg.rs`.

**The interpreter** runs a call as `program.begin(memory)` -> `Call` (`set_input`,
`provide` memory for outputs) -> `run(ctx)` -> `Done` (`output`, `take_output`). The
arrays of the call are borrowed until the `Done` is dropped, which also releases what
the call holds. Tasks are `KernelHandle`s: the function and its config, vouched for
once, when the task is added.

## 7. `pytensor_llvm.ir`

A small LLVM wrapper (`rust/pytensor-llvm/src/ir`) on top of `llvm-sys`:

* `Module` owns an LLVM context; `Function`, `Value` and `Builder` refer to it.
* Every builder method checks its operands (types, module, terminated blocks) before
  calling LLVM, and raises a Python exception otherwise: with assertions off, LLVM's
  behavior on malformed IR is undefined.
* Operands may be Python literals: `b.add(i, 1)`, `b.select(c, x, 0.0)`, `b.loop(4)`.
  A literal takes the type of the other operand, or the one the method asks for.
  `b.i64(v)`, `b.f64(v)`, ... make constants.
* Integer arithmetic simplifies what it sees: constants fold, and so do `x + 0`,
  `x * 1`, `x * 0`, `x & 0`, a select on a constant condition, `b.smin`/`b.smax` of
  constants. A static shape therefore stays a constant (`value.const_int`) through the
  arithmetic of an emitter without the emitter doing anything for it.
* Comparisons are `b.slt(a, b)`, `b.eq(a, b)`, ... (and `b.icmp(ir.ICmp.SLT, a, b)`).
* Control flow comes as `with b.loop(n) as i:`, `with b.if_(cond):` followed by
  `with b.else_():`, and closures for values that flow out: `b.fold(n, init, body)`,
  `b.if_else(cond, then, else_)`.
* `value.named("n")` names a value in the IR text, for reading dumps.
* Things that need care are Rust methods: libm calls with vector variants,
  intrinsics, runtime calls.
* `Module.compile()` verifies, optimizes, generates machine code and adds it to the JIT;
  objects are cached by a hash of the IR text and the target.
* The wheel ships type stubs (`pytensor_llvm-stubs`), so editors and type checkers know
  the builder.

On top of it, in Python (`codegen/`):

* `types.ir` is the module itself, imported on first use: emitters write `ir.i64`.
* `num.Num` is a value with its dtype. LLVM integers have no sign and floats have their
  own instructions, so code that depends on the dtype (the scalar ops) is written with
  operators, which pick the instruction: `x + y`, `x // y`, `x < y`, `x.astype(dtype)`,
  `cond.select(a, b)`, `x.maximum(y)`. They follow numpy for the dtype. Operands must
  have the same dtype; `==` stays identity (`x.eq(y)` compares) and a `Num` can't be a
  Python condition. Index arithmetic, which is always i64, uses the builder directly.
* `types.Slot` and `types.PtArray` name the fields of the two structs emitters reach
  into: `b.field_ptr(slot_type(nd), slot, Slot.OWNER)`.

## 8. Programs

`program.py` plans a FunctionGraph into a program, a list of tasks for the Rust
interpreter (`rust/pytensor-llvm/src/interp`). Planning runs in stages, each a module
of `plan/` with its own data, so that each can be read (and tested) alone:

1. **Flatten** (`plan/flatten.py`): the graph becomes a list of items over value ids
   (`plan/values.py`). OpFromGraphs without a unit are inlined, compiled ones become
   unit items, interpreted ones subprogram items; nodes touching ScalarType values are
   grouped into units; FusedElemwise stays one node when its pattern is supported.
2. **Resolve** (`plan/resolve.py`): which implementation runs each item, given the
   layouts of its inputs: generated code (`codegen/`), a Rust kernel
   (`rust_kernels.py`), a numba kernel, a subprogram. For a unit, the body of a
   generated Scan or the core of a Blockwise this resolves all the items inside, in
   order, layouts flowing from one to the next: the result (`Resolved`, `Body`) is
   everything code generation needs. `build_unit_function` and the Scan and Blockwise
   emitters take it as an argument (`Codegen.body`) and never ask the planner.
3. **Tasks** (`Planner.task`): one task per item, with its inputs, outputs, how each
   output comes to be and what it may alias (section 3). This is where memory is
   provided, layouts are recorded and in-place work is granted.
4. **Compile**: all generated modules in one parallel batch.
5. **Memory** (`plan/memory.py`): from the tasks alone, the kind of every value
   (input, constant, provided, returned), lifetimes (a value lives as long as anything
   that may be a view of it), arena offsets by first fit, release points.

Stages 2 and 3 are interleaved for the tasks of the program (a task's output layouts
are the next one's input layouts), but nothing in stage 2 depends on the planner's
state.

`program.dump()` prints a program: every value with its type and storage (input,
constant, arena offset, fresh output, allocated by its task, view of another value),
then every task with its arguments and releases.

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
called kernels and the body's scratch are freed after every iteration. The interpreter itself
has no control flow.

The Rust Scan kernel (`interp/scan.rs`), running the loop body as a subprogram, remains for
`generate_scan=False` and as the fallback if generating fails.

Blockwise is generated the same way (`codegen/blockwise.py`): a loop over the batch
whose body is the core graph (the core op's node and what its `make_node` adds),
resolved for the inputs at one batch index. The outputs are allocated before the loop
when their core lengths are static or named by an input in the signature; then core
outputs are produced straight into their rows, and the loop may run in parallel
(estimated by the core's `llvm_work`). A core length only the core knows (`(m,n)->(k)`)
comes from the first iteration, which allocates the output; that loop is serial, and
an empty batch fails, as the length isn't known.

`ScalarLoop` (inside Elemwise or FusedElemwise) is generated like Composite: the loop
is a `b.fold(n, init, body, until=...)` whose body is the inner scalar graph emitted
inline, so LLVM sees one loop with no calls. A while loop stops after the first
iteration whose until flag is set and returns the flag as its last output.

## 9. Calling from C

`program.c_function()` returns a new `PtFunction*` reference and
`pytensor_llvm.interp.c_api_table()` (also `program.c_api_table()`) the matching
`PtFunctionApi`; the header is `pytensor_llvm.c_header()` (a `PtSession` per caller, provided or runtime-allocated
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

With `PYTENSOR_LLVM_PERF=1` generated functions keep frame pointers (part of the cache
key) and every linked object is written to `/tmp/perf-<pid>.map`. The planner labels
each module with its task (`ptl:<task description>`), and functions appear as
`ptl:<task>::<function> [<hash prefix>]`, e.g.
`ptl:FusedElemwise{Exp, reduce[add@(0,)]}::kernel [39ddfa22]`.

The same functions, with their machine code, go to a jitdump, `jit-<pid>.dump` in
`PYTENSOR_LLVM_JITDUMP_DIR` (default `/tmp`), for `perf annotate`. With debug
information (section 10a) each function also gets its source lines: a jitdump holds one
file:line per address, so of the Python call stack in the debug information, the line
of the innermost frame outside the generic helpers (`emit.py`, `num.py`, `types.py`),
usually the op's emitter. The line tables are read from the object's DWARF (`gimli`,
relocations applied, `addr2line` for the inlined frames). Usage:

```
PYTENSOR_LLVM_PERF=1 PYTENSOR_LLVM_DEBUG_INFO=1 perf record -k 1 -g python script.py
perf inject --jit -i perf.data -o perf.jit.data
perf report -i perf.jit.data        # perf annotate -i perf.jit.data: the Python lines
```

## 10a. Reading the generated code

* `linker.ir()` is the IR as generated, `linker.ir(optimized=True)` after the
  optimization pipeline (recomputed from the IR, per module also
  `stats.modules[k].optimized_ir()`).
* `remarks=True` starts each module with LLVM's optimization remarks (what the loop
  vectorizer did or why not, inlining, unrolling...), each under the instruction of
  the unoptimized IR it is about. The C API passes on remarks of every pass, unlabeled;
  they need LLVM's process-wide `-pass-remarks*` options, which are switched on only
  while one module is optimized for them, with other compilations held off.
* The `debug_info` option (or `PYTENSOR_LLVM_DEBUG_INFO=1`) gives the generated code
  debug information (`ir/debug.rs`): every function a subprogram named after its node
  (unit, parallel body), every instruction the Python call stack that emitted it, as
  Python functions inlined into each other, up to the `build_*_function` that built
  the LLVM function. Node functions are inlined, and LLVM keeps the chain, so in the
  optimized code every instruction still knows its Python code and its node.
  `locations=True` prints that as a comment above the instructions (`with_locations`);
  the remarks show it too. It is part of the IR, so of the cache key.

## 11. History

An earlier version of this backend described the graph to Rust as an op specification
and generated and interpreted the code there, so the supported ops were fixed in Rust.
This design replaced it: ops are registered in Python, Rust only provides the compiler,
the interpreter and the Rust kernels. The earlier version is in the git history.
