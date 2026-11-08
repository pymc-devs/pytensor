# The function-based LLVM backend (LLVM_V2)

A rewrite of the LLVM backend around one idea: every node is a function with a
typed signature, and how much gets compiled together is decided in the PyTensor graph.
It lives next to the first backend (`pytensor/link/llvm`, `rust/pytensor-llvm/src/*.rs`)
until it covers the same ops. Mode: `"LLVM_V2"`.

## Pieces

| | Python | Rust (`rust/pytensor-llvm/src/v2`) |
|---|---|---|
| graph | `dispatch.py` converts a FunctionGraph | `graph.rs`: values, nodes, constants |
| ops | one converter per Op, returns a typed op object | `op.rs` traits, `ops/*` implementations |
| scalar ops | class → class table | `scalar.rs`: one type per op, Composite |
| grouping | `tensor/rewriting/units.py` (`llvm_units`) | `ops/ofg.rs` |
| running | `linker.py` | `interp.rs` (interpreter), `unit.rs` + `compile.rs` (compiled units) |

## Signatures

A node's signature is derived from its types and the op's `view_map`, never inferred
by a planner:

* `Caller`: the output's shape is static, so the caller provides the memory;
* `Callee`: the shape is dynamic, so the callee allocates and hands over the buffer;
* `ViewOf(i)`: the output may share input `i`'s memory (the callee may still return a
  fresh buffer, e.g. a reshape of a non-contiguous array).

Indices are newtypes (`InputIdx`, `OutputIdx`, `Axis`, `ValueId`, `NodeId`, `CalleeIdx`,
...) in `IndexVec`s, so an output index can't be used where an input index is expected.
Axis lengths are `AxisLen::{Static(n), Dynamic}`.

## Ops

An op is a Rust type implementing `TensorOp`, exposed to Python as a class
(`v2.Dot()`, `v2.Elemwise(v2.Add())`). It offers up to two implementations:

* `Kernel`: precompiled Rust. A method on the op's struct, so parameters (`axis`,
  `lower`, ...) are plain fields; there are no parameter blobs. `TensorOp::kernel(sig)`
  instantiates it for the node's types (dtype dispatch happens once).
* `Generate`: LLVM IR emitted into a compiled unit (Elemwise, CAReduce, DimShuffle,
  DeepCopy, compiled OpFromGraph).

Scalar ops are their own trait (`ScalarOp`) and are always generated, inlined into the
loop calling them. Composite is a scalar op whose body is a scalar graph. Elemwise holds
a scalar op and supplies the loop nest; CAReduce holds a binary scalar op and an
identity.

## Interpreted and compiled graphs

`OpFromGraph` gets a `unit: UnitKind | None` parameter:

* `COMPILED`: the inner graph becomes one LLVM function. Generated ops are inlined;
  ops that only have a kernel are called through one trampoline
  (`ptl_v2_call_kernel`) with the unit's callee table.
* `INTERPRETED`: the inner graph is a `Program`, which is itself a `Kernel`.
* `None`: inlined into the surrounding graph during conversion.

The function itself is interpreted. A node without a kernel becomes a one-node
compiled unit. The `llvm_units` rewrite wraps convex groups of small, static
Elemwise/CAReduce/DimShuffle nodes in compiled OpFromGraphs. It replaces the old
segments, and the result shows up in `dprint`.

Calling conventions:

* external ABI: `int32 f(PtCtx*, PtArray* args, const EnvRepr* env)` for compiled units
  (inputs then outputs), used by the interpreter and the trampoline;
* inside a unit, an array is an `ArrVal`: data pointer plus one LLVM value per axis
  length and stride, constants where the type knows them;
* Rust kernels are called by the interpreter directly through the trait, with no ABI
  in between.

## Memory

* Interpreter: static temporaries get fixed arena offsets, reused by liveness. Static
  graph outputs get a fresh buffer per call, or the caller's memory when the program
  runs as a kernel. Dynamic outputs are allocated by the kernel and released after
  their last use. Views keep what they view alive.
* Compiled units: static temporaries are on the stack (up to 16 KiB) or in runtime
  buffers. Everything a unit allocates is released at its single exit block; on errors
  that includes the outputs it allocated.
* There is no buffer reuse for dynamic shapes.

## Caching

A unit's object code is cached by a hash of its graph structure, types, the keys of its
generated ops, 0-d constant values, the signature and the target. The callee and
constant tables are rebuilt from the graph each time (`EnvBuilder`), in the order the
code generator consumes them. The key doesn't cover the code generator itself: bump
`UNIT_CODEGEN_VERSION` when generated code changes.

## Status

Ported: Elemwise (all scalar ops of the first backend), CAReduce, DimShuffle, ViewOp,
SpecifyShape, ScalarFromTensor/TensorFromScalar, DeepCopyOp, Dot, OpFromGraph (both
kinds, nested), and the numba fallback for everything else. That includes calling it
from inside compiled units.

Not yet ported: FusedElemwise (currently inlined as plain Elemwise and CAReduce),
subtensor and indexing ops, Join, Reshape, Alloc, ARange, Shape/Shape_i/MakeVector,
CheckAndRaise, Sort, CumOp, ExtractDiag, Cholesky, SolveTriangular, Scan, the C API,
profiling, and the perf map.

## Open questions

* Layout in `TensorType`: types carry `layout: Option<Layout>`, but Python always
  passes `None`, so generated code checks contiguity at run time. Still to decide: how
  a rewrite sets the layout (output types come from `make_node`).
* In-place: inplace rewrites stay excluded, so there is no `destroy_map` handling yet.
* An interpreted OpFromGraph compiles its units separately from the outer program's
  (not in one parallel batch).
* One-node units take constants as inputs, so 0-d constants aren't folded into the
  code there (they are inside grouped units).
