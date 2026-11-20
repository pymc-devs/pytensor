"""Code generation from Python.

* `interface`: what an op registers (`llvm_emit`, `llvm_layout`) and what its emitter
  gets (its inputs, `NodeOutput`s) -- start here;
* `types`: LLVM representations of PyTensor types;
* `emit`: arrays, errors, temporary memory, casts, loops;
* `signature`: how a node's function is called (`Signature`, `node_outputs`);
* `node`: the function of a node, around its emitter or kernel;
* `kernel`: the kernel ABI wrapper around a function;
* `scalar`: scalar op emitters;
* `elemwise`: Elemwise, scalar-op nodes, CAReduce;
* `subtensor`: Subtensor and IncSubtensor;
* `advanced`: gathers and scatters with integer index arrays;
* `extra`: CumOp, ExtractDiag, Split;
* `fused`: FusedElemwise (indexed reads/writes and reductions in one loop);
* `basic`: Shape, Shape_i, MakeVector, Alloc, AllocEmpty, ARange, Reshape, Join, CheckAndRaise;
* `shape`: views (DimShuffle, ViewOp, SpecifyShape) and DeepCopyOp;
* `random`: random variables (draws from the call context's random state);
* `items`: emitting the items of a graph into one function (`emit_items`);
* `call`: calls from generated code to kernels (Rust, numba, compiled on their own);
* `graph`: running a whole graph from generated code (a Scan's loop body, a Blockwise's
  core, a compiled unit, the program's function);
* `scan`: Scan as a generated loop;
* `blockwise`: Blockwise as a generated loop over the batch, running the core node.
"""

from pytensor.link.llvm.codegen import (
    advanced,
    basic,
    blockwise,
    elemwise,
    extra,
    fused,
    graph,
    random,
    scalar,
    scan,
    shape,
    subtensor,
)
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.kernel import compile_node
from pytensor.link.llvm.codegen.signature import (
    OutputMode,
    new_output,
    node_outputs,
    node_signature,
)


scan.register()
