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
* `unit`: compiled units (several nodes in one function), OpFromGraph layouts;
* `call`: calls from generated code to kernels, numba and subprograms;
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
    random,
    scalar,
    scan,
    shape,
    subtensor,
    unit,
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
