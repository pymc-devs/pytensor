"""Code generation from Python: see ../DESIGN.md.

* `types`: LLVM representations of PyTensor types;
* `emit`: arrays, outputs, errors, workspace, casts, loops;
* `function`: signatures, node functions, C ABI wrappers;
* `scalar`: scalar op emitters;
* `elemwise`: Elemwise, scalar-op nodes, CAReduce;
* `subtensor`: Subtensor and IncSubtensor;
* `advanced`: gathers and scatters with integer index arrays;
* `extra`: CumOp, ExtractDiag, Split;
* `fused`: FusedElemwise (indexed reads/writes and reductions in one loop);
* `basic`: Shape, Shape_i, MakeVector, Alloc, AllocEmpty, ARange, Reshape, Join, CheckAndRaise;
* `shape`: views (DimShuffle, ViewOp, SpecifyShape) and DeepCopyOp;
* `random`: random variables (draws from the call context's random state);
* `unit`: compiled units (several nodes in one function);
* `call`: calls from generated code to kernels, numba and nested programs;
* `scan`: Scan as a generated loop.
"""

from pytensor.link.llvm_v2.codegen import (  # noqa: F401 (registers emitters)
    advanced,
    basic,
    elemwise,
    extra,
    fused,
    random,
    scalar,
    scan,
    shape,
    subtensor,
)
from pytensor.link.llvm_v2.codegen.function import (
    OutMode,
    compile_node,
    llvm_emit,
    fresh_output,
    llvm_outputs,
    node_outputs,
    view_output,
    node_signature,
)

scan.register()
