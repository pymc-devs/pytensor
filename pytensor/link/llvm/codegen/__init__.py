"""Code generation from Python: see ../DESIGN.md.

* `types`: LLVM representations of PyTensor types;
* `emit`: arrays, outputs, errors, temporary memory, casts, loops;
* `function`: signatures, node functions, kernel ABI wrappers;
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
* `call`: calls from generated code to kernels, numba and subprograms;
* `scan`: Scan as a generated loop.
"""

from pytensor.link.llvm.codegen import (  # noqa: F401 (registers emitters)
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
from pytensor.link.llvm.codegen.function import (
    OutputMode,
    compile_node,
    llvm_emit,
    new_output,
    llvm_layout,
    node_outputs,
    node_signature,
)

scan.register()
