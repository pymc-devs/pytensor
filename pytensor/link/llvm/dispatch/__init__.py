"""The LLVM implementations of ops: what each op registers (`llvm_emit`, `llvm_layout`,
see codegen/interface.py). Importing this package registers them all.

* `scalar`: scalar ops (`gamma`: GammaLn and Psi);
* `elemwise`: Elemwise, scalar-op nodes, CAReduce;
* `fused`: FusedElemwise (indexed reads/writes and reductions in one loop);
* `subtensor`: Subtensor and IncSubtensor;
* `advanced`: gathers and scatters with integer index arrays;
* `basic`: Shape, Shape_i, MakeVector, Alloc, AllocEmpty, ARange, Reshape, Join,
  CheckAndRaise;
* `shape`: views (DimShuffle, ViewOp, SpecifyShape) and DeepCopyOp;
* `extra`: CumOp, ExtractDiag, Split;
* `random`: random variables (draws from the call context's random state);
* `scan`: Scan as a generated loop;
* `blockwise`: Blockwise as a generated loop over the batch, running the core graph;
* `ofg`: compiled units (OpFromGraph), which run their inner graph.
"""

from pytensor.link.llvm.dispatch import (
    advanced,
    basic,
    blockwise,
    elemwise,
    extra,
    fused,
    ofg,
    random,
    scalar,
    scan,
    shape,
    subtensor,
)


scan.register()
