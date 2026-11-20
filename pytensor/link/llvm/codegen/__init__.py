"""The toolkit for generating code: nothing here knows about specific ops (their
implementations are in `dispatch/`).

* `interface`: what an op registers (`llvm_emit`, `llvm_layout`) and what its emitter
  gets (its inputs, `NodeOutput`s) -- start here;
* `types`: LLVM representations of PyTensor types;
* `num`: numbers in generated code (an IR value with its dtype);
* `emit`: arrays, errors, temporary memory, casts, loops;
* `signature`: how a node's function is called (`Signature`, `node_outputs`);
* `node`: the function of a node, around its emitter or kernel;
* `kernel`: the kernel ABI wrapper around a function;
* `call`: calls from generated code to kernels (Rust, numba, compiled on their own);
* `graph`: running a whole graph from generated code (a Scan's loop body, a Blockwise's
  core, a compiled unit, the program's function).
"""

# The ops' implementations register themselves with `llvm_emit` and `llvm_layout` when
# imported. Code generation always imports this package first, so importing them here
# makes sure they are registered wherever code is generated.
import pytensor.link.llvm.dispatch
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.kernel import compile_node
from pytensor.link.llvm.codegen.signature import (
    OutputMode,
    new_output,
    node_outputs,
    node_signature,
)
