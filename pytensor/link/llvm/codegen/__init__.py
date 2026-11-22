"""The toolkit for generating code: nothing here knows about specific ops (their
implementations are in `dispatch/`).

This package exports what an op's implementation is written against (see
doc/extending/creating_an_llvm_op.md): register with `llvm_emit` (and `llvm_layout`),
emit with `Codegen`, `Array` and `Num`, test with `compile_node`.

The modules:

* `interface`: what an op registers (`llvm_emit`, `llvm_layout`) and what its emitter
  gets (its inputs, one handle per output) -- start here;
* `context`: the state of generating code (`Codegen`): errors, temporary memory, loops;
* `array`: arrays in generated code (`Array`) and the output slots nodes fill (`Slot`);
* `num`: numbers in generated code (an IR value with its type), and loading, storing
  and converting elements;
* `types`: LLVM representations of PyTensor types;
* `destination`: where results go (caller memory, output slots, scalars);
* `signature`: how a node's function is called (`Signature`, `node_outputs`);
* `node`: the function of a node, around its emitter or kernel;
* `kernel`: the kernel ABI wrapper around a function;
* `call`: calls from generated code to kernels (Rust, numba, compiled on their own);
* `graph`: running a whole graph from generated code (a Scan's loop body, a Blockwise's
  core, a compiled unit, the program's function).

What is public: a module's `__all__` lists what code outside this package uses (op
implementations, and the planner and program, which build functions with `node`,
`graph`, `kernel` and `signature`). Other names without an underscore are shared
between the modules here (or used by tests), and a leading underscore marks what only
its own module uses.
"""

# The ops' implementations register themselves with `llvm_emit` and `llvm_layout` when
# imported. Code generation always imports this package first, so importing them here
# makes sure they are registered wherever code is generated.
import pytensor.link.llvm.dispatch
from pytensor.link.llvm.codegen.array import Array, advance, c_strides, size
from pytensor.link.llvm.codegen.context import (
    Codegen,
    ElementBody,
    IndexBounds,
    index_bounds,
    loop_order,
)
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
    EmitterOutput,
    ScalarOutput,
    TensorOutput,
    llvm_emit,
    llvm_layout,
)
from pytensor.link.llvm.codegen.kernel import compile_node
from pytensor.link.llvm.codegen.num import Num, NumLike, cast, constant, load, store
from pytensor.link.llvm.codegen.types import (
    element_type,
    ir,
    is_absent,
    memory_type,
    value_bits,
    value_type,
)


__all__ = [
    "Array",
    "Codegen",
    "ElementBody",
    "EmitterInput",
    "EmitterOutput",
    "IndexBounds",
    "Num",
    "NumLike",
    "ScalarOutput",
    "TensorOutput",
    "advance",
    "c_strides",
    "cast",
    "compile_node",
    "constant",
    "element_type",
    "index_bounds",
    "ir",
    "is_absent",
    "llvm_emit",
    "llvm_layout",
    "load",
    "loop_order",
    "memory_type",
    "size",
    "store",
    "value_bits",
    "value_type",
]
