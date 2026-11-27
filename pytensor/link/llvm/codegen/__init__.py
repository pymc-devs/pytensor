"""The toolkit for generating code: nothing here knows about specific ops (their
implementations are in `dispatch/`).

This package exports what an op's emitter is written against (see
doc/extending/creating_an_llvm_op.md): emit with `Codegen`, `Array` and `Num`, test with
`compile_node`. Emitters are registered in the registry (registry.py).
Ops that run an inner graph (`Codegen.graph_function`) also use `OutputDestination`
and `Captured`.

The modules:

* `interface`: what an op's emitter gets (its inputs, one handle per output) -- start
  here;
* `context`: the state of generating code (`Codegen`): errors, temporary memory, loops;
* `array`: arrays in generated code (`Array`) and the output slots nodes fill (`Slot`);
* `num`: numbers in generated code (an IR value with its type), and loading, storing
  and converting elements;
* `types`: the LLVM types of values (their representations are in ../types.py), and
  the kernel ABI;
* `destination`: where results go (caller memory, output slots, scalars);
* `node`: the function of a node, around its emitter or kernel;
* `kernel`: the kernel ABI wrapper around a function, and the modules of steps compiled
  on their own (`build_own_module`);
* `call`: calls from generated code to kernels (Rust, numba, compiled on their own);
* `graph`: running a whole graph from generated code (a Scan's loop body, a Blockwise's
  core, a compiled unit, the program's function).

What is public: a module's `__all__` lists what code outside this package uses (op
implementations, and the program, which builds functions with `node`, `graph` and
`kernel`). How nodes run and how their functions are called is decided before, in
`planning/`. Other names without an underscore are shared
between the modules here (or used by tests), and a leading underscore marks what only
its own module uses.
"""

from pytensor.link.llvm.codegen.array import Array, advance, c_strides, size
from pytensor.link.llvm.codegen.context import (
    Codegen,
    ElementBody,
    IndexBounds,
    index_bounds,
    loop_order,
)
from pytensor.link.llvm.codegen.destination import Captured, OutputDestination
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
    EmitterOutput,
    ScalarOutput,
    TensorOutput,
)
from pytensor.link.llvm.codegen.kernel import compile_node
from pytensor.link.llvm.codegen.num import Num, NumLike, cast, constant, load, store
from pytensor.link.llvm.codegen.types import ir, memory_type, value_bits, value_type
from pytensor.link.llvm.types import element_type, is_absent


__all__ = [
    "Array",
    "Captured",
    "Codegen",
    "ElementBody",
    "EmitterInput",
    "EmitterOutput",
    "IndexBounds",
    "Num",
    "NumLike",
    "OutputDestination",
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
    "load",
    "loop_order",
    "memory_type",
    "size",
    "store",
    "value_bits",
    "value_type",
]
