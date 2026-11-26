"""What ops register with the LLVM backend, and what planning and codegen ask about it.

An op registers:

* `llvm_emit(op, node, codegen, inputs, outputs)`, which emits the node's code
  (codegen/interface.py says what an emitter gets);
* optionally `llvm_layout(op, node, input_layouts)`, which says the layout of each
  output given the inputs' layouts. That is the only thing about its outputs that the
  op doesn't already say;
* optionally `llvm_supports(op, node)`, for an emitter that handles only some nodes of
  its op.

For example, `dispatch/elemwise.py` registers `llvm_emit` and `llvm_layout` for
Elemwise. The planner asks `can_generate(node)` to decide whether a node's code can be
generated, and `planning/signature.py` combines `llvm_layout` with the op's `view_map`
and `destroy_map` into the outputs of the node's function.

Importing this module registers the ops' implementations (`dispatch/`), so that what it
answers never depends on what else happens to be imported.
"""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.types import element_type, is_absent, repr_of


if TYPE_CHECKING:
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from pytensor.link.llvm.codegen.context import Codegen
    from pytensor.link.llvm.codegen.interface import EmitterInput, EmitterOutput


@singledispatch
def llvm_layout(
    op: Op, node: Apply, input_layouts: list[Layout | None]
) -> list[Layout | None]:
    """Return the layout of each output of `node`, given its inputs' layouts.

    None leaves an output to the default: C-contiguous new memory, or nothing known for
    a view (an output the `view_map` lists). An emitter may write an output into an
    input it destroys (`TensorOutput.overwrite`) only where the input's layout implies
    the one declared here: one that does declares the input's (`overwritten_layout`).

    New memory is allocated in a packed layout; one that isn't packed means any strides,
    and the memory is allocated C-contiguous. A view's layout must be true of the view
    the emitter builds: `TensorOutput.set_view` relies on it."""
    return [None] * len(node.outputs)


@singledispatch
def llvm_emit(
    op: Op,
    node: Apply,
    cg: Codegen,
    inputs: list[EmitterInput],
    outputs: list[EmitterOutput | None],
) -> None:
    """Emit the code of `node` (see codegen/interface.py for what it gets).

    Raise NotImplementedError for a node it can't handle: something else runs it
    then."""
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


def has_emitter(op: Op) -> bool:
    """Is an emitter registered for the type of `op`?"""
    return llvm_emit.dispatch(type(op)) is not llvm_emit.dispatch(object)


@singledispatch
def llvm_supports(op: Op, node: Apply) -> bool:
    """Can `llvm_emit` generate `node`? Asked before the node's code is generated
    (when the planner chooses how it runs), so an emitter never refuses: one that
    handles only some nodes of its op registers a check here. A check depends on the
    node alone, not on layouts.

    Default: every input and output has a representation (`supported_types`)."""
    return supported_types(node)


def supported_types(node: Apply) -> bool:
    """Do the node's inputs and outputs all have a representation, with element types
    the backend supports?"""
    for var in (*node.inputs, *node.outputs):
        t = var.type
        if is_absent(t):
            continue
        try:
            repr_of(t)
            if hasattr(t, "dtype"):
                element_type(t.dtype)
        except NotImplementedError:
            return False
    return True


def can_generate(node: Apply) -> bool:
    """Has `node` an emitter that supports it?"""
    return has_emitter(node.op) and llvm_supports(node.op, node)


__all__ = [
    "can_generate",
    "has_emitter",
    "llvm_emit",
    "llvm_layout",
    "llvm_supports",
    "supported_types",
]


# The ops' implementations register themselves when imported. They import this module,
# which is complete by now.
import pytensor.link.llvm.dispatch  # noqa: E402, F401
