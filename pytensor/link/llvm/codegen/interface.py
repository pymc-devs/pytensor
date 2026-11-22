"""What an op registers to have its code generated, and what its emitter gets.

An op registers:

* `llvm_emit(op, node, codegen, inputs, outputs)`, which emits the node's code;
* optionally `llvm_layout(op, node, input_layouts)`, which says the layout of each
  output given the inputs' layouts. That is the only thing about its outputs that the
  op doesn't already say.

Everything else about an output comes from the op itself: its `view_map` says which
inputs the output may be a view of, its `destroy_map` which input it may overwrite.
`node_outputs` (signature.py) combines the three into the output's layout and how it is
passed, and the emitter gets one handle per output that knows the result:

* `TensorOutput` for a tensor: `allocate(shape)` for new memory, `view(array)` for a
  view of an input the `view_map` lists, `buffer(input)` / `buffer_holding(input)` for
  an output that may overwrite an input the `destroy_map` lists;
* `ScalarOutput` for a ScalarType value: `set(value)`;
* None for a value without data (a random generator).

The emitter's inputs are, per node input, an `Array` for a tensor (read-only unless the
planner lets the node overwrite it: `writable`), an `ir.Value` for a ScalarType value,
or None for a value without data. Constants the planner folds into the code arrive the
same way.

Ops that produce their outputs by running an inner graph (Scan, Blockwise, compiled
units) hand on each handle's `destination` (see `destination.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.emit import Array, Codegen, check_strides, copy
from pytensor.link.llvm.codegen.types import ir, itemsize


if TYPE_CHECKING:
    from pytensor.graph.basic import Apply
    from pytensor.link.llvm.codegen.destination import (
        OutputDestination,
        ScalarDestination,
    )


#: What an emitter gets for one node input.
EmitterInput = Array | ir.Value | None


@singledispatch
def llvm_layout(op, node: Apply, input_layouts: list[Layout | None]) -> list:
    """Return the layout of each output of `node`, given its inputs' layouts.

    None leaves an output to the default: C-contiguous new memory, or nothing known for
    a view (an output the `view_map` lists). An output that may overwrite an input has
    that input's layout instead, whatever is declared here (`new_memory_layout`).

    New memory is allocated in a packed layout; one that isn't packed means any strides,
    and the memory is allocated C-contiguous. A view's layout must be true of the view
    the emitter builds: `TensorOutput.view` relies on it."""
    return [None] * len(node.outputs)


@singledispatch
def llvm_emit(op, node: Apply, codegen: Codegen, inputs: list, outputs: list) -> None:
    """Emit the code of `node` (see the module's docstring).

    Raise NotImplementedError for a node it can't handle: something else runs it
    then."""
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


def has_emitter(op) -> bool:
    """Is an emitter registered for the type of `op`?"""
    return llvm_emit.dispatch(type(op)) is not llvm_emit.dispatch(object)


# ------------------------------------------------------------------ handles


@dataclass
class TensorOutput:
    """What an emitter gets for a tensor output."""

    destination: OutputDestination
    #: the output's layout, as the planner computed it
    layout: Layout
    #: positions of the inputs the op's `view_map` says the output may be a view of
    view_inputs: tuple[int, ...] = ()
    #: positions of the inputs the op's `destroy_map` says the output may overwrite
    destroyed_inputs: tuple[int, ...] = ()

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return new memory of `shape` for the result, in the output's layout."""
        return self.destination.allocate(shape)

    def view(self, array: Array) -> Array:
        """Make `array`, a view of an input, the result.

        The view gets the output's layout: the one `llvm_layout` declares is trusted
        (verified at run time with the `check_layouts` option), and an error while
        generating if `array`'s own layout contradicts it."""
        if not self.view_inputs:
            raise ValueError(
                "view() of an output that isn't a view: the op's view_map doesn't "
                "list it"
            )
        codegen = self.destination.codegen
        shape = array.static_shape()
        size = itemsize(array.dtype)
        if not array.layout.implies(self.layout, shape, size):
            if _contradicts(array.layout, self.layout, shape, size):
                raise ValueError(
                    f"the view has layout {array.layout.describe(shape)}, but the "
                    f"output is declared {self.layout.describe(shape)}"
                )
            array = array.with_layout(self.layout)
            if codegen.check_layouts:
                check_strides(codegen, array, self.layout, "a view")
        return self.destination.assign(array)

    def reuses(self, input: Array) -> bool:
        """Does the result go into `input`'s buffer? (The op destroys it for this
        output, and the planner lets the node overwrite it.)"""
        return input.writable and input.base in self.destroyed_inputs

    def buffer(self, input: Array) -> Array:
        """Return memory for the result: `input`'s buffer, still holding the input,
        if the result may go there (`reuses`), else new uninitialized memory."""
        if self.reuses(input):
            return self.destination.assign(input)
        return self.destination.allocate(input.shape)

    def buffer_holding(self, input: Array) -> Array:
        """Return memory for the result that holds `input`'s values: `input`'s buffer
        if the result may go there (`reuses`), else a copy of it."""
        if self.reuses(input):
            return self.destination.assign(input)
        result = self.destination.allocate(input.shape)
        copy(self.destination.codegen, input, result)
        return result


def _contradicts(actual: Layout, declared: Layout, shape, size: int) -> bool:
    """Do the two layouts fix different strides for an axis of length other than 1?"""
    for axis, length in enumerate(shape):
        if length == 1:
            continue
        mine = actual.static_stride(axis, shape, size)
        theirs = declared.static_stride(axis, shape, size)
        if mine is not None and theirs is not None and mine != theirs:
            return True
    return False


@dataclass
class ScalarOutput:
    """What an emitter gets for a ScalarType output."""

    destination: ScalarDestination

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def set(self, value: ir.Value) -> None:
        """Make `value` the result."""
        self.destination.set(value)


#: What an emitter gets for one node output (None for a value without data).
EmitterOutput = TensorOutput | ScalarOutput


def destinations(outputs: list) -> list[OutputDestination | None]:
    """Return the destination of each handle in `outputs`, for an op that hands its
    outputs on to an inner graph (`GraphFunction.emit_call`)."""
    return [None if output is None else output.destination for output in outputs]


__all__ = [
    "EmitterInput",
    "EmitterOutput",
    "ScalarOutput",
    "TensorOutput",
    "destinations",
    "has_emitter",
    "llvm_emit",
    "llvm_layout",
]
