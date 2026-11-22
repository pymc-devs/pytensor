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

* `TensorOutput` for a tensor: `allocate(shape)` for new memory, `set_view(array)` for
  a view of an input the `view_map` lists, and for an input the `destroy_map` lists,
  `may_overwrite(input)` to ask whether the result may go into its memory and
  `overwrite(input)` to put it there;
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

import numpy as np

from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.emit import Array, Codegen, check_strides
from pytensor.link.llvm.codegen.types import ir
from pytensor.scalar.basic import ScalarType


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
    the emitter builds: `TensorOutput.set_view` relies on it."""
    return [None] * len(node.outputs)


@singledispatch
def llvm_emit(op, node: Apply, cg: Codegen, inputs: list, outputs: list) -> None:
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
    allocated: bool = False

    @property
    def scalar_type(self) -> ScalarType:
        return self.destination.scalar_type

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return memory of `shape` for the result, in the output's layout. Call it
        once.

        The memory is the caller's or new: the planner decides, and the emitter writes
        the result into it either way."""
        if self.allocated:
            raise ValueError("allocate() called twice for the same output")
        self.allocated = True
        return self.destination.allocate(shape)

    def set_view(self, array: Array) -> Array:
        """Make `array`, a view of an input, the result.

        The view gets the output's layout: the one `llvm_layout` declares is trusted
        (verified at run time with the `check_layouts` option), and an error while
        generating if `array`'s own layout contradicts it."""
        if not self.view_inputs:
            raise ValueError(
                "set_view() of an output that isn't a view: the op's view_map doesn't "
                "list it"
            )
        cg = self.destination.cg
        shape = array.static_shape()
        size = np.dtype(array.scalar_type.dtype).itemsize
        if not array.layout.implies(self.layout, shape, size):
            if _contradicts(array.layout, self.layout, shape, size):
                raise ValueError(
                    f"the view has layout {array.layout.describe(shape)}, but the "
                    f"output is declared {self.layout.describe(shape)}"
                )
            array = array.with_layout(self.layout)
            if cg.check_layouts:
                check_strides(cg, array, self.layout, "a view")
        return self.destination.assign(array)

    def may_overwrite(self, input: Array) -> bool:
        """May the result go into `input`'s memory?

        Yes if the op destroys that input for this output (`destroy_map`) and the
        planner lets the node overwrite it. This is known while generating the code."""
        return input.writable and input.base in self.destroyed_inputs

    def overwrite(self, input: Array) -> Array:
        """Make `input`'s memory the result, and return it to be written into.

        Only where `may_overwrite(input)`."""
        if not self.may_overwrite(input):
            raise ValueError(
                "overwrite() of an input the output may not overwrite (ask "
                "may_overwrite first)"
            )
        return self.destination.assign(input)


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
    def scalar_type(self) -> ScalarType:
        return self.destination.scalar_type

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
