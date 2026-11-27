"""What an op's emitter gets: its inputs, and one handle per output.

An op registers its emitter `llvm_emit` and its output layouts `llvm_layout` in the
registry (registry.py). Everything else about an output comes from the op itself: its
`view_map` says which inputs the output may be a view of, its `destroy_map` which input
it may overwrite. `node_outputs` (planning/signature.py) combines the three into the
output's layout and how it is passed, and the emitter gets one handle per output that
knows the result:

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
units) hand their handles on to `GraphFunction.emit_call`, and use a few more methods
of `TensorOutput` for outputs they build up over a loop: `allocate_once`, `current`,
`reallocate`, `truncate` (see destination.py for results that go somewhere else).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytensor.link.llvm.codegen.array import Array
from pytensor.link.llvm.codegen.types import ir
from pytensor.link.llvm.layout import Layout
from pytensor.scalar.basic import ScalarType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.destination import (
        OutputDestination,
        ScalarDestination,
    )


#: What an emitter gets for one node input.
type EmitterInput = Array | ir.Value | None


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
        size = array.itemsize
        if not array.layout.implies(self.layout, shape, size):
            if _contradicts(array.layout, self.layout, shape, size):
                raise ValueError(
                    f"the view has layout {array.layout.describe(shape)}, but the "
                    f"output is declared {self.layout.describe(shape)}"
                )
            array = array.with_layout(self.layout)
            if cg.check_layouts:
                array.check_strides(cg, self.layout, "a view")
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

    # ---- for outputs built up over a loop (ops that run an inner graph)

    def allocate_once(self, shape: list[ir.Value]) -> Array:
        """Return the result's memory, allocated now with `shape` unless it already
        was when this code runs (by an earlier iteration of a loop, or `allocate`).

        For a shape only an iteration knows: the first one allocates, the others get
        the same memory."""
        return self.destination.allocate_once(shape)

    def current(self) -> Array:
        """Return the result's memory as it is when this code runs (after `allocate`,
        `allocate_once` or `reallocate`), in `allocated_layout`."""
        return self.destination.current()

    @property
    def allocated_layout(self) -> Layout:
        """The layout of the memory `allocate` returns (and `current`)."""
        return self.destination.allocated_layout

    def reallocate(self, shape: list[ir.Value]) -> Array:
        """Return memory of `shape` for the result: new memory if the result had
        another shape when this code runs, else the memory it has. What it held is
        lost; the old memory stays readable until the innermost `scratch_scope` ends.

        For a result whose shape may change from one iteration to the next (a Scan's
        untraced state). Fails at run time for memory the caller provides."""
        return self.destination.reallocate(shape)

    def truncate(self, length: ir.Value) -> None:
        """Shorten the result's first axis to `length` (when this code runs; its
        strides stay). For a result allocated longer than it turned out (a while-loop's
        trace)."""
        self.destination.truncate(length)


def _contradicts(
    actual: Layout, declared: Layout, shape: Sequence[int | None], size: int
) -> bool:
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
type EmitterOutput = TensorOutput | ScalarOutput


__all__ = [
    "EmitterInput",
    "EmitterOutput",
    "ScalarOutput",
    "TensorOutput",
]
