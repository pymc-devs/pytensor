"""What an op registers to have its code generated, and what its emitter gets.

An op registers two functions:

* `llvm_outputs(op, node, input_layouts)` declares the storage of each output: where
  its memory comes from (`OutputStorage`). Optional; the default is new memory, or a
  possible view of the inputs listed in the op's `view_map`.
* `llvm_emit(op, node, codegen, inputs, outputs)` emits the node's code.

The storage declaration is the single source of truth about an output. The planner
derives the output's layout and how it is passed from it, and the emitter gets a handle
that matches it:

=================  =====================  =========================================
storage            handle                 the emitter
=================  =====================  =========================================
`NewBuffer`        `NewBufferOutput`      `allocate(shape)`, then writes the result
`InputView`        `InputViewOutput`      `emit(codegen, inputs)`, or registers
                                          `emit_input_views` as its emitter
`ViewOrCopy`       `ViewOrCopyOutput`     `view(array)` or `allocate(shape)`
`InPlaceUpdate`    `InPlaceOutput`        `buffer()` or `buffer_holding_input()`
(ScalarType)       `ScalarOutput`         `set(value)`
(no data)          None
=================  =====================  =========================================

The emitter's inputs are, per node input, an `Array` for a tensor (read-only unless an
`InPlaceUpdate` output may reuse it), an `ir.Value` for a ScalarType value, or None for
a value without data (a random generator). Constants the planner folds into the code
arrive the same way.

Ops that produce their outputs by running an inner graph (Scan, Blockwise, compiled
units) hand on each handle's `destination` (see `destination.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Protocol

from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.emit import Array, Codegen, check_strides, copy
from pytensor.link.llvm.codegen.types import ir


if TYPE_CHECKING:
    from pytensor.graph.basic import Apply
    from pytensor.link.llvm.codegen.destination import (
        OutputDestination,
        ScalarDestination,
    )


#: What an emitter gets for one node input.
EmitterInput = Array | ir.Value | None


# ------------------------------------------------------------------ storage


class ViewMapping(Protocol):
    """How a view is made from its input, once for the planner and once for codegen.

    Both methods describe the same view: `layout` is what the planner relies on before
    the program runs, and `apply` builds the view in generated code. Keeping them in
    one object keeps them next to each other; with the `check_layouts` option, the
    generated code also verifies at run time that the view's strides match `layout`.
    """

    def layout(self, source: Layout) -> Layout:
        """Return the view's layout, given the layout of the input it views."""
        ...

    def apply(self, codegen: Codegen, inputs: list[EmitterInput]) -> Array:
        """Build the view from the node's inputs.

        The returned array's layout doesn't matter: the handle replaces it with
        `layout(...)`."""
        ...


@dataclass(frozen=True)
class NewBuffer:
    """The output is written into new memory, packed in `layout`.

    None means C-contiguous. A layout that isn't packed means any strides: the memory
    is then allocated C-contiguous, which must imply it."""

    layout: Layout | None = None


@dataclass(frozen=True)
class InputView:
    """The output is always a view of the node input at position `input`."""

    input: int
    mapping: ViewMapping


@dataclass(frozen=True)
class ViewOrCopy:
    """The output is a view of one of `inputs` or new memory, decided by the emitter
    (possibly at run time). Both have `layout`."""

    inputs: tuple[int, ...]
    layout: Layout


@dataclass(frozen=True)
class InPlaceUpdate:
    """The output is the node input at position `input`, with changes.

    It is written into the input's buffer when the op destroys that input for this
    output (`destroy_map`), their types match, and the planner allows it; the output
    then has the input's layout, where new memory can have it too
    (`new_memory_layout`). Otherwise it is new C-contiguous memory."""

    input: int


OutputStorage = NewBuffer | InputView | ViewOrCopy | InPlaceUpdate


@singledispatch
def llvm_outputs(op, node: Apply, input_layouts: list[Layout | None]) -> list:
    """Declare the storage of each output of `node`, given its inputs' layouts.

    Return one `OutputStorage` per output; entries for outputs that aren't tensors are
    ignored. The default is `NewBuffer()` (C-contiguous), or `ViewOrCopy` with nothing
    known about the layout for an output the op's `view_map` says may be a view.
    """
    parents = op_view_parents(op, node)
    storage: list[OutputStorage] = []
    for output, output_parents in zip(node.outputs, parents, strict=True):
        viewed = tuple(i for i, var in enumerate(node.inputs) if var in output_parents)
        if viewed and hasattr(output.type, "ndim"):
            storage.append(ViewOrCopy(viewed, Layout.dynamic(output.type.ndim)))
        else:
            storage.append(NewBuffer())
    return storage


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
class NewBufferOutput:
    """The handle of a `NewBuffer` output."""

    destination: OutputDestination

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return new memory of `shape` for the result, in the declared layout."""
        return self.destination.allocate(shape)


@dataclass
class InputViewOutput:
    """The handle of an `InputView` output."""

    destination: OutputDestination
    storage: InputView
    #: the view's layout, as the planner computed it
    layout: Layout

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def emit(self, codegen: Codegen, inputs: list[EmitterInput]) -> Array:
        """Build the view and make it the result."""
        view = self.storage.mapping.apply(codegen, inputs).with_layout(self.layout)
        if codegen.check_layouts:
            check_strides(codegen, view, self.layout, "a view")
        return self.destination.assign(view)


@dataclass
class ViewOrCopyOutput:
    """The handle of a `ViewOrCopy` output."""

    destination: OutputDestination
    layout: Layout

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def view(self, array: Array) -> Array:
        """Make `array`, a view of one of the declared inputs, the result.

        Its layout must imply the declared one."""
        return self.destination.assign(array)

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return new memory of `shape` for the result, in the declared layout."""
        return self.destination.allocate(shape)


@dataclass
class InPlaceOutput:
    """The handle of an `InPlaceUpdate` output."""

    destination: OutputDestination
    #: the node input the output may reuse
    input: Array

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    @property
    def reuses_input(self) -> bool:
        """Does the result go into the input's buffer? (Decided by the planner.)"""
        return self.input.writable

    def buffer(self) -> Array:
        """Return memory for the result: the input's buffer, still holding the input,
        if it is reused, else new uninitialized memory."""
        if self.reuses_input:
            return self.destination.assign(self.input)
        return self.destination.allocate(self.input.shape)

    def buffer_holding_input(self) -> Array:
        """Return memory for the result that holds the input's values: the input's
        buffer if it is reused, else a copy of the input."""
        if self.reuses_input:
            return self.destination.assign(self.input)
        result = self.destination.allocate(self.input.shape)
        copy(self.destination.codegen, self.input, result)
        return result


@dataclass
class ScalarOutput:
    """The handle of a ScalarType output."""

    destination: ScalarDestination

    @property
    def dtype(self) -> str:
        return self.destination.dtype

    def set(self, value: ir.Value) -> None:
        """Make `value` the result."""
        self.destination.set(value)


#: What an emitter gets for one node output (None for a value without data).
EmitterOutput = (
    NewBufferOutput | InputViewOutput | ViewOrCopyOutput | InPlaceOutput | ScalarOutput
)


def destinations(outputs: list) -> list[OutputDestination | None]:
    """Return the destination of each handle in `outputs`, for an op that hands its
    outputs on to an inner graph (`GraphFunction.emit_call`)."""
    return [None if output is None else output.destination for output in outputs]


def emit_input_views(
    op, node: Apply, codegen: Codegen, inputs: list, outputs: list
) -> None:
    """The emitter of an op whose outputs are all `InputView`s: register it as
    `llvm_emit.register(SomeOp, emit_input_views)`."""
    for output in outputs:
        if not isinstance(output, InputViewOutput):
            raise TypeError(
                f"{op}: emit_input_views needs InputView outputs, got "
                f"{type(output).__name__}"
            )
        output.emit(codegen, inputs)


__all__ = [
    "EmitterInput",
    "EmitterOutput",
    "InPlaceOutput",
    "InPlaceUpdate",
    "InputView",
    "InputViewOutput",
    "NewBuffer",
    "NewBufferOutput",
    "OutputStorage",
    "ScalarOutput",
    "ViewMapping",
    "ViewOrCopy",
    "ViewOrCopyOutput",
    "destinations",
    "emit_input_views",
    "has_emitter",
    "llvm_emit",
    "llvm_outputs",
]
