"""How a node's function is called: the representation of each argument, and how each
output is passed (`Signature`).

`node_outputs` turns what an op declares about its outputs (`llvm_outputs`) into the
`Output`s of its function: their layouts and how they are passed. This is the only
place where those rules live.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.analysis.layout import Layout, new_memory_layout
from pytensor.link.llvm.codegen.interface import (
    InPlaceUpdate,
    InputView,
    NewBuffer,
    OutputStorage,
    ViewOrCopy,
    llvm_outputs,
)
from pytensor.link.llvm.codegen.types import (
    NoneRepr,
    ScalarRepr,
    TensorRepr,
    ir,
    is_absent,
    repr_of,
)


Representation = TensorRepr | ScalarRepr | NoneRepr


class Passing(enum.Enum):
    """How a tensor output is passed to the function that produces it."""

    #: The caller passes memory to write into, with its own strides.
    CALLER_BUFFER = "caller buffer"
    #: The function fills an output slot: with memory it allocated (handing over one
    #: reference), or with a view of an input the output may alias.
    CALLEE_SLOT = "callee slot"


@dataclass(frozen=True)
class Output:
    """One output of a function: what it is, where its memory comes from, and how it is
    passed. For a ScalarType output (stored through a pointer) or one without data
    (no argument), only `representation` is set."""

    representation: Representation
    storage: OutputStorage | None = None
    passing: Passing | None = None
    #: Positions of the node inputs the result may be a view of.
    aliased_inputs: tuple[int, ...] = ()

    @classmethod
    def new_buffer(cls, var, layout: Layout | None, passing: Passing) -> Output:
        """Return the output for `var` in new memory of `layout` (a kernel's output)."""
        if is_absent(var.type):
            return cls(NoneRepr())
        return cls(repr_of(var.type, layout), NewBuffer(layout), passing)

    @property
    def tensor(self) -> TensorRepr:
        """The representation of a tensor output (TypeError for other outputs)."""
        if not isinstance(self.representation, TensorRepr):
            raise TypeError(f"not a tensor output: {self.representation}")
        return self.representation

    @property
    def is_tensor(self) -> bool:
        return isinstance(self.representation, TensorRepr)

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.representation, ScalarRepr)

    @property
    def is_absent(self) -> bool:
        return isinstance(self.representation, NoneRepr)

    @property
    def layout(self) -> Layout | None:
        """The output's layout (None for non-tensors).

        Passed in a caller buffer, it is the memory the op asks for (the caller may
        provide other memory, see `with_layout`). Passed in a slot, it is what the
        result has, a view or new memory."""
        return getattr(self.representation, "layout", None)

    def with_layout(self, layout: Layout) -> Output:
        """Return the same output with another layout (the memory actually provided)."""
        representation = self.tensor
        return replace(
            self,
            representation=TensorRepr(
                representation.dtype, representation.shape, layout
            ),
        )

    @property
    def ir_type(self):
        """The type of the output's parameter: the array itself for a caller buffer,
        else a pointer (to the slot, or to the scalar)."""
        if self.passing is Passing.CALLER_BUFFER:
            return self.tensor.ir_type
        return ir.ptr


@dataclass(frozen=True)
class Signature:
    """The parameters of a node's function: `inputs` (one representation per input
    argument) and `outputs` (one per node output; those without data have no
    parameter)."""

    inputs: tuple[Representation, ...]
    outputs: tuple[Output, ...]
    #: For each node input, its argument position; None for inputs that aren't
    #: arguments (folded constants, values without data). None: the inputs are the
    #: arguments, in order.
    argument_of_input: tuple[int | None, ...] | None = None

    def param_types(self) -> list:
        input_types = [r.ir_type for r in self.inputs if not isinstance(r, NoneRepr)]
        output_types = [o.ir_type for o in self.outputs if not o.is_absent]
        return [ir.ptr, ir.ptr, *input_types, *output_types]

    @property
    def has_scalars(self) -> bool:
        return any(isinstance(r, ScalarRepr) for r in self.inputs) or any(
            o.is_scalar for o in self.outputs
        )

    def alias_arguments(self, output: Output) -> tuple[int, ...]:
        """Return the argument positions of the inputs `output` may be a view of."""
        if self.argument_of_input is None:
            return output.aliased_inputs
        arguments = []
        for position in output.aliased_inputs:
            argument = self.argument_of_input[position]
            if argument is None:
                raise NotImplementedError("a view of a folded constant")
            arguments.append(argument)
        return tuple(arguments)


# ------------------------------------------------------------------ resolution


def known_layouts(node, input_layouts) -> list[Layout | None]:
    """Return `input_layouts`, with DYNAMIC for tensors whose layout isn't known."""
    if input_layouts is None:
        input_layouts = [None] * len(node.inputs)
    return [
        layout
        if layout is not None or not hasattr(var.type, "ndim")
        else Layout.dynamic(var.type.ndim)
        for var, layout in zip(node.inputs, input_layouts, strict=True)
    ]


def can_update_in_place(node, output: int, input: int) -> bool:
    """May `node`'s output at position `output` be written into the input at position
    `input`?

    The op must destroy that input for this output (`destroy_map`), and both must be
    tensors of the same dtype and broadcastable axes."""
    if input not in node.op.destroy_map.get(output, ()):
        return False
    in_type, out_type = node.inputs[input].type, node.outputs[output].type
    return (
        isinstance(repr_of(in_type), TensorRepr)
        and isinstance(repr_of(out_type), TensorRepr)
        and in_type.dtype == out_type.dtype
        and in_type.broadcastable == out_type.broadcastable
    )


def node_outputs(
    node, input_layouts=None, writable: frozenset[int] = frozenset()
) -> list[Output]:
    """Return the `Output` of each output of `node`, from what its op declares
    (`llvm_outputs`).

    `input_layouts`: the layout of each input (None where unknown). `writable`: the
    positions of the inputs the planner lets the node overwrite; an `InPlaceUpdate`
    output may only be its input's memory if that input is writable."""
    input_layouts = known_layouts(node, input_layouts)
    declared = llvm_outputs(node.op, node, input_layouts)
    return [
        _resolve(node, position, storage, input_layouts, writable)
        for position, storage in enumerate(declared)
    ]


def _resolve(
    node,
    position: int,
    storage: OutputStorage,
    input_layouts: list[Layout | None],
    writable: frozenset[int],
) -> Output:
    var = node.outputs[position]
    representation = repr_of(var.type)
    if not isinstance(representation, TensorRepr):
        return Output(representation)

    # new memory of a static size is provided by the caller
    new_memory = Passing.CALLER_BUFFER if representation.static else Passing.CALLEE_SLOT

    match storage:
        case NewBuffer(layout=layout):
            if layout is None:
                layout = Layout.c(var.type.ndim)
            return Output(repr_of(var.type, layout), storage, new_memory)
        case InputView(input=input, layout=layout):
            _check_view(node, position, (input,))
            return Output(
                repr_of(var.type, layout), storage, Passing.CALLEE_SLOT, (input,)
            )
        case ViewOrCopy(inputs=inputs, layout=layout):
            _check_view(node, position, inputs)
            return Output(
                repr_of(var.type, layout), storage, Passing.CALLEE_SLOT, inputs
            )
        case InPlaceUpdate(input=input):
            if not can_update_in_place(node, position, input):
                # the op doesn't destroy the input for this output: new memory
                layout = Layout.c(var.type.ndim)
                return Output(repr_of(var.type, layout), storage, new_memory)
            layout = new_memory_layout(input_layouts[input], var.type)
            if input in writable:
                return Output(
                    repr_of(var.type, layout), storage, Passing.CALLEE_SLOT, (input,)
                )
            return Output(repr_of(var.type, layout), storage, new_memory)
    raise TypeError(f"{node.op}: output {position}: not an OutputStorage: {storage!r}")


def _check_view(node, position: int, inputs: tuple[int, ...]) -> None:
    """Fail unless the graph knows output `position` may be a view of `inputs`."""
    parents = op_view_parents(node.op, node)[position]
    for input in inputs:
        if node.inputs[input] not in parents:
            raise ValueError(
                f"{node.op}: output {position} is declared a view of input {input}, "
                "but the op's view_map doesn't say so"
            )


# ------------------------------------------------------------------ signatures


def skipped_inputs(node, folded=frozenset()) -> frozenset[int]:
    """Return the positions of the inputs that aren't arguments: folded constants and
    values without data."""
    return frozenset(folded) | {
        k for k, var in enumerate(node.inputs) if is_absent(var.type)
    }


def node_signature(
    node,
    folded=frozenset(),
    input_layouts=None,
    provided=None,
    outputs=None,
    writable: frozenset[int] = frozenset(),
) -> Signature:
    """Return the signature of the function of `node`.

    * `folded`: positions of the inputs that are constants folded into the code;
    * `input_layouts`: the layout of each input;
    * `provided`: per output, a layout replacing the declared one of a caller buffer
      (for a caller that fixes the memory, e.g. a test writing into a slice);
    * `outputs`: the outputs, instead of what `node_outputs` gives (a kernel's);
    * `writable`: see `node_outputs`.
    """
    skipped = skipped_inputs(node, folded)
    if input_layouts is None:
        input_layouts = [None] * len(node.inputs)
    inputs = tuple(
        repr_of(var.type, layout)
        for k, (var, layout) in enumerate(zip(node.inputs, input_layouts, strict=True))
        if k not in skipped
    )
    argument_of_input = [
        None if k in skipped else k - sum(1 for s in skipped if s < k)
        for k in range(len(node.inputs))
    ]
    if outputs is None:
        outputs = node_outputs(node, input_layouts, writable)
    resolved = []
    for k, output in enumerate(outputs):
        if (
            output.passing is Passing.CALLER_BUFFER
            and provided is not None
            and provided[k] is not None
        ):
            output = output.with_layout(provided[k])
        resolved.append(output)
    return Signature(inputs, tuple(resolved), tuple(argument_of_input))


__all__ = [
    "Output",
    "Passing",
    "Representation",
    "Signature",
    "can_update_in_place",
    "known_layouts",
    "node_outputs",
    "node_signature",
    "skipped_inputs",
]
