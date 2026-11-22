"""How a node's function is called: the representation of each argument, and how each
output is passed (`Signature`).

`node_outputs` turns what an op says about its outputs (its `view_map`, `destroy_map`
and `llvm_layout`) into the `Output`s of its function: their layouts and how they are
passed. This is the only place where those rules live.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.analysis.layout import Layout, new_memory_layout
from pytensor.link.llvm.codegen.interface import has_emitter, llvm_layout
from pytensor.link.llvm.codegen.types import (
    NoneRepr,
    Representation,
    ScalarRepr,
    TensorRepr,
    ir,
    is_absent,
    repr_of,
)


class Passing(enum.Enum):
    """How a tensor output is passed to the function that produces it."""

    #: The caller passes memory to write into, with its own strides.
    CALLER_BUFFER = "caller buffer"
    #: The function fills an output slot: with memory it allocated (handing over one
    #: reference), or with a view of an input the output may alias.
    CALLEE_SLOT = "callee slot"


@dataclass(frozen=True)
class Output:
    """One output of a function: what it is, and how it is passed. For a ScalarType
    output (stored through a pointer) or one without data (no argument), only
    `representation` is set."""

    representation: Representation
    passing: Passing | None = None
    #: Positions of the node inputs the result may be a view of: those the op's
    #: `view_map` lists, and an input it may overwrite where the planner allows it.
    aliased_inputs: tuple[int, ...] = ()

    @classmethod
    def new_buffer(
        cls, var: Variable, layout: Layout | None, passing: Passing
    ) -> Output:
        """Return the output for `var` in new memory of `layout` (a kernel's output)."""
        if is_absent(var.type):
            return cls(NoneRepr())
        return cls(repr_of(var.type, layout), passing)

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
        if isinstance(self.representation, TensorRepr):
            return self.representation.layout
        return None

    def with_layout(self, layout: Layout) -> Output:
        """Return the same output with another layout (the memory actually provided)."""
        representation = self.tensor
        return replace(
            self,
            representation=TensorRepr(
                representation.scalar_type, representation.shape, layout
            ),
        )

    @property
    def ir_type(self) -> ir.Type:
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

    def param_types(self) -> list[ir.Type]:
        input_types = [
            representation.ir_type
            for representation in self.inputs
            if not isinstance(representation, NoneRepr)
        ]
        output_types = [
            output.ir_type for output in self.outputs if not output.is_absent
        ]
        return [ir.ptr, ir.ptr, *input_types, *output_types]

    @property
    def has_scalars(self) -> bool:
        return any(
            isinstance(representation, ScalarRepr) for representation in self.inputs
        ) or any(output.is_scalar for output in self.outputs)

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


def known_layouts(
    node: Apply, input_layouts: Sequence[Layout | None] | None
) -> list[Layout | None]:
    """Return `input_layouts`, with DYNAMIC for tensors whose layout isn't known."""
    if input_layouts is None:
        input_layouts = [None] * len(node.inputs)
    return [
        layout
        if layout is not None or not hasattr(var.type, "ndim")
        else Layout.dynamic(var.type.ndim)
        for var, layout in zip(node.inputs, input_layouts, strict=True)
    ]


def can_update_in_place(node: Apply, output: int, input: int) -> bool:
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


def view_inputs(node: Apply, output: int) -> tuple[int, ...]:
    """Return the positions of the inputs output `output` of `node` may be a view of
    (the op's `view_map`)."""
    parents = op_view_parents(node.op, node)[output]
    return tuple(k for k, var in enumerate(node.inputs) if var in parents)


def destroyed_inputs(node: Apply, output: int) -> tuple[int, ...]:
    """Return the position of the input output `output` of `node` may be written into
    (the first the op's `destroy_map` lists that `can_update_in_place`), or ().

    Only for generated code, which can choose to: a kernel's outputs are new memory."""
    if not has_emitter(node.op):
        return ()
    for k in node.op.destroy_map.get(output, ()):
        if can_update_in_place(node, output, k):
            return (k,)
    return ()


def node_outputs(
    node: Apply,
    input_layouts: Sequence[Layout | None] | None = None,
    writable: Collection[int] = frozenset(),
) -> list[Output]:
    """Return the `Output` of each output of `node`.

    * Its layout is what the op's `llvm_layout` declares, or the default: C for new
      memory, nothing known for a view. An output that may overwrite an input has that
      input's layout, where new memory can have it too (`new_memory_layout`), so that
      it is the same whether the planner lets it overwrite the input or not.
    * It may alias the inputs its `view_map` lists, and the input it may overwrite if
      that input is `writable` (the planner lets the node overwrite it).
    * It is passed in a slot if it may alias an input or its size is only known at run
      time, else in a buffer the caller provides.

    `input_layouts`: the layout of each input (None where unknown)."""
    input_layouts = known_layouts(node, input_layouts)
    declared = llvm_layout(node.op, node, input_layouts)
    outputs: list[Output] = []
    for position, (var, layout) in enumerate(zip(node.outputs, declared, strict=True)):
        representation = repr_of(var.type)
        if not isinstance(representation, TensorRepr):
            outputs.append(Output(representation))
            continue
        viewed = view_inputs(node, position)
        destroyed = destroyed_inputs(node, position)
        if destroyed:
            layout = new_memory_layout(input_layouts[destroyed[0]], var.type)
        elif layout is None:
            ndim = var.type.ndim
            layout = Layout.dynamic(ndim) if viewed else Layout.c(ndim)
        aliased = viewed + tuple(
            k for k in destroyed if k in writable and k not in viewed
        )
        if aliased or not representation.static:
            passing = Passing.CALLEE_SLOT
        else:
            passing = Passing.CALLER_BUFFER
        outputs.append(Output(repr_of(var.type, layout), passing, aliased))
    return outputs


# ------------------------------------------------------------------ signatures


def skipped_inputs(
    node: Apply, folded: Collection[int] = frozenset()
) -> frozenset[int]:
    """Return the positions of the inputs that aren't arguments: folded constants and
    values without data."""
    return frozenset(folded) | {
        k for k, var in enumerate(node.inputs) if is_absent(var.type)
    }


def node_signature(
    node: Apply,
    folded: Collection[int] = frozenset(),
    input_layouts: Sequence[Layout | None] | None = None,
    provided: Sequence[Layout | None] | None = None,
    outputs: Sequence[Output] | None = None,
    writable: Collection[int] = frozenset(),
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
    resolved: list[Output] = []
    for k, output in enumerate(outputs):
        layout = None if provided is None else provided[k]
        if output.passing is Passing.CALLER_BUFFER and layout is not None:
            output = output.with_layout(layout)
        resolved.append(output)
    return Signature(inputs, tuple(resolved), tuple(argument_of_input))


__all__ = [
    "Output",
    "Passing",
    "Representation",
    "Signature",
    "can_update_in_place",
    "destroyed_inputs",
    "known_layouts",
    "node_outputs",
    "node_signature",
    "skipped_inputs",
    "view_inputs",
]
