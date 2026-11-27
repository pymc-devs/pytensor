"""How a node's function is called: the representation of each argument, and how each
output is passed (`Signature`).

`node_outputs` turns what an op says about its outputs (its `view_map`, `destroy_map`
and `llvm_layout`) into the `Output`s of its function: their layouts and how they are
passed. This is the only place where those rules live.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from pytensor.graph.basic import Apply
from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.layout import Layout, new_memory_layout
from pytensor.link.llvm.planning.implementation import GENERATED, Implementation
from pytensor.link.llvm.registry import has_emitter
from pytensor.link.llvm.types import (
    NoneRepr,
    Representation,
    ScalarRepr,
    TensorRepr,
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
    #: the constant inputs compiled into the code: (position, value) pairs
    folded: tuple[tuple[int, float], ...] = ()
    #: the inputs the function may overwrite (the planner allows it)
    writable: frozenset[int] = frozenset()

    @property
    def folded_values(self) -> dict[int, float]:
        """{input position: value} of the folded constants."""
        return dict(self.folded)

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


def overwritten_layout(
    node: Apply, output: int, input_layouts: Sequence[Layout | None]
) -> Layout | None:
    """For an op's `llvm_layout`: the layout output `output` declares if its emitter
    writes it into the memory of an input it destroys (`TensorOutput.overwrite`),
    which it may only where the input's layout implies the declared one. That is the
    input's layout, where new memory can have it too (`new_memory_layout`). None if
    the output can't overwrite an input."""
    destroyed = destroyed_inputs(node, output)
    if not destroyed:
        return None
    layouts = known_layouts(node, input_layouts)
    return new_memory_layout(layouts[destroyed[0]], node.outputs[output].type)


def node_outputs(
    node: Apply,
    input_layouts: Sequence[Layout | None] | None = None,
    writable: Collection[int] = frozenset(),
    implementation: Implementation = GENERATED,
) -> list[Output]:
    """Return the `Output` of each output of `node`, run by `implementation`.

    * Its layout is what the implementation declares (for generated code, the op's
      `llvm_layout`), or the default: C for new memory, nothing known for a view.
    * It may alias the inputs its `view_map` lists (if the implementation returns
      views), and the input it may overwrite: one that is `writable` (the planner lets
      the node overwrite it, and the implementation does) and whose layout implies the
      declared one, so that the layout is the same whether it is overwritten or not
      (an op whose emitter writes in place declares `overwritten_layout`).
    * It is passed in a slot if it may alias an input, its size is only known at run
      time, or the implementation can't write memory the caller provides; else in a
      buffer the caller provides.

    `input_layouts`: the layout of each input (None where unknown)."""
    input_layouts = known_layouts(node, input_layouts)
    declared = implementation.outputs(node, input_layouts)
    outputs: list[Output] = []
    for position, (var, declaration) in enumerate(
        zip(node.outputs, declared, strict=True)
    ):
        representation = repr_of(var.type)
        if not isinstance(representation, TensorRepr):
            outputs.append(Output(representation))
            continue
        viewed = view_inputs(node, position) if implementation.may_view else ()
        destroyed = (
            destroyed_inputs(node, position) if implementation.can_overwrite else ()
        )
        layout = declaration.layout
        if layout is None:
            ndim = var.type.ndim
            layout = Layout.dynamic(ndim) if viewed else Layout.c(ndim)
        shape, itemsize = var.type.shape, var.type.numpy_dtype.itemsize
        aliased = viewed + tuple(
            k
            for k in destroyed
            if k in writable
            and k not in viewed
            and (overwritten := input_layouts[k]) is not None
            and overwritten.implies(layout, shape, itemsize)
        )
        if aliased or not representation.static or not declaration.writes_provided:
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
    input_layouts: Sequence[Layout | None] | None = None,
    *,
    implementation: Implementation = GENERATED,
    folded: Mapping[int, float] | None = None,
    writable: Collection[int] = frozenset(),
    provided: Sequence[Layout | None] | None = None,
) -> Signature:
    """Return the signature of the function of `node`, run by `implementation`: the
    whole description of that function.

    * `input_layouts`: the layout of each input;
    * `folded`: {position: value} of the inputs that are constants compiled into the
      code;
    * `writable`: the inputs the planner lets it overwrite (see `node_outputs`);
    * `provided`: per output, a layout replacing the declared one of a caller buffer
      (for a caller that fixes the memory, e.g. a test writing into a slice).
    """
    folded = folded or {}
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
    writable = frozenset(writable)
    resolved: list[Output] = []
    for k, output in enumerate(
        node_outputs(node, input_layouts, writable, implementation)
    ):
        layout = None if provided is None else provided[k]
        if output.passing is Passing.CALLER_BUFFER and layout is not None:
            output = output.with_layout(layout)
        resolved.append(output)
    return Signature(
        inputs,
        tuple(resolved),
        tuple(argument_of_input),
        tuple(sorted(folded.items())),
        writable,
    )


__all__ = [
    "Output",
    "Passing",
    "Signature",
    "known_layouts",
    "node_outputs",
    "node_signature",
    "overwritten_layout",
    "skipped_inputs",
]
