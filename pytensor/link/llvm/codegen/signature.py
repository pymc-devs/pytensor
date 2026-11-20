"""How a node's function is called: the representation of each argument, and how each
output comes to be (`Signature`)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.codegen.interface import llvm_layout
from pytensor.link.llvm.codegen.types import (
    NoneRepr,
    ScalarRepr,
    TensorRepr,
    ir,
    is_absent,
    repr_of,
)
from pytensor.link.llvm.layout import Layout, new_memory_layout


class OutputMode(enum.Enum):
    #: the caller provides the memory (with its own strides)
    PROVIDED = "provided"
    #: the function returns the array through a slot: memory it allocated (it hands
    #: over one reference), or a view of an input the output may alias
    RETURNED = "returned"
    #: a ScalarType output, stored through a pointer
    SCALAR = "scalar"
    #: no value (a random generator token): no argument
    ABSENT = "absent"


@dataclass(frozen=True)
class Output:
    repr: TensorRepr | ScalarRepr | NoneRepr
    mode: OutputMode
    #: the inputs the output may be a view of (RETURNED; none: it is new memory)
    aliases: tuple = ()

    @property
    def layout(self):
        """The output's layout (None for scalars). For PROVIDED it is the memory the
        op asks for (the provider may give it other memory, see `with_layout`); for
        RETURNED it is what the result has, a view or new memory."""
        return getattr(self.repr, "layout", None)

    def with_layout(self, layout: Layout):
        """The same output with another layout (e.g. the memory actually provided)."""
        return replace(self, repr=TensorRepr(self.repr.dtype, self.repr.shape, layout))

    @property
    def ir_type(self):
        if self.mode is OutputMode.PROVIDED:
            return self.repr.ir_type
        return ir.ptr


@dataclass(frozen=True)
class Signature:
    """The arguments of a node's function: `inputs` (representations, one per argument),
    `outputs` (one per node output; ABSENT ones have no argument)."""

    inputs: tuple
    outputs: tuple[Output, ...]

    def param_types(self):
        outs = [o.ir_type for o in self.outputs if o.mode is not OutputMode.ABSENT]
        return [ir.ptr, ir.ptr, *(r.ir_type for r in self.inputs), *outs]

    @property
    def has_scalars(self):
        return any(isinstance(r, ScalarRepr) for r in self.inputs) or any(
            isinstance(o.repr, ScalarRepr) for o in self.outputs
        )


def destroyed_input(node, k: int) -> int | None:
    """The input output `k` may be written over: one the op destroys for it
    (`destroy_map`) that has its type (dtype, number of axes, broadcastable ones)."""
    out = node.outputs[k]
    for i in node.op.destroy_map.get(k, ()):
        t = node.inputs[i].type
        if (
            isinstance(repr_of(t), TensorRepr)
            and isinstance(repr_of(out.type), TensorRepr)
            and t.dtype == out.type.dtype
            and t.broadcastable == out.type.broadcastable
        ):
            return i
    return None


def new_output(var, layout=None) -> Output:
    """An output in new memory: SCALAR, PROVIDED for static shapes, else RETURNED; in
    `layout` (default C-contiguous). ABSENT for values without data."""
    r = repr_of(var.type)
    if isinstance(r, NoneRepr):
        return Output(r, OutputMode.ABSENT)
    if isinstance(r, ScalarRepr):
        return Output(r, OutputMode.SCALAR)
    lay = layout if layout is not None else Layout.c(var.type.ndim)
    return Output(
        repr_of(var.type, lay), OutputMode.PROVIDED if r.static else OutputMode.RETURNED
    )


def known_layouts(node, in_layouts) -> list:
    """`in_layouts` with DYNAMIC for tensors whose layout isn't known."""
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    return [
        lay
        if lay is not None or not hasattr(i.type, "ndim")
        else Layout.dynamic(i.type.ndim)
        for i, lay in zip(node.inputs, in_layouts, strict=True)
    ]


def node_outputs(node, in_layouts=None, writable=frozenset()) -> list[Output]:
    """How each output of `node` comes to be when its code is generated. This follows
    from the output's type, from `llvm_layout`, and from the inputs the output may be a
    view of (`op_view_parents`: the op's `view_map`, or for an OpFromGraph, whatever
    its inner graph may pass through).

    An output the op may write over an input (`destroyed_input`) has that input's
    layout, where new memory can have it too (`new_memory_layout`): it is the input's
    memory or new memory, as its emitter decides. It may be the input's memory only if
    the input is `writable` (positions the planner lets the node overwrite)."""
    op = node.op
    in_layouts = known_layouts(node, in_layouts)
    layouts = llvm_layout(op, node, in_layouts)
    parents = op_view_parents(op, node)
    outs = []
    for k, (o, lay) in enumerate(zip(node.outputs, layouts, strict=True)):
        aliases: tuple = ()
        if isinstance(repr_of(o.type), TensorRepr):
            aliases = tuple(
                i for i, var in enumerate(node.inputs) if var in parents[k]
            )
            i = destroyed_input(node, k)
            if i is not None:
                lay = new_memory_layout(in_layouts[i], o.type)
                if i in writable and i not in aliases:
                    aliases = (*aliases, i)
        if not aliases:
            outs.append(new_output(o, lay))
            continue
        if lay is None:
            lay = Layout.dynamic(o.type.ndim)
        outs.append(Output(repr_of(o.type, lay), OutputMode.RETURNED, aliases))
    return outs


def skipped_inputs(node, folded=frozenset()):
    """Input positions that aren't arguments: folded constants and None inputs."""
    return frozenset(folded) | {
        k for k, i in enumerate(node.inputs) if is_absent(i.type)
    }


def node_signature(
    node,
    folded=frozenset(),
    in_layouts=None,
    provided=None,
    declared=None,
    writable=frozenset(),
) -> Signature:
    """`folded`: positions of inputs that are constants folded into the code (not
    arguments). None inputs aren't arguments either.

    `in_layouts`: the `Layout` of each input; outputs as `node_outputs` gives them, with
    `aliases` counting the function's arguments.
    The declared layout of a PROVIDED output is binding: whoever provides the memory
    provides it in that layout. `provided[k]` replaces the declaration, for a caller
    that generates a function for memory it fixes (tests writing into a slice).
    `declared`: the outputs instead (a kernel's, see call.py; `aliases` counting the
    node's inputs). `writable`: see `node_outputs`."""
    folded = skipped_inputs(node, folded)
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    inputs = tuple(
        repr_of(i.type, lay)
        for k, (i, lay) in enumerate(zip(node.inputs, in_layouts, strict=True))
        if k not in folded
    )

    def arg_index(pos):
        # position among the function's (non-folded) inputs
        if pos in folded:
            raise NotImplementedError("a view of a folded constant")
        return pos - sum(1 for p in folded if p < pos)

    outputs = []
    for k, o in enumerate(
        declared if declared is not None else node_outputs(node, in_layouts, writable)
    ):
        if (
            o.mode is OutputMode.PROVIDED
            and provided is not None
            and provided[k] is not None
        ):
            o = o.with_layout(provided[k])
        if o.aliases:
            o = Output(o.repr, o.mode, tuple(arg_index(pos) for pos in o.aliases))
        outputs.append(o)
    return Signature(inputs, tuple(outputs))


__all__ = [
    "Output",
    "OutputMode",
    "Signature",
    "destroyed_input",
    "known_layouts",
    "new_output",
    "node_outputs",
    "node_signature",
    "skipped_inputs",
]
