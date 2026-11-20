"""Running a whole graph from generated code.

Some nodes contain a graph of their own: a Scan has its loop body, a Blockwise has its
core, a compiled unit (an OpFromGraph) has its inner graph. The program itself is also
just a graph of items. All of them are run the same way:

1. `cg.graph_function(inputs, outputs, in_layouts)` decides how every node of the graph
   runs (generated code, or a call of a kernel) and returns a `GraphFunction`.
2. `GraphFunction.emit_call(cg, inputs, outputs)` emits the code that runs the graph.
   It is called like an emitter (see interface.py): it gets the values of the inputs
   and one `NodeOutput` per output, and the `NodeOutput` decides where the output's
   memory comes from.

`build_graph_function` wraps this into a function of its own: the program's top level.
A compiled unit's emitter (an OpFromGraph with `execution=COMPILED`) is here too: its
code runs its inner graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.link.llvm.codegen.emit import Array, Codegen, copy, unpack
from pytensor.link.llvm.codegen.interface import (
    ProvidedOutput,
    SlotOutput,
    llvm_emit,
    llvm_layout,
)
from pytensor.link.llvm.codegen.node import NodeFunction
from pytensor.link.llvm.codegen.signature import OutputMode, Signature
from pytensor.link.llvm.codegen.types import ScalarRepr, ir, repr_of
from pytensor.link.llvm.layout import infer_layouts, new_memory_layout
from pytensor.link.llvm.codegen.items import (
    Destination,
    bind_constants,
    emit_items,
    temporaries_of,
)


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.interface import NodeOutput
    from pytensor.link.llvm.plan.resolve import Body


@dataclass
class GraphFunction:
    """A graph, ready to be run from the function being generated.

    Get one from `Codegen.graph_function`, then call `emit_call` to emit the code that
    runs it."""

    #: the graph, with the decision for each of its nodes (`plan.resolve.Body`)
    body: Body
    #: prefix for the names of the LLVM functions made for its nodes
    name: str
    #: the values of the graph's constants (vid -> Array or IR value)
    constants: dict
    #: for each value a node produces, how the node produces it (vid -> `Output`)
    produced_by: dict = field(default_factory=dict)
    #: where each array constant is in the link table (vid -> entry)
    entries: dict = field(default_factory=dict)
    _calls: int = 0

    @classmethod
    def bind(cls, cg, body: Body, name: str) -> GraphFunction:
        """Prepare the graph `body` to be run from the function `cg` is building.

        This emits code for the graph's constants at the current point, so call it
        before a loop that runs the graph, not inside the loop."""
        constants: dict = {}
        entries = bind_constants(cg, body.constants, constants)
        produced_by = {
            vid: o
            for r in body.items
            for vid, o in zip(r.item.outputs, r.signature.outputs, strict=True)
        }
        return cls(body, name, constants, produced_by, entries)

    @property
    def nodes(self) -> list:
        """The nodes of the graph, in the order they run."""
        return [r.item.node for r in self.body.items]

    def emit_call(
        self, cg, inputs: list, outputs: list[NodeOutput | None], profile=None
    ) -> tuple[frozenset, tuple[dict, int]]:
        """Emit the code that runs the graph, into the function `cg` is building.

        Nodes with generated code are inlined; the others become calls of their
        kernels. This works like an emitter (see interface.py):

        * `inputs`: one value per graph input: an `Array` for a tensor (its layout must
          imply the layout the graph was prepared for), an IR value for a ScalarType,
          or None for a value without data. Inputs that are constants may be None.
        * `outputs`: one `NodeOutput` per graph output, or None for a value without
          data.

        How each output gets its memory:

        * If the node computing it writes into memory it is given, the output's
          `allocate` is called before the graph runs, so the node can write the result
          there directly.
        * If the node allocates the memory itself, and the output offers its slot
          (`offer_slot`), the node allocates directly into that slot.
        * Otherwise the output's `view` (or `set`, for a scalar) is called with the
          result after the graph has run.

        Every output's `finish` is called last, after all outputs were produced. Until
        then the graph's inputs may still be read, so outputs must not overwrite them
        before `finish`.

        Memory that kernels allocate inside the graph is released when the innermost
        `scratch_scope` ends: use the results before that.

        `profile`: the link table entry of the profile counters, if profiling.

        Returns where the graph's temporary values went: the vids of all of them, and
        the arena that holds those not on the stack ({vid: offset}, size in bytes)."""
        body = self.body
        b = cg.b
        vals = dict(self.constants)
        for vid, v in zip(body.in_ids, inputs, strict=True):
            if v is not None and vid not in vals:
                vals[vid] = v

        allocated: dict[int, Array] = {}
        slots: dict[int, Destination] = {}
        targets: dict = {}
        for k, (vid, out) in enumerate(zip(body.out_ids, outputs, strict=True)):
            o = self.produced_by.get(vid)
            if out is None or o is None:
                continue
            if o.mode is OutputMode.PROVIDED:
                a = out.allocate([b.i64(n) for n in o.repr.shape])
                allocated[k] = a
                targets.setdefault(vid, Destination(array=a, layout=a.layout))
            elif o.mode is OutputMode.RETURNED and not o.aliases:
                offered = out.offer_slot()
                if offered is not None:
                    slots[k] = Destination(slot=offered[0], layout=offered[1])
                    targets.setdefault(vid, slots[k])

        memory, arena = temporaries_of(cg, body.items, targets, body.out_ids)

        name = self.name if self._calls == 0 else f"{self.name}{self._calls}"
        self._calls += 1
        written, owned = emit_items(
            cg,
            cg.module,
            name,
            body.items,
            vals,
            targets,
            memory=memory,
            profile=profile,
        )
        for slot, ndim in owned:
            cg.release_later(slot, ndim)

        for k, (vid, out) in enumerate(zip(body.out_ids, outputs, strict=True)):
            if out is None:
                continue
            v = vals[vid]
            if k in slots and vid in written and targets[vid] is slots[k]:
                out.slot_filled()
            elif k in allocated:
                if not (vid in written and targets[vid].array is allocated[k]):
                    copy(cg, v, allocated[k])
            elif isinstance(v, Array):
                out.view(v)
            else:
                out.set(v)
        for out in outputs:
            if out is not None:
                out.finish()
        return frozenset(memory), arena


def build_graph_function(
    module,
    name: str,
    body: Body,
    in_types: list,
    in_layouts: list,
    outputs: list,
    parallel: bool = True,
    links=None,
    profile=None,
    resolver=None,
) -> NodeFunction:
    """Build an LLVM function that runs the graph `body` (the program's top level).

    The function has the same kind of signature as a node's function, but is not
    inlined: the program's kernel wrapper calls it.

    * `in_types`, `in_layouts`: the type of each graph input (tensors only), and the
      layout the graph was prepared for;
    * `outputs`: one `Output` per graph output: memory the caller provides
      (PROVIDED), or new memory the function returns (RETURNED);
    * `links`: the function's link table; the kernels it calls and its array
      constants are added to it;
    * `profile`: if given, an array of counters (`PROFILE_ITEMS + len(body.items)`
      uint64 values) that the code adds the time of each item to;
    * `resolver`: decides how the nodes of inner graphs run (a Scan's body, ...).

    The returned `NodeFunction` also says where the temporary values went (`arena`,
    `temporaries`) and where the array constants are in the link table
    (`constants`)."""
    in_reprs = [repr_of(t, lay) for t, lay in zip(in_types, in_layouts, strict=True)]
    for r in [*in_reprs, *(o.repr for o in outputs)]:
        if isinstance(r, ScalarRepr):
            raise NotImplementedError("ScalarType values can't cross the boundary")
    for o in outputs:
        if o.mode not in (OutputMode.PROVIDED, OutputMode.RETURNED) or o.aliases:
            raise NotImplementedError(f"function outputs can't be {o.mode}")
    sig = Signature(tuple(in_reprs), tuple(outputs))
    fn = module.function(name, sig.param_types(), ir.i32, debug_name=name)
    b = fn.builder()
    params = iter(fn.params)
    ctx, config = next(params), next(params)
    cg = Codegen(
        b,
        ctx,
        config,
        parallel=parallel,
        links=links,
        resolver=resolver,
        module=module,
        name=name,
    )
    inputs = []
    for pos, r in enumerate(in_reprs):
        a = unpack(b, r, next(params))
        a.base, a.writable = pos, False
        inputs.append(a)
    outs: list[NodeOutput | None] = [
        ProvidedOutput(cg, unpack(b, o.repr, v))
        if o.mode is OutputMode.PROVIDED
        else SlotOutput.for_output(cg, v, o)
        for o, v in zip(outputs, params, strict=True)
    ]
    # (temporaries are freed before the function returns)
    with cg.scratch_scope(function=True):
        graph = GraphFunction.bind(cg, body, name)
        cells = None if profile is None else links.link_array(profile, share=False)
        temporaries, arena = graph.emit_call(cg, inputs, outs, profile=cells)
    b.ret(0)
    return NodeFunction(
        fn,
        sig,
        links,
        arena=arena,
        temporaries=temporaries,
        constants=graph.entries,
    )


# ------------------------------------------------------------------ compiled units


@llvm_layout.register(OpFromGraph)
def _(op, node, in_layouts):
    """The layouts of the inner outputs. Inlined that is what they are; a compiled unit
    has them where new memory can (it copies an output that isn't one of its inputs'
    memory: `new_memory_layout`)."""
    inner = infer_layouts(op.fgraph, in_layouts)
    outs = [inner[o] for o in op.fgraph.outputs]
    if op.execution is Execution.COMPILED:
        return [
            None if lay is None else new_memory_layout(lay, o.type)
            for lay, o in zip(outs, node.outputs, strict=True)
        ]
    return outs


@llvm_emit.register(OpFromGraph)
def _(op, node, cg, inputs, outputs):
    """A compiled unit (an OpFromGraph with `execution=COMPILED`): its code runs its
    inner graph. (Other OpFromGraphs never get here: they are inlined.)"""
    layouts = [a.layout if isinstance(a, Array) else None for a in inputs]
    graph = cg.graph_function(op.fgraph.inputs, op.fgraph.outputs, layouts, "unit")
    graph.emit_call(cg, inputs, outputs)


__all__ = ["GraphFunction", "build_graph_function"]
