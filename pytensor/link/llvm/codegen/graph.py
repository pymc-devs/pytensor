"""Running a whole graph from generated code.

Some nodes contain a graph of their own: a Scan has its loop body, a Blockwise has its
core, a compiled unit (an OpFromGraph) has its inner graph. The program itself is also
just a graph of steps. All of them are run the same way:

1. `cg.graph_function(inputs, outputs, in_layouts)` decides how every node of the graph
   runs (generated code, or a call of a kernel) and returns a `GraphFunction`.
2. `GraphFunction.emit_call(cg, inputs, outputs)` emits the code that runs the graph.
   It gets the values of the inputs and one `OutputDestination` per output, which
   decides where the output's memory comes from (see destination.py).

`build_graph_function` wraps this into a function of its own: the program's top level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.analysis.liveness import temporary_lives
from pytensor.link.llvm.codegen.destination import (
    ProvidedDestination,
    SlotDestination,
)
from pytensor.link.llvm.codegen.emit import (
    Array,
    Codegen,
    c_array,
    constant,
    copy,
    pack,
    packed_array,
    read_slot,
    store,
    unpack,
)
from pytensor.link.llvm.codegen.node import NodeFunction, build_node_function
from pytensor.link.llvm.codegen.signature import Output, Passing, Signature
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    Slot,
    ir,
    itemsize,
    repr_of,
    slot_type,
)


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.destination import OutputDestination
    from pytensor.link.llvm.planner import PlannedGraph


@dataclass
class GraphFunction:
    """A graph, ready to be run from the function being generated.

    Get one from `Codegen.graph_function`, then call `emit_call` to emit the code that
    runs it."""

    #: the graph, with the decision for each of its nodes (`planner.PlannedGraph`)
    planned: PlannedGraph
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
    def bind(cls, cg, planned: PlannedGraph, name: str) -> GraphFunction:
        """Prepare the graph `planned` to be run from the function `cg` is building.

        This emits code for the graph's constants at the current point, so call it
        before a loop that runs the graph, not inside the loop."""
        constants: dict = {}
        entries = bind_constants(cg, planned.constants, constants)
        produced_by = {
            vid: o
            for r in planned.steps
            for vid, o in zip(r.outputs, r.signature.outputs, strict=True)
        }
        return cls(planned, name, constants, produced_by, entries)

    @property
    def nodes(self) -> list:
        """The nodes of the graph, in the order they run."""
        return [r.node for r in self.planned.steps]

    def emit_call(
        self, cg, inputs: list, outputs: list[OutputDestination | None], profile=None
    ) -> tuple[frozenset, tuple[dict, int]]:
        """Emit the code that runs the graph, into the function `cg` is building.

        Nodes with generated code are inlined; the others become calls of their
        kernels.

        * `inputs`: one value per graph input: an `Array` for a tensor (its layout must
          imply the layout the graph was prepared for), an IR value for a ScalarType,
          or None for a value without data. Inputs that are constants may be None.
        * `outputs`: one `OutputDestination` per graph output, or None for a value
          without data.

        How each output gets its memory:

        * If the node computing it writes into memory it is given, the output's
          `allocate` is called before the graph runs, so the node can write the result
          there directly.
        * If the node allocates the memory itself, and the output offers its slot
          (`offer_slot`), the node allocates directly into that slot.
        * Otherwise the output's `assign` (or `set`, for a scalar) is called with the
          result after the graph has run.

        Every output's `finish` is called last, after all outputs were produced. Until
        then the graph's inputs may still be read, so outputs must not overwrite them
        before `finish`.

        Memory that kernels allocate inside the graph is released when the innermost
        `scratch_scope` ends: use the results before that.

        `profile`: the link table entry of the profile counters, if profiling.

        Returns where the graph's temporary values went: the vids of all of them, and
        the arena that holds those not on the stack ({vid: offset}, size in bytes)."""
        planned = self.planned
        b = cg.b
        vals = dict(self.constants)
        for vid, v in zip(planned.in_ids, inputs, strict=True):
            if v is not None and vid not in vals:
                vals[vid] = v

        allocated: dict[int, Array] = {}
        slots: dict[int, Target] = {}
        targets: dict = {}
        for k, (vid, out) in enumerate(zip(planned.out_ids, outputs, strict=True)):
            o = self.produced_by.get(vid)
            if out is None or o is None:
                continue
            if o.passing is Passing.CALLER_BUFFER:
                a = out.allocate([b.i64(n) for n in o.representation.shape])
                allocated[k] = a
                targets.setdefault(vid, Target(array=a, layout=a.layout))
            elif o.passing is Passing.CALLEE_SLOT and not o.aliased_inputs:
                offered = out.offer_slot()
                if offered is not None:
                    slots[k] = Target(slot=offered[0], layout=offered[1])
                    targets.setdefault(vid, slots[k])

        memory, arena = temporaries_of(cg, planned.steps, targets, planned.out_ids)

        name = self.name if self._calls == 0 else f"{self.name}{self._calls}"
        self._calls += 1
        written, owned = emit_steps(
            cg,
            cg.module,
            name,
            planned.steps,
            vals,
            targets,
            memory=memory,
            profile=profile,
        )
        for slot, ndim in owned:
            cg.release_later(slot, ndim)

        for k, (vid, out) in enumerate(zip(planned.out_ids, outputs, strict=True)):
            if out is None:
                continue
            v = vals[vid]
            if k in slots and vid in written and targets[vid] is slots[k]:
                out.slot_filled()
            elif k in allocated:
                if not (vid in written and targets[vid].array is allocated[k]):
                    copy(cg, v, allocated[k])
            elif isinstance(v, Array):
                out.assign(v)
            else:
                out.set(v)
        for out in outputs:
            if out is not None:
                out.finish()
        return frozenset(memory), arena


def build_graph_function(
    module,
    name: str,
    planned: PlannedGraph,
    in_types: list,
    in_layouts: list,
    outputs: list,
    parallel: bool = True,
    links=None,
    profile=None,
    planner=None,
) -> NodeFunction:
    """Build an LLVM function that runs the graph `planned` (the program's top level).

    The function has the same kind of signature as a node's function, but is not
    inlined: the program's kernel wrapper calls it.

    * `in_types`, `in_layouts`: the type of each graph input (tensors only), and the
      layout the graph was prepared for;
    * `outputs`: one `Output` per graph output: memory the caller provides
      (PROVIDED), or new memory the function returns (RETURNED);
    * `links`: the function's link table; the kernels it calls and its array
      constants are added to it;
    * `profile`: if given, an array of counters (`PROFILE_ITEMS + len(planned.steps)`
      uint64 values) that the code adds the time of each step to;
    * `planner`: decides how the nodes of inner graphs run (a Scan's body, ...).

    The returned `NodeFunction` also says where the temporary values went (`arena`,
    `temporaries`) and where the array constants are in the link table
    (`constants`)."""
    in_reprs = [repr_of(t, lay) for t, lay in zip(in_types, in_layouts, strict=True)]
    for r in [*in_reprs, *(o.representation for o in outputs)]:
        if isinstance(r, ScalarRepr):
            raise NotImplementedError("ScalarType values can't cross the boundary")
    for o in outputs:
        if not o.is_tensor or o.aliased_inputs:
            raise NotImplementedError(f"function outputs can't be {o}")
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
        planner=planner,
        module=module,
        name=name,
    )
    inputs = []
    for pos, r in enumerate(in_reprs):
        a = unpack(b, r, next(params))
        a.base, a.writable = pos, False
        inputs.append(a)
    outs: list[OutputDestination | None] = [
        ProvidedDestination(cg, unpack(b, o.representation, v))
        if o.passing is Passing.CALLER_BUFFER
        else SlotDestination.for_output(cg, v, o)
        for o, v in zip(outputs, params, strict=True)
    ]
    # (temporaries are freed before the function returns)
    with cg.scratch_scope(function=True):
        graph = GraphFunction.bind(cg, planned, name)
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


# ------------------------------------------------------------------ emitting the steps
#
# The machinery `GraphFunction.emit_call` uses: it emits the steps of a graph one after
# the other (`emit_steps`). For each step it calls the step's function, which is
# inlined: either the node's generated code, or a call of the step's kernel. Where the
# steps' results go:
#
# * a result the step writes into memory it is given goes into memory the caller asked
#   for (`targets`, e.g. a row of a Scan's buffer), else into a temporary
#   (`temporaries_of`: on the stack, or in an arena shared by values that are never
#   needed at the same time);
# * a result the step allocates itself goes straight into an output slot of the caller
#   if it asked for that (`targets`), else into memory that is released afterwards;
# * views and scalars are read back from the step's output slots;
# * 0-d constants are written into the code.


#: the cells of a profile (`emit_steps`'s `profile`): profiling on (nonzero),
#: number of calls, then nanoseconds per step
PROFILE_ON, PROFILE_CALLS, PROFILE_ITEMS = 0, 1, 2


@dataclass
class Target:
    """Where a value may be produced directly: memory (`array`, in `layout`), or an
    output slot (`slot`) for a value allocated in new memory (in `layout`)."""

    array: Array | None = None
    layout: Layout | None = None
    slot: object = None


def fits(t: Target | None, o: Output) -> bool:
    """Can a step produce its output `o` straight in `t`? Provided memory whose layout
    implies the declared one, or a slot for new memory in a layout implying the
    slot's."""
    if t is None or t.layout is None:
        return False
    tensor = o.tensor
    shape, size = tensor.shape, itemsize(tensor.dtype)
    if o.passing is Passing.CALLER_BUFFER:
        return t.array is not None and t.layout.implies(tensor.layout, shape, size)
    return (
        o.passing is Passing.CALLEE_SLOT
        and not o.aliased_inputs
        and t.slot is not None
        and tensor.layout.implies(t.layout, shape, size)
    )


def temporaries_of(cg, steps, targets, kept) -> tuple[dict, tuple[dict, int]]:
    """Memory for the temporary values of the steps `steps` (`Codegen.temporaries`).

    A temporary is a result a step writes into memory it is given, when that memory
    isn't one the caller asked for (`targets`, vid -> `Target`). `kept`: the values
    still needed after the last step (the outputs). Returns what `Codegen.temporaries`
    returns."""
    sizes = {
        vid: int(np.prod(o.representation.shape, dtype=np.int64))
        * itemsize(o.representation.dtype)
        for r in steps
        for vid, o in zip(r.outputs, r.signature.outputs, strict=True)
        if o.passing is Passing.CALLER_BUFFER
        and o.representation.static
        and not fits(targets.get(vid), o)
    }
    return cg.temporaries(sizes, temporary_lives(steps, sizes, kept))


def release(cg, owned):
    """Give up the buffers kernels allocated into `owned` slots ([(slot, ndim)])."""
    for slot, nd in owned:
        cg.b.rt_decref(
            cg.ctx, cg.b.load(ir.ptr, cg.b.field_ptr(slot_type(nd), slot, Slot.OWNER))
        )


def bind_constants(cg, constants, vals) -> dict:
    """Values of `constants` ({vid: (type, ndarray)}): 0-d ones folded into the code,
    arrays from the links, in their memory order (`Layout.of_const`; equal ones share
    their memory). Returns {vid: link entry} of the arrays."""
    b = cg.b
    entries = {}
    for vid, (t, data) in constants.items():
        r = repr_of(t)
        data = np.asarray(data, dtype=t.dtype)
        if isinstance(r, ScalarRepr) or data.ndim == 0:
            value = constant(b, t.dtype, data.item())
            if isinstance(r, ScalarRepr):
                vals[vid] = value
            else:
                slot = b.alloca(_mem_type(t.dtype))
                store(b, t.dtype, value, slot)
                vals[vid] = c_array(t.dtype, slot, [], [])
        else:
            if cg.links is None:
                raise NotImplementedError("array constants need a links")
            from pytensor.link.llvm.codegen.call import linked

            layout = Layout.of_const(data)
            order = list(layout.order)
            memory = np.ascontiguousarray(data.transpose(order))
            arr = memory.transpose(np.argsort(order))
            entries[vid] = cg.links.link_array(memory)
            vals[vid] = Array(
                t.dtype,
                linked(cg, entries[vid]),
                [b.i64(n) for n in arr.shape],
                [b.i64(st) for st in arr.strides],
                layout,
            )
    return entries


def emit_steps(
    cg,
    module,
    name,
    steps,
    vals,
    targets=None,
    memory=None,
    profile=None,
):
    """Emit the steps `steps` (`planner.Step`, in order) into the function
    being built in `cg`: each node's function is called (and inlined), or its kernel.

    * `vals`: vid -> Array (tensors) or value (scalars); must hold every value the
      steps read but don't produce, with the layouts they were planned for; filled
      with what they produce;
    * `targets`: vid -> `Target`: where a value may be produced directly (PROVIDED
      outputs whose declared layout the target's implies; returned new C-contiguous
      memory into a slot); other values go into temporary memory or kernel buffers;
    * `memory`: {vid: pointer}: where the provided outputs that aren't produced in their
      targets go (`temporaries_of`); others get temporary memory of their own;
    * `profile`: the link entry of the profile cells (`PROFILE_*`): while profiling is
      on, each step's wall time is added to its cell.

    Returns (written, owned): the vids produced in their targets, and the slots of
    buffers kernels allocated for other values, for the caller to `release` when they
    are no longer used (on failure, they are released here)."""
    from pytensor.link.llvm.codegen.call import build_call_function, linked

    b = cg.b
    ctx, config = cg.ctx, cg.config
    targets = targets or {}
    written = set()
    owned = []  # (slot, ndim): buffers kernels allocated
    lays = {}  # vid -> Layout of what the steps produced
    if profile is not None:
        cells = linked(cg, profile)
        profiling = b.ne(b.load(ir.i64, cells), b.i64(0))
        with b.if_(profiling):
            b.rt_counter_add(b.gep(ir.i64, cells, PROFILE_CALLS), b.i64(1))
        start = b.alloca(ir.i64)
    for n, step in enumerate(steps):
        node, kernel, nsig = step.node, step.kernel, step.signature
        present = step.arguments
        # provided outputs: their target if its layout implies the declared one; else a
        # temporary in the declared layout (copied by the caller if needed)
        in_place = {
            j: targets[vid].array
            for j, (vid, o) in enumerate(zip(step.outputs, nsig.outputs, strict=True))
            if o.passing is Passing.CALLER_BUFFER
            and vid not in written
            and fits(targets.get(vid), o)
        }
        for vid, o in zip(step.outputs, nsig.outputs, strict=True):
            if o.layout is not None:
                lays[vid] = o.layout

        def into_slot(vid, o):
            # an allocated value with a slot target: allocated straight into it if the
            # node allocates it in a layout the slot takes
            return (
                o.passing is Passing.CALLEE_SLOT
                and vid not in written
                and fits(targets.get(vid), o)
            )

        temporaries = {
            j
            for j, (vid, o) in enumerate(zip(step.outputs, nsig.outputs, strict=True))
            if o.passing is Passing.CALLEE_SLOT and not into_slot(vid, o)
        }
        if node is None:
            nf = build_call_function(module, f"{name}_n{n}", nsig, kernel, cg.links)
        else:
            nf = build_node_function(
                module,
                node,
                f"{name}_n{n}",
                scratch_outputs=frozenset(temporaries),
                input_layouts=step.in_layouts,
                parallel=cg.parallel,
                check_layouts=cg.check_layouts,
                kernel=kernel,
                links=cg.links,
                planner=cg.planner,
            )
        args = [ctx, config]
        for vid, rep in zip(present, nsig.inputs, strict=True):
            if vid not in vals:
                raise ValueError(
                    f"{name}: {step.description} reads a value that isn't computed yet"
                )
            v = vals[vid]
            if isinstance(rep, ScalarRepr):
                args.append(v)
                continue
            if not v.layout.implies(rep.layout, rep.shape, itemsize(rep.dtype)):
                raise AssertionError(
                    f"{name}: {step.description} was planned for an input with layout "
                    f"{rep.layout}, got {v.layout}"
                )
            args.append(pack(b, rep, v))
        reads = []  # (vid, slot, how)
        for j, (vid, o) in enumerate(zip(step.outputs, nsig.outputs, strict=True)):
            if o.is_absent:
                continue
            if o.passing is Passing.CALLER_BUFFER:
                if j in in_place:
                    a = in_place[j]
                    written.add(vid)
                else:
                    representation = o.representation
                    lay = (
                        o.layout
                        if o.layout.fully_packed(representation.shape)
                        else None
                    )  # (C implies any)
                    shape = list(representation.shape)
                    if memory is not None and vid in memory:
                        dims = [b.i64(m) for m in shape]
                        a = packed_array(
                            b, representation.dtype, memory[vid], dims, lay
                        )
                    else:
                        a = cg.temporary(representation.dtype, shape, lay)
                vals[vid] = a
                args.append(pack(b, o.representation, a))
            elif o.is_scalar:
                slot = b.alloca(o.representation.ir_type)
                args.append(slot)
                reads.append((vid, slot, "scalar", o))
            else:
                if into_slot(vid, o):
                    slot = targets[vid].slot
                    written.add(vid)
                else:
                    ndim = o.representation.ndim
                    slot = b.alloca(slot_type(ndim))
                    b.store(b.zero(slot_type(ndim)), slot)
                    if kernel is not None:
                        owned.append((slot, ndim))
                args.append(slot)
                # a temporary of a generated node is scratch memory it took (or a view)
                how = "scratch" if kernel is None and j in temporaries else "array"
                reads.append((vid, slot, how, o))
        if profile is not None:
            with b.if_(profiling):
                b.store(b.rt_clock_ns(), start)
        rc = b.call(nf.function, args)
        if profile is not None:
            with b.if_(profiling):
                b.rt_counter_add(
                    b.gep(ir.i64, cells, PROFILE_ITEMS + n),
                    b.sub(b.rt_clock_ns(), b.load(ir.i64, start)),
                )
        failed = b.ne(rc, 0)
        with b.if_(failed):
            b.rt_error_context(ctx, " ".join(step.description.split())[:200])
            release(cg, owned)
            b.ret(rc)
        for vid, slot, how, o in reads:
            if how == "scalar":
                vals[vid] = b.load(o.representation.ir_type, slot)
            else:
                representation = o.representation
                vals[vid] = read_slot(
                    b,
                    slot,
                    representation.dtype,
                    list(representation.shape),
                    lays.get(vid),
                )
                if o.aliased_inputs:
                    sources = [vals[present[a]] for a in nsig.alias_arguments(o)]
                    _maybe_view(b, vals[vid], sources, slot, representation.ndim)
                if how == "scratch":
                    # (`SlotDestination.in_scratch_memory`: the owner is the scratch)
                    cg.free_later(
                        b.load(
                            ir.ptr,
                            b.field_ptr(
                                slot_type(representation.ndim), slot, Slot.OWNER
                            ),
                        )
                    )
    return written, owned


def _maybe_view(b, a: Array, sources: list, slot, ndim: int) -> None:
    """Remember that `a` may be a view of an input of the graph.

    `a` was returned through `slot` by a node that may return a view of `sources`
    (some of its inputs). If all of `sources` lie in the memory of the same graph input,
    `a` may too: `a.base` is set to that input. Whether it really is a view is only
    known at run time: when the slot has no owner (a view has none, new memory does).
    `a.base_if` is set to that condition."""
    bases = {s.base if isinstance(s, Array) else None for s in sources}
    if len(bases) != 1 or None in bases:
        return
    owner = b.load(ir.ptr, b.field_ptr(slot_type(ndim), slot, Slot.OWNER))
    really = b.eq(owner, b.null())
    for s in sources:
        if s.base_if is not None:
            really = b.and_(really, s.base_if)
    a.base, a.writable, a.base_if = bases.pop(), False, really


def _mem_type(dtype):
    from pytensor.link.llvm.codegen.types import memory_type

    return memory_type(dtype)


__all__ = [
    "PROFILE_CALLS",
    "PROFILE_ITEMS",
    "PROFILE_ON",
    "GraphFunction",
    "Target",
    "build_graph_function",
    "emit_steps",
]
