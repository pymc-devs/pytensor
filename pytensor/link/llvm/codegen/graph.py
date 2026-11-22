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
            vid: output
            for step in planned.steps
            for vid, output in zip(step.outputs, step.signature.outputs, strict=True)
        }
        return cls(planned, name, constants, produced_by, entries)

    @property
    def nodes(self) -> list:
        """The nodes of the graph, in the order they run."""
        return [step.node for step in self.planned.steps]

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
          `allocate` is called before the graph runs. The node writes the result there
          directly if that memory has the layout it needs, else the result is copied
          there afterwards.
        * If the node allocates the memory itself, it allocates directly into the
          output's slot where the output takes that (`slot_for_new_memory`).
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
        values = dict(self.constants)
        for vid, v in zip(planned.in_ids, inputs, strict=True):
            if v is not None and vid not in values:
                values[vid] = v

        # Where results may be produced directly: memory a graph output provides
        # (vid -> Array), or a graph output's slot (vid -> its destination). A value
        # that is several outputs is produced directly into the first one's.
        memory_targets: dict[int, Array] = {}
        slot_targets: dict[int, OutputDestination] = {}
        target_of: dict[int, int] = {}  # vid -> the output whose target it is
        given: dict[int, Array] = {}  # output -> the memory it provides
        for k, (vid, out) in enumerate(zip(planned.out_ids, outputs, strict=True)):
            produced = self.produced_by.get(vid)
            if out is None or produced is None or vid in target_of:
                continue
            if produced.passing is Passing.CALLER_BUFFER:
                shape = [b.i64(n) for n in produced.tensor.shape]
                given[k] = memory_targets[vid] = out.allocate(shape)
                target_of[vid] = k
            elif (
                produced.passing is Passing.CALLEE_SLOT and not produced.aliased_inputs
            ):
                slot_targets[vid] = out
                target_of[vid] = k

        memory, arena = temporaries_of(
            cg, planned.steps, memory_targets, planned.out_ids
        )

        name = self.name if self._calls == 0 else f"{self.name}{self._calls}"
        self._calls += 1
        written, owned = emit_steps(
            cg,
            cg.module,
            name,
            planned.steps,
            values,
            memory_targets,
            slot_targets,
            memory=memory,
            profile=profile,
        )
        for slot, ndim in owned:
            cg.release_later(slot, ndim)

        for k, (vid, out) in enumerate(zip(planned.out_ids, outputs, strict=True)):
            if out is None:
                continue
            if vid in written and target_of[vid] == k:
                continue  # produced directly in the output
            value = values[vid]
            if k in given:
                copy(cg, value, given[k])
            elif isinstance(value, Array):
                out.assign(value)
            else:
                out.set(value)
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
      (`CALLER_BUFFER`), or new memory the function returns (`CALLEE_SLOT`);
    * `links`: the function's link table; the kernels it calls and its array
      constants are added to it;
    * `profile`: if given, an array of counters (`PROFILE_ITEMS + len(planned.steps)`
      uint64 values) that the code adds the time of each step to;
    * `planner`: decides how the nodes of inner graphs run (a Scan's body, ...).

    The returned `NodeFunction` also says where the temporary values went (`arena`,
    `temporaries`) and where the array constants are in the link table
    (`constants`)."""
    in_reprs = [
        repr_of(t, layout) for t, layout in zip(in_types, in_layouts, strict=True)
    ]
    for representation in [*in_reprs, *(output.representation for output in outputs)]:
        if isinstance(representation, ScalarRepr):
            raise NotImplementedError("ScalarType values can't cross the boundary")
    for output in outputs:
        if not output.is_tensor or output.aliased_inputs:
            raise NotImplementedError(f"function outputs can't be {output}")
    signature = Signature(tuple(in_reprs), tuple(outputs))
    fn = module.function(name, signature.param_types(), ir.i32, debug_name=name)
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
    for pos, representation in enumerate(in_reprs):
        a = unpack(b, representation, next(params))
        a.base, a.writable = pos, False
        inputs.append(a)
    outs: list[OutputDestination | None] = [
        ProvidedDestination(cg, unpack(b, output.representation, v))
        if output.passing is Passing.CALLER_BUFFER
        else SlotDestination.for_output(cg, v, output)
        for output, v in zip(outputs, params, strict=True)
    ]
    # (temporaries are freed before the function returns)
    with cg.scratch_scope(function=True):
        graph = GraphFunction.bind(cg, planned, name)
        cells = None if profile is None else links.link_array(profile, share=False)
        temporaries, arena = graph.emit_call(cg, inputs, outs, profile=cells)
    b.ret(0)
    return NodeFunction(
        fn,
        signature,
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
# * a result the step writes into memory it is given goes into memory a graph output
#   provides (`memory_targets`, e.g. a row of a Scan's buffer) if that memory has the
#   layout the step needs, else into a temporary (`temporaries_of`: on the stack, or in
#   an arena shared by values that are never needed at the same time);
# * a result the step allocates itself goes straight into a graph output's slot if the
#   output takes it (`slot_targets`, `OutputDestination.slot_for_new_memory`), else
#   into memory that is released afterwards;
# * views and scalars are read back from the step's output slots;
# * 0-d constants are written into the code.


#: the cells of a profile (`emit_steps`'s `profile`): profiling on (nonzero),
#: number of calls, then nanoseconds per step
PROFILE_ON, PROFILE_CALLS, PROFILE_ITEMS = 0, 1, 2


def _writes_into(memory: Array, output: Output) -> bool:
    """Can a step write its output `output`, passed in a caller buffer, straight into
    `memory`? If `memory`'s layout implies the one the step needs."""
    tensor = output.tensor
    size = itemsize(tensor.dtype)
    return memory.layout.implies(tensor.layout, tensor.shape, size)


def temporaries_of(
    cg, steps, memory_targets: dict, kept
) -> tuple[dict, tuple[dict, int]]:
    """Memory for the temporary values of the steps `steps` (`Codegen.temporaries`).

    A temporary is a result a step writes into memory it is given, when it can't be
    memory a graph output provides (`memory_targets`, vid -> Array, see
    `_writes_into`). `kept`: the values still needed after the last step (the outputs).
    Returns what `Codegen.temporaries` returns."""
    sizes = {}
    for step in steps:
        for vid, output in zip(step.outputs, step.signature.outputs, strict=True):
            if output.passing is not Passing.CALLER_BUFFER:
                continue
            tensor = output.tensor
            if not tensor.static:
                continue
            target = memory_targets.get(vid)
            if target is not None and _writes_into(target, output):
                continue
            sizes[vid] = int(np.prod(tensor.shape, dtype=np.int64)) * itemsize(
                tensor.dtype
            )
    return cg.temporaries(sizes, temporary_lives(steps, sizes, kept))


@dataclass
class _SlotRead:
    """A result to read back from a step's output slot after the step ran."""

    vid: int
    slot: ir.Value
    output: Output
    #: it is scratch memory of the generated node (freed with the surrounding scope)
    in_scratch: bool = False


def release(cg, owned):
    """Give up the buffers kernels allocated into `owned` slots ([(slot, ndim)])."""
    for slot, nd in owned:
        cg.b.rt_decref(
            cg.ctx, cg.b.load(ir.ptr, cg.b.field_ptr(slot_type(nd), slot, Slot.OWNER))
        )


def bind_constants(cg, constants, values) -> dict:
    """Values of `constants` ({vid: (type, ndarray)}): 0-d ones folded into the code,
    arrays from the links, in their memory order (`Layout.of_const`; equal ones share
    their memory). Returns {vid: link entry} of the arrays."""
    b = cg.b
    entries = {}
    for vid, (t, data) in constants.items():
        representation = repr_of(t)
        data = np.asarray(data, dtype=t.dtype)
        if isinstance(representation, ScalarRepr) or data.ndim == 0:
            value = constant(b, t.dtype, data.item())
            if isinstance(representation, ScalarRepr):
                values[vid] = value
            else:
                slot = b.alloca(_mem_type(t.dtype))
                store(b, t.dtype, value, slot)
                values[vid] = c_array(t.dtype, slot, [], [])
        else:
            if cg.links is None:
                raise NotImplementedError("array constants need a links")
            from pytensor.link.llvm.codegen.call import linked

            layout = Layout.of_const(data)
            order = list(layout.order)
            memory = np.ascontiguousarray(data.transpose(order))
            arr = memory.transpose(np.argsort(order))
            entries[vid] = cg.links.link_array(memory)
            values[vid] = Array(
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
    values,
    memory_targets: dict | None = None,
    slot_targets: dict | None = None,
    memory=None,
    profile=None,
):
    """Emit the steps `steps` (`planner.Step`, in order) into the function
    being built in `cg`: each node's function is called (and inlined), or its kernel.

    * `vals`: vid -> Array (tensors) or value (scalars); must hold every value the
      steps read but don't produce, with the layouts they were planned for; filled
      with what they produce;
    * `memory_targets`: vid -> Array: memory a value written into given memory may be
      produced in directly, if its layout implies the one the step needs
      (`_writes_into`);
    * `slot_targets`: vid -> `OutputDestination`: whose slot a value a step allocates
      may be allocated into, if the destination takes it (`slot_for_new_memory`);
    * `memory`: {vid: pointer}: where the other values written into given memory go
      (`temporaries_of`); others get temporary memory of their own;
    * `profile`: the link entry of the profile cells (`PROFILE_*`): while profiling is
      on, each step's wall time is added to its cell.

    Returns (written, owned): the vids produced directly in their targets, and the
    slots of buffers kernels allocated for other values, for the caller to `release`
    when they are no longer used (on failure, they are released here)."""
    from pytensor.link.llvm.codegen.call import build_call_function, linked

    b = cg.b
    ctx, config = cg.ctx, cg.config
    memory_targets = memory_targets or {}
    slot_targets = slot_targets or {}
    written: set[int] = set()
    owned = []  # (slot, ndim): buffers kernels allocated
    layouts = {}  # vid -> Layout of what the steps produced
    if profile is not None:
        cells = linked(cg, profile)
        profiling = b.ne(b.load(ir.i64, cells), b.i64(0))
        with b.if_(profiling):
            b.rt_counter_add(b.gep(ir.i64, cells, PROFILE_CALLS), b.i64(1))
        start = b.alloca(ir.i64)
    for n, step in enumerate(steps):
        node, kernel, signature = step.node, step.kernel, step.signature
        results = list(zip(step.outputs, signature.outputs, strict=True))
        for vid, output in results:
            if output.layout is not None:
                layouts[vid] = output.layout

        # Where each result goes, decided before the node's function is built (it
        # allocates differently into scratch memory): given memory of a graph output,
        # a graph output's slot, or neither.
        into_memory: dict[int, Array] = {}
        into_slot: dict[int, ir.Value] = {}
        for j, (vid, output) in enumerate(results):
            if vid in written:
                continue
            if output.passing is Passing.CALLER_BUFFER:
                target = memory_targets.get(vid)
                if target is not None and _writes_into(target, output):
                    into_memory[j] = target
                    written.add(vid)
            elif output.passing is Passing.CALLEE_SLOT and not output.aliased_inputs:
                destination = slot_targets.get(vid)
                if destination is not None:
                    slot = destination.slot_for_new_memory(output.tensor)
                    if slot is not None:
                        into_slot[j] = slot
                        written.add(vid)
        # results the node allocates that go nowhere else: its scratch memory
        scratch = {
            j
            for j, (_, output) in enumerate(results)
            if output.passing is Passing.CALLEE_SLOT and j not in into_slot
        }

        if node is None:
            node_function = build_call_function(
                module, f"{name}_n{n}", signature, kernel, cg.links
            )
        else:
            node_function = build_node_function(
                module,
                node,
                f"{name}_n{n}",
                scratch_outputs=frozenset(scratch),
                input_layouts=step.in_layouts,
                parallel=cg.parallel,
                check_layouts=cg.check_layouts,
                kernel=kernel,
                links=cg.links,
                planner=cg.planner,
            )
        args = [ctx, config]
        for vid, representation in zip(step.arguments, signature.inputs, strict=True):
            if vid not in values:
                raise ValueError(
                    f"{name}: {step.description} reads a value that isn't computed yet"
                )
            value = values[vid]
            if isinstance(representation, ScalarRepr):
                args.append(value)
                continue
            if not value.layout.implies(
                representation.layout,
                representation.shape,
                itemsize(representation.dtype),
            ):
                raise AssertionError(
                    f"{name}: {step.description} was planned for an input with layout "
                    f"{representation.layout}, got {value.layout}"
                )
            args.append(pack(b, representation, value))

        scalar_reads = []  # (vid, pointer, output)
        slot_reads: list[_SlotRead] = []
        for j, (vid, output) in enumerate(results):
            if output.is_absent:
                continue
            if output.passing is Passing.CALLER_BUFFER:
                tensor = output.tensor
                if j in into_memory:
                    array = into_memory[j]
                else:
                    # a temporary, packed in the declared layout (C implies any)
                    layout = (
                        tensor.layout
                        if tensor.layout.fully_packed(tensor.shape)
                        else None
                    )
                    if memory is not None and vid in memory:
                        lengths = [b.i64(m) for m in tensor.shape]
                        array = packed_array(
                            b, tensor.dtype, memory[vid], lengths, layout
                        )
                    else:
                        array = cg.temporary(tensor.dtype, list(tensor.shape), layout)
                values[vid] = array
                args.append(pack(b, tensor, array))
            elif output.is_scalar:
                pointer = b.alloca(output.representation.ir_type)
                args.append(pointer)
                scalar_reads.append((vid, pointer, output))
            else:
                ndim = output.tensor.ndim
                if j in into_slot:
                    slot = into_slot[j]
                else:
                    slot = b.alloca(slot_type(ndim))
                    b.store(b.zero(slot_type(ndim)), slot)
                    if kernel is not None:
                        owned.append((slot, ndim))
                args.append(slot)
                # (a scratch result of a generated node is scratch memory it took, or
                # a view)
                in_scratch = kernel is None and j in scratch
                slot_reads.append(_SlotRead(vid, slot, output, in_scratch))

        if profile is not None:
            with b.if_(profiling):
                b.store(b.rt_clock_ns(), start)
        rc = b.call(node_function.function, args)
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

        for vid, pointer, output in scalar_reads:
            values[vid] = b.load(output.representation.ir_type, pointer)
        for read in slot_reads:
            tensor = read.output.tensor
            array = read_slot(
                b, read.slot, tensor.dtype, list(tensor.shape), layouts.get(read.vid)
            )
            values[read.vid] = array
            if read.output.aliased_inputs:
                sources = [
                    values[step.arguments[a]]
                    for a in signature.alias_arguments(read.output)
                ]
                _maybe_view(b, array, sources, read.slot, tensor.ndim)
            if read.in_scratch:
                # (`SlotDestination.in_scratch_memory`: the owner is the scratch)
                owner = b.field_ptr(slot_type(tensor.ndim), read.slot, Slot.OWNER)
                cg.free_later(b.load(ir.ptr, owner))
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
    "build_graph_function",
    "emit_steps",
]
