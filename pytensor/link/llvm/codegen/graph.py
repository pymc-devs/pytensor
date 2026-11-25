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

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.analysis.liveness import temporary_lives
from pytensor.link.llvm.codegen.array import Array, Slot
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.destination import (
    ProvidedDestination,
    SlotDestination,
)
from pytensor.link.llvm.codegen.interface import (
    EmitterOutput,
    ScalarOutput,
    TensorOutput,
)
from pytensor.link.llvm.codegen.node import NodeFunction, node_function
from pytensor.link.llvm.codegen.num import constant, store
from pytensor.link.llvm.codegen.signature import Output, Passing, Signature
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    TensorRepr,
    element_type,
    ir,
    memory_type,
    repr_of,
)


if TYPE_CHECKING:
    from pytensor.graph.type import Type
    from pytensor.link.llvm.codegen.destination import OutputDestination
    from pytensor.link.llvm.codegen.kernel_module import KernelModule
    from pytensor.link.llvm.codegen.plan import PlannedGraph, Step, ValueId
    from pytensor.scalar.basic import ScalarType
    from pytensor.tensor.type import TensorType


@dataclass
class GraphFunction:
    """A graph, ready to be run from the function being generated.

    Get one from `Codegen.graph_function`, then call `emit_call` to emit the code that
    runs it."""

    #: the graph, with the decision for each of its nodes (`planner.PlannedGraph`)
    planned: PlannedGraph
    #: prefix for the names of the LLVM functions made for its nodes
    name: str
    #: the values of the graph's constants (value id -> Array or IR value)
    constants: dict[ValueId, Array | ir.Value]
    #: for each value a node produces, how the node produces it (value id -> `Output`)
    produced_by: dict[ValueId, Output] = field(default_factory=dict)
    #: where each array constant is in the link table (value id -> entry)
    entries: dict[ValueId, int] = field(default_factory=dict)
    _calls: int = 0

    @classmethod
    def bind(cls, cg: Codegen, planned: PlannedGraph, name: str) -> GraphFunction:
        """Prepare the graph `planned` to be run from the function `cg` is building.

        This emits code for the graph's constants at the current point, so call it
        before a loop that runs the graph, not inside the loop."""
        constants: dict[ValueId, Array | ir.Value] = {}
        entries = _bind_constants(cg, planned.constants, constants)
        produced_by = {
            value_id: output
            for step in planned.steps
            for value_id, output in zip(
                step.outputs, step.signature.outputs, strict=True
            )
        }
        return cls(planned, name, constants, produced_by, entries)

    @property
    def nodes(self) -> list[Apply | None]:
        """The nodes of the graph, in the order they run."""
        return [step.node for step in self.planned.steps]

    def emit_call(
        self,
        cg: Codegen,
        inputs: Sequence[Array | ir.Value | None],
        outputs: Sequence[EmitterOutput | OutputDestination | None],
        profile: int | None = None,
    ) -> tuple[frozenset[ValueId], tuple[dict[ValueId, int], int]]:
        """Emit the code that runs the graph, into the function `cg` is building.

        Nodes with generated code are inlined; the others become calls of their
        kernels.

        * `inputs`: one value per graph input: an `Array` for a tensor (its layout must
          imply the layout the graph was prepared for), an IR value for a ScalarType,
          or None for a value without data. Inputs that are constants may be None.
        * `outputs`: per graph output, where it goes: an output handle of the op
          running the graph (its `TensorOutput` or `ScalarOutput`), an
          `OutputDestination` of the op's own, or None for a value without data.

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

        Returns where the graph's temporary values went: the value ids of all of them,
        and the arena that holds those not on the stack ({value id: offset}, size in
        bytes)."""
        planned = self.planned
        b = cg.b
        destinations = [
            out.destination if isinstance(out, TensorOutput | ScalarOutput) else out
            for out in outputs
        ]
        values: dict[ValueId, Array | ir.Value] = dict(self.constants)
        for value_id, v in zip(planned.in_ids, inputs, strict=True):
            if v is not None and value_id not in values:
                values[value_id] = v

        # Where results may be produced directly: memory a graph output provides
        # (value id -> Array), or a graph output's slot (value id -> its destination).
        # A value that is several outputs is produced directly into the first one's.
        memory_targets: dict[ValueId, Array] = {}
        slot_targets: dict[ValueId, OutputDestination] = {}
        target_of: dict[ValueId, int] = {}  # value id -> the output whose target it is
        given: dict[int, Array] = {}  # output -> the memory it provides
        for k, (value_id, out) in enumerate(
            zip(planned.out_ids, destinations, strict=True)
        ):
            produced = self.produced_by.get(value_id)
            if out is None or produced is None or value_id in target_of:
                continue
            if produced.passing is Passing.CALLER_BUFFER:
                shape = [b.i64(n) for n in produced.tensor.fixed_shape]
                given[k] = memory_targets[value_id] = out.allocate(shape)
                target_of[value_id] = k
            elif (
                produced.passing is Passing.CALLEE_SLOT and not produced.aliased_inputs
            ):
                slot_targets[value_id] = out
                target_of[value_id] = k

        memory, arena = _temporaries_of(
            cg, planned.steps, memory_targets, planned.out_ids
        )

        name = self.name if self._calls == 0 else f"{self.name}{self._calls}"
        self._calls += 1
        written, owned = _emit_steps(
            cg,
            name,
            planned.steps,
            values,
            memory_targets,
            slot_targets,
            memory=memory,
            profile=profile,
        )
        for slot in owned:
            cg.release_later(slot.field(b, Slot.OWNER))

        for k, (value_id, out) in enumerate(
            zip(planned.out_ids, destinations, strict=True)
        ):
            if out is None:
                continue
            if value_id in written and target_of[value_id] == k:
                continue  # produced directly in the output
            value = values[value_id]
            if not isinstance(value, Array):
                out.set(value)
            elif k in given:
                given[k].copy_from(cg, value)
            else:
                out.assign(value)
        for out in destinations:
            if out is not None:
                out.finish()
        return frozenset(memory), arena


def build_graph_function(
    unit: KernelModule,
    name: str,
    planned: PlannedGraph,
    in_types: Sequence[Type],
    in_layouts: Sequence[Layout | None],
    outputs: Sequence[Output],
    profile: np.ndarray | None = None,
) -> NodeFunction:
    """Build an LLVM function in `unit` that runs the graph `planned` (the program's
    top level). The kernels it calls and its array constants go into `unit`'s link
    table, and inner graphs are planned with its `plan_graph`.

    The function has the same kind of signature as a node's function, but is not
    inlined: the program's kernel wrapper calls it.

    * `in_types`, `in_layouts`: the type of each graph input (tensors only), and the
      layout the graph was prepared for;
    * `outputs`: one `Output` per graph output: memory the caller provides
      (`CALLER_BUFFER`), or new memory the function returns (`CALLEE_SLOT`);
    * `profile`: if given, an array of counters (`PROFILE_ITEMS + len(planned.steps)`
      uint64 values) that the code adds the time of each step to;

    The returned `NodeFunction` also says where the temporary values went (`arena`,
    `temporaries`) and where the array constants are in the link table
    (`constants`)."""
    in_reprs: list[TensorRepr] = []
    for t, layout in zip(in_types, in_layouts, strict=True):
        representation = repr_of(t, layout)
        if not isinstance(representation, TensorRepr):
            raise NotImplementedError(f"function inputs can't be {representation}")
        in_reprs.append(representation)
    for output in outputs:
        if not output.is_tensor or output.aliased_inputs:
            raise NotImplementedError(f"function outputs can't be {output}")
    signature = Signature(tuple(in_reprs), tuple(outputs))
    fn = unit.module.function(name, signature.param_types(), ir.i32, debug_name=name)
    b = fn.builder()
    params = iter(fn.params)
    ctx, config = next(params), next(params)
    cg = Codegen(b, ctx, config, unit, name=name)
    inputs: list[Array | ir.Value | None] = []
    for pos, representation in enumerate(in_reprs):
        a = Array.unpack(b, representation, next(params))
        a.base, a.writable = pos, False
        inputs.append(a)
    outs: list[OutputDestination | None] = [
        ProvidedDestination(cg, Array.unpack(b, output.tensor, v))
        if output.passing is Passing.CALLER_BUFFER
        else SlotDestination.for_output(cg, v, output)
        for output, v in zip(outputs, params, strict=True)
    ]
    # (temporaries are freed before the function returns)
    with cg.scratch_scope(function=True):
        graph = GraphFunction.bind(cg, planned, name)
        cells = None if profile is None else unit.links.link_array(profile, share=False)
        temporaries, arena = graph.emit_call(cg, inputs, outs, profile=cells)
    b.ret(0)
    return NodeFunction(
        fn,
        signature,
        unit.links,
        arena=arena,
        temporaries=temporaries,
        constants=graph.entries,
    )


# ------------------------------------------------------------------ emitting the steps
#
# The machinery `GraphFunction.emit_call` uses: it emits the steps of a graph one after
# the other (`_emit_steps`). For each step it calls the step's function, which is
# inlined: either the node's generated code, or a call of the step's kernel. Where the
# steps' results go:
#
# * a result the step writes into memory it is given goes into memory a graph output
#   provides (`memory_targets`, e.g. a row of a Scan's buffer) if that memory has the
#   layout the step needs, else into a temporary (`_temporaries_of`: on the stack, or in
#   an arena shared by values that are never needed at the same time);
# * a result the step allocates itself goes straight into a graph output's slot if the
#   output takes it (`slot_targets`, `OutputDestination.slot_for_new_memory`), else
#   into memory that is released afterwards;
# * views and scalars are read back from the step's output slots;
# * 0-d constants are written into the code.


#: the cells of a profile (`_emit_steps`'s `profile`): profiling on (nonzero),
#: number of calls, then nanoseconds per step
PROFILE_ON, PROFILE_CALLS, PROFILE_ITEMS = 0, 1, 2


def _writes_into(memory: Array, output: Output) -> bool:
    """Can a step write its output `output`, passed in a caller buffer, straight into
    `memory`? If `memory`'s layout implies the one the step needs."""
    tensor = output.tensor
    return memory.layout.implies(tensor.layout, tensor.shape, memory.itemsize)


def _temporaries_of(
    cg: Codegen,
    steps: Sequence[Step],
    memory_targets: Mapping[ValueId, Array],
    kept: Collection[ValueId],
) -> tuple[dict[ValueId, ir.Value], tuple[dict[ValueId, int], int]]:
    """Memory for the temporary values of the steps `steps` (`Codegen.temporaries`).

    A temporary is a result a step writes into memory it is given, when it can't be
    memory a graph output provides (`memory_targets`, value id -> Array, see
    `_writes_into`). `kept`: the values still needed after the last step (the outputs).
    Returns what `Codegen.temporaries` returns."""
    sizes: dict[ValueId, int] = {}
    for step in steps:
        for value_id, output in zip(step.outputs, step.signature.outputs, strict=True):
            if output.passing is not Passing.CALLER_BUFFER:
                continue
            tensor = output.tensor
            if not tensor.static:
                continue
            target = memory_targets.get(value_id)
            if target is not None and _writes_into(target, output):
                continue
            elements = int(np.prod(tensor.fixed_shape, dtype=np.int64))
            sizes[value_id] = elements * np.dtype(tensor.scalar_type.dtype).itemsize
    return cg.temporaries(sizes, temporary_lives(steps, sizes, kept))


@dataclass
class _SlotRead:
    """A result to read back from a step's output slot after the step ran."""

    value_id: ValueId
    slot: Slot
    output: Output
    #: it is scratch memory of the generated node (freed with the surrounding scope)
    in_scratch: bool = False


def _bind_constants(
    cg: Codegen,
    constants: Mapping[ValueId, tuple[TensorType | ScalarType, Any]],
    values: dict[ValueId, Array | ir.Value],
) -> dict[ValueId, int]:
    """Values of `constants` ({value id: (type, ndarray)}): 0-d ones folded into the
    code, arrays from the links, in their memory order (`Layout.of_const`; equal ones
    share their memory). Returns {value id: link entry} of the arrays."""
    b = cg.b
    entries: dict[ValueId, int] = {}
    for value_id, (t, data) in constants.items():
        representation = repr_of(t)
        data = np.asarray(data, dtype=t.dtype)
        scalar_type = element_type(t)
        if isinstance(representation, ScalarRepr) or data.ndim == 0:
            value = constant(b, scalar_type, data.item())
            if isinstance(representation, ScalarRepr):
                values[value_id] = value
            else:
                slot = b.alloca(memory_type(scalar_type))
                store(b, scalar_type, value, slot)
                values[value_id] = Array.c(scalar_type, slot, [], [])
        else:
            from pytensor.link.llvm.codegen.call import linked

            layout = Layout.of_const(data)
            order = list(layout.order)
            memory = np.ascontiguousarray(data.transpose(order))
            arr = memory.transpose(np.argsort(order))
            entries[value_id] = cg.links.link_array(memory)
            values[value_id] = Array(
                scalar_type,
                linked(cg, entries[value_id]),
                [b.i64(n) for n in arr.shape],
                [b.i64(st) for st in arr.strides],
                layout,
            )
    return entries


def _emit_steps(
    cg: Codegen,
    name: str,
    steps: Sequence[Step],
    values: dict[ValueId, Array | ir.Value],
    memory_targets: Mapping[ValueId, Array] | None = None,
    slot_targets: Mapping[ValueId, OutputDestination] | None = None,
    memory: Mapping[ValueId, ir.Value] | None = None,
    profile: int | None = None,
) -> tuple[set[ValueId], list[Slot]]:
    """Emit the steps `steps` (`planner.Step`, in order) into the function
    being built in `cg`: each node's function is called (and inlined), or its kernel.

    * `values`: value id -> Array (tensors) or value (scalars); must hold every value
      the steps read but don't produce, with the layouts they were planned for; filled
      with what they produce;
    * `memory_targets`: value id -> Array: memory a value written into given memory may
      be produced in directly, if its layout implies the one the step needs
      (`_writes_into`);
    * `slot_targets`: value id -> `OutputDestination`: whose slot a value a step
      allocates may be allocated into, if the destination takes it
      (`slot_for_new_memory`);
    * `memory`: {value id: pointer}: where the other values written into given memory
      go (`_temporaries_of`); others get temporary memory of their own;
    * `profile`: the link entry of the profile cells (`PROFILE_*`): while profiling is
      on, each step's wall time is added to its cell.

    Returns (written, owned): the value ids produced directly in their targets, and
    the slots of buffers kernels allocated for other values, for the caller to
    `release` when they are no longer used (on failure, they are released here)."""
    from pytensor.link.llvm.codegen.call import linked

    b = cg.b
    ctx, config = cg.ctx, cg.config
    memory_targets = memory_targets or {}
    slot_targets = slot_targets or {}
    written: set[ValueId] = set()
    owned: list[Slot] = []  # whose owner is a runtime buffer
    layouts: dict[ValueId, Layout] = {}  # of what the steps produced
    if profile is not None:
        cells = linked(cg, profile)
        profiling = b.ne(b.load(ir.i64, cells), b.i64(0))
        with b.if_(profiling):
            b.rt_counter_add(b.gep(ir.i64, cells, PROFILE_CALLS), b.i64(1))
        start = b.alloca(ir.i64)
    for n, step in enumerate(steps):
        signature = step.signature
        results = list(zip(step.outputs, signature.outputs, strict=True))
        for value_id, output in results:
            if output.layout is not None:
                layouts[value_id] = output.layout

        # Where each result goes, decided before the node's function is built (it
        # allocates differently into scratch memory): given memory of a graph output,
        # a graph output's slot, or neither.
        into_memory: dict[int, Array] = {}
        into_slot: dict[int, Slot] = {}
        for j, (value_id, output) in enumerate(results):
            if value_id in written:
                continue
            if output.passing is Passing.CALLER_BUFFER:
                target = memory_targets.get(value_id)
                if target is not None and _writes_into(target, output):
                    into_memory[j] = target
                    written.add(value_id)
            elif output.passing is Passing.CALLEE_SLOT and not output.aliased_inputs:
                destination = slot_targets.get(value_id)
                if destination is not None:
                    slot = destination.slot_for_new_memory(output.tensor)
                    if slot is not None:
                        into_slot[j] = slot
                        written.add(value_id)
        # results the node allocates that go nowhere else: its scratch memory
        scratch = {
            j
            for j, (_, output) in enumerate(results)
            if output.passing is Passing.CALLEE_SLOT and j not in into_slot
        }

        function = node_function(
            cg.unit,
            f"{name}_n{n}",
            step.node,
            step.implementation,
            signature,
            scratch_outputs=frozenset(scratch),
            parallel=cg.parallel,
        )
        args: list[ir.Value] = [ctx, config]
        for value_id, representation in zip(
            step.arguments, signature.inputs, strict=True
        ):
            if value_id not in values:
                raise ValueError(
                    f"{name}: {step.description} reads a value that isn't computed yet"
                )
            value = values[value_id]
            if not isinstance(value, Array):
                args.append(value)
                continue
            if not isinstance(representation, TensorRepr):
                raise TypeError(f"{step.description}: an array for {representation}")
            if not value.layout.implies(
                representation.layout, representation.shape, value.itemsize
            ):
                raise AssertionError(
                    f"{name}: {step.description} was planned for an input with layout "
                    f"{representation.layout}, got {value.layout}"
                )
            args.append(value.pack(b, representation))

        scalar_reads: list[tuple[ValueId, ir.Value, ir.Type]] = []
        slot_reads: list[_SlotRead] = []
        for j, (value_id, output) in enumerate(results):
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
                    if memory is not None and value_id in memory:
                        lengths = [b.i64(m) for m in tensor.fixed_shape]
                        array = Array.packed(
                            b, tensor.scalar_type, memory[value_id], lengths, layout
                        )
                    else:
                        array = cg.temporary(
                            tensor.scalar_type, list(tensor.fixed_shape), layout
                        )
                values[value_id] = array
                args.append(array.pack(b, tensor))
            elif isinstance(output.representation, ScalarRepr):
                ir_type = output.representation.ir_type
                pointer = b.alloca(ir_type)
                args.append(pointer)
                scalar_reads.append((value_id, pointer, ir_type))
            else:
                ndim = output.tensor.ndim
                # (a result in scratch memory, or a view, has no runtime buffer)
                in_scratch = j in function.in_scratch
                if j in into_slot:
                    slot = into_slot[j]
                else:
                    slot = Slot.alloca(b, ndim)
                    if not in_scratch:
                        owned.append(slot)
                args.append(slot.pointer)
                slot_reads.append(_SlotRead(value_id, slot, output, in_scratch))

        if profile is not None:
            with b.if_(profiling):
                b.store(b.rt_clock_ns(), start)
        rc = b.call(function.function, args)
        assert rc is not None  # (it returns an i32)
        if profile is not None:
            with b.if_(profiling):
                b.rt_counter_add(
                    b.gep(ir.i64, cells, PROFILE_ITEMS + n),
                    b.sub(b.rt_clock_ns(), b.load(ir.i64, start)),
                )
        failed = b.ne(rc, 0)
        with b.if_(failed):
            b.rt_error_context(ctx, " ".join(step.description.split())[:200])
            for slot in owned:
                slot.release(b, ctx)
            b.ret(rc)

        for value_id, pointer, ir_type in scalar_reads:
            values[value_id] = b.load(ir_type, pointer)
        for read in slot_reads:
            tensor = read.output.tensor
            array = read.slot.read(
                b, tensor.scalar_type, list(tensor.shape), layouts.get(read.value_id)
            )
            values[read.value_id] = array
            if read.output.aliased_inputs:
                sources = [
                    values[step.arguments[a]]
                    for a in signature.alias_arguments(read.output)
                ]
                _maybe_view(b, array, sources, read.slot)
            if read.in_scratch:
                # (`SlotDestination.in_scratch_memory`: the owner is the scratch)
                cg.free_later(read.slot.owner(b))
    return written, owned


def _maybe_view(
    b: ir.Builder, a: Array, sources: Sequence[Array | ir.Value], slot: Slot
) -> None:
    """Remember that `a` may be a view of an input of the graph.

    `a` was returned through `slot` by a node that may return a view of `sources`
    (some of its inputs). If all of `sources` lie in the memory of the same graph input,
    `a` may too: `a.base` is set to that input. Whether it really is a view is only
    known at run time: when the slot has no owner (a view has none, new memory does).
    `a.base_if` is set to that condition."""
    bases = {s.base if isinstance(s, Array) else None for s in sources}
    if len(bases) != 1 or None in bases:
        return
    really = b.eq(slot.owner(b), b.null())
    for s in sources:
        if isinstance(s, Array) and s.base_if is not None:
            really = b.and_(really, s.base_if)
    a.base, a.writable, a.base_if = bases.pop(), False, really


__all__ = [
    "PROFILE_CALLS",
    "PROFILE_ITEMS",
    "PROFILE_ON",
    "GraphFunction",
    "build_graph_function",
]
