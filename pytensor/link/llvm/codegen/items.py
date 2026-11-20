"""Emitting a list of items into one function (`emit_items`).

`emit_items` is the machinery behind `GraphFunction.emit_call` (graph.py): it emits the
items of a graph one after the other. For each item it calls the item's function, which
is inlined: either the node's generated code, or a call of the item's kernel. Where the
items' results go:

* a result the item writes into memory it is given goes into memory the caller asked
  for (`targets`, e.g. a row of a Scan's buffer), else into a temporary
  (`temporaries_of`: on the stack, or in an arena shared by values that are never
  needed at the same time);
* a result the item allocates itself goes straight into an output slot of the caller
  if it asked for that (`targets`), else into memory that is released afterwards;
* views and scalars are read back from the item's output slots;
* 0-d constants are written into the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.link.llvm.codegen.emit import (
    Array,
    c_array,
    constant,
    pack,
    packed_array,
    read_slot,
    store,
)
from pytensor.link.llvm.codegen.node import build_node_function
from pytensor.link.llvm.codegen.signature import Output, OutputMode
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    Slot,
    ir,
    itemsize,
    repr_of,
    slot_type,
)
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.plan.memory import temporary_lives


#: the cells of a profile (`emit_items`'s `profile`): profiling on (nonzero),
#: number of calls, then nanoseconds per item
PROFILE_ON, PROFILE_CALLS, PROFILE_ITEMS = 0, 1, 2


@dataclass
class Destination:
    """Where a value may be produced directly: memory (`array`, in `layout`), or an
    output slot (`slot`) for a value allocated in new memory (in `layout`)."""

    array: Array | None = None
    layout: Layout | None = None
    slot: object = None


def fits(t: Destination | None, o: Output) -> bool:
    """Can an item produce its output `o` straight in `t`? Provided memory whose layout
    implies the declared one, or a slot for new memory in a layout implying the
    slot's."""
    if t is None or t.layout is None:
        return False
    shape, size = o.repr.shape, itemsize(o.repr.dtype)
    if o.mode is OutputMode.PROVIDED:
        return t.array is not None and t.layout.implies(o.layout, shape, size)
    return (
        o.mode is OutputMode.RETURNED
        and not o.aliases
        and t.slot is not None
        and o.layout.implies(t.layout, shape, size)
    )


def temporaries_of(cg, resolved, targets, kept) -> tuple[dict, tuple[dict, int]]:
    """Memory for the temporary values of the items `resolved` (`Codegen.temporaries`).

    A temporary is a result an item writes into memory it is given, when that memory
    isn't one the caller asked for (`targets`, vid -> `Destination`). `kept`: the values
    still needed after the last item (the outputs). Returns what `Codegen.temporaries`
    returns."""
    sizes = {
        vid: int(np.prod(o.repr.shape, dtype=np.int64)) * itemsize(o.repr.dtype)
        for r in resolved
        for vid, o in zip(r.item.outputs, r.signature.outputs, strict=True)
        if o.mode is OutputMode.PROVIDED
        and o.repr.static
        and not fits(targets.get(vid), o)
    }
    return cg.temporaries(sizes, temporary_lives(resolved, sizes, kept))


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


def emit_items(
    cg,
    module,
    name,
    resolved,
    vals,
    targets=None,
    memory=None,
    profile=None,
):
    """Emit the items `resolved` (`plan.resolve.Resolved`, in order) into the function
    being built in `cg`: each node's function is called (and inlined), or its kernel.

    * `vals`: vid -> Array (tensors) or value (scalars); must hold every value the
      items read but don't produce, with the layouts they were resolved for; filled
      with what they produce;
    * `targets`: vid -> `Destination`: where a value may be produced directly (PROVIDED
      outputs whose declared layout the target's implies; returned new C-contiguous
      memory into a slot); other values go into temporary memory or kernel buffers;
    * `memory`: {vid: pointer}: where the provided outputs that aren't produced in their
      targets go (`temporaries_of`); others get temporary memory of their own;
    * `profile`: the link entry of the profile cells (`PROFILE_*`): while profiling is
      on, each item's wall time is added to its cell.

    Returns (written, owned): the vids produced in their targets, and the slots of
    buffers kernels allocated for other values, for the caller to `release` when they
    are no longer used (on failure, they are released here)."""
    from pytensor.link.llvm.codegen.call import build_call_function, linked

    b = cg.b
    ctx, config = cg.ctx, cg.config
    targets = targets or {}
    written = set()
    owned = []  # (slot, ndim): buffers kernels allocated
    lays = {}  # vid -> Layout of what the items produced
    if profile is not None:
        cells = linked(cg, profile)
        profiling = b.ne(b.load(ir.i64, cells), b.i64(0))
        with b.if_(profiling):
            b.rt_counter_add(b.gep(ir.i64, cells, PROFILE_CALLS), b.i64(1))
        start = b.alloca(ir.i64)
    for n, r in enumerate(resolved):
        item, kernel, nsig, in_lays = r.item, r.kernel, r.signature, r.in_layouts
        node = getattr(item, "node", None)
        present = r.arguments
        # provided outputs: their target if its layout implies the declared one; else a
        # temporary in the declared layout (copied by the caller if needed)
        in_place = {
            j: targets[vid].array
            for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if o.mode is OutputMode.PROVIDED
            and vid not in written
            and fits(targets.get(vid), o)
        }
        for vid, o in zip(item.outputs, nsig.outputs, strict=True):
            if o.layout is not None:
                lays[vid] = o.layout

        def into_slot(vid, o):
            # an allocated value with a slot target: allocated straight into it if the
            # node allocates it in a layout the slot takes
            return (
                o.mode is OutputMode.RETURNED
                and vid not in written
                and fits(targets.get(vid), o)
            )

        temporaries = {
            j
            for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if o.mode is OutputMode.RETURNED and not into_slot(vid, o)
        }
        if node is None:
            nf = build_call_function(module, f"{name}_n{n}", nsig, kernel, cg.links)
        else:
            nf = build_node_function(
                module,
                node,
                f"{name}_n{n}",
                temporary_outputs=frozenset(temporaries),
                in_layouts=in_lays,
                parallel=cg.parallel,
                kernel=kernel,
                links=cg.links,
                resolver=cg.resolver,
            )
        args = [ctx, config]
        for vid, rep in zip(present, nsig.inputs, strict=True):
            if vid not in vals:
                raise ValueError(
                    f"{name}: {r.description} reads a value that isn't computed yet"
                )
            v = vals[vid]
            if isinstance(rep, ScalarRepr):
                args.append(v)
                continue
            if not v.layout.implies(rep.layout, rep.shape, itemsize(rep.dtype)):
                raise AssertionError(
                    f"{name}: {r.description} was resolved for an input with layout {rep.layout}, got {v.layout}"
                )
            args.append(pack(b, rep, v))
        reads = []  # (vid, slot, how)
        for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True)):
            if o.mode is OutputMode.ABSENT:
                continue
            if o.mode is OutputMode.PROVIDED:
                if j in in_place:
                    a = in_place[j]
                    written.add(vid)
                else:
                    lay = (
                        o.layout if o.layout.fully_packed(o.repr.shape) else None
                    )  # (C implies any)
                    shape = list(o.repr.shape)
                    if memory is not None and vid in memory:
                        dims = [b.i64(m) for m in shape]
                        a = packed_array(b, o.repr.dtype, memory[vid], dims, lay)
                    else:
                        a = cg.temporary(o.repr.dtype, shape, lay)
                vals[vid] = a
                args.append(pack(b, o.repr, a))
            elif o.mode is OutputMode.SCALAR:
                slot = b.alloca(o.repr.ir_type)
                args.append(slot)
                reads.append((vid, slot, "scalar", o))
            else:
                if into_slot(vid, o):
                    slot = targets[vid].slot
                    written.add(vid)
                else:
                    slot = b.alloca(slot_type(o.repr.ndim))
                    b.store(b.zero(slot_type(o.repr.ndim)), slot)
                    if kernel is not None:
                        owned.append((slot, o.repr.ndim))
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
            b.rt_error_context(ctx, " ".join(r.description.split())[:200])
            release(cg, owned)
            b.ret(rc)
        for vid, slot, how, o in reads:
            if how == "scalar":
                vals[vid] = b.load(o.repr.ir_type, slot)
            else:
                vals[vid] = read_slot(
                    b, slot, o.repr.dtype, list(o.repr.shape), lays.get(vid)
                )
                if o.aliases:
                    sources = [vals[present[a]] for a in o.aliases]
                    _maybe_view(b, vals[vid], sources, slot, o.repr.ndim)
                if how == "scratch":
                    # (`SlotOutput.temporary`: the owner is the scratch memory)
                    cg.free_later(
                        b.load(
                            ir.ptr,
                            b.field_ptr(slot_type(o.repr.ndim), slot, Slot.OWNER),
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
    "Destination",
    "bind_constants",
    "emit_items",
    "release",
]
