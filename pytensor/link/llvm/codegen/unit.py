"""Compiled units: several nodes generated as one function.

A unit has the signature of a node: its inputs and outputs are the values that cross
its boundary. Inside, every node's function is called (and inlined):

* provided outputs of inner nodes are temporaries in the unit's temporary memory, or the
  unit's own provided output memory when the value is a unit output;
* allocated outputs of inner nodes go into temporary memory too, except for unit outputs,
  which the node allocates straight into the unit's output slot;
* views and scalars are read back from their slots;
* 0-d constants are folded into the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.link.llvm.codegen.emit import (
    Array,
    Codegen,
    SlotOutput,
    c_array,
    constant,
    copy,
    pack,
    read_slot,
    store,
    unpack,
)
from pytensor.link.llvm.codegen.function import (
    NodeFunction,
    Output,
    OutputMode,
    Signature,
    build_node_function,
    debug_name,
)
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    Slot,
    TensorRepr,
    ir,
    is_absent,
    itemsize,
    repr_of,
    slot_type,
)
from pytensor.link.llvm.layout import Layout


def build_unit_function(
    module,
    name,
    inputs,
    outputs,
    resolved,
    constants,
    input_layouts=None,
    output_layouts=None,
    output_aliases=None,
    parallel=True,
    links=None,
) -> NodeFunction:
    """A function computing `outputs` from `inputs` by running the items `resolved`.

    * `inputs`: [(vid, PyTensor type)] -- tensors only;
    * `input_layouts`: a `Layout` per input (None: nothing known), the ones the items
      were resolved for (`unit_input_layouts`);
    * `outputs`: [(vid, PyTensor type, OutputMode)] with modes PROVIDED or RETURNED;
    * `output_layouts`: the layout of the memory provided for each PROVIDED output
      (None: any strides), and of the result of a RETURNED one that may be a view;
    * `output_aliases`: per output, the positions of the inputs it may be a view of
      (RETURNED only): it is returned as a view when it lies in one of their memory,
      else as a copy in its layout;
    * `resolved`: the unit's items in order, resolved (`plan.resolve.Resolved`): each
      says whether its code is generated or which kernel it calls. A kernel's handle
      goes into `links` (the returned function's `links`: call it with that config).
      Buffers kernels allocate for values inside the unit are released when it returns;
    * `constants`: {vid: (type, ndarray)} (0-d ones folded, arrays from the link table).
    """
    from pytensor.link.llvm.codegen.call import LinkTable

    if links is None:
        links = LinkTable()
    if input_layouts is None:
        input_layouts = [None] * len(inputs)
    in_reprs = [
        repr_of(t, lay) for (_, t), lay in zip(inputs, input_layouts, strict=True)
    ]
    if output_layouts is None:
        output_layouts = [None] * len(outputs)
    if output_aliases is None:
        output_aliases = [()] * len(outputs)
    outs = [
        Output(repr_of(t, lay if mode is OutputMode.PROVIDED or al else None), mode, al)
        for (_, t, mode), lay, al in zip(
            outputs, output_layouts, output_aliases, strict=True
        )
    ]
    for r in [*in_reprs, *(o.repr for o in outs)]:
        if isinstance(r, ScalarRepr):
            raise NotImplementedError("ScalarType values can't cross a unit's boundary")
    for o in outs:
        if o.mode not in (OutputMode.PROVIDED, OutputMode.RETURNED):
            raise NotImplementedError(f"unit outputs can't be {o.mode}")
    sig = Signature(tuple(in_reprs), tuple(outs))
    nodes = ", ".join(debug_name(r.item.node) for r in resolved)
    fn = module.function(
        name,
        sig.param_types(),
        ir.i32,
        inline=ir.Inline.Always,
        debug_name=f"unit: {nodes}"[:300],
    )
    b = fn.builder()
    params = fn.params
    ctx, config = params[:2]
    cg = Codegen(
        b, ctx, config, parallel=parallel, links=links, module=module, name=name
    )
    vals = {}  # vid -> Array (tensors) or value (scalars)
    in_arrays = []
    k = 2
    for (vid, _), r in zip(inputs, in_reprs, strict=True):
        vals[vid] = unpack(b, r, params[k])
        in_arrays.append(vals[vid])
        k += 1
    targets = {}  # vid -> Destination, first occurrence
    out_targets = []
    for (vid, _, mode), o in zip(outputs, outs, strict=True):
        v = params[k]
        k += 1
        if mode is OutputMode.PROVIDED:
            t = Destination(array=unpack(b, o.repr, v), layout=o.repr.layout)
        else:
            t = Destination(slot=v)
        out_targets.append(t)
        targets.setdefault(vid, t)
    # (temporaries are freed before the function returns)
    with cg.scratch_scope(function=True):
        bind_constants(cg, constants, vals)

        written, owned = emit_items(cg, module, name, resolved, vals, targets)

        # unit outputs that weren't produced in place: views of the inputs where they are
        # (and may be), else copies
        for (vid, t, mode), target, o in zip(outputs, out_targets, outs, strict=True):
            if vid in written and targets[vid] is target:
                continue
            src = vals[vid]
            if mode is OutputMode.PROVIDED:
                copy(cg, src, target.array)
                continue
            out = SlotOutput(
                cg,
                target.slot,
                t.dtype,
                t.ndim,
                aliases=o.aliases,
                # a copy is in the declared layout if packed, else C-contiguous (which
                # implies it, see `program._view_layout`)
                layout=(
                    o.layout
                    if o.aliases and o.layout.fully_packed(o.repr.shape)
                    else None
                ),
            )
            if not o.aliases:
                copy(cg, src, out.allocate(src.shape))
                continue
            inside = b.const_bool(False)
            for a in o.aliases:
                inside = b.or_(inside, _lies_in(b, src, in_arrays[a]))

            def view(src=src, out=out):
                out.set_view(src)

            def copied(src=src, out=out):
                copy(cg, src, out.allocate(src.shape))

            b.if_else(inside, view, copied)
        release(cg, owned)
    b.ret(0)
    return NodeFunction(fn, sig, links)


def _bounds(b, a: Array):
    """(first, end) byte addresses of the memory `a` covers (end exclusive), as
    pointers; and whether it is empty."""
    lo, hi = b.i64(0), b.i64(itemsize(a.dtype))
    empty = b.const_bool(False)
    for n, st in zip(a.shape, a.strides, strict=True):
        ext = b.mul(b.sub(n, 1), st)
        lo = b.add(lo, b.smin(ext, 0))
        hi = b.add(hi, b.smax(ext, 0))
        empty = b.or_(empty, b.eq(n, 0))
    return b.gep_bytes(a.data, lo), b.gep_bytes(a.data, hi), empty


def _lies_in(b, a: Array, base: Array):
    """Whether the memory of `a` lies in the memory of `base` (i1)."""
    a_lo, a_hi, a_empty = _bounds(b, a)
    lo, hi, empty = _bounds(b, base)
    inside = b.and_(b.uge(a_lo, lo), b.ule(a_hi, hi))
    return b.and_(inside, b.not_(b.or_(a_empty, empty)))


@dataclass
class Destination:
    """Where a value may be produced directly: memory (`array`, in `layout`), or an
    output slot (`slot`) for a value allocated C-contiguous."""

    array: Array | None = None
    layout: Layout | None = None
    slot: object = None


def release(cg, owned):
    """Give up the buffers kernels allocated into `owned` slots ([(slot, ndim)])."""
    for slot, nd in owned:
        cg.b.rt_decref(
            cg.ctx, cg.b.load(ir.ptr, cg.b.field_ptr(slot_type(nd), slot, Slot.OWNER))
        )


def bind_constants(cg, constants, vals):
    """Values of `constants` ({vid: (type, ndarray)}): 0-d ones folded into the code,
    arrays from the links (C-contiguous copies kept alive with it)."""
    b = cg.b
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

            arr = np.ascontiguousarray(data)
            ptr = linked(cg, cg.links.link_array(arr))
            shape = [b.i64(n) for n in arr.shape]
            strides = [b.i64(st) for st in arr.strides]
            vals[vid] = c_array(t.dtype, ptr, shape, strides)


def unit_input_layouts(types, layouts) -> list:
    """The layouts a unit's inputs have inside it, given what is known outside: what
    their representations keep."""
    return [repr_of(t, lay).layout for t, lay in zip(types, layouts, strict=True)]


def emit_items(cg, module, name, resolved, vals, targets=None, on_fail=None):
    """Emit the items `resolved` (`plan.resolve.Resolved`, in order) into the function
    being built in `cg`: each node's function is called (and inlined), or its kernel.

    * `vals`: vid -> Array (tensors) or value (scalars); must hold every value the
      items read but don't produce, with the layouts they were resolved for; filled
      with what they produce;
    * `targets`: vid -> `Destination`: where a value may be produced directly (PROVIDED
      outputs whose declared layout the target's implies; returned new C-contiguous
      memory into a slot); other values go into temporary memory or kernel buffers;
    * `on_fail()`: emits the caller's cleanup before an error is returned.

    Returns (written, owned): the vids produced in their targets, and the slots of
    buffers kernels allocated for other values, for the caller to `release` when they
    are no longer used (on failure, they are released here)."""
    b = cg.b
    ctx, config = cg.ctx, cg.config
    targets = targets or {}
    written = set()
    owned = []  # (slot, ndim): buffers kernels allocated
    lays = {}  # vid -> Layout of what the items produced
    for n, r in enumerate(resolved):
        item, node, kernel, nsig, in_lays = (
            r.item,
            r.item.node,
            r.kernel,
            r.signature,
            r.in_layouts,
        )
        # provided outputs: their target if its layout implies the declared one; else a
        # temporary in the declared layout (copied by the caller if needed)
        in_place = {}
        for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True)):
            t = targets.get(vid)
            if (
                o.mode is OutputMode.PROVIDED
                and t is not None
                and t.array is not None
                and vid not in written
                and t.layout is not None
                and t.layout.implies(o.layout, o.repr.shape, itemsize(o.repr.dtype))
            ):
                in_place[j] = t.array
        for vid, o in zip(item.outputs, nsig.outputs, strict=True):
            if o.layout is not None:
                lays[vid] = o.layout

        def into_slot(vid, o):
            # an allocated value with a slot target: allocated straight into it if the
            # node allocates it C-contiguous
            t = targets.get(vid)
            return (
                o.mode is OutputMode.RETURNED
                and not o.aliases
                and t is not None
                and t.slot is not None
                and vid not in written
                and o.layout.is_c(o.repr.shape)
            )

        temporaries = {
            j
            for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if o.mode is OutputMode.RETURNED and not into_slot(vid, o)
        }
        nf = build_node_function(
            module,
            node,
            f"{name}_n{n}",
            temporary_outputs=frozenset(temporaries),
            in_layouts=in_lays,
            parallel=cg.parallel,
            kernel=kernel,
            links=cg.links,
            body=r.body,
        )
        args = [ctx, config]
        present = [
            vid
            for vid, var in zip(item.inputs, node.inputs, strict=True)
            if not is_absent(var.type)
        ]
        for vid, rep in zip(present, nsig.inputs, strict=True):
            if vid not in vals:
                raise ValueError(
                    f"{name}: {node.op} reads a value that isn't computed yet"
                )
            v = vals[vid]
            if isinstance(rep, ScalarRepr):
                args.append(v)
                continue
            if not v.layout.implies(rep.layout, rep.shape, itemsize(rep.dtype)):
                raise AssertionError(
                    f"{name}: {node.op} was resolved for an input with layout {rep.layout}, got {v.layout}"
                )
            args.append(pack(b, rep, v))
        reads = []  # (vid, slot, how)
        for j, (vid, var, o) in enumerate(
            zip(item.outputs, node.outputs, nsig.outputs, strict=True)
        ):
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
                    a = cg.temporary(var.type.dtype, list(var.type.shape), lay)
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
        rc = b.call(nf.function, args)
        failed = b.ne(rc, 0)
        with b.if_(failed):
            release(cg, owned)
            if on_fail is not None:
                on_fail()
            b.ret(rc)
        for vid, slot, how, o in reads:
            if how == "scalar":
                vals[vid] = b.load(o.repr.ir_type, slot)
            else:
                var = node.outputs[item.outputs.index(vid)]
                vals[vid] = read_slot(
                    b, slot, var.type.dtype, list(var.type.shape), lays.get(vid)
                )
                if how == "scratch":
                    # (`SlotOutput.temporary`: the owner is the scratch memory)
                    cg.free_later(
                        b.load(
                            ir.ptr,
                            b.field_ptr(slot_type(o.repr.ndim), slot, Slot.OWNER),
                        )
                    )
    return written, owned


def _mem_type(dtype):
    from pytensor.link.llvm.codegen.types import memory_type

    return memory_type(dtype)


__all__ = [
    "Destination",
    "TensorRepr",
    "bind_constants",
    "build_unit_function",
    "emit_items",
    "release",
    "unit_input_layouts",
]
