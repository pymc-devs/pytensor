"""Compiled units: several nodes generated as one function.

A unit has the signature of a node (DESIGN.md, section 2): its inputs and outputs are the
values that cross its boundary. Inside, every node's function is called (and inlined):

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
    c_array,
    Array,
    Codegen,
    ProvidedOutput,
    SlotOutput,
    constant,
    copy,
    pack,
    read_slot,
    store,
    unpack,
)
from pytensor.link.llvm.codegen.function import (
    NodeFunction,
    OutMode,
    Output,
    Signature,
    build_node_function,
    node_outputs,
    node_signature,
)
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.codegen.types import ScalarRepr, TensorRepr, ir, is_absent, itemsize, repr_of, slot_type


@dataclass
class NodeItem:
    """A node the planner schedules, with the value ids (vids: the planner's names for
    values) of what it reads and writes. The same PyTensor node can appear with
    different vids (inlined OpFromGraphs)."""

    node: object
    inputs: list
    outputs: list


def build_unit_function(
    module,
    name,
    inputs,
    outputs,
    items,
    constants,
    input_layouts=None,
    output_layouts=None,
    parallel=True,
    kernel_for=None,
    links=None,
    planner=None,
) -> NodeFunction:
    """A function computing `outputs` from `inputs` by running `items` in order.

    * `inputs`: [(vid, PyTensor type)] -- tensors only;
    * `input_layouts`: a `Layout` per input (None: nothing known); the layouts of the
      values inside follow from how they are produced (`llvm_outputs`);
    * `output_layouts`: the layout of the memory provided for each PROVIDED output
      (None: any strides);
    * `outputs`: [(vid, PyTensor type, OutMode)] with modes PROVIDED or ALLOCATED;
    * `constants`: {vid: (type, ndarray)} (0-d ones folded, arrays from the link table);
    * `kernel_for(node, in_layouts)`: a `Kernel` for a node the unit calls instead of
      generating (a kernel, numba, ...), or None; its handle goes into `links` (the
      returned function's `links`: call it with that config). Buffers kernels allocate
      for values inside the unit are released when it returns.
    """
    from pytensor.link.llvm.codegen.call import LinkTable

    if links is None:
        links = LinkTable()
    m = ir()
    if input_layouts is None:
        input_layouts = [None] * len(inputs)
    in_reprs = [repr_of(t, lay) for (_, t), lay in zip(inputs, input_layouts, strict=True)]
    lays = {vid: r.layout for (vid, _), r in zip(inputs, in_reprs, strict=True)}
    if output_layouts is None:
        output_layouts = [None] * len(outputs)
    outs = [Output(repr_of(t, lay if mode is OutMode.PROVIDED else None), mode) for (_, t, mode), lay in zip(outputs, output_layouts, strict=True)]
    for r in [*in_reprs, *(o.repr for o in outs)]:
        if isinstance(r, ScalarRepr):
            raise NotImplementedError("ScalarType values can't cross a unit's boundary")
    for o in outs:
        if o.mode not in (OutMode.PROVIDED, OutMode.ALLOCATED):
            raise NotImplementedError(f"unit outputs can't be {o.mode}")
    sig = Signature(tuple(in_reprs), tuple(outs))
    fn = module.function(name, sig.param_types(), m.i32, inline=m.Inline.Always)
    b = fn.builder()
    params = fn.params
    ctx, config, frame = params[:3]
    cg = Codegen(b, ctx, config, frame, parallel=parallel, links=links, planner=planner, module=module, name=name)
    vals = {}  # vid -> Array (tensors) or value (scalars)
    k = 3
    for (vid, _), r in zip(inputs, in_reprs, strict=True):
        vals[vid] = unpack(b, r, params[k])
        k += 1
    targets = {}  # vid -> Target, first occurrence
    out_targets = []
    for (vid, _, mode), o in zip(outputs, outs, strict=True):
        v = params[k]
        k += 1
        if mode is OutMode.PROVIDED:
            t = Target(array=unpack(b, o.repr, v), layout=o.repr.layout)
        else:
            t = Target(slot=v)
        out_targets.append(t)
        targets.setdefault(vid, t)
    bind_constants(cg, constants, vals)

    written, owned = emit_items(cg, module, name, items, vals, lays, targets, kernel_for)

    # unit outputs that weren't produced in place: copy
    for (vid, t, mode), target in zip(outputs, out_targets, strict=True):
        if vid in written and targets[vid] is target:
            continue
        src = vals[vid]
        if mode is OutMode.PROVIDED:
            copy(cg, src, target.array)
        else:
            out = SlotOutput(cg, target.slot, t.dtype, t.ndim)
            copy(cg, src, out.allocate(src.shape))
    release(cg, owned)
    b.ret(b.const_int(m.i32, 0))
    return NodeFunction(fn, sig, cg.frame_size, links)


@dataclass
class Target:
    """Where a value may be produced directly: memory (`array`, in `layout`), or an
    output slot (`slot`) for a value allocated C-contiguous."""

    array: Array | None = None
    layout: Layout | None = None
    slot: object = None


def release(cg, owned):
    """Give up the buffers kernels allocated into `owned` slots ([(slot, ndim)])."""
    m = ir()
    for slot, nd in owned:
        cg.b.rt_decref(cg.ctx, cg.b.load(m.ptr, cg.b.field_ptr(slot_type(nd), slot, 3)))


def bind_constants(cg, constants, vals):
    """Values of `constants` ({vid: (type, ndarray)}): 0-d ones folded into the code,
    arrays from the links (C-contiguous copies kept alive with it)."""
    b, m = cg.b, ir()
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
            shape = [b.const_int(m.i64, n) for n in arr.shape]
            strides = [b.const_int(m.i64, st) for st in arr.strides]
            vals[vid] = c_array(t.dtype, ptr, shape, strides)


def emit_items(cg, module, name, items, vals, lays, targets=None, kernel_for=None, on_fail=None):
    """Emit `items` (in order) into the function being built in `cg`: each node's
    function is called (and inlined), or its kernel.

    * `vals`: vid -> Array (tensors) or value (scalars); must hold every value the
      items read but don't produce; filled with what they produce;
    * `lays`: vid -> Layout of the values (filled too);
    * `targets`: vid -> `Target`: where a value may be produced directly (PROVIDED
      outputs whose declared layout the target's implies; ALLOCATED C-contiguous
      ones into a slot); other values go into temporary memory or kernel buffers;
    * `on_fail()`: emits the caller's cleanup before an error is returned.

    Returns (written, owned): the vids produced in their targets, and the slots of
    buffers kernels allocated for other values, for the caller to `release` when they
    are no longer used (on failure, they are released here)."""
    b, m = cg.b, ir()
    ctx, config = cg.ctx, cg.config
    targets = targets or {}
    written = set()
    owned = []  # (slot, ndim): buffers kernels allocated
    for n, item in enumerate(items):
        node = item.node
        in_lays = [lays.get(vid) for vid in item.inputs]
        kernel = kernel_for(node, in_lays) if kernel_for is not None else None
        nsig = node_signature(node, frozenset(), in_lays, declared=None if kernel is None else kernel.outputs)
        # provided outputs: their target if its layout implies the declared one; else a
        # temporary in the declared layout (copied by the caller if needed)
        in_place = {}
        for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True)):
            t = targets.get(vid)
            if (
                o.mode is OutMode.PROVIDED
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
                o.mode is OutMode.ALLOCATED
                and t is not None
                and t.slot is not None
                and vid not in written
                and o.layout.is_c(o.repr.shape)
            )

        temporaries = {
            j
            for j, (vid, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if (o.mode is OutMode.ALLOCATED and not into_slot(vid, o)) or o.mode is OutMode.VIEW
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
            planner=cg.planner,
        )
        args = [ctx, config, _frame_part(cg, nf.frame_size)]
        present = [vid for vid, var in zip(item.inputs, node.inputs, strict=True) if not is_absent(var.type)]
        for vid, r in zip(present, nsig.inputs, strict=True):
            if vid not in vals:
                raise ValueError(f"{name}: {node.op} reads a value that isn't computed yet")
            v = vals[vid]
            args.append(v if isinstance(r, ScalarRepr) else pack(b, r, v))
        reads = []  # (vid, slot, how)
        for j, (vid, var, o) in enumerate(zip(item.outputs, node.outputs, nsig.outputs, strict=True)):
            if o.mode is OutMode.ABSENT:
                continue
            if o.mode is OutMode.PROVIDED:
                if j in in_place:
                    a = in_place[j]
                    written.add(vid)
                else:
                    lay = o.layout if o.layout.fully_packed(o.repr.shape) else None  # (C implies any)
                    a = cg.temporary(var.type.dtype, [b.const_int(m.i64, s) for s in var.type.shape], lay)
                vals[vid] = a
                args.append(pack(b, o.repr, a))
            elif o.mode is OutMode.SCALAR:
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
                reads.append((vid, slot, "array", o))
        rc = b.call(nf.function, args)
        failed = b.icmp(m.ICmp.NE, rc, b.const_int(m.i32, 0))
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
                vals[vid] = read_slot(b, slot, var.type.dtype, list(var.type.shape), lays.get(vid))
    return written, owned


def _frame_part(cg, nbytes):
    """A part of the unit's frame for one inner node."""
    m = ir()
    if nbytes == 0:
        return cg.b.null()
    off = cg.frame_size
    cg.frame_size += -(-nbytes // 64) * 64
    return cg.b.gep_bytes(cg.frame, cg.b.const_int(m.i64, off))


def _mem_type(dtype):
    from pytensor.link.llvm.codegen.types import memory_type

    return memory_type(dtype)


__all__ = ["NodeItem", "Target", "TensorRepr", "bind_constants", "build_unit_function", "emit_items", "release"]
