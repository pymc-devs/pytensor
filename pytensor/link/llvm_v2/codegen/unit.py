"""Compiled units: several nodes generated as one function.

A unit has the signature of a node (DESIGN.md, section 2): its inputs and outputs are the
values that cross its boundary. Inside, every node's function is called (and inlined):

* provided outputs of inner nodes are temporaries in the unit's workspace, or the
  unit's own provided output memory when the value is a unit output;
* allocated outputs of inner nodes go into the workspace too, except for unit outputs,
  which the node allocates straight into the unit's output slot;
* views and scalars are read back from their slots;
* 0-d constants are folded into the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.link.llvm_v2.codegen.emit import (
    c_array,
    Array,
    EmitContext,
    ProvidedOutput,
    SlotOutput,
    constant,
    copy,
    pack,
    read_slot,
    store,
    unpack,
)
from pytensor.link.llvm_v2.codegen.function import (
    NodeFunction,
    OutMode,
    Output,
    Signature,
    build_node_function,
    node_outputs,
    node_signature,
)
from pytensor.link.llvm_v2.layout import Layout
from pytensor.link.llvm_v2.codegen.types import ScalarRepr, TensorRepr, ir, is_absent, repr_of, slot_type


@dataclass
class Item:
    """A node with the keys of the values it reads and writes (keys are the planner's
    names for values; the same PyTensor node can appear with different keys)."""

    node: object
    inputs: list
    outputs: list


def build_unit_function(
    module, name, inputs, outputs, items, constants, input_layouts=None, output_layouts=None
) -> NodeFunction:
    """A function computing `outputs` from `inputs` by running `items` in order.

    * `inputs`: [(key, PyTensor type)] -- tensors only;
    * `input_layouts`: a `Layout` per input (None: nothing known); the layouts of the
      values inside follow from how they are produced (`llvm_outputs`);
    * `output_layouts`: the layout of the memory provided for each PROVIDED output
      (None: any strides);
    * `outputs`: [(key, PyTensor type, OutMode)] with modes PROVIDED or ALLOCATED;
    * `constants`: {key: (type, ndarray)}, 0-d only (folded).
    """
    m = ir()
    if input_layouts is None:
        input_layouts = [None] * len(inputs)
    in_reprs = [repr_of(t, lay) for (_, t), lay in zip(inputs, input_layouts, strict=True)]
    lays = {key: r.layout for (key, _), r in zip(inputs, in_reprs, strict=True)}
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
    ctx, env, ws = params[:3]
    cx = EmitContext(b, ctx, env, ws)

    vals = {}  # key -> Array (tensors) or value (scalars)
    k = 3
    for (key, _), r in zip(inputs, in_reprs, strict=True):
        vals[key] = unpack(b, r, params[k])
        k += 1
    unit_out = {}  # key -> (index, ProvidedOutput | slot pointer), first occurrence
    unit_out_list = []
    for idx, ((key, _, mode), o) in enumerate(zip(outputs, outs, strict=True)):
        v = params[k]
        k += 1
        target = ProvidedOutput(unpack(b, o.repr, v)) if mode is OutMode.PROVIDED else v
        unit_out_list.append(target)
        unit_out.setdefault(key, (idx, target))
    for key, (t, data) in constants.items():
        r = repr_of(t)
        value = constant(b, t.dtype, np.asarray(data).item())
        if isinstance(r, ScalarRepr):
            vals[key] = value
        else:
            slot = b.alloca(_mem_type(t.dtype))
            store(b, t.dtype, value, slot)
            vals[key] = c_array(t.dtype, slot, [], [])

    written = set()  # unit outputs whose memory was written in place
    for n, item in enumerate(items):
        node = item.node
        in_lays = [lays.get(key) for key in item.inputs]
        declared = node_outputs(node, in_lays)
        # provided outputs: the unit's own output memory (with its layout) the first
        # time a unit output is produced, else a temporary in the layout asked for
        in_place, provided = {}, []
        for j, (key, o) in enumerate(zip(item.outputs, declared, strict=True)):
            lay = None
            if o.mode is OutMode.PROVIDED:
                target = unit_out.get(key)
                if target is not None and isinstance(target[1], ProvidedOutput) and key not in written:
                    in_place[j] = target[1].array
                    lay = outs[target[0]].repr.layout
                else:
                    lay = o.layout if o.layout.fully_packed(o.repr.shape) else Layout.c(o.repr.ndim)
            provided.append(lay)
        nsig = node_signature(node, frozenset(), in_lays, provided)
        for key, o in zip(item.outputs, nsig.outputs, strict=True):
            if o.layout is not None:
                lays[key] = o.layout

        def into_unit_slot(key, o):
            # an allocated unit output (C-contiguous) can be allocated straight into the
            # unit's slot if the node allocates it C-contiguous
            return (
                o.mode is OutMode.ALLOCATED
                and _unit_alloc_target(unit_out, key) is not None
                and key not in written
                and o.layout.is_c(o.repr.shape)
            )

        temporaries = {
            j
            for j, (key, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if (o.mode is OutMode.ALLOCATED and not into_unit_slot(key, o)) or o.mode is OutMode.VIEW
        }
        nf = build_node_function(
            module,
            node,
            f"{name}_n{n}",
            temporary_outputs=frozenset(temporaries),
            in_layouts=in_lays,
            provided=provided,
        )
        args = [ctx, env, _ws_part(cx, nf.workspace)]
        present = [key for key, var in zip(item.inputs, node.inputs, strict=True) if not is_absent(var.type)]
        for key, r in zip(present, nsig.inputs, strict=True):
            if key not in vals:
                raise ValueError(f"unit {name}: {node.op} reads a value that isn't computed yet")
            v = vals[key]
            args.append(v if isinstance(r, ScalarRepr) else pack(b, r, v))
        reads = []  # (key, slot, how)
        for j, (key, var, o) in enumerate(zip(item.outputs, node.outputs, nsig.outputs, strict=True)):
            if o.mode is OutMode.PROVIDED:
                if j in in_place:
                    a = in_place[j]
                    written.add(key)
                else:
                    a = cx.temporary(var.type.dtype, [b.const_int(m.i64, s) for s in var.type.shape], o.layout)
                vals[key] = a
                args.append(pack(b, o.repr, a))
            elif o.mode is OutMode.SCALAR:
                slot = b.alloca(o.repr.ir_type)
                args.append(slot)
                reads.append((key, slot, "scalar", o))
            else:
                if into_unit_slot(key, o):
                    slot = _unit_alloc_target(unit_out, key)
                    written.add(key)
                else:
                    slot = b.alloca(slot_type(o.repr.ndim))
                    b.store(b.zero(slot_type(o.repr.ndim)), slot)
                args.append(slot)
                reads.append((key, slot, "array", o))
        rc = b.call(nf.function, args)
        failed = b.icmp(m.ICmp.NE, rc, b.const_int(m.i32, 0))
        with b.if_(failed):
            b.ret(rc)
        for key, slot, how, o in reads:
            if how == "scalar":
                vals[key] = b.load(o.repr.ir_type, slot)
            else:
                var = node.outputs[item.outputs.index(key)]
                vals[key] = read_slot(b, slot, var.type.dtype, list(var.type.shape), lays.get(key))

    # unit outputs that weren't produced in place: copy
    for idx, ((key, t, mode), target) in enumerate(zip(outputs, unit_out_list, strict=True)):
        first = unit_out[key][0] == idx
        if first and key in written:
            continue
        src = vals[key]
        if mode is OutMode.PROVIDED:
            copy(cx, src, target.array)
        else:
            out = SlotOutput(cx, target, t.dtype, t.ndim)
            copy(cx, src, out.allocate(src.shape))
    b.ret(b.const_int(m.i32, 0))
    return NodeFunction(fn, sig, cx.ws_static)


def _unit_alloc_target(unit_out, key):
    t = unit_out.get(key)
    if t is None or isinstance(t[1], ProvidedOutput):
        return None
    return t[1]


def _ws_part(cx, nbytes):
    """A region of the unit's static workspace for one inner node."""
    m = ir()
    if nbytes == 0:
        return cx.b.null()
    off = cx.ws_static
    cx.ws_static += -(-nbytes // 64) * 64
    return cx.b.gep_bytes(cx.ws, cx.b.const_int(m.i64, off))


def _mem_type(dtype):
    from pytensor.link.llvm_v2.codegen.types import memory_type

    return memory_type(dtype)


__all__ = ["Item", "TensorRepr", "build_unit_function"]
