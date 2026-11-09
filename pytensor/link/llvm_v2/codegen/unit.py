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
from pytensor.link.llvm_v2.codegen.types import ScalarRepr, TensorRepr, ir, is_absent, itemsize, repr_of, slot_type


@dataclass
class Item:
    """A node with the keys of the values it reads and writes (keys are the planner's
    names for values; the same PyTensor node can appear with different keys)."""

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
    callee_for=None,
    closure=None,
    planner=None,
) -> NodeFunction:
    """A function computing `outputs` from `inputs` by running `items` in order.

    * `inputs`: [(key, PyTensor type)] -- tensors only;
    * `input_layouts`: a `Layout` per input (None: nothing known); the layouts of the
      values inside follow from how they are produced (`llvm_outputs`);
    * `output_layouts`: the layout of the memory provided for each PROVIDED output
      (None: any strides);
    * `outputs`: [(key, PyTensor type, OutMode)] with modes PROVIDED or ALLOCATED;
    * `constants`: {key: (type, ndarray)} (0-d ones folded, arrays from the closure);
    * `callee_for(node, in_layouts)`: a `Callee` for a node the unit calls instead of
      generating (a kernel, numba, ...), or None; its handle goes into `closure` (the
      returned function's `closure`: call it with that env). Buffers callees allocate
      for values inside the unit are released when it returns.
    """
    from pytensor.link.llvm_v2.codegen.call import Closure

    if closure is None:
        closure = Closure()
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
    cx = EmitContext(b, ctx, env, ws, parallel=parallel, closure=closure, planner=planner, module=module, name=name)
    vals = {}  # key -> Array (tensors) or value (scalars)
    k = 3
    for (key, _), r in zip(inputs, in_reprs, strict=True):
        vals[key] = unpack(b, r, params[k])
        k += 1
    targets = {}  # key -> Target, first occurrence
    out_targets = []
    for (key, _, mode), o in zip(outputs, outs, strict=True):
        v = params[k]
        k += 1
        if mode is OutMode.PROVIDED:
            t = Target(array=unpack(b, o.repr, v), layout=o.repr.layout)
        else:
            t = Target(slot=v)
        out_targets.append(t)
        targets.setdefault(key, t)
    bind_constants(cx, constants, vals)

    written, owned = emit_items(cx, module, name, items, vals, lays, targets, callee_for)

    # unit outputs that weren't produced in place: copy
    for (key, t, mode), target in zip(outputs, out_targets, strict=True):
        if key in written and targets[key] is target:
            continue
        src = vals[key]
        if mode is OutMode.PROVIDED:
            copy(cx, src, target.array)
        else:
            out = SlotOutput(cx, target.slot, t.dtype, t.ndim)
            copy(cx, src, out.allocate(src.shape))
    release(cx, owned)
    b.ret(b.const_int(m.i32, 0))
    return NodeFunction(fn, sig, cx.ws_static, closure)


@dataclass
class Target:
    """Where a value may be produced directly: memory (`array`, in `layout`), or an
    output slot (`slot`) for a value allocated C-contiguous."""

    array: Array | None = None
    layout: Layout | None = None
    slot: object = None


def release(cx, owned):
    """Give up the buffers callees allocated into `owned` slots ([(slot, ndim)])."""
    m = ir()
    for slot, nd in owned:
        cx.b.rt_decref(cx.ctx, cx.b.load(m.ptr, cx.b.field_ptr(slot_type(nd), slot, 3)))


def bind_constants(cx, constants, vals):
    """Values of `constants` ({key: (type, ndarray)}): 0-d ones folded into the code,
    arrays from the closure (C-contiguous copies kept alive with it)."""
    b, m = cx.b, ir()
    for key, (t, data) in constants.items():
        r = repr_of(t)
        data = np.asarray(data, dtype=t.dtype)
        if isinstance(r, ScalarRepr) or data.ndim == 0:
            value = constant(b, t.dtype, data.item())
            if isinstance(r, ScalarRepr):
                vals[key] = value
            else:
                slot = b.alloca(_mem_type(t.dtype))
                store(b, t.dtype, value, slot)
                vals[key] = c_array(t.dtype, slot, [], [])
        else:
            if cx.closure is None:
                raise NotImplementedError("array constants need a closure")
            from pytensor.link.llvm_v2.codegen.call import load_slot

            arr = np.ascontiguousarray(data)
            ptr = load_slot(cx, cx.closure.add_data(arr))
            shape = [b.const_int(m.i64, n) for n in arr.shape]
            strides = [b.const_int(m.i64, st) for st in arr.strides]
            vals[key] = c_array(t.dtype, ptr, shape, strides)


def emit_items(cx, module, name, items, vals, lays, targets=None, callee_for=None, on_fail=None):
    """Emit `items` (in order) into the function being built in `cx`: each node's
    function is called (and inlined), or its callee.

    * `vals`: key -> Array (tensors) or value (scalars); must hold every value the
      items read but don't produce; filled with what they produce;
    * `lays`: key -> Layout of the values (filled too);
    * `targets`: key -> `Target`: where a value may be produced directly (PROVIDED
      outputs whose declared layout the target's implies; ALLOCATED C-contiguous
      ones into a slot); other values go into the workspace or callee buffers;
    * `on_fail()`: emits the caller's cleanup before an error is returned.

    Returns (written, owned): the keys produced in their targets, and the slots of
    buffers callees allocated for other values, for the caller to `release` when they
    are no longer used (on failure, they are released here)."""
    b, m = cx.b, ir()
    ctx, env = cx.ctx, cx.env
    targets = targets or {}
    written = set()
    owned = []  # (slot, ndim): buffers callees allocated
    for n, item in enumerate(items):
        node = item.node
        in_lays = [lays.get(key) for key in item.inputs]
        callee = callee_for(node, in_lays) if callee_for is not None else None
        nsig = node_signature(node, frozenset(), in_lays, declared=None if callee is None else callee.outputs)
        # provided outputs: their target if its layout implies the declared one; else a
        # temporary in the declared layout (copied by the caller if needed)
        in_place = {}
        for j, (key, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True)):
            t = targets.get(key)
            if (
                o.mode is OutMode.PROVIDED
                and t is not None
                and t.array is not None
                and key not in written
                and t.layout is not None
                and t.layout.implies(o.layout, o.repr.shape, itemsize(o.repr.dtype))
            ):
                in_place[j] = t.array
        for key, o in zip(item.outputs, nsig.outputs, strict=True):
            if o.layout is not None:
                lays[key] = o.layout

        def into_slot(key, o):
            # an allocated value with a slot target: allocated straight into it if the
            # node allocates it C-contiguous
            t = targets.get(key)
            return (
                o.mode is OutMode.ALLOCATED
                and t is not None
                and t.slot is not None
                and key not in written
                and o.layout.is_c(o.repr.shape)
            )

        temporaries = {
            j
            for j, (key, o) in enumerate(zip(item.outputs, nsig.outputs, strict=True))
            if (o.mode is OutMode.ALLOCATED and not into_slot(key, o)) or o.mode is OutMode.VIEW
        }
        nf = build_node_function(
            module,
            node,
            f"{name}_n{n}",
            temporary_outputs=frozenset(temporaries),
            in_layouts=in_lays,
            parallel=cx.parallel,
            callee=callee,
            closure=cx.closure,
            planner=cx.planner,
        )
        args = [ctx, env, _ws_part(cx, nf.workspace)]
        present = [key for key, var in zip(item.inputs, node.inputs, strict=True) if not is_absent(var.type)]
        for key, r in zip(present, nsig.inputs, strict=True):
            if key not in vals:
                raise ValueError(f"{name}: {node.op} reads a value that isn't computed yet")
            v = vals[key]
            args.append(v if isinstance(r, ScalarRepr) else pack(b, r, v))
        reads = []  # (key, slot, how)
        for j, (key, var, o) in enumerate(zip(item.outputs, node.outputs, nsig.outputs, strict=True)):
            if o.mode is OutMode.ABSENT:
                continue
            if o.mode is OutMode.PROVIDED:
                if j in in_place:
                    a = in_place[j]
                    written.add(key)
                else:
                    lay = o.layout if o.layout.fully_packed(o.repr.shape) else None  # (C implies any)
                    a = cx.temporary(var.type.dtype, [b.const_int(m.i64, s) for s in var.type.shape], lay)
                vals[key] = a
                args.append(pack(b, o.repr, a))
            elif o.mode is OutMode.SCALAR:
                slot = b.alloca(o.repr.ir_type)
                args.append(slot)
                reads.append((key, slot, "scalar", o))
            else:
                if into_slot(key, o):
                    slot = targets[key].slot
                    written.add(key)
                else:
                    slot = b.alloca(slot_type(o.repr.ndim))
                    b.store(b.zero(slot_type(o.repr.ndim)), slot)
                    if callee is not None:
                        owned.append((slot, o.repr.ndim))
                args.append(slot)
                reads.append((key, slot, "array", o))
        rc = b.call(nf.function, args)
        failed = b.icmp(m.ICmp.NE, rc, b.const_int(m.i32, 0))
        with b.if_(failed):
            release(cx, owned)
            if on_fail is not None:
                on_fail()
            b.ret(rc)
        for key, slot, how, o in reads:
            if how == "scalar":
                vals[key] = b.load(o.repr.ir_type, slot)
            else:
                var = node.outputs[item.outputs.index(key)]
                vals[key] = read_slot(b, slot, var.type.dtype, list(var.type.shape), lays.get(key))
    return written, owned


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


__all__ = ["Item", "Target", "TensorRepr", "bind_constants", "build_unit_function", "emit_items", "release"]
