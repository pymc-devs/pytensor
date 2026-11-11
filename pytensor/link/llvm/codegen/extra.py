"""CumOp, ExtractDiag, Split."""

from __future__ import annotations

import pytensor.scalar.basic as ps
from pytensor.link.llvm.codegen.elemwise import output_array
from pytensor.link.llvm.codegen.emit import advance, Array, cast, constant, copy, load, store
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_outputs, view_output
from pytensor.link.llvm.layout import split_layouts
from pytensor.link.llvm.codegen.scalar import emit_scalar
from pytensor.tensor.basic import ExtractDiag, Split
from pytensor.tensor.extra_ops import CumOp


def _set(out, cg, a: Array):
    """Make output `out` the array `a`: as a view if it is one, else by copying."""
    if hasattr(out, "set_view") and out.view_of is not None:
        out.set_view(a)
    else:
        copy(cg, a, output_array(out, a.shape))


@llvm_emit.register(CumOp)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    if x.ndim == 0:
        raise NotImplementedError("CumOp on a 0-d input")
    axis = op.axis % x.ndim
    out = output_array(outputs[0], x.shape)
    dt = out.dtype
    combine = ps.add if op.mode == "add" else ps.mul
    init = constant(b, dt, 0 if op.mode == "add" else 1)
    others = [a for a in range(x.ndim) if a != axis]

    def level(k, px, po):
        if k == len(others):

            def step(i, acc):
                v = cast(b, load(b, x.dtype, advance(b, px, i, x.strides[axis], x.dtype)), x.dtype, dt)
                new = emit_scalar(combine, b, [(acc, dt), (v, dt)], [dt])[0]
                store(b, dt, new, advance(b, po, i, out.strides[axis], out.dtype))
                return [new]

            b.fold(x.shape[axis], [init], step)
            return
        a = others[k]
        with b.loop(x.shape[a]) as i:
            level(k + 1, advance(b, px, i, x.strides[a], x.dtype), advance(b, po, i, out.strides[a], out.dtype))

    level(0, x.data, out.data)


@llvm_emit.register(ExtractDiag)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    a1, a2 = op.axis1 % x.ndim, op.axis2 % x.ndim
    off = int(op.offset)
    n1, n2 = x.shape[a1], x.shape[a2]
    zero = b.i64(0)
    # length = max(0, min(n1 - max(0, -off), n2 - max(0, off)))
    l1 = b.sub(n1, b.i64(max(0, -off)))
    l2 = b.sub(n2, b.i64(max(0, off)))
    length = b.select(b.slt(l1, l2), l1, l2)
    length = b.select(b.sgt(length, zero), length, zero)
    start = b.mul(b.i64(off), x.strides[a2], nsw=True) if off >= 0 else b.mul(b.i64(-off), x.strides[a1], nsw=True)
    start = b.select(b.sgt(length, zero), start, zero)
    shape = [x.shape[a] for a in range(x.ndim) if a not in (a1, a2)] + [length]
    strides = [x.strides[a] for a in range(x.ndim) if a not in (a1, a2)] + [b.add(x.strides[a1], x.strides[a2])]
    _set(outputs[0], cg, Array(x.dtype, b.gep_bytes(x.data, start), shape, strides))


@llvm_outputs.register(Split)
def _(op, node, in_layouts):
    return [view_output(o, lay) for o, lay in zip(node.outputs, split_layouts(op, node, in_layouts[0]), strict=True)]


@llvm_emit.register(Split)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x, splits = inputs
    axis = op.axis
    offset = b.i64(0)
    parts = []
    for k in range(op.len_splits):
        p = b.gep_bytes(splits.data, b.mul(b.i64(k), splits.strides[0], nsw=True))
        n = cast(b, load(b, splits.dtype, p), splits.dtype, "int64")
        cg.check(b.sge(n, b.i64(0)), "Split: negative split size")
        static = node.outputs[k].type.shape[axis]
        if static is not None:
            cg.check(b.eq(n, b.i64(static)), "Split: split size differs from the static shape")
            n = b.i64(static)
        parts.append((offset, n))
        offset = b.add(offset, n)
    cg.check(b.eq(offset, x.shape[axis]), "Split: the split sizes don't add up to the length of the axis")
    layouts = split_layouts(op, node, x.layout)
    for out, (start, n), lay in zip(outputs, parts, layouts, strict=True):
        shape = list(x.shape)
        shape[axis] = n
        # an empty part may start at the end: keep the pointer inside the array
        start = b.select(b.sgt(n, b.i64(0)), start, b.i64(0))
        _set(out, cg, Array(x.dtype, b.gep_bytes(x.data, b.mul(start, x.strides[axis], nsw=True)), shape, list(x.strides), lay))
