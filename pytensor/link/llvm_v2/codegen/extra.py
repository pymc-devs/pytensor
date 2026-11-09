"""CumOp, ExtractDiag, Split."""

from __future__ import annotations

import pytensor.scalar.basic as ps
from pytensor.link.llvm_v2.codegen.elemwise import output_array
from pytensor.link.llvm_v2.codegen.emit import Array, cast, constant, copy, load, mul, store
from pytensor.link.llvm_v2.codegen.function import llvm_emit, llvm_outputs, view_output
from pytensor.link.llvm_v2.layout import split_layouts
from pytensor.link.llvm_v2.codegen.scalar import emit_scalar
from pytensor.link.llvm_v2.codegen.types import ir
from pytensor.tensor.basic import ExtractDiag, Split
from pytensor.tensor.extra_ops import CumOp


def _set(out, cx, a: Array):
    """Make output `out` the array `a`: as a view if it is one, else by copying."""
    if hasattr(out, "set_view") and out.view_of is not None:
        out.set_view(a)
    else:
        copy(cx, a, output_array(out, a.shape))


@llvm_emit.register(CumOp)
def _(op, node, cx, inputs, outputs):
    b = cx.b
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
                v = cast(b, load(b, x.dtype, b.gep_bytes(px, mul(b, i, x.strides[axis]))), x.dtype, dt)
                new = emit_scalar(combine, b, [(acc, dt), (v, dt)], [dt])[0]
                store(b, dt, new, b.gep_bytes(po, mul(b, i, out.strides[axis])))
                return [new]

            b.fold(x.shape[axis], [init], step)
            return
        a = others[k]
        with b.loop(x.shape[a]) as i:
            level(k + 1, b.gep_bytes(px, mul(b, i, x.strides[a])), b.gep_bytes(po, mul(b, i, out.strides[a])))

    level(0, x.data, out.data)


@llvm_emit.register(ExtractDiag)
def _(op, node, cx, inputs, outputs):
    b, m = cx.b, ir()
    i64 = m.i64
    x = inputs[0]
    a1, a2 = op.axis1 % x.ndim, op.axis2 % x.ndim
    off = int(op.offset)
    n1, n2 = x.shape[a1], x.shape[a2]
    zero = b.const_int(i64, 0)
    # length = max(0, min(n1 - max(0, -off), n2 - max(0, off)))
    l1 = b.sub(n1, b.const_int(i64, max(0, -off)))
    l2 = b.sub(n2, b.const_int(i64, max(0, off)))
    length = b.select(b.icmp(m.ICmp.SLT, l1, l2), l1, l2)
    length = b.select(b.icmp(m.ICmp.SGT, length, zero), length, zero)
    start = mul(b, b.const_int(i64, off), x.strides[a2]) if off >= 0 else mul(b, b.const_int(i64, -off), x.strides[a1])
    start = b.select(b.icmp(m.ICmp.SGT, length, zero), start, zero)
    shape = [x.shape[a] for a in range(x.ndim) if a not in (a1, a2)] + [length]
    strides = [x.strides[a] for a in range(x.ndim) if a not in (a1, a2)] + [b.add(x.strides[a1], x.strides[a2])]
    _set(outputs[0], cx, Array(x.dtype, b.gep_bytes(x.data, start), shape, strides))


@llvm_outputs.register(Split)
def _(op, node, in_layouts):
    return [view_output(o, lay) for o, lay in zip(node.outputs, split_layouts(op, node, in_layouts[0]), strict=True)]


@llvm_emit.register(Split)
def _(op, node, cx, inputs, outputs):
    b, m = cx.b, ir()
    i64 = m.i64
    x, splits = inputs
    axis = op.axis
    offset = b.const_int(i64, 0)
    parts = []
    for k in range(op.len_splits):
        p = b.gep_bytes(splits.data, mul(b, b.const_int(i64, k), splits.strides[0]))
        n = cast(b, load(b, splits.dtype, p), splits.dtype, "int64")
        cx.check(b.icmp(m.ICmp.SGE, n, b.const_int(i64, 0)), "Split: negative split size")
        static = node.outputs[k].type.shape[axis]
        if static is not None:
            cx.check(b.icmp(m.ICmp.EQ, n, b.const_int(i64, static)), "Split: split size differs from the static shape")
            n = b.const_int(i64, static)
        parts.append((offset, n))
        offset = b.add(offset, n)
    cx.check(b.icmp(m.ICmp.EQ, offset, x.shape[axis]), "Split: the split sizes don't add up to the length of the axis")
    layouts = split_layouts(op, node, x.layout)
    for out, (start, n), lay in zip(outputs, parts, layouts, strict=True):
        shape = list(x.shape)
        shape[axis] = n
        # an empty part may start at the end: keep the pointer inside the array
        start = b.select(b.icmp(m.ICmp.SGT, n, b.const_int(i64, 0)), start, b.const_int(i64, 0))
        _set(out, cx, Array(x.dtype, b.gep_bytes(x.data, mul(b, start, x.strides[axis])), shape, list(x.strides), lay))
