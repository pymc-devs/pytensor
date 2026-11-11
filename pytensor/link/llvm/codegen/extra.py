"""CumOp, ExtractDiag, Split."""

from __future__ import annotations

import operator

from pytensor.link.llvm.codegen.emit import Array, cast, constant, load, set_output
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.num import Num
from pytensor.link.llvm.layout import split_layouts
from pytensor.tensor.basic import ExtractDiag, Split
from pytensor.tensor.extra_ops import CumOp


@llvm_emit.register(CumOp)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    if x.ndim == 0:
        raise NotImplementedError("CumOp on a 0-d input")
    out = outputs[0].allocate(x.shape)
    combine = operator.add if op.mode == "add" else operator.mul
    init = constant(b, out.dtype, 0 if op.mode == "add" else 1)

    def level(src: Array, dst: Array, axis: int):
        """Accumulate along `axis`, for every index of the other axes."""
        if src.ndim == 1:

            def step(i, acc):
                new = combine(Num(b, acc, dst.dtype), src.load(b, i).astype(dst.dtype))
                dst.store(b, new, i)
                return new.value

            b.fold(src.shape[0], [init], step)
            return
        other = 1 if axis == 0 else 0
        with b.loop(src.shape[other]) as i:
            level(src.at(b, i, other), dst.at(b, i, other), axis - (other < axis))

    level(x, out, op.axis % x.ndim)


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
    start = (
        b.mul(b.i64(off), x.strides[a2], nsw=True)
        if off >= 0
        else b.mul(b.i64(-off), x.strides[a1], nsw=True)
    )
    start = b.select(b.sgt(length, zero), start, zero)
    shape = [x.shape[a] for a in range(x.ndim) if a not in (a1, a2)] + [length]
    strides = [x.strides[a] for a in range(x.ndim) if a not in (a1, a2)] + [
        b.add(x.strides[a1], x.strides[a2])
    ]
    set_output(cg, outputs[0], x.view(b.gep_bytes(x.data, start), shape, strides))


@llvm_layout.register(Split)
def _(op, node, in_layouts):
    return split_layouts(op, node, in_layouts[0])


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
            cg.check(
                b.eq(n, b.i64(static)),
                "Split: split size differs from the static shape",
            )
            n = b.i64(static)
        parts.append((offset, n))
        offset = b.add(offset, n)
    cg.check(
        b.eq(offset, x.shape[axis]),
        "Split: the split sizes don't add up to the length of the axis",
    )
    layouts = split_layouts(op, node, x.layout)
    for out, (start, n), lay in zip(outputs, parts, layouts, strict=True):
        shape = list(x.shape)
        shape[axis] = n
        # an empty part may start at the end: keep the pointer inside the array
        start = b.select(b.sgt(n, b.i64(0)), start, b.i64(0))
        set_output(
            cg,
            out,
            x.view(
                b.gep_bytes(x.data, b.mul(start, x.strides[axis], nsw=True)),
                shape,
                list(x.strides),
                lay,
            ),
        )
