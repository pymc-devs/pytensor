"""CumOp, ExtractDiag, Split, Argmax, Nonzero."""

from __future__ import annotations

import operator

from pytensor.link.llvm.codegen.emit import Array, constant, set_output, size
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.num import Num
from pytensor.link.llvm.codegen.types import ir
from pytensor.link.llvm.layout import split_layouts
from pytensor.tensor.basic import ExtractDiag, Nonzero, Split
from pytensor.tensor.extra_ops import CumOp
from pytensor.tensor.math import Argmax


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
    # length = max(0, min(n1 - max(0, -off), n2 - max(0, off)))
    l1 = b.sub(n1, max(0, -off))
    l2 = b.sub(n2, max(0, off))
    length = b.smax(b.smin(l1, l2), 0)
    start = (
        b.mul(x.strides[a2], off, nsw=True)
        if off >= 0
        else b.mul(x.strides[a1], -off, nsw=True)
    )
    start = b.select(b.sgt(length, 0), start, 0)
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
        n = splits.load(b, k).astype("int64").value
        cg.check(b.sge(n, 0), "Split: negative split size")
        static = node.outputs[k].type.shape[axis]
        if static is not None:
            cg.check(
                b.eq(n, static),
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
        start = b.select(b.sgt(n, 0), start, 0)
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


@llvm_emit.register(Argmax)
def _(op, node, cg, inputs, outputs):
    """numpy's argmax over the axes, flattened in C order: the first maximum, or the
    first NaN."""
    b = cg.b
    (x,) = inputs
    reduced = range(x.ndim) if op.axis is None else op.axis
    kept = [a for a in range(x.ndim) if a not in reduced]
    out = outputs[0].allocate([x.shape[a] for a in kept])
    cg.check(
        b.sgt(size(b, [x.shape[a] for a in reduced]), 0),
        "attempt to get argmax of an empty sequence",
    )
    nan_aware = x.dtype.startswith("float")

    def better(v: Num, best: Num) -> Num:
        if nan_aware:
            return (v > best) | (v.ne(v) & best.eq(best))
        return v > best

    def walk(a: Array, carried):
        """Fold (best, its index, the flat index) over `a`, in C order."""
        if a.ndim == 1:

            def step(i, best, index, count):
                v, best = a.load(b, i), Num(b, best, a.dtype)
                take = better(v, best)
                return [
                    take.select(v, best).value,
                    b.select(take.value, count, index),
                    b.add(count, 1),
                ]

            return b.fold(a.shape[0], carried, step)
        return b.fold(a.shape[0], carried, lambda i, *c: walk(a.at(b, i), list(c)))

    def level(src: Array, dst: Array, positions):
        """For every index of the kept axes (at `positions` in `src`)."""
        if positions:
            first, *rest = positions
            with b.loop(src.shape[first]) as i:
                level(src.at(b, i, first), dst.at(b, i), [p - 1 for p in rest])
        elif src.ndim == 0:
            dst.store(b, 0)
        else:
            start = src.load(b, *([0] * src.ndim)).value
            dst.store(b, walk(src, [start, b.i64(0), b.i64(0)])[1])

    level(x, out, kept)


@llvm_emit.register(Nonzero)
def _(op, node, cg, inputs, outputs):
    """numpy.nonzero: per axis, the indices of the non-zero elements (NaN is non-zero),
    in C order. One pass counts them, for the length of the outputs; the other writes
    them."""
    b = cg.b
    (x,) = inputs

    def nonzero(v: Num) -> Num:
        return v if v.dtype == "bool" else v.ne(0)

    def walk(a: Array, carried, index, visit):
        """Fold `visit(index, element, *carried)` over the elements of `a`, in C order."""
        if a.ndim == 0:
            return visit(index, a.load(b), *carried)
        return b.fold(
            a.shape[0],
            carried,
            lambda i, *c: walk(a.at(b, i), list(c), [*index, i], visit),
        )

    def count(index, v, n):
        return [b.add(n, b.zext(nonzero(v).value, ir.i64))]

    (n,) = walk(x, [b.i64(0)], [], count)
    outs = [o.allocate([n]) for o in outputs]

    def write(index, v, k):
        take = nonzero(v).value
        with b.if_(take):
            for out, i in zip(outs, index, strict=True):
                out.store(b, i, k)
        return [b.add(k, b.zext(take, ir.i64))]

    walk(x, [b.i64(0)], [], write)
