"""CumOp, ExtractDiag, Split, Argmax, Nonzero."""

from __future__ import annotations

import operator

from pytensor.link.llvm.analysis.layout import split_layouts
from pytensor.link.llvm.codegen.emit import Array, Codegen, constant, copy, size
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.num import Num
from pytensor.link.llvm.codegen.types import ir
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


def _diagonal(codegen: Codegen, op: ExtractDiag, source: Array) -> Array:
    """Return the diagonal of two axes of `source`, as the last axis of a view."""
    builder = codegen.b
    axis1, axis2 = op.axis1 % source.ndim, op.axis2 % source.ndim
    offset = int(op.offset)
    # length = max(0, min(n1 - max(0, -offset), n2 - max(0, offset)))
    length1 = builder.sub(source.shape[axis1], max(0, -offset))
    length2 = builder.sub(source.shape[axis2], max(0, offset))
    length = builder.smax(builder.smin(length1, length2), 0)
    if offset >= 0:
        start = builder.mul(source.strides[axis2], offset, nsw=True)
    else:
        start = builder.mul(source.strides[axis1], -offset, nsw=True)
    start = builder.select(builder.sgt(length, 0), start, 0)
    kept = [a for a in range(source.ndim) if a not in (axis1, axis2)]
    shape = [source.shape[a] for a in kept] + [length]
    strides = [source.strides[a] for a in kept] + [
        builder.add(source.strides[axis1], source.strides[axis2])
    ]
    return source.view(builder.gep_bytes(source.data, start), shape, strides)


@llvm_emit.register(ExtractDiag)
def _(op, node, codegen, inputs, outputs):
    (output,) = outputs
    diagonal = _diagonal(codegen, op, inputs[0])
    if op.view:
        output.set_view(diagonal)
    else:
        copy(codegen, diagonal, output.allocate(diagonal.shape))


@llvm_layout.register(Split)
def _(op, node, input_layouts):
    return split_layouts(op, node, input_layouts[0])


@llvm_emit.register(Split)
def _(op, node, codegen, inputs, outputs):
    builder = codegen.b
    source, splits = inputs
    axis = op.axis
    offset = builder.i64(0)
    parts = []
    for k, var in enumerate(node.outputs):
        length = splits.load(builder, k).astype("int64").value
        codegen.check(builder.sge(length, 0), "Split: negative split size")
        static = var.type.shape[axis]
        if static is not None:
            codegen.check(
                builder.eq(length, static),
                "Split: split size differs from the static shape",
            )
            length = builder.i64(static)
        parts.append((offset, length))
        offset = builder.add(offset, length)
    codegen.check(
        builder.eq(offset, source.shape[axis]),
        "Split: the split sizes don't add up to the length of the axis",
    )
    for output, (start, length) in zip(outputs, parts, strict=True):
        shape = list(source.shape)
        shape[axis] = length
        # an empty part may start at the end: keep the pointer inside the array
        start = builder.select(builder.sgt(length, 0), start, 0)
        byte_offset = builder.mul(start, source.strides[axis], nsw=True)
        data = builder.gep_bytes(source.data, byte_offset)
        output.set_view(source.view(data, shape, list(source.strides)))


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
