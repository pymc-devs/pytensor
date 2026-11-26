"""Shapes, allocation, ranges, reshape, join, assertions."""

from __future__ import annotations

import pytensor.scalar as ps
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, c_strides, size
from pytensor.link.llvm.codegen.interface import ScalarOutput, llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.num import cast, load, store
from pytensor.link.llvm.codegen.types import element_type, ir
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.basic import Alloc, AllocEmpty, ARange, Join, MakeVector
from pytensor.tensor.shape import Reshape, Shape, Shape_i


def _scalar(b, a):
    """The element of a 0-d array (or a scalar value itself)."""
    if not isinstance(a, Array):
        return a
    return a.load(b).value


@llvm_emit.register(Shape_i)
def _(op, node, cg, inputs, outputs):
    out = outputs[0].allocate([])
    out.store(cg.b, cast(cg.b, inputs[0].shape[op.i], ps.int64, out.scalar_type))


@llvm_emit.register(Shape)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    out = outputs[0].allocate([b.i64(x.ndim)])
    for a, n in enumerate(x.shape):
        out.store(b, cast(b, n, ps.int64, out.scalar_type), a)


@llvm_emit.register(MakeVector)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    out = outputs[0].allocate([b.i64(len(inputs))])
    for k, x in enumerate(inputs):
        out.store(b, cast(b, _scalar(b, x), x.scalar_type, out.scalar_type), k)


def _shape_from(cg, node, values, first):
    """Output lengths from 0-d integer inputs, checked against the static type."""
    b = cg.b
    shape = []
    static = node.outputs[0].type.shape
    for a, x in enumerate(values[first:]):
        n = cast(b, _scalar(b, x), x.scalar_type, ps.int64)
        cg.check(b.sge(n, 0), "negative dimensions are not allowed")
        if static[a] is not None:
            cg.check(
                b.eq(n, static[a]),
                f"the shape on axis {a} doesn't match the static shape {static[a]}",
            )
            n = b.i64(static[a])
        shape.append(n)
    return shape


@llvm_emit.register(Alloc)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    v = inputs[0]
    shape = _shape_from(cg, node, inputs, 1)
    out = outputs[0].allocate(shape)
    # the value broadcasts against the output, aligned on the right
    off = len(shape) - v.ndim
    vt = node.inputs[0].type
    strides = [b.i64(0)] * off
    for a in range(v.ndim):
        ln, on = v.shape[a], shape[off + a]
        if vt.shape[a] == 1:
            strides.append(b.i64(0))
            continue
        if not (ln.const_int is not None and ln.const_int == on.const_int):
            one = b.eq(ln, 1)
            cg.check(
                b.or_(b.eq(ln, on), one),
                "Alloc: the value can't be broadcast to the shape",
            )
            strides.append(b.select(one, 0, v.strides[a]))
        else:
            strides.append(v.strides[a])
    source = [None] * off + list(range(v.ndim))
    src = Array(
        v.scalar_type, v.data, shape, strides, v.view_layout(shape, strides, source)
    )

    def body(p):
        store(
            b,
            out.scalar_type,
            cast(b, load(b, v.scalar_type, p[0]), v.scalar_type, out.scalar_type),
            p[1],
        )

    cg.for_each(shape, [src, out], body)


@llvm_emit.register(AllocEmpty)
def _(op, node, cg, inputs, outputs):
    outputs[0].allocate(_shape_from(cg, node, inputs, 0))


@llvm_emit.register(ARange)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    start, stop, step = (_scalar(b, x) for x in inputs)
    types = [x.scalar_type for x in inputs]
    out_type = element_type(node.outputs[0].type)
    i64 = ir.i64
    if all(d not in ps.float_types for d in types):
        start, stop, step = (
            cast(b, v, d, ps.int64)
            for v, d in zip((start, stop, step), types, strict=True)
        )
        cg.check(b.ne(step, 0), "arange: step can't be zero")
        # length = ceil((stop - start) / step), at least 0
        span = b.sub(stop, start)
        adj = b.select(b.sgt(step, 0), b.sub(step, 1), b.add(step, 1))
        n = b.smax(b.sdiv(b.add(span, adj), step), 0)
        start_v, step_v, dt = start, step, ps.int64
    else:
        start, stop, step = (
            cast(b, v, d, ps.float64)
            for v, d in zip((start, stop, step), types, strict=True)
        )
        cg.check(b.fcmp(ir.FCmp.UNE, step, 0.0), "arange: step can't be zero")
        q = b.intrinsic(ir.Intrinsic.Ceil, [b.fdiv(b.fsub(stop, start), step)])
        ok = b.fcmp(ir.FCmp.OLT, b.intrinsic(ir.Intrinsic.Fabs, [q]), 2.0**62)
        cg.check(ok, "arange: invalid or too large range")
        q = b.select(b.fcmp(ir.FCmp.OGT, q, 0.0), q, 0.0)
        n = b.fptosi(q, i64)
        start_v, step_v, dt = start, step, ps.float64
    static = node.outputs[0].type.shape[0]
    if static is not None:
        cg.check(b.eq(n, static), "arange: length differs from the static shape")
    out = outputs[0].allocate([n])
    with b.loop(n) as i:
        if dt == ps.int64:
            v = b.add(start_v, b.mul(i, step_v))
        else:
            v = b.fadd(start_v, b.fmul(b.sitofp(i, ir.f64), step_v))
        out.store(b, cast(b, v, dt, out_type), i)


@llvm_layout.register(Reshape)
def _(op, node, input_layouts) -> list[Layout | None]:
    # a view of a C-contiguous input, else a C-contiguous copy: C either way
    return [Layout.c(node.outputs[0].type.ndim)]


@llvm_emit.register(Reshape)
def _(op, node, cg, inputs, outputs):
    """A view if the input is contiguous, else a copy (C order)."""
    b = cg.b
    i64 = ir.i64
    x, shp = inputs
    total = size(b, x.shape)
    static = node.outputs[0].type.shape
    raw = [shp.load(b, a).astype(ps.int64).value for a in range(op.ndim)]
    # one -1 is inferred from the size
    known = b.i64(1)
    n_infer = b.i64(0)
    for v in raw:
        neg = b.eq(v, -1)
        known = b.select(neg, known, b.mul(known, v))
        n_infer = b.add(n_infer, b.zext(neg, i64))
        cg.check(b.sge(v, -1), "reshape: negative length")
    cg.check(b.sle(n_infer, 1), "reshape: can only infer one length")
    nz = b.ne(known, 0)
    inferred = b.select(nz, b.sdiv(total, b.select(nz, known, 1)), 0)
    shape = []
    for a, v in enumerate(raw):
        n = b.select(b.eq(v, -1), inferred, v)
        if static[a] is not None:
            n = b.i64(static[a])
        shape.append(n)
    cg.check(b.eq(size(b, shape), total), "cannot reshape: the total size changes")
    out = outputs[0]
    contig = x.is_contiguous(b, x.shape)

    def view():
        strides = c_strides(b, x.scalar_type, shape)
        out.set_view(x.view(x.data, shape, strides, Layout.c(len(shape))))
        return []

    def copied():
        _copy_reshaped(cg, x, out.allocate(shape))
        return []

    c = contig.const_int
    if c == 1:
        view()
    elif c == 0:
        copied()
    else:
        b.if_else(contig, view, copied)


def _copy_reshaped(cg, src: Array, dst: Array):
    """Copy `src` into C-contiguous `dst` (same size, other shape), in C order: into
    `dst`'s memory seen with `src`'s shape."""
    shape = src.shape
    strides = c_strides(cg.b, dst.scalar_type, shape)
    Array.c(dst.scalar_type, dst.data, shape, strides).copy_from(cg, src)


@llvm_emit.register(Join)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    axis = op.axis
    first = inputs[0]
    nd = first.ndim
    for k, x in enumerate(inputs[1:], 1):
        for a in range(nd):
            if a == axis:
                continue
            p, q = x.shape[a], first.shape[a]
            if not (p.const_int is not None and p.const_int == q.const_int):
                cg.check(
                    b.eq(p, q),
                    f"Join: all inputs must have the same shape except on the join axis (input {k} differs on axis {a})",
                )
    total = first.shape[axis]
    for x in inputs[1:]:
        total = (
            b.add(total, x.shape[axis], nsw=True)
            if total.const_int is None or x.shape[axis].const_int is None
            else b.i64(total.const_int + x.shape[axis].const_int)
        )
    shape = [total if a == axis else first.shape[a] for a in range(nd)]
    out = outputs[0].allocate(shape)
    offset = b.i64(0)
    for x in inputs:
        dst = Array(
            out.scalar_type,
            b.gep_bytes(out.data, b.mul(offset, out.strides[axis], nsw=True)),
            x.shape,
            out.strides,
            out.view_layout(x.shape, out.strides, range(nd)),
        )
        dst.copy_from(cg, x)
        offset = (
            b.add(offset, x.shape[axis])
            if offset.const_int is None or x.shape[axis].const_int is None
            else b.i64(offset.const_int + x.shape[axis].const_int)
        )


@llvm_layout.register(CheckAndRaise)
def _(op, node, input_layouts) -> list[Layout | None]:
    """A view that is its input, for the output's possibly more static shape."""
    return [input_layouts[0].for_shape(node.outputs[0].type.shape)]


@llvm_emit.register(CheckAndRaise)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    for condition, var in zip(inputs[1:], node.inputs[1:], strict=True):
        value = _scalar(b, condition)
        scalar_type = element_type(var.type)
        ok = value if scalar_type == ps.bool else cast(b, value, scalar_type, ps.bool)
        cg.check(ok, op.msg or "Assertion failed", op.exc_type)
    if isinstance(outputs[0], ScalarOutput):
        outputs[0].set(inputs[0])
    else:
        outputs[0].set_view(inputs[0])
