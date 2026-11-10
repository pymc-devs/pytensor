"""Shapes, allocation, ranges, reshape, join, assertions."""

from __future__ import annotations

from pytensor.link.llvm.codegen.elemwise import output_array
from pytensor.link.llvm.codegen.emit import (
    advance,
    c_array,
    Array,
    ProvidedOutput,
    c_strides,
    cast,
    copy,
    for_each,
    is_contiguous,
    load,
    mul,
    size,
    store,
)
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_outputs, view_output
from pytensor.link.llvm.layout import Layout, same_layout
from pytensor.link.llvm.codegen.types import ir, is_float
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.basic import ARange, Alloc, AllocEmpty, Join, MakeVector
from pytensor.tensor.shape import Reshape, Shape, Shape_i


def _scalar(b, a):
    """The element of a 0-d array (or a scalar value itself)."""
    if not isinstance(a, Array):
        return a
    return load(b, a.dtype, a.data)


@llvm_emit.register(Shape_i)
def _(op, node, cg, inputs, outputs):
    out = output_array(outputs[0], [])
    store(cg.b, out.dtype, cast(cg.b, inputs[0].shape[op.i], "int64", out.dtype), out.data)


@llvm_emit.register(Shape)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    x = inputs[0]
    out = output_array(outputs[0], [b.const_int(m.i64, x.ndim)])
    for a, n in enumerate(x.shape):
        p = b.gep_bytes(out.data, mul(b, b.const_int(m.i64, a), out.strides[0]))
        store(b, out.dtype, cast(b, n, "int64", out.dtype), p)


@llvm_emit.register(MakeVector)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    out = output_array(outputs[0], [b.const_int(m.i64, len(inputs))])
    for k, x in enumerate(inputs):
        p = b.gep_bytes(out.data, mul(b, b.const_int(m.i64, k), out.strides[0]))
        store(b, out.dtype, cast(b, _scalar(b, x), x.dtype, out.dtype), p)


def _shape_from(cg, node, values, first):
    """Output lengths from 0-d integer inputs, checked against the static type."""
    b, m = cg.b, ir()
    shape = []
    static = node.outputs[0].type.shape
    for a, x in enumerate(values[first:]):
        n = cast(b, _scalar(b, x), x.dtype, "int64")
        cg.check(b.icmp(m.ICmp.SGE, n, b.const_int(m.i64, 0)), "negative dimensions are not allowed")
        if static[a] is not None:
            cg.check(
                b.icmp(m.ICmp.EQ, n, b.const_int(m.i64, static[a])),
                f"the shape on axis {a} doesn't match the static shape {static[a]}",
            )
            n = b.const_int(m.i64, static[a])
        shape.append(n)
    return shape


@llvm_emit.register(Alloc)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    v = inputs[0]
    shape = _shape_from(cg, node, inputs, 1)
    out = output_array(outputs[0], shape)
    # the value broadcasts against the output, aligned on the right
    off = len(shape) - v.ndim
    vt = node.inputs[0].type
    strides = [b.const_int(m.i64, 0)] * off
    for a in range(v.ndim):
        ln, on = v.shape[a], shape[off + a]
        if vt.shape[a] == 1:
            strides.append(b.const_int(m.i64, 0))
            continue
        if not (ln.const_int is not None and ln.const_int == on.const_int):
            one = b.icmp(m.ICmp.EQ, ln, b.const_int(m.i64, 1))
            cg.check(b.or_(b.icmp(m.ICmp.EQ, ln, on), one), "Alloc: the value can't be broadcast to the shape")
            strides.append(b.select(one, b.const_int(m.i64, 0), v.strides[a]))
        else:
            strides.append(v.strides[a])
    src = Array(v.dtype, v.data, shape, strides)

    def body(p):
        store(b, out.dtype, cast(b, load(b, v.dtype, p[0]), v.dtype, out.dtype), p[1])

    for_each(b, shape, [src, out], body)


@llvm_emit.register(AllocEmpty)
def _(op, node, cg, inputs, outputs):
    output_array(outputs[0], _shape_from(cg, node, inputs, 0))


@llvm_emit.register(ARange)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    start, stop, step = (_scalar(b, x) for x in inputs)
    dts = [x.dtype for x in inputs]
    out_dtype = node.outputs[0].type.dtype
    i64 = m.i64
    if all(not is_float(d) for d in dts):
        start, stop, step = (cast(b, v, d, "int64") for v, d in zip((start, stop, step), dts, strict=True))
        cg.check(b.icmp(m.ICmp.NE, step, b.const_int(i64, 0)), "arange: step can't be zero")
        # length = ceil((stop - start) / step), at least 0
        span = b.sub(stop, start)
        pos = b.icmp(m.ICmp.SGT, step, b.const_int(i64, 0))
        adj = b.select(pos, b.sub(step, b.const_int(i64, 1)), b.add(step, b.const_int(i64, 1)))
        n = b.sdiv(b.add(span, adj), step)
        n = b.select(b.icmp(m.ICmp.SGT, n, b.const_int(i64, 0)), n, b.const_int(i64, 0))
        start_v, step_v, dt = start, step, "int64"
    else:
        f64 = m.f64
        start, stop, step = (cast(b, v, d, "float64") for v, d in zip((start, stop, step), dts, strict=True))
        cg.check(b.fcmp(m.FCmp.UNE, step, b.const_float(f64, 0.0)), "arange: step can't be zero")
        q = b.intrinsic(m.Intrinsic.Ceil, [b.fdiv(b.fsub(stop, start), step)])
        ok = b.fcmp(m.FCmp.OLT, b.intrinsic(m.Intrinsic.Fabs, [q]), b.const_float(f64, 2.0**62))
        cg.check(ok, "arange: invalid or too large range")
        q = b.select(b.fcmp(m.FCmp.OGT, q, b.const_float(f64, 0.0)), q, b.const_float(f64, 0.0))
        n = b.fptosi(q, i64)
        start_v, step_v, dt = start, step, "float64"
    static = node.outputs[0].type.shape[0]
    if static is not None:
        cg.check(b.icmp(m.ICmp.EQ, n, b.const_int(i64, static)), "arange: length differs from the static shape")
    out = output_array(outputs[0], [n])
    with b.loop(n) as i:
        if dt == "int64":
            v = b.add(start_v, b.mul(i, step_v))
        else:
            v = b.fadd(start_v, b.fmul(b.sitofp(i, m.f64), step_v))
        store(b, out_dtype, cast(b, v, dt, out_dtype), advance(b, out.data, i, out.strides[0], out.dtype))


@llvm_outputs.register(Reshape)
def _(op, node, in_layouts):
    # a view of a C-contiguous input, else a C-contiguous copy: C either way
    return [view_output(node.outputs[0], Layout.c(node.outputs[0].type.ndim))]


@llvm_emit.register(Reshape)
def _(op, node, cg, inputs, outputs):
    """A view if the input is contiguous, else a copy (C order)."""
    b, m = cg.b, ir()
    i64 = m.i64
    x, shp = inputs
    total = size(b, x.shape)
    static = node.outputs[0].type.shape
    raw = []
    for a in range(op.ndim):
        p = b.gep_bytes(shp.data, mul(b, b.const_int(i64, a), shp.strides[0]))
        raw.append(cast(b, load(b, shp.dtype, p), shp.dtype, "int64"))
    # one -1 is inferred from the size
    minus1 = b.const_int(i64, -1)
    known = b.const_int(i64, 1)
    n_infer = b.const_int(i64, 0)
    for v in raw:
        neg = b.icmp(m.ICmp.EQ, v, minus1)
        known = b.select(neg, known, b.mul(known, v))
        n_infer = b.add(n_infer, b.zext(neg, i64))
        cg.check(b.icmp(m.ICmp.SGE, v, minus1), "reshape: negative length")
    cg.check(b.icmp(m.ICmp.SLE, n_infer, b.const_int(i64, 1)), "reshape: can only infer one length")
    nz = b.icmp(m.ICmp.NE, known, b.const_int(i64, 0))
    inferred = b.select(nz, b.sdiv(total, b.select(nz, known, b.const_int(i64, 1))), b.const_int(i64, 0))
    shape = []
    for a, v in enumerate(raw):
        n = b.select(b.icmp(m.ICmp.EQ, v, minus1), inferred, v)
        if static[a] is not None:
            n = b.const_int(i64, static[a])
        shape.append(n)
    cg.check(b.icmp(m.ICmp.EQ, size(b, shape), total), "cannot reshape: the total size changes")
    out = outputs[0]
    if isinstance(out, ProvidedOutput):
        copy(cg, x, out.array)
        return
    contig = is_contiguous(b, x.shape, x)

    def view():
        out.set_view(c_array(x.dtype, x.data, shape, c_strides(b, x.dtype, shape)))
        return []

    def fresh():
        dst = out.allocate(shape)
        src = Array(x.dtype, x.data, x.shape, x.strides)
        _copy_reshaped(cg, src, dst)
        return []

    c = contig.const_int
    if c == 1:
        view()
    elif c == 0:
        fresh()
    else:
        b.if_else(contig, view, fresh)
        out.done = True


def _copy_reshaped(cg, src: Array, dst: Array):
    """Copy `src` into contiguous `dst` (same size, other shape), in C order."""
    b, m = cg.b, ir()
    k = [b.const_int(m.i64, 0)]

    def body(p):
        i = b.load(m.i64, kptr)
        store(b, dst.dtype, load(b, src.dtype, p[0]), b.gep(_mem(dst.dtype), dst.data, i))
        b.store(b.add(i, b.const_int(m.i64, 1)), kptr)

    kptr = b.alloca(m.i64)
    b.store(k[0], kptr)
    for_each(b, src.shape, [Array(src.dtype, src.data, src.shape, src.strides)], body)


def _mem(dtype):
    from pytensor.link.llvm.codegen.types import memory_type

    return memory_type(dtype)


@llvm_emit.register(Join)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
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
                    b.icmp(m.ICmp.EQ, p, q),
                    f"Join: all inputs must have the same shape except on the join axis (input {k} differs on axis {a})",
                )
    total = first.shape[axis]
    for x in inputs[1:]:
        total = b.add(total, x.shape[axis], nsw=True) if total.const_int is None or x.shape[axis].const_int is None else b.const_int(m.i64, total.const_int + x.shape[axis].const_int)
    shape = [total if a == axis else first.shape[a] for a in range(nd)]
    out = output_array(outputs[0], shape)
    offset = b.const_int(m.i64, 0)
    for x in inputs:
        dst = Array(out.dtype, b.gep_bytes(out.data, mul(b, offset, out.strides[axis])), x.shape, out.strides)
        copy(cg, x, dst)
        offset = b.add(offset, x.shape[axis]) if offset.const_int is None or x.shape[axis].const_int is None else b.const_int(m.i64, offset.const_int + x.shape[axis].const_int)


@llvm_outputs.register(CheckAndRaise)
def _(op, node, in_layouts):
    return [view_output(node.outputs[0], same_layout(node, in_layouts[0]))]


@llvm_emit.register(CheckAndRaise)
def _(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    for c, var in zip(inputs[1:], node.inputs[1:], strict=True):
        v = _scalar(b, c)
        dt = var.type.dtype
        ok = v if dt == "bool" else cast(b, v, dt, "bool")
        cg.check(ok, op.msg or "Assertion failed")
    outputs[0].set_view(inputs[0])
