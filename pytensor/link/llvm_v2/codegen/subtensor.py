"""Basic indexing: Subtensor (a view) and IncSubtensor (set or add into a copy)."""

from __future__ import annotations

from pytensor.link.llvm_v2.codegen.elemwise import output_array
from pytensor.link.llvm_v2.codegen.emit import Array, cast, copy, for_each, load, mul, store
from pytensor.link.llvm_v2.codegen.function import fresh_output, llvm_emit, llvm_outputs, view_output
from pytensor.link.llvm_v2.layout import subtensor_layout
from pytensor.link.llvm_v2.codegen.scalar import emit_scalar
from pytensor.link.llvm_v2.codegen.types import ir
from pytensor.scalar.basic import add as scalar_add
from pytensor.tensor.subtensor import IncSubtensor, Subtensor, unflatten_index_variables


def _i64(b, v, dtype):
    return cast(b, v, dtype, "int64")


def _select(b, c, x, y):
    if c.const_int is not None:
        return x if c.const_int else y
    return b.select(c, x, y)


def _min(b, x, y):
    m = ir()
    return _select(b, b.icmp(m.ICmp.SLT, x, y), x, y)


def _or(b, x, y):
    if x.const_int == 1 or y.const_int == 1:
        return b.const_bool(True)
    if x.const_int == 0:
        return y
    if y.const_int == 0:
        return x
    return b.or_(x, y)


def _max(b, x, y):
    m = ir()
    return _select(b, b.icmp(m.ICmp.SGT, x, y), x, y)


def index(cx, x: Array, entries) -> Array:
    """`x[entries]` as a view: `entries` hold (value, dtype) for integer indices and
    slices of (value, dtype) / None, like `unflatten_index_variables` returns them."""
    b, m = cx.b, ir()
    i64 = m.i64
    zero, one = b.const_int(i64, 0), b.const_int(i64, 1)
    offset = zero
    shape, strides = [], []
    empty = b.const_bool(False)
    axis = 0
    for e in entries:
        n, st = x.shape[axis], x.strides[axis]
        if isinstance(e, slice):
            step = one if e.step is None else _i64(b, *e.step)
            cx.check(b.icmp(m.ICmp.NE, step, zero), "slice step cannot be zero")
            pos = b.icmp(m.ICmp.SGT, step, zero)
            minus1 = b.const_int(i64, -1)
            nm1 = b.sub(n, one)
            lower = _select(b, pos, zero, minus1)
            upper = _select(b, pos, n, nm1)

            def bound(v, default):
                if v is None:
                    return default
                v = _i64(b, *v)
                neg = b.icmp(m.ICmp.SLT, v, zero)
                return _select(b, neg, _max(b, b.add(v, n), lower), _min(b, v, upper))

            start = bound(e.start, _select(b, pos, lower, upper))
            stop = bound(e.stop, _select(b, pos, upper, lower))
            # length = ceil((stop - start) / step), at least 0
            span = _select(b, pos, b.sub(stop, start), b.sub(start, stop))
            astep = _select(b, pos, step, b.neg(step))
            length = _select(
                b,
                b.icmp(m.ICmp.SGT, span, zero),
                b.add(b.sdiv(b.sub(span, one), astep), one),
                zero,
            )
            empty = _or(b, empty, b.icmp(m.ICmp.EQ, length, zero))
            offset = b.add(offset, mul(b, start, st))
            shape.append(length)
            strides.append(mul(b, st, step))
        else:
            i = _i64(b, *e)
            i = _select(b, b.icmp(m.ICmp.SLT, i, zero), b.add(i, n), i)
            ok = b.and_(b.icmp(m.ICmp.SGE, i, zero), b.icmp(m.ICmp.SLT, i, n))
            cx.check(ok, f"index out of bounds for axis {axis}")
            offset = b.add(offset, mul(b, i, st))
        axis += 1
    shape += x.shape[axis:]
    strides += x.strides[axis:]
    # an empty view may start out of range: keep the pointer inside the array
    offset = _select(b, empty, zero, offset)
    data = x.data if offset.const_int == 0 else b.gep_bytes(x.data, offset)
    return Array(x.dtype, data, shape, strides)


def _entries(op, node, inputs, first):
    values = [(v, var.type.dtype) for v, var in zip(inputs[first:], node.inputs[first:], strict=True)]
    return unflatten_index_variables(values, op.idx_list)


@llvm_emit.register(Subtensor)
def _(op, node, cx, inputs, outputs):
    view = index(cx, inputs[0], _entries(op, node, inputs, 1))
    view.layout = subtensor_layout(op, node, inputs[0].layout)
    out = outputs[0]
    if hasattr(out, "set_view"):
        out.set_view(view)
    else:
        copy(cx, view, out.array)


@llvm_outputs.register(Subtensor)
def _(op, node, in_layouts):
    return [view_output(node.outputs[0], subtensor_layout(op, node, in_layouts[0]))]


@llvm_outputs.register(IncSubtensor)
def _(op, node, in_layouts):
    return [fresh_output(node.outputs[0])]


@llvm_emit.register(IncSubtensor)
def _(op, node, cx, inputs, outputs):
    b, m = cx.b, ir()
    x, y = inputs[0], inputs[1]
    out = output_array(outputs[0], x.shape)
    copy(cx, x, out)
    view = index(cx, out, _entries(op, node, inputs, 2))
    # y broadcasts against the view, aligned on the right, by its static type
    ytype = node.inputs[1].type
    off = view.ndim - ytype.ndim
    if off < 0:
        raise NotImplementedError("IncSubtensor: y has more axes than the indexed view")
    ystrides = [b.const_int(m.i64, 0)] * off
    for a, s in enumerate(ytype.shape):
        va = off + a
        if s == 1:
            ystrides.append(b.const_int(m.i64, 0))
        else:
            # like numpy, a length-1 axis of y broadcasts even if its type doesn't say so
            ln, vn = y.shape[a], view.shape[va]
            if ln.const_int is not None and ln.const_int == vn.const_int:
                ystrides.append(y.strides[a])
                continue
            one = b.icmp(m.ICmp.EQ, ln, b.const_int(m.i64, 1))
            cx.check(
                _or(b, b.icmp(m.ICmp.EQ, ln, vn), one),
                "IncSubtensor: shape mismatch between the value and the indexed view",
            )
            ystrides.append(_select(b, one, b.const_int(m.i64, 0), y.strides[a]))
    ys = Array(y.dtype, y.data, view.shape, ystrides)
    dt = out.dtype

    def body(p):
        v = cast(b, load(b, y.dtype, p[1]), y.dtype, dt)
        if not op.set_instead_of_inc:
            v = emit_scalar(scalar_add, b, [(load(b, dt, p[0]), dt), (v, dt)], [dt])[0]
        store(b, dt, v, p[0])

    for_each(b, view.shape, [view, ys], body)


__all__ = ["index"]
