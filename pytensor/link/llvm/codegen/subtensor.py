"""Basic indexing: Subtensor (a view) and IncSubtensor (set or add into a copy)."""

from __future__ import annotations

from pytensor.link.llvm.codegen.emit import (
    Array,
    cast,
    for_each,
    load,
    set_output,
    store,
)
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_inplace, llvm_layout
from pytensor.link.llvm.codegen.scalar import emit_scalar
from pytensor.link.llvm.layout import subtensor_layout
from pytensor.scalar.basic import add as scalar_add
from pytensor.tensor.subtensor import IncSubtensor, Subtensor, unflatten_index_variables


def _i64(b, v, dtype):
    return cast(b, v, dtype, "int64")


def index(cg, x: Array, entries) -> Array:
    """`x[entries]` as a view: `entries` hold (value, dtype) for integer indices and
    slices of (value, dtype) / None, like `unflatten_index_variables` returns them."""
    b = cg.b
    offset = b.i64(0)
    shape, strides = [], []
    empty = b.const_bool(False)
    axis = 0
    for e in entries:
        n, st = x.shape[axis], x.strides[axis]
        if isinstance(e, slice):
            step = b.i64(1) if e.step is None else _i64(b, *e.step)
            cg.check(b.ne(step, 0), "slice step cannot be zero")
            pos = b.sgt(step, 0)
            lower = b.select(pos, b.i64(0), -1)
            upper = b.select(pos, n, b.sub(n, 1))

            def bound(v, default):
                if v is None:
                    return default
                v = _i64(b, *v)
                return b.select(
                    b.slt(v, 0), b.smax(b.add(v, n), lower), b.smin(v, upper)
                )

            start = bound(e.start, b.select(pos, lower, upper))
            stop = bound(e.stop, b.select(pos, upper, lower))
            # length = ceil((stop - start) / step), at least 0
            span = b.select(pos, b.sub(stop, start), b.sub(start, stop))
            astep = b.select(pos, step, b.neg(step))
            length = b.select(
                b.sgt(span, 0), b.add(b.sdiv(b.sub(span, 1), astep), 1), 0
            )
            empty = b.or_(empty, b.eq(length, 0))
            offset = b.add(offset, b.mul(start, st, nsw=True))
            shape.append(length)
            strides.append(b.mul(st, step, nsw=True))
        else:
            i = _i64(b, *e)
            i = b.select(b.slt(i, 0), b.add(i, n), i)
            cg.check(
                b.and_(b.sge(i, 0), b.slt(i, n)), f"index out of bounds for axis {axis}"
            )
            offset = b.add(offset, b.mul(i, st, nsw=True))
        axis += 1
    shape += x.shape[axis:]
    strides += x.strides[axis:]
    # an empty view may start out of range: keep the pointer inside the array
    offset = b.select(empty, 0, offset)
    data = x.data if offset.const_int == 0 else b.gep_bytes(x.data, offset)
    return x.view(data, shape, strides)


def _entries(op, node, inputs, first):
    values = [
        (v, var.type.dtype)
        for v, var in zip(inputs[first:], node.inputs[first:], strict=True)
    ]
    return unflatten_index_variables(values, op.idx_list)


@llvm_emit.register(Subtensor)
def _(op, node, cg, inputs, outputs):
    view = index(cg, inputs[0], _entries(op, node, inputs, 1))
    view.layout = subtensor_layout(op, node, inputs[0].layout)
    set_output(cg, outputs[0], view)


@llvm_layout.register(Subtensor)
def _(op, node, in_layouts):
    return [subtensor_layout(op, node, in_layouts[0])]


@llvm_inplace.register(IncSubtensor)
def _(op, node):
    return {0: 0} if op.inplace else {}


@llvm_emit.register(IncSubtensor)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x, y = inputs[0], inputs[1]
    out = outputs[0].overwrite(cg, x)  # x itself if it may be destroyed, else a copy
    view = index(cg, out, _entries(op, node, inputs, 2))
    # y broadcasts against the view, aligned on the right, by its static type
    ytype = node.inputs[1].type
    off = view.ndim - ytype.ndim
    if off < 0:
        raise NotImplementedError("IncSubtensor: y has more axes than the indexed view")
    ystrides = [b.i64(0)] * off
    for a, s in enumerate(ytype.shape):
        va = off + a
        if s == 1:
            ystrides.append(b.i64(0))
        else:
            # like numpy, a length-1 axis of y broadcasts even if its type doesn't say so
            ln, vn = y.shape[a], view.shape[va]
            if ln.const_int is not None and ln.const_int == vn.const_int:
                ystrides.append(y.strides[a])
                continue
            one = b.eq(ln, 1)
            cg.check(
                b.or_(b.eq(ln, vn), one),
                "IncSubtensor: shape mismatch between the value and the indexed view",
            )
            ystrides.append(b.select(one, 0, y.strides[a]))
    ys = Array(y.dtype, y.data, view.shape, ystrides)
    dt = out.dtype

    def body(p):
        v = cast(b, load(b, y.dtype, p[1]), y.dtype, dt)
        if not op.set_instead_of_inc:
            v = emit_scalar(scalar_add, b, [(load(b, dt, p[0]), dt), (v, dt)], [dt])[0]
        store(b, dt, v, p[0])

    for_each(b, view.shape, [view, ys], body)


__all__ = ["index"]
