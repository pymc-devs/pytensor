"""Basic indexing: Subtensor (a view) and IncSubtensor (set or add into a copy)."""

from __future__ import annotations

import numpy as np

import pytensor.scalar as ps
from pytensor.graph.basic import Constant
from pytensor.link.llvm.codegen.array import Array
from pytensor.link.llvm.codegen.num import cast, load, store
from pytensor.link.llvm.dispatch.scalar import emit_scalar
from pytensor.link.llvm.layout import DYNAMIC, Layout, Stride
from pytensor.link.llvm.planning.signature import overwritten_layout
from pytensor.link.llvm.registry import (
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.types import element_type
from pytensor.scalar.basic import add as scalar_add
from pytensor.tensor.subtensor import IncSubtensor, Subtensor, unflatten_index_variables


def _i64(b, v, scalar_type):
    return cast(b, v, scalar_type, ps.int64)


def index(cg, x: Array, entries) -> Array:
    """`x[entries]` as a view: `entries` hold (value, ScalarType) for integer indices
    and slices of (value, ScalarType) / None, like `unflatten_index_variables` returns them."""
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
            i = cg.checked_index(
                _i64(b, *e),
                n,
                f"index out of bounds for axis {axis}",
                (0, None) if e[1] in ps.uint_types else (None, None),
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
        (v, element_type(var.type))
        for v, var in zip(inputs[first:], node.inputs[first:], strict=True)
    ]
    return unflatten_index_variables(values, op.idx_list)


@llvm_layout.register(Subtensor)
def _(op, node, input_layouts) -> list[Layout | None]:
    """A view of the input. An integer index removes its axis; a slice that may not
    cover its whole axis, or has a step, makes the packed axes outside it no longer
    follow from the lengths inside them, and a step changes the axis's own stride."""
    x = input_layouts[0]
    xv = node.inputs[0]
    shape, itemsize = xv.type.shape, xv.type.numpy_dtype.itemsize
    out_shape = node.outputs[0].type.shape
    entries = unflatten_index_variables(node.inputs[1:], op.idx_list)
    demoted: set[int] = set()
    removed: list[int] = []
    stepped: dict[int, Stride] = {}
    for a, entry in enumerate(entries):
        if not isinstance(entry, slice):
            # The axis disappears: packed axes outside it now skip its length.
            removed.append(a)
            demoted.update(x.outer_in_block(a, shape))
            continue
        # The step: 1 if there is none, None if it is only known when running.
        if entry.step is None:
            step: int | None = 1
        elif isinstance(entry.step, Constant):
            step = int(np.asarray(entry.step.data))
        else:
            step = None
        if entry.start is None and entry.stop is None and step == 1:
            continue
        n_out = out_shape[a - len(removed)]
        if not (shape[a] is not None and n_out == shape[a]):
            demoted.update(x.outer_in_block(a, shape))
        if step != 1:
            stride = x.static_stride(a, shape, itemsize)
            stepped[a] = DYNAMIC if step is None or stride is None else stride * step
            demoted.update(x.outer_in_block(a, shape))
    strides = x.demoted_strides(demoted, shape, itemsize)
    for a, stride in stepped.items():
        strides[a] = stride
    keep = [a for a in range(x.ndim) if a not in removed]
    renumber = {a: k for k, a in enumerate(keep)}
    order = [renumber[a] for a in x.order if a in renumber]
    return [Layout.make(out_shape, order, [strides[a] for a in keep])]


@llvm_emit.register(Subtensor)
def _(op, node, cg, inputs, outputs):
    outputs[0].set_view(index(cg, inputs[0], _entries(op, node, inputs, 1)))


@llvm_supports.register(IncSubtensor)
def _(op, node) -> bool:
    # y broadcasts against the indexed view: it can't have more axes (an integer
    # index takes an axis away)
    scalars = sum(1 for entry in op.idx_list if not isinstance(entry, slice))
    view_ndim = node.inputs[0].type.ndim - scalars
    return supported_types(node) and node.inputs[1].type.ndim <= view_ndim


@llvm_layout.register(IncSubtensor)
def _(op, node, input_layouts) -> list[Layout | None]:
    # x's, when the result can be written into it (else C)
    return [overwritten_layout(node, 0, input_layouts)]


@llvm_emit.register(IncSubtensor)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x, y = inputs[0], inputs[1]
    out = outputs[0]
    if out.may_overwrite(x):
        result = out.overwrite(x)
    else:
        result = out.allocate(x.shape)
        result.copy_from(cg, x)
    view = index(cg, result, _entries(op, node, inputs, 2))
    # y broadcasts against the view, aligned on the right, by its static type
    ytype = node.inputs[1].type
    off = view.ndim - ytype.ndim
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
    source = [None] * off + list(range(ytype.ndim))
    ys = Array(
        y.scalar_type,
        y.data,
        view.shape,
        ystrides,
        y.view_layout(view.shape, ystrides, source),
    )
    dt = result.scalar_type

    def body(p):
        v = cast(b, load(b, y.scalar_type, p[1]), y.scalar_type, dt)
        if not op.set_instead_of_inc:
            v = emit_scalar(scalar_add, b, [(load(b, dt, p[0]), dt), (v, dt)], [dt])[0]
        store(b, dt, v, p[0])

    cg.for_each(view.shape, [view, ys], body)


__all__ = ["index"]
