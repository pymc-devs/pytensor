"""Elemwise, bare scalar ops and CAReduce."""

from __future__ import annotations

import numpy as np

import pytensor.scalar.basic as ps
from pytensor.link.llvm.analysis.layout import PACKED, Layout, common_order
from pytensor.link.llvm.analysis.work import scalar_cost
from pytensor.link.llvm.codegen.array import Array, advance, size
from pytensor.link.llvm.codegen.context import loop_order
from pytensor.link.llvm.codegen.interface import (
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.num import cast, constant, load, store
from pytensor.link.llvm.codegen.signature import overwritten_layout
from pytensor.link.llvm.codegen.types import element_type, memory_type
from pytensor.link.llvm.dispatch.scalar import emit_scalar, scalar_supported
from pytensor.tensor.elemwise import CAReduce, Elemwise


def broadcast_shape(cg, node, inputs):
    """The loop shape of an Elemwise. Broadcasting follows the static types (PyTensor
    doesn't broadcast at run time): non-broadcastable lengths must agree."""
    b = cg.b
    out_shape = node.outputs[0].type.shape
    shape = []
    for a, static in enumerate(out_shape):
        length = b.i64(static) if static is not None else None
        for k, (var, x) in enumerate(zip(node.inputs, inputs, strict=True)):
            if var.type.shape[a] == 1:
                continue
            ln = x.shape[a]
            if length is None:
                length = ln
            elif not (
                length.const_int is not None and length.const_int == ln.const_int
            ):
                cg.check(
                    b.eq(length, ln),
                    f"Elemwise: shape mismatch: input {k} has a different length on axis {a}",
                )
        shape.append(length if length is not None else b.i64(1))
    return shape


@llvm_layout.register(Elemwise)
def _(op, node, input_layouts):
    """New memory in the memory order most inputs are packed in (C if there is none):
    then all operands share one order and the loop is flat. Elemwise on transposed
    inputs gives a transposed output instead of a strided loop. Outputs in
    `inplace_pattern` have their input's layout, so that they can be written into it."""
    layouts = []
    for k, output in enumerate(node.outputs):
        overwritten = overwritten_layout(node, k, input_layouts)
        if overwritten is not None:
            layouts.append(overwritten)
            continue
        shape = tuple(None if s is None else int(s) for s in output.type.shape)
        order = common_order(input_layouts, shape)
        if order is None:
            layouts.append(None)
        else:
            layouts.append(Layout.make(shape, order, [PACKED] * len(shape)))
    return layouts


@llvm_emit.register(Elemwise)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    shape = broadcast_shape(cg, node, inputs)
    # Every element is read before it is written, so an output can be the memory of
    # the input it updates.
    outs = []
    for k, output in enumerate(outputs):
        updated = inputs[op.inplace_pattern[k]] if k in op.inplace_pattern else None
        if updated is not None and output.may_overwrite(updated):
            outs.append(output.overwrite(updated))
        else:
            outs.append(output.allocate(shape))
    in_types = [element_type(v.type) for v in node.inputs]
    out_types = [element_type(v.type) for v in node.outputs]
    # inputs that are one element for the whole loop are read once, before it
    fixed, operands, slot = [], [], []
    for var, x in zip(node.inputs, inputs, strict=True):
        if all(s == 1 for s in var.type.shape):
            fixed.append(load(b, x.scalar_type, x.data))
            slot.append(None)
        else:
            slot.append(len(operands))
            operands.append(x.loop_view(b, shape))
            fixed.append(None)
    n_in = len(operands)
    operands += outs

    def body(ptrs):
        args = [
            (fixed[k] if slot[k] is None else load(b, d, ptrs[slot[k]]), d)
            for k, d in enumerate(in_types)
        ]
        res = emit_scalar(op.scalar_op, b, args, out_types)
        for k, (v, d) in enumerate(zip(res, out_types, strict=True)):
            store(b, d, v, ptrs[n_in + k])

    cg.for_each(shape, operands, body, parallel_cost=scalar_cost(op.scalar_op))


@llvm_supports.register(Elemwise)
@llvm_supports.register(ps.ScalarOp)
def _(op, node) -> bool:
    scalar_op = op.scalar_op if isinstance(op, Elemwise) else op
    return supported_types(node) and scalar_supported(
        scalar_op, [element_type(var.type) for var in node.outputs]
    )


@llvm_emit.register(ps.ScalarOp)
def _(op, node, cg, inputs, outputs):
    """A scalar op applied to ScalarType values."""
    args = [
        (v, element_type(var.type)) for v, var in zip(inputs, node.inputs, strict=True)
    ]
    res = emit_scalar(op, cg.b, args, [element_type(var.type) for var in node.outputs])
    for out, v in zip(outputs, res, strict=True):
        out.set(v)


# ------------------------------------------------------------------ CAReduce


NO_EMPTY_REDUCTION = (ps.Maximum, ps.Minimum, ps.ScalarMaximum, ps.ScalarMinimum)


def reduce_identity(op, acc_type):
    return identity_value(op.scalar_op, op.scalar_op.identity, acc_type)


def identity_value(scalar_op, ident, acc_type):
    """The start value of a reduction, representable in `acc_type`."""
    if ident is None:
        raise NotImplementedError(
            f"LLVM backend: reduction with {scalar_op} has no identity"
        )
    ident = np.asarray(ident).item()
    dt = np.dtype(acc_type.dtype)
    if dt.kind in "iu" and not np.isfinite(ident):
        info = np.iinfo(dt)
        ident = info.min if ident < 0 else info.max
    return ident


@llvm_supports.register(CAReduce)
def _(op, node) -> bool:
    if not supported_types(node) or op.scalar_op.identity is None:
        return False
    out_type = element_type(node.outputs[0].type)
    try:
        acc = element_type(getattr(op, "acc_dtype", None) or out_type)
    except NotImplementedError:
        return False
    return scalar_supported(op.scalar_op, [acc])


@llvm_emit.register(CAReduce)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (out,) = outputs
    nd = x.ndim
    axes = sorted(range(nd) if op.axis is None else [a % nd for a in op.axis])
    reduced = [a in axes for a in range(nd)]
    out_type = element_type(node.outputs[0].type)
    acc = element_type(getattr(op, "acc_dtype", None) or out_type)
    init = constant(b, acc, reduce_identity(op, acc))

    def combine(a, v):
        return emit_scalar(
            op.scalar_op, b, [(a, acc), (cast(b, v, x.scalar_type, acc), acc)], [acc]
        )[0]

    out_shape = [x.shape[a] for a in range(nd) if not reduced[a]]
    if isinstance(op.scalar_op, NO_EMPTY_REDUCTION):
        # like numpy: maximum/minimum have no identity for empty reductions
        empty = b.eq(size(b, [x.shape[a] for a in axes]), 0)
        some_out = b.ne(size(b, out_shape), 0)
        cg.check(
            b.not_(b.and_(empty, some_out)),
            "zero-size array to reduction operation which has no identity",
        )
    res = out.allocate(out_shape)

    if not out_shape:
        # full reduction: accumulate in a register
        total = reduce_all(b, x, init, combine)
        store(b, out_type, cast(b, total, acc, out_type), res.data)
        return

    # partial: accumulate into an array of the accumulator dtype
    acc_arr = res if acc == out_type else cg.temporary(acc, out_shape)
    cg.for_each(out_shape, [acc_arr], lambda p: store(b, acc, init, p[0]))
    acc_strides, source, k = [], [], 0
    for is_reduced in reduced:
        if is_reduced:
            acc_strides.append(b.i64(0))
            source.append(None)
        else:
            acc_strides.append(acc_arr.strides[k])
            source.append(k)
            k += 1
    # loop in x's memory order; if its innermost memory axis is reduced, that axis is
    # accumulated in a register (for a transposed x and axis=0 that's the unit-stride one)
    order = loop_order(x.shape, [x])
    inner = order[-1]
    layout = acc_arr.view_layout(x.shape, acc_strides, source)
    accs = Array(acc, acc_arr.data, x.shape, acc_strides, layout)

    def body(p):
        cur = load(b, acc, p[1])
        if reduced[inner]:
            new = reduce_row(b, x, inner, p[0], cur, combine)
        else:
            new = combine(cur, load(b, x.scalar_type, p[0]))
        store(b, acc, new, p[1])

    loops = order[:-1] if reduced[inner] else order
    # the outermost loop may run in parallel when each iteration has its own
    # accumulators (it isn't over a reduced axis)
    cost = scalar_cost(op.scalar_op) if loops and not reduced[loops[0]] else None
    cg.loop_nest(x.shape, [x, accs], body, loops, parallel_cost=cost)
    if acc != out_type:
        res.copy_from(cg, acc_arr)


def reduce_row(b, x: Array, axis, row, init, combine):
    """Combine the elements along `axis` of `x`, starting at `row`, into `init`. A unit
    stride gets its own loop: the vectorizer would otherwise use gathers."""
    n, st = x.shape[axis], x.strides[axis]
    et = memory_type(x.scalar_type)

    def unit():
        return b.fold(
            n,
            [init],
            lambda i, a: [combine(a, load(b, x.scalar_type, b.gep(et, row, i)))],
        )

    def strided():
        return b.fold(
            n,
            [init],
            lambda i, a: [
                combine(
                    a, load(b, x.scalar_type, advance(b, row, i, st, x.scalar_type))
                )
            ],
        )

    isz = np.dtype(x.scalar_type.dtype).itemsize
    if st.const_int is not None:
        return (unit() if st.const_int == isz else strided())[0]
    return b.if_else(b.eq(st, isz), unit, strided)[0]


def reduce_all(b, x: Array, init, combine):
    """Combine all elements of `x` into one value: a flat loop if `x` is contiguous in
    any axis order (the order doesn't matter here), a nest of folds otherwise (both,
    chosen at run time, when the layout doesn't settle it)."""

    def flat():
        (result,) = b.fold(
            size(b, x.shape),
            [init],
            lambda i, a: [
                combine(
                    a,
                    load(
                        b, x.scalar_type, b.gep(memory_type(x.scalar_type), x.data, i)
                    ),
                )
            ],
        )
        return [result]

    def nest():
        def level(axis, p, a):
            if axis == x.ndim:
                return combine(a, load(b, x.scalar_type, p))
            (result,) = b.fold(
                x.shape[axis],
                [a],
                lambda i, acc: [
                    level(
                        axis + 1, advance(b, p, i, x.strides[axis], x.scalar_type), acc
                    )
                ],
            )
            return result

        return [level(0, x.data, init)]

    c = (
        b.const_bool(True)
        if x.layout.fully_packed(x.static_shape())
        else x.is_contiguous(b, x.shape)
    )
    if c.const_int == 1:
        return flat()[0]
    if c.const_int == 0:
        return nest()[0]
    return b.if_else(c, flat, nest)[0]


__all__ = ["broadcast_shape", "reduce_all"]
