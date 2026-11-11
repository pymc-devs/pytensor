"""Elemwise, bare scalar ops and CAReduce."""

from __future__ import annotations

import numpy as np

import pytensor.scalar.basic as ps
from pytensor.link.llvm.codegen.emit import (
    advance,
    loop_order,
    loop_view,
    Array,
    ProvidedOutput,
    _strided,
    cast,
    constant,
    copy,
    for_each,
    is_contiguous,
    load,
    memory_type,
    size,
    store,
)
from pytensor.link.llvm.codegen.function import new_output, llvm_emit, llvm_outputs
from pytensor.link.llvm.layout import PACKED, Layout, common_order
from pytensor.link.llvm.work import scalar_cost
from pytensor.link.llvm.codegen.scalar import emit_scalar
from pytensor.link.llvm.codegen.types import check_dtype, ir
from pytensor.tensor.elemwise import CAReduce, Elemwise


def output_array(out, shape):
    """The memory of an output: the caller's, or freshly allocated."""
    if isinstance(out, ProvidedOutput):
        return out.array
    return out.allocate(shape)


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
            elif not (length.const_int is not None and length.const_int == ln.const_int):
                cg.check(
                    b.eq(length, ln),
                    f"Elemwise: shape mismatch: input {k} has a different length on axis {a}",
                )
        shape.append(length if length is not None else b.i64(1))
    return shape


@llvm_outputs.register(Elemwise)
def _(op, node, in_layouts):
    """New memory in the memory order most inputs are packed in (C if there is none):
    then all operands share one order and the loop is flat. Elemwise on transposed
    inputs gives a transposed output instead of a strided loop."""
    outs = []
    for o in node.outputs:
        shape = tuple(None if s is None else int(s) for s in o.type.shape)
        order = common_order(in_layouts, shape)
        lay = None if order is None else Layout.make(shape, order, [PACKED] * len(shape))
        outs.append(new_output(o, lay))
    return outs


@llvm_emit.register(Elemwise)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    shape = broadcast_shape(cg, node, inputs)
    outs = [output_array(o, shape) for o in outputs]
    in_dtypes = [check_dtype(v.type.dtype) for v in node.inputs]
    out_dtypes = [check_dtype(v.type.dtype) for v in node.outputs]
    # inputs that are one element for the whole loop are read once, before it
    fixed, operands, slot = [], [], []
    for var, x in zip(node.inputs, inputs, strict=True):
        if all(s == 1 for s in var.type.shape):
            fixed.append(load(b, x.dtype, x.data))
            slot.append(None)
        else:
            slot.append(len(operands))
            operands.append(loop_view(b, x, shape))
            fixed.append(None)
    n_in = len(operands)
    operands += outs

    def body(ptrs):
        args = [(fixed[k] if slot[k] is None else load(b, d, ptrs[slot[k]]), d) for k, d in enumerate(in_dtypes)]
        res = emit_scalar(op.scalar_op, b, args, out_dtypes)
        for k, (v, d) in enumerate(zip(res, out_dtypes, strict=True)):
            store(b, d, v, ptrs[n_in + k])

    for_each(b, shape, operands, body, cg.par(scalar_cost(op.scalar_op)))


@llvm_emit.register(ps.ScalarOp)
def _(op, node, cg, inputs, outputs):
    """A scalar op applied to ScalarType values."""
    args = [(v, check_dtype(var.type.dtype)) for v, var in zip(inputs, node.inputs, strict=True)]
    res = emit_scalar(op, cg.b, args, [check_dtype(o.type.dtype) for o in node.outputs])
    for out, v in zip(outputs, res, strict=True):
        out.set(cg.b, v)


# ------------------------------------------------------------------ CAReduce


NO_EMPTY_REDUCTION = (ps.Maximum, ps.Minimum, ps.ScalarMaximum, ps.ScalarMinimum)


def reduce_identity(op, acc_dtype):
    return identity_value(op.scalar_op, op.scalar_op.identity, acc_dtype)


def identity_value(scalar_op, ident, acc_dtype):
    """The start value of a reduction, representable in `acc_dtype`."""
    if ident is None:
        raise NotImplementedError(f"LLVM backend: reduction with {scalar_op} has no identity")
    ident = np.asarray(ident).item()
    dt = np.dtype(acc_dtype)
    if dt.kind in "iu" and not np.isfinite(ident):
        info = np.iinfo(dt)
        ident = info.min if ident < 0 else info.max
    return ident


@llvm_emit.register(CAReduce)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (out,) = outputs
    nd = x.ndim
    axes = sorted(range(nd) if op.axis is None else [a % nd for a in op.axis])
    reduced = [a in axes for a in range(nd)]
    out_dtype = check_dtype(node.outputs[0].type.dtype)
    acc = check_dtype(getattr(op, "acc_dtype", None) or out_dtype)
    init = constant(b, acc, reduce_identity(op, acc))

    def combine(a, v):
        return emit_scalar(op.scalar_op, b, [(a, acc), (cast(b, v, x.dtype, acc), acc)], [acc])[0]

    out_shape = [x.shape[a] for a in range(nd) if not reduced[a]]
    if isinstance(op.scalar_op, NO_EMPTY_REDUCTION):
        # like numpy: maximum/minimum have no identity for empty reductions
        empty = b.eq(size(b, [x.shape[a] for a in axes]), b.i64(0))
        some_out = b.ne(size(b, out_shape), b.i64(0))
        cg.check(b.xor(b.and_(empty, some_out), b.const_bool(True)), "zero-size array to reduction operation which has no identity")
    res = output_array(out, out_shape)

    if not out_shape:
        # full reduction: accumulate in a register
        total = reduce_all(b, x, init, combine)
        store(b, out_dtype, cast(b, total, acc, out_dtype), res.data)
        return

    # partial: accumulate into an array of the accumulator dtype
    acc_arr = res if acc == out_dtype else cg.temporary(acc, out_shape)
    for_each(b, out_shape, [acc_arr], lambda p: store(b, acc, init, p[0]))
    acc_strides, k = [], 0
    for r in reduced:
        if r:
            acc_strides.append(b.i64(0))
        else:
            acc_strides.append(acc_arr.strides[k])
            k += 1
    # loop in x's memory order; if its innermost memory axis is reduced, that axis is
    # accumulated in a register (for a transposed x and axis=0 that's the unit-stride one)
    order = loop_order(x.shape, [x])
    inner = order[-1]
    accs = Array(acc, acc_arr.data, x.shape, acc_strides)

    def body(p):
        cur = load(b, acc, p[1])
        if reduced[inner]:
            new = reduce_row(b, x, inner, p[0], cur, combine)
        else:
            new = combine(cur, load(b, x.dtype, p[0]))
        store(b, acc, new, p[1])

    loops = order[:-1] if reduced[inner] else order
    # the outermost loop may run in parallel when each iteration has its own
    # accumulators (it isn't over a reduced axis)
    par = cg.par(scalar_cost(op.scalar_op)) if loops and not reduced[loops[0]] else None
    if par is not None and not par.worth_it(size(b, x.shape)):
        par = None
    _strided(b, x.shape, [x, accs], body, loops, par=par)
    if acc != out_dtype:
        copy(cg, acc_arr, res)


def reduce_row(b, x: Array, axis, row, init, combine):
    """Combine the elements along `axis` of `x`, starting at `row`, into `init`. A unit
    stride gets its own loop: the vectorizer would otherwise use gathers."""
    n, st = x.shape[axis], x.strides[axis]
    et = memory_type(x.dtype)

    def unit():
        return b.fold(n, [init], lambda i, a: [combine(a, load(b, x.dtype, b.gep(et, row, i)))])

    def strided():
        return b.fold(n, [init], lambda i, a: [combine(a, load(b, x.dtype, advance(b, row, i, st, x.dtype)))])

    isz = b.i64(np.dtype(x.dtype).itemsize)
    if st.const_int is not None:
        return (unit() if st.const_int == isz.const_int else strided())[0]
    return b.if_else(b.eq(st, isz), unit, strided)[0]


def reduce_all(b, x: Array, init, combine):
    """Combine all elements of `x` into one value: a flat loop if `x` is contiguous in
    any axis order (the order doesn't matter here), a nest of folds otherwise (both,
    chosen at run time, when the layout doesn't settle it)."""

    def flat():
        (r,) = b.fold(size(b, x.shape), [init], lambda i, a: [combine(a, load(b, x.dtype, b.gep(memory_type(x.dtype), x.data, i)))])
        return [r]

    def nest():
        def level(axis, p, a):
            if axis == x.ndim:
                return combine(a, load(b, x.dtype, p))
            (r,) = b.fold(x.shape[axis], [a], lambda i, acc: [level(axis + 1, advance(b, p, i, x.strides[axis], x.dtype), acc)])
            return r

        return [level(0, x.data, init)]

    c = b.const_bool(True) if x.layout.fully_packed(x.static_shape()) else is_contiguous(b, x.shape, x)
    if c.const_int == 1:
        return flat()[0]
    if c.const_int == 0:
        return nest()[0]
    return b.if_else(c, flat, nest)[0]


__all__ = ["broadcast_shape", "reduce_all"]
