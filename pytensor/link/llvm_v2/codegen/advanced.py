"""Advanced indexing with integer index arrays: gathers and scatters.

Handled: integer index arrays (broadcast together) on adjacent axes, every other axis
taken in full (`x[:, idx]`, `x[i, j]`, `x[idx_a, idx_b, :]`). The indexed axes are
replaced by the broadcast shape of the indices, in place (as numpy does for adjacent
advanced indices). Other patterns (boolean masks, slices with bounds, separated index
arrays) raise NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytensor.scalar.basic as ps
from pytensor.link.llvm_v2.codegen.elemwise import output_array
from pytensor.link.llvm_v2.codegen.emit import Array, cast, copy, for_each, load, mul, store
from pytensor.link.llvm_v2.codegen.function import OutMode, fresh_output, llvm_emit, llvm_outputs
from pytensor.link.llvm_v2.codegen.scalar import emit_scalar
from pytensor.link.llvm_v2.codegen.types import ir, repr_of
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedSubtensor, unflatten_index_variables


@dataclass
class Pattern:
    first: int  # first indexed axis of x
    indices: list  # (Array, PyTensor variable) of the index arrays, in axis order
    bshape: list  # broadcast shape of the indices (i64 values)
    view_shape: list  # shape of x[indices]


def pattern(cx, x: Array, node, index_values, index_vars, op) -> Pattern:
    b, m = cx.b, ir()
    entries = unflatten_index_variables(list(zip(index_values, index_vars, strict=True)), op.idx_list)
    first = None
    indices = []
    for a, e in enumerate(entries):
        if isinstance(e, slice):
            if e != slice(None):
                raise NotImplementedError("advanced indexing with bounded slices")
            if first is not None and len(indices) and a == first + len(indices):
                pass  # a slice after the indices
            continue
        arr, var = e
        if var.type.dtype == "bool" or not var.type.dtype.startswith(("int", "uint")):
            raise NotImplementedError("advanced indexing with non-integer index arrays")
        if first is None:
            first = a
        elif a != first + len(indices):
            raise NotImplementedError("advanced indexing with separated index arrays")
        indices.append((arr, var))
    if first is None:
        raise NotImplementedError("advanced indexing without index arrays")
    # broadcast the index shapes (right-aligned, as numpy)
    nb = max(v.type.ndim for _, v in indices)
    bshape = [None] * nb
    for arr, var in indices:
        off = nb - var.type.ndim
        for d in range(var.type.ndim):
            if var.type.shape[d] == 1:
                continue
            n = arr.shape[d]
            if bshape[off + d] is None:
                bshape[off + d] = n
            elif not (n.const_int is not None and n.const_int == bshape[off + d].const_int):
                cx.check(b.icmp(m.ICmp.EQ, n, bshape[off + d]), "shape mismatch: indexing arrays could not be broadcast together")
    bshape = [n if n is not None else b.const_int(m.i64, 1) for n in bshape]
    k = len(indices)
    view_shape = list(x.shape[:first]) + bshape + list(x.shape[first + k :])
    return Pattern(first, indices, bshape, view_shape)


def operands(cx, base: Array, pat: Pattern):
    """`base` with strides over the view's loop axes (0 on the index axes), and the index
    arrays with strides over the same axes."""
    b, m = cx.b, ir()
    zero = b.const_int(m.i64, 0)
    k, nb = len(pat.indices), len(pat.bshape)
    first = pat.first
    bst = list(base.strides[:first]) + [zero] * nb + list(base.strides[first + k :])
    base_op = Array(base.dtype, base.data, pat.view_shape, bst)
    idx_ops = []
    for arr, var in pat.indices:
        off = nb - var.type.ndim
        st = [zero] * len(pat.view_shape)
        for d in range(var.type.ndim):
            if var.type.shape[d] != 1:
                st[first + off + d] = arr.strides[d]
        idx_ops.append(Array(arr.dtype, arr.data, pat.view_shape, st))
    return base_op, idx_ops


def element(cx, base: Array, pat: Pattern, base_ptr, idx_ptrs, what):
    """Pointer to the indexed element of `base`: base_ptr plus the index offsets
    (negative indices wrap; out-of-range ones fail)."""
    b, m = cx.b, ir()
    zero = b.const_int(m.i64, 0)
    p = base_ptr
    for j, ((arr, var), ip) in enumerate(zip(pat.indices, idx_ptrs, strict=True)):
        axis = pat.first + j
        n = base.shape[axis]
        i = cast(b, load(b, arr.dtype, ip), arr.dtype, "int64")
        i = b.select(b.icmp(m.ICmp.SLT, i, zero), b.add(i, n), i)
        ok = b.and_(b.icmp(m.ICmp.SGE, i, zero), b.icmp(m.ICmp.SLT, i, n))
        cx.check(ok, f"index out of bounds for axis {axis} ({what})")
        p = b.gep_bytes(p, mul(b, i, base.strides[axis]))
    return p


@llvm_outputs.register(AdvancedSubtensor)
@llvm_outputs.register(AdvancedIncSubtensor)
def _(op, node, in_layouts):
    return [fresh_output(node.outputs[0])]


@llvm_emit.register(AdvancedSubtensor)
def _(op, node, cx, inputs, outputs):
    b = cx.b
    x = inputs[0]
    pat = pattern(cx, x, node, inputs[1:], node.inputs[1:], op)
    out = output_array(outputs[0], pat.view_shape)
    base_op, idx_ops = operands(cx, x, pat)
    k = len(idx_ops)

    def body(p):
        src = element(cx, x, pat, p[0], p[1 : 1 + k], "read")
        store(b, out.dtype, load(b, x.dtype, src), p[-1])

    for_each(b, pat.view_shape, [base_op, *idx_ops, out], body)


@llvm_emit.register(AdvancedIncSubtensor)
def _(op, node, cx, inputs, outputs):
    b, m = cx.b, ir()
    x, y = inputs[0], inputs[1]
    out = output_array(outputs[0], x.shape)
    copy(cx, x, out)
    pat = pattern(cx, out, node, inputs[2:], node.inputs[2:], op)
    base_op, idx_ops = operands(cx, out, pat)
    # y broadcasts against the indexed view, aligned on the right; like numpy, a
    # length-1 axis broadcasts even when its type doesn't say so
    vs = pat.view_shape
    yt = node.inputs[1].type
    off = len(vs) - yt.ndim
    if off < 0:
        raise NotImplementedError("advanced inc/set: the value has more axes than the indexed view")
    zero = b.const_int(m.i64, 0)
    yst = [zero] * off
    for a in range(yt.ndim):
        if yt.shape[a] == 1:
            yst.append(zero)
            continue
        ln, vn = y.shape[a], vs[off + a]
        if ln.const_int is not None and ln.const_int == vn.const_int:
            yst.append(y.strides[a])
            continue
        one = b.icmp(m.ICmp.EQ, ln, b.const_int(m.i64, 1))
        cx.check(b.or_(b.icmp(m.ICmp.EQ, ln, vn), one), "shape mismatch: value array could not be broadcast to the indexing result")
        yst.append(b.select(one, zero, y.strides[a]))
    ys = Array(y.dtype, y.data, vs, yst)
    k = len(idx_ops)
    dt = out.dtype

    def body(p):
        dst = element(cx, out, pat, p[0], p[1 : 1 + k], "write")
        v = cast(b, load(b, y.dtype, p[-1]), y.dtype, dt)
        if not op.set_instead_of_inc:
            v = emit_scalar(ps.add, b, [(load(b, dt, dst), dt), (v, dt)], [dt])[0]
        store(b, dt, v, dst)

    # sequential, so duplicate indices accumulate (inc) or the last one wins (set)
    for_each(b, vs, [base_op, *idx_ops, ys], body)
