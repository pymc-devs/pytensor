"""Advanced indexing with integer index arrays: gathers and scatters.

Handled: integer index arrays (broadcast together) on adjacent axes, every other axis
taken in full (`x[:, idx]`, `x[i, j]`, `x[idx_a, idx_b, :]`). The indexed axes are
replaced by the broadcast shape of the indices, in place (as numpy does for adjacent
advanced indices). Other patterns (boolean masks, slices with bounds, separated index
arrays) aren't (`llvm_supports`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytensor.scalar.basic as ps
from pytensor.link.llvm.codegen.array import Array, advance, size
from pytensor.link.llvm.codegen.context import index_bounds
from pytensor.link.llvm.codegen.interface import (
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.num import cast, load, store
from pytensor.link.llvm.codegen.signature import overwritten_layout
from pytensor.link.llvm.dispatch.scalar import emit_scalar
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    unflatten_index_variables,
)


@dataclass
class Pattern:
    first: int  # first indexed axis of x
    indices: list  # (Array, PyTensor variable) of the index arrays, in axis order
    bshape: list  # broadcast shape of the indices (i64 values)
    view_shape: list  # shape of x[indices]
    bounds: list  # index_bounds of each index array


def index_arrays(op, index_vars) -> tuple[int, list[int]] | None:
    """Where the index arrays of an advanced indexing node are: (the axis of x the
    first one indexes, their positions in `index_vars`), or None for a pattern that
    isn't handled (bounded slices, non-integer or separated index arrays, none)."""
    entries = unflatten_index_variables(list(enumerate(index_vars)), op.idx_list)
    first = None
    found: list[int] = []
    for a, e in enumerate(entries):
        if isinstance(e, slice):
            if e != slice(None):
                return None
            continue
        k, var = e
        if var.type.dtype == "bool" or not var.type.dtype.startswith(("int", "uint")):
            return None
        if first is None:
            first = a
        elif a != first + len(found):
            return None
        found.append(k)
    return None if first is None else (first, found)


def _view_ndim(x_var, index_vars, found) -> int:
    """The number of axes of x[indices]."""
    index_ndim = max(int(index_vars[k].type.ndim) for k in found)
    return int(x_var.type.ndim) - len(found) + index_ndim


@llvm_supports.register(AdvancedSubtensor)
def _(op, node) -> bool:
    return supported_types(node) and index_arrays(op, node.inputs[1:]) is not None


@llvm_supports.register(AdvancedIncSubtensor)
def _(op, node) -> bool:
    found = index_arrays(op, node.inputs[2:])
    if not supported_types(node) or found is None:
        return False
    # y broadcasts against the indexed view: it can't have more axes
    view_ndim = _view_ndim(node.inputs[0], node.inputs[2:], found[1])
    return int(node.inputs[1].type.ndim) <= view_ndim


def pattern(cg, x: Array, node, index_values, index_vars, op) -> Pattern:
    b = cg.b
    found = index_arrays(op, index_vars)
    if found is None:
        raise ValueError(f"{op}: an indexing pattern `llvm_supports` refuses")
    first = found[0]
    indices = [(index_values[k], index_vars[k]) for k in found[1]]
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
            elif not (
                n.const_int is not None and n.const_int == bshape[off + d].const_int
            ):
                cg.check(
                    b.eq(n, bshape[off + d]),
                    "shape mismatch: indexing arrays could not be broadcast together",
                )
    bshape = [n if n is not None else b.i64(1) for n in bshape]
    k = len(indices)
    view_shape = list(x.shape[:first]) + bshape + list(x.shape[first + k :])
    bounds = [index_bounds(var) for _, var in indices]
    return Pattern(first, indices, bshape, view_shape, bounds)


def operands(cg, base: Array, pat: Pattern):
    """`base` with strides over the view's loop axes (0 on the index axes), and the index
    arrays with strides over the same axes."""
    b = cg.b
    zero = b.i64(0)
    k, nb = len(pat.indices), len(pat.bshape)
    first = pat.first
    bst = list(base.strides[:first]) + [zero] * nb + list(base.strides[first + k :])
    source = [*range(first), *[None] * nb, *range(first + k, base.ndim)]
    layout = base.view_layout(pat.view_shape, bst, source)
    base_op = Array(base.scalar_type, base.data, pat.view_shape, bst, layout)
    idx_ops = []
    for arr, var in pat.indices:
        off = nb - var.type.ndim
        st = [zero] * len(pat.view_shape)
        source_shape: list[int | None] = [None] * len(pat.view_shape)
        for d in range(var.type.ndim):
            if var.type.shape[d] != 1:
                st[first + off + d] = arr.strides[d]
                source_shape[first + off + d] = d
        layout = arr.view_layout(pat.view_shape, st, source_shape)
        idx_ops.append(Array(arr.scalar_type, arr.data, pat.view_shape, st, layout))
    return base_op, idx_ops


def element(cg, base: Array, pat: Pattern, base_ptr, idx_ptrs, what, defer=False):
    """Pointer to the indexed element of `base`: base_ptr plus the index offsets
    (negative indices wrap; out-of-range ones fail, when the loop is over with
    `defer`: see `Codegen.checked_index`)."""
    b = cg.b
    p = base_ptr
    for j, ((arr, var), ip) in enumerate(zip(pat.indices, idx_ptrs, strict=True)):
        axis = pat.first + j
        n = base.shape[axis]
        i = cast(b, load(b, arr.scalar_type, ip), arr.scalar_type, ps.int64)
        i = cg.checked_index(
            i, n, f"index out of bounds for axis {axis} ({what})", pat.bounds[j], defer
        )
        p = advance(b, p, i, base.strides[axis], base.scalar_type)
    return p


@llvm_emit.register(AdvancedSubtensor)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    pat = pattern(cg, x, node, inputs[1:], node.inputs[1:], op)
    out = outputs[0].allocate(pat.view_shape)
    base_op, idx_ops = operands(cg, x, pat)
    k = len(idx_ops)

    def body(p):
        src = element(cg, x, pat, p[0], p[1 : 1 + k], "read", defer=True)
        store(b, out.scalar_type, load(b, x.scalar_type, src), p[-1])

    # out-of-range indices fail after the loop, so that it can be vectorized; they are
    # clamped meanwhile, so an indexed axis must not be empty if anything is read
    reads = size(b, pat.view_shape)
    axes = range(pat.first, pat.first + k)
    messages = [f"index out of bounds for axis {axis} (read)" for axis in axes]
    for axis, message in zip(axes, messages, strict=True):
        cg.check(b.or_(b.ne(x.shape[axis], 0), b.eq(reads, 0)), message)
    with cg.deferred_checks(messages):
        cg.for_each(pat.view_shape, [base_op, *idx_ops, out], body)


@llvm_layout.register(AdvancedIncSubtensor)
def _(op, node, input_layouts):
    # x's, when the result can be written into it (else C)
    return [overwritten_layout(node, 0, input_layouts)]


@llvm_emit.register(AdvancedIncSubtensor)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x, y = inputs[0], inputs[1]
    out = outputs[0]
    if out.may_overwrite(x):
        result = out.overwrite(x)
    else:
        result = out.allocate(x.shape)
        result.copy_from(cg, x)
    pat = pattern(cg, result, node, inputs[2:], node.inputs[2:], op)
    base_op, idx_ops = operands(cg, result, pat)
    # y broadcasts against the indexed view, aligned on the right; like numpy, a
    # length-1 axis broadcasts even when its type doesn't say so
    vs = pat.view_shape
    yt = node.inputs[1].type
    off = len(vs) - yt.ndim
    zero = b.i64(0)
    yst = [zero] * off
    for a in range(yt.ndim):
        if yt.shape[a] == 1:
            yst.append(zero)
            continue
        ln, vn = y.shape[a], vs[off + a]
        if ln.const_int is not None and ln.const_int == vn.const_int:
            yst.append(y.strides[a])
            continue
        one = b.eq(ln, 1)
        cg.check(
            b.or_(b.eq(ln, vn), one),
            "shape mismatch: value array could not be broadcast to the indexing result",
        )
        yst.append(b.select(one, 0, y.strides[a]))
    source = [None] * off + list(range(yt.ndim))
    ys = Array(y.scalar_type, y.data, vs, yst, y.view_layout(vs, yst, source))
    k = len(idx_ops)
    dt = result.scalar_type

    if op.ignore_duplicates and not op.set_instead_of_inc:
        # as numpy's x[idx] += y: every x[idx] is read before any is written, so a
        # duplicate index gets one sum (the last), not all of them
        sums = cg.temporary(dt, vs)

        def gather(p):
            src = element(cg, result, pat, p[0], p[1 : 1 + k], "read")
            v = cast(b, load(b, y.scalar_type, p[-2]), y.scalar_type, dt)
            v = emit_scalar(ps.add, b, [(load(b, dt, src), dt), (v, dt)], [dt])[0]
            store(b, dt, v, p[-1])

        def scatter(p):
            dst = element(cg, result, pat, p[0], p[1 : 1 + k], "write")
            store(b, dt, load(b, dt, p[-1]), dst)

        cg.for_each(vs, [base_op, *idx_ops, ys, sums], gather)
        cg.for_each(vs, [base_op, *idx_ops, sums], scatter)
        return

    def body(p):
        dst = element(cg, result, pat, p[0], p[1 : 1 + k], "write")
        v = cast(b, load(b, y.scalar_type, p[-1]), y.scalar_type, dt)
        if not op.set_instead_of_inc:
            v = emit_scalar(ps.add, b, [(load(b, dt, dst), dt), (v, dt)], [dt])[0]
        store(b, dt, v, dst)

    # sequential, so duplicate indices accumulate (inc) or the last one wins (set)
    cg.for_each(vs, [base_op, *idx_ops, ys], body)
