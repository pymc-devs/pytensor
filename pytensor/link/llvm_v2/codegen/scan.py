"""Scan as generated code: the loop, its buffers and its body in one function.

The conventions are PyTensor's (and the Rust kernel's, rt/scan.rs): every traced output
(mit-mot, mit-sot, sit-sot) has a buffer whose first rows hold the initial taps; the
body reads tap `k` at row `(i + lookback + k) % len` and writes at
`(i + lookback + out_tap) % len`. After the loop, buffers shorter than the trace are
rotated so the last value is at the end, while-loops are truncated to the iterations run,
and unused rows of regular loops are zeroed. nit-sot buffers whose rows have no static
shape are allocated by the first iteration.

The loop body is the inner graph's items, emitted (`emit_items`): nodes we
generate are inlined, the others called (kernels, numba, subprograms). Body outputs
are produced directly in their rows when that is safe (a static buffer length that
can't make the row coincide with a row the body reads), else copied there. Buffers
kernels allocate and scratch memory are released after every iteration.
"""

from __future__ import annotations

from pytensor.link.llvm_v2.codegen.emit import Array, cast, constant, copy, for_each, load, store, advance
from pytensor.link.llvm_v2.codegen.function import OutMode, Output, llvm_emit, llvm_outputs
from pytensor.link.llvm_v2.codegen.types import ir, is_absent, repr_of
from pytensor.link.llvm_v2.codegen.unit import Target, bind_constants, emit_items, release
from pytensor.link.llvm_v2.layout import Layout, remove_axis


def scan_supported(op) -> bool:
    """Can we generate this Scan? (Every Scan so far: nodes we can't generate are
    called.)"""
    return True


def _n_inner_traced_outs(info):
    return sum(len(t) for t in info.mit_mot_out_slices) + len(info.mit_sot_in_slices) + len(info.sit_sot_in_slices)


def register():
    from pytensor.scan.op import Scan

    @llvm_outputs.register(Scan)
    def _(op, node, in_layouts):
        # buffers allocated by the loop, C-contiguous (as the Rust kernel's)
        return [
            Output(repr_of(o.type), OutMode.ABSENT)
            if is_absent(o.type)
            else Output(repr_of(o.type, Layout.c(o.type.ndim)), OutMode.ALLOCATED)
            for o in node.outputs
        ]

    @llvm_emit.register(Scan)
    def _(op, node, cg, inputs, outputs):
        if cg.planner is None or cg.links is None or cg.module is None or not scan_supported(op):
            raise NotImplementedError(f"generated {op}")
        emit_scan(op, node, cg, inputs, outputs)


def _i64(b, a: Array | object, dtype):
    """An integer 0-d array (or scalar value) as i64."""
    v = load(b, dtype, a.data) if isinstance(a, Array) else a
    return cast(b, v, dtype, "int64")


def _smax0(b, v):
    m = ir()
    zero = b.const_int(m.i64, 0)
    return b.select(b.icmp(m.ICmp.SGT, v, zero), v, zero)


def _rows(b, buf: Array, start, count):
    """Rows `start .. start + count` of a C-contiguous buffer."""
    p = advance(b, buf.data, start, buf.strides[0], buf.dtype)
    return Array(buf.dtype, p, [count, *buf.shape[1:]], list(buf.strides), Layout.c(len(buf.shape)))


def _row(b, buf: Array, r, layout=None):
    """Row `r` of `buf` (C-contiguous unless `layout` says what it is)."""
    p = advance(b, buf.data, r, buf.strides[0], buf.dtype)
    nd = len(buf.shape)
    return Array(buf.dtype, p, list(buf.shape[1:]), list(buf.strides[1:]), layout if layout is not None else Layout.c(nd - 1))


def _fill_zero(cg, a: Array):
    b = cg.b
    zero = constant(b, a.dtype, 0)
    for_each(b, a.shape, [a], lambda p: store(b, a.dtype, zero, p[0]))


class _Traced:
    def __init__(self, out, buf, lookback, in_taps, out_taps, post_process):
        self.out = out  # SlotOutput (the outer output)
        self.buf = buf  # Array: the buffer
        self.lookback = lookback
        self.in_taps = in_taps
        self.out_taps = out_taps
        self.post_process = post_process

    def static_len(self):
        return self.buf.shape[0].const_int

    def row_index(self, b, i, tap):
        m = ir()
        n = self.buf.shape[0]
        off = b.add(i, b.const_int(m.i64, self.lookback + tap))
        if n.const_int is not None:
            return b.urem(off, b.const_int(m.i64, max(n.const_int, 1)))
        one = b.const_int(m.i64, 1)
        return b.urem(off, b.select(b.icmp(m.ICmp.SGT, n, one), n, one))

    def direct_write_ok(self, k_out):
        """Can the body produce out tap `k_out` straight into its row? Only if that row
        is never a row the same iteration reads (a static length decides it)."""
        n = self.static_len()
        if n is None or n < 1:
            return False
        w = self.lookback + self.out_taps[k_out]
        reads = {(self.lookback + t) % n for t in self.in_taps}
        others = {(self.lookback + t) % n for j, t in enumerate(self.out_taps) if j != k_out}
        return (w % n) not in reads and (w % n) not in others


def emit_scan(op, node, cg, inputs, outputs):
    b, m = cg.b, ir()
    info = op.info
    planner = cg.planner
    i64 = m.i64
    n_mm, n_ms, n_ss = len(info.mit_mot_in_slices), len(info.mit_sot_in_slices), len(info.sit_sot_in_slices)
    o_seq = 1
    o_mm = o_seq + info.n_seqs
    o_ms = o_mm + n_mm
    o_ss = o_ms + n_ms
    o_ut = o_ss + n_ss
    o_nit = o_ut + info.n_untraced_sit_sot
    o_ns = o_nit + info.n_nit_sot
    dt = [getattr(v.type, "dtype", None) for v in node.inputs]

    n_steps = _smax0(b, _i64(b, inputs[0], dt[0]))

    # ---- sequences: long enough?
    seqs = inputs[o_seq:o_mm]
    for k, a in enumerate(seqs):
        if len(a.shape) < 1:
            raise NotImplementedError("a 0-d sequence")
        cg.check(b.icmp(m.ICmp.SGE, a.shape[0], n_steps), f"Scan: sequence {k} is shorter than n_steps")

    # ---- traced buffers: copies of the outer buffers, allocated into the outputs
    q = 0  # next outer output
    traced = []
    groups = [(o_mm, info.mit_mot_in_slices, True), (o_ms, info.mit_sot_in_slices, False), (o_ss, info.sit_sot_in_slices, False)]
    for base, taps, is_mm in groups:
        for j, in_taps in enumerate(taps):
            a = inputs[base + j]
            if len(a.shape) < 1:
                raise NotImplementedError("Scan: state buffers need at least one dimension")
            out = outputs[q]
            q += 1
            buf = out.allocate(list(a.shape))
            copy(cg, a, buf)
            in_taps = [int(t) for t in in_taps]
            lookback = -min(min(in_taps, default=0), 0)
            out_taps = [int(t) for t in info.mit_mot_out_slices[j]] if is_mm else [0]
            traced.append(_Traced(out, buf, lookback, in_taps, out_taps, not is_mm))

    # ---- nit-sot buffers: allocated now if their rows have a static shape
    inner_outs = op.fgraph.outputs
    first_nit = _n_inner_traced_outs(info)
    nit = []  # (out SlotOutput, length, buf Array or None, inner output var)
    for j in range(info.n_nit_sot):
        out = outputs[q + j]
        length = _smax0(b, _i64(b, inputs[o_nit + j], dt[o_nit + j]))
        var = inner_outs[first_nit + j]
        if all(s is not None for s in var.type.shape):
            buf = out.allocate([length, *(b.const_int(i64, s) for s in var.type.shape)])
        else:
            buf = None
        nit.append([out, length, buf, var])
    q += info.n_nit_sot
    # ---- untraced states: copies, replaced by every iteration's output (reallocated when
    # its shape changes, so read from the output slot in every iteration)
    untraced = []
    for j in range(info.n_untraced_sit_sot):
        out = outputs[q + j]
        a = inputs[o_ut + j]
        if out is None:  # no data (a random generator)
            untraced.append(None)
            continue
        copy(cg, a, out.allocate(list(a.shape)))
        untraced.append(out)

    # ---- the loop body's graph
    fg = op.fgraph
    scope = {}
    in_ids = []
    for v in fg.inputs:
        k = planner.new_value(v.type)
        scope[v] = k
        in_ids.append(k)
    items = planner.flatten(fg.toposort(), scope, inside_unit=True)
    out_ids = [planner.value_of(o, scope) for o in fg.outputs]
    consts = {}
    for it in items:
        for k in it.inputs:
            if k in planner.consts:
                consts[k] = (planner.types[k], planner.consts[k])
    for k in out_ids:
        if k in planner.consts:
            consts[k] = (planner.types[k], planner.consts[k])
    n_traced_outs = first_nit

    def body(i, *acc):
        mark = b.rt_scratch_mark(cg.ctx)
        vals, lays = {}, {}
        bind_constants(cg, consts, vals)
        vids = iter(in_ids)
        for a in seqs:
            k = next(vids)
            r = _row(b, a, i, remove_axis(a.layout, 0, a.static_shape(), _isz(a.dtype)))
            vals[k], lays[k] = r, r.layout
        for t in traced:
            for tap in t.in_taps:
                k = next(vids)
                r = _row(b, t.buf, t.row_index(b, i, tap))
                vals[k], lays[k] = r, r.layout
        cur_untraced = []
        for out in untraced:
            k = next(vids)
            cur = None if out is None else out.read()
            cur_untraced.append(cur)
            if cur is not None:
                vals[k], lays[k] = cur, cur.layout
        for a in inputs[o_ns:]:
            k = next(vids)
            if a is not None:
                vals[k] = a
                lays[k] = a.layout if isinstance(a, Array) else None
        # where outputs may be produced directly
        targets = {}
        rows = []  # per inner output: (buffer row Array or None, how)
        oi = 0
        for t in traced:
            for kk, tap in enumerate(t.out_taps):
                r = _row(b, t.buf, t.row_index(b, i, tap))
                rows.append(r)
                if t.direct_write_ok(kk):
                    targets.setdefault(out_ids[oi], Target(array=r, layout=r.layout))
                oi += 1
        for out, length, buf, var in nit:
            if buf is not None:
                r = _row(b, buf, i)
                rows.append(r)
                targets.setdefault(out_ids[oi], Target(array=r, layout=r.layout))
            else:
                rows.append(None)
            oi += 1

        written, owned = emit_items(cg, cg.module, f"{cg.name}_body", items, vals, lays, targets, planner.kernel_for)

        def value(k):
            if k not in vals:
                raise ValueError(f"Scan body: no value for output vid {k}")
            return vals[k]

        # copy what wasn't produced in place into its row
        for oi in range(n_traced_outs):
            k = out_ids[oi]
            t = targets.get(k)
            if k in written and t is not None and t.array is rows[oi]:
                continue
            copy(cg, value(k), rows[oi])
        for j, (out, length, buf, var) in enumerate(nit):
            k = out_ids[n_traced_outs + j]
            src = value(k)
            if buf is not None:
                t = targets.get(k)
                if not (k in written and t is not None and t.array is rows[n_traced_outs + j]):
                    copy(cg, src, rows[n_traced_outs + j])
                continue
            # rows without a static shape: the buffer is allocated by the first iteration
            first = b.icmp(m.ICmp.EQ, i, b.const_int(i64, 0))
            with b.if_(first):
                out.allocate([length, *src.shape])
            cur = out.read()
            for d, n in enumerate(src.shape):
                cg.check(b.icmp(m.ICmp.EQ, n, cur.shape[1 + d]), "Scan: nit-sot body outputs have different shapes")
            copy(cg, src, _row(b, cur, i))
        for out, cur, k in zip(untraced, cur_untraced, out_ids[n_traced_outs + info.n_nit_sot :], strict=False):
            if out is None:
                continue
            src = value(k)
            differs = b.const_bool(False)
            for d, n in enumerate(src.shape):
                differs = b.or_(differs, b.icmp(m.ICmp.NE, n, cur.shape[d]))
            if differs.const_int == 0:
                copy(cg, src, cur)
                continue
            # a new shape: a new buffer (the old one is released after the copy: the
            # output may be a view of it)
            old = b.alloca(m.ptr)
            b.store(b.null(), old)
            with b.if_(differs):
                b.store(out.owner(), old)
                out.allocate(list(src.shape))
            copy(cg, src, out.read())
            b.rt_decref(cg.ctx, b.load(m.ptr, old))
        stop = None
        if info.as_while:
            c = value(out_ids[-1])
            v = load(b, c.dtype, c.data) if isinstance(c, Array) else c
            stop = b.icmp(m.ICmp.NE, cast(b, v, fg.outputs[-1].type.dtype, "int8"), b.const_int(m.i8, 0))
        release(cg, owned)
        b.rt_scratch_release(cg.ctx, mark)
        if info.as_while:
            return [b.add(acc[0], b.const_int(i64, 1)), stop]
        return []

    if info.as_while:
        res = b.fold(n_steps, [b.const_int(i64, 0)], body, until=True)
        taken = res[0]
    else:
        b.fold(n_steps, [], body)
        taken = n_steps

    # ---- post-processing
    def finish(out, buf, max_offset):
        n = buf.shape[0]
        used = b.add(taken, b.const_int(i64, max_offset))
        one = b.const_int(i64, 1)
        rotate = b.and_(b.icmp(m.ICmp.SGT, n, one), b.icmp(m.ICmp.SLT, n, used))
        with b.if_(rotate):
            shift = b.urem(used, n)
            with b.if_(b.icmp(m.ICmp.SGT, shift, b.const_int(i64, 0))):
                old_owner = out.owner()
                nb = out.allocate(list(buf.shape))
                rest = b.sub(n, shift)
                copy(cg, _rows(b, buf, shift, rest), _rows(b, nb, b.const_int(i64, 0), rest))
                copy(cg, _rows(b, buf, b.const_int(i64, 0), shift), _rows(b, nb, rest, shift))
                b.rt_decref(cg.ctx, old_owner)
        tail = b.and_(b.xor(rotate, b.const_bool(True)), b.icmp(m.ICmp.SGT, n, used))
        with b.if_(tail):
            if info.as_while:
                out.set_length(0, used)
            else:
                _fill_zero(cg, _rows(b, buf, used, b.sub(n, used)))

    for t in traced:
        if t.post_process:
            finish(t.out, t.buf, t.lookback + max(t.out_taps, default=0))
    for out, length, buf, var in nit:
        if buf is None:
            # allocated by the first iteration; if none ran: an empty array
            none_ran = b.icmp(m.ICmp.EQ, out.data(), b.null())
            with b.if_(none_ran):
                out.allocate([length, *(b.const_int(i64, 0) for _ in range(var.type.ndim))])
            buf = out.read()
        finish(out, buf, 0)


def _isz(dtype):
    import numpy as np

    return np.dtype(dtype).itemsize


__all__ = ["emit_scan", "register", "scan_supported"]
