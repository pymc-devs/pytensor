"""Scan as generated code: the loop, its buffers and its body in one function.

The conventions are PyTensor's (and the Rust kernel's, interp/scan.rs). Every traced
state (mit-mot, mit-sot, sit-sot) has a buffer whose first `lookback` rows hold the
initial values; iteration `i` reads tap `t` at row `(i + lookback + t) % len` and writes
its output taps the same way. nit-sot outputs go to row `i % len`. After the loop,
buffers shorter than the trace are rotated (in place) so the last value is at the end,
while-loops are truncated to the iterations run, and unused rows of regular loops are
zeroed. nit-sot buffers whose rows have no static shape are allocated by the first
iteration.

A state's buffer is the outer buffer itself when the Scan may destroy it (`inplace`,
not for while-loops); else a new one, into which only the initial rows are copied.

The loop body is the inner graph's items, emitted (`emit_items`): nodes we generate are
inlined, the others called (kernels, numba, subprograms). Body outputs are produced
directly in their rows unless the row is one the iteration also reads (only in short
circular buffers; decided once, before the loop), else copied there. Outputs that may
be views of the states are copied aside first, before any row is overwritten. Buffers
kernels allocate and scratch memory the body takes are released after every iteration.
The rows the loop visits are counters carried through it (`_RowCounters`).
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.codegen.emit import (
    Array,
    InplaceOutput,
    SlotOutput,
    advance,
    cast,
    constant,
    copy,
    for_each,
    load,
    store,
)
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_inplace, llvm_layout
from pytensor.link.llvm.codegen.types import ir, is_absent, repr_of
from pytensor.link.llvm.codegen.unit import (
    Destination,
    bind_constants,
    emit_items,
    release,
)
from pytensor.link.llvm.layout import Layout, remove_axis


def register():
    from pytensor.scan.op import Scan

    @llvm_layout.register(Scan)
    def _(op, node, in_layouts):
        # buffers allocated by the loop, C-contiguous (as the Rust kernel's)
        return [
            None if is_absent(o.type) else Layout.c(o.type.ndim) for o in node.outputs
        ]

    @llvm_inplace.register(Scan)
    def _(op, node):
        # a traced state's buffer can be the outer one (output k destroys input
        # 1 + n_seqs + k); not in while-loops, whose outputs are truncated
        info = op.info
        if info.as_while:
            return {}
        n_traced = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
        return {k: 1 + info.n_seqs + k for k in range(n_traced)}

    @llvm_emit.register(Scan)
    def _(op, node, cg, inputs, outputs):
        if cg.body is None or cg.links is None or cg.module is None:
            raise NotImplementedError(f"generated {op}")
        emit_scan(op, node, cg, inputs, outputs)


# ------------------------------------------------------------------ the buffers


@dataclass
class _StateBuffer:
    """The buffer of a traced state (mit-mot, mit-sot, sit-sot): the outer output."""

    output: SlotOutput | InplaceOutput
    buffer: Array
    #: rows before the one iteration 0 reads as tap 0 (the initial values)
    lookback: int
    read_taps: list[int]
    write_taps: list[int]
    #: mit-mot buffers are returned as they are, the others rotated and trimmed
    is_mit_mot: bool

    @property
    def first_write_row(self) -> int:
        """The row of the trace iteration 0's last write tap is in: after `n` steps
        the trace has `n + first_write_row` rows."""
        return self.lookback + max(self.write_taps, default=0)

    def row_key(self, tap):
        """The key of the row of tap `tap` in the loop's `_RowCounters`."""
        return (id(self), tap)

    def add_rows(self, b, counters: _RowCounters):
        """Register the rows of its taps: tap `t` of iteration `i` is in row
        `(i + lookback + t) % len`."""
        n = self.buffer.shape[0]
        n = b.smax(n, 1) if n.const_int is None else max(n.const_int, 1)
        for tap in {*self.read_taps, *self.write_taps}:
            counters.add(b, self.row_key(tap), n, self.lookback + tap)

    def write_in_place(self, b, k: int):
        """i1: may the body produce write tap `k` straight into its row? Only if no
        other tap of the iteration (read or written) is in that row. The rows of taps
        `t` and `u` coincide when the buffer length divides `t - u`: only in circular
        buffers no longer than the taps reach. A constant if the length is static."""
        w = self.write_taps[k]
        others = [
            *self.read_taps,
            *(t for j, t in enumerate(self.write_taps) if j != k),
        ]
        distances = sorted({abs(w - t) for t in others})
        n = self.buffer.shape[0]
        if 0 in distances:
            return b.const_bool(False)
        if n.const_int is not None:
            return b.const_bool(
                n.const_int >= 1 and all(d % n.const_int for d in distances)
            )
        ok = b.sgt(n, 0)
        for d in distances:
            ok = b.and_(ok, b.ne(b.urem(b.i64(d), b.smax(n, 1)), 0))
        return ok


@dataclass
class _NitSotBuffer:
    """The buffer of a nit-sot output: `n_rows` rows (maybe fewer than the steps:
    written circularly), one more if `n_rows` is 0 (trimmed after the loop), so that
    every iteration has a row to write."""

    output: SlotOutput
    #: the length the output has (i64, >= 0)
    n_rows: object
    #: the rows the buffer has: `max(n_rows, 1)`; iteration `i` writes row
    #: `i % stored_rows`
    stored_rows: object
    #: None: the rows have no static shape, the first iteration allocates it
    buffer: Array | None
    row_ndim: int

    def row_key(self):
        """The key of its row in the loop's `_RowCounters`."""
        return id(self)


class _RowCounters:
    """The rows the loop visits, `(i + offset) % n` in iteration `i`: carried through
    the loop as counters that wrap around instead of a division per iteration."""

    def __init__(self):
        self._position = {}
        self.moduli = []
        #: the rows of iteration 0
        self.initial = []

    def add(self, b, key, n, offset: int):
        """Register row `(i + offset) % n` under `key`; `n` (i64 or int) is >= 1."""
        if key in self._position:
            return
        self._position[key] = len(self.moduli)
        if not isinstance(n, int) and n.const_int is not None:
            n = n.const_int
        if isinstance(n, int):
            self.initial.append(b.i64(offset % n))
            n = b.i64(n)
        else:
            self.initial.append(b.urem(b.i64(offset), n))
        self.moduli.append(n)

    def row(self, rows, key):
        """The row of `key`, of the current values `rows` (the carried ones)."""
        return rows[self._position[key]]

    def advance(self, b, rows) -> list:
        """The rows of the next iteration."""
        next_rows = []
        for r, n in zip(rows, self.moduli, strict=True):
            r = b.add(r, 1)
            next_rows.append(b.select(b.eq(r, n), b.i64(0), r))
        return next_rows


def _as_i64(b, a, dtype):
    """An integer 0-d array (or scalar value) as i64."""
    v = load(b, dtype, a.data) if isinstance(a, Array) else a
    return cast(b, v, dtype, "int64")


def _row_range(b, buf: Array, start, count) -> Array:
    """Rows `start .. start + count` of `buf`."""
    p = advance(b, buf.data, start, buf.strides[0], buf.dtype)
    c = buf.layout.is_c(buf.static_shape())
    layout = Layout.c(buf.ndim) if c else Layout.dynamic(buf.ndim)
    return buf.view(p, [count, *buf.shape[1:]], buf.strides, layout)


def _select(b, cond, a: Array, other: Array) -> Array:
    """`a` if `cond` else `other`: arrays of one shape."""
    strides = [
        b.select(cond, s, t) for s, t in zip(a.strides, other.strides, strict=True)
    ]
    return a.view(b.select(cond, a.data, other.data), a.shape, strides, a.layout)


def _fill_zero(cg, a: Array):
    b = cg.b
    zero = constant(b, a.dtype, 0)
    for_each(b, a.shape, [a], lambda p: store(b, a.dtype, zero, p[0]))


def _allocate_state_buffers(cg, op, inputs, outputs) -> list[_StateBuffer]:
    """Buffers of the traced states: the outer buffers if the Scan may destroy them,
    else new ones holding their initial values."""
    b = cg.b
    info = op.info
    initial = [
        *op.outer_mitmot(inputs),
        *op.outer_mitsot(inputs),
        *op.outer_sitsot(inputs),
    ]
    outs = [
        *op.outer_mitmot_outs(outputs),
        *op.outer_mitsot_outs(outputs),
        *op.outer_sitsot_outs(outputs),
    ]
    read_taps = [
        *info.mit_mot_in_slices,
        *info.mit_sot_in_slices,
        *info.sit_sot_in_slices,
    ]
    write_taps = [*info.mit_mot_out_slices] + [(0,)] * (info.n_mit_sot + info.n_sit_sot)
    states = []
    for k, (init, out, reads, writes) in enumerate(
        zip(initial, outs, read_taps, write_taps, strict=True)
    ):
        if init.ndim < 1:
            raise NotImplementedError("Scan: state buffers need at least one dimension")
        reads = [int(t) for t in reads]
        lookback = -min(min(reads, default=0), 0)
        is_mit_mot = k < info.n_mit_mot
        if isinstance(out, InplaceOutput):
            buffer = out.overwrite(cg, init)
        else:
            buffer = out.allocate(list(init.shape))
            if is_mit_mot:
                copy(cg, init, buffer)
            else:
                # the loop only reads the initial rows before it writes a row; the
                # rows it doesn't write are zeroed after it
                n = b.smin(init.shape[0], lookback)
                copy(cg, _row_range(b, init, 0, n), _row_range(b, buffer, 0, n))
        states.append(
            _StateBuffer(
                output=out,
                buffer=buffer,
                lookback=lookback,
                read_taps=reads,
                write_taps=[int(t) for t in writes],
                is_mit_mot=is_mit_mot,
            )
        )
    return states


def _allocate_nit_sot_buffers(cg, op, node, inputs, outputs) -> list[_NitSotBuffer]:
    """Buffers of the nit-sot outputs, allocated now if their rows have a static
    shape."""
    b = cg.b
    nit_sots = []
    for length, length_var, out, inner in zip(
        op.outer_nitsot(inputs),
        op.outer_nitsot(node.inputs),
        op.outer_nitsot_outs(outputs),
        op.inner_nitsot_outs(op.fgraph.outputs),
        strict=True,
    ):
        n_rows = b.smax(_as_i64(b, length, length_var.type.dtype), 0)
        nit = _NitSotBuffer(
            output=out,
            n_rows=n_rows,
            stored_rows=b.smax(n_rows, 1),
            buffer=None,
            row_ndim=inner.type.ndim,
        )
        if all(s is not None for s in inner.type.shape):
            row_shape = [b.i64(s) for s in inner.type.shape]
            nit.buffer = out.allocate([nit.stored_rows, *row_shape])
        nit_sots.append(nit)
    return nit_sots


def _copy_untraced_states(cg, op, inputs, outputs) -> list[SlotOutput | None]:
    """The untraced states, copied into their outputs (None: no data, a random
    generator). Each iteration's output replaces them; an output of another shape is
    reallocated, so they are read from the slot in every iteration."""
    untraced = []
    for init, out in zip(
        op.outer_untraced_sit_sot(inputs),
        op.outer_untraced_sit_sot_outs(outputs),
        strict=True,
    ):
        if out is not None:
            copy(cg, init, out.allocate(list(init.shape)))
        untraced.append(out)
    return untraced


# ------------------------------------------------------------------ the loop body


def _may_view_states(body, state_vids) -> set:
    """The vids of the body's values that may be views of the memory of the states
    (their inputs, and what views of them may be views of)."""
    views = set(state_vids)
    for r in body.items:
        present = [
            vid
            for vid, var in zip(r.item.inputs, r.item.node.inputs, strict=True)
            if not is_absent(var.type)
        ]
        for vid, o in zip(r.item.outputs, r.signature.outputs, strict=True):
            if any(a is not None and present[a] in views for a in o.aliases):
                views.add(vid)
    return views


def _store_nit_sot_row_of_dynamic_shape(cg, nit: _NitSotBuffer, i, row, src: Array):
    """Write the output of iteration `i` into row `row` of a nit-sot whose buffer the
    first iteration allocates (its rows have no static shape); all rows must have the
    same shape."""
    b = cg.b
    with b.if_(b.eq(i, 0)):
        nit.output.allocate([nit.stored_rows, *src.shape])
    buffer = nit.output.read()
    for d, n in enumerate(src.shape):
        cg.check(
            b.eq(n, buffer.shape[1 + d]),
            "Scan: nit-sot body outputs have different shapes",
        )
    copy(cg, src, buffer.at(b, row))


def _replace_untraced_state(cg, out: SlotOutput, current: Array, src: Array):
    """Make `src` the untraced state: copied into its memory, or into new memory if
    its shape differs."""
    b = cg.b
    differs = b.const_bool(False)
    for n, m in zip(src.shape, current.shape, strict=True):
        differs = b.or_(differs, b.ne(n, m))
    if differs.const_int == 0:
        copy(cg, src, current)
        return
    old_owner = b.alloca(ir.ptr)
    b.store(b.null(), old_owner)
    with b.if_(differs):
        b.store(out.owner(), old_owner)
        out.allocate(list(src.shape))
    copy(cg, src, out.read())
    b.rt_decref(cg.ctx, b.load(ir.ptr, old_owner))


# ------------------------------------------------------------------ after the loop


def _rotate_and_trim(cg, out, buffer: Array, trace_len, is_while: bool):
    """Put the rows of a buffer that held `trace_len` rows of a trace in order: rotate
    a circular one (shorter than the trace) so the last row is at the end; truncate
    (while-loops) or zero (regular loops) the rows past the trace of a longer one."""
    b = cg.b
    n = buffer.shape[0]
    circular = b.and_(b.sgt(n, 1), b.slt(n, trace_len))
    with b.if_(circular):
        shift = b.urem(trace_len, n)
        with b.if_(b.sgt(shift, 0)):
            # in place, through a copy in temporary memory
            aside = cg.temporary(buffer.dtype, list(buffer.shape))
            copy(cg, buffer, aside)
            rest = b.sub(n, shift)
            copy(cg, _row_range(b, aside, shift, rest), _row_range(b, buffer, 0, rest))
            copy(cg, _row_range(b, aside, 0, shift), _row_range(b, buffer, rest, shift))
    with b.if_(b.and_(b.not_(circular), b.sgt(n, trace_len))):
        if is_while:
            out.set_length(0, trace_len)
        else:
            _fill_zero(cg, _row_range(b, buffer, trace_len, b.sub(n, trace_len)))


# ------------------------------------------------------------------ the scan


def scan_body_layouts(op, node, in_layouts) -> list:
    """The layouts the inner graph's inputs have in the generated loop (`emit_scan`),
    one per inner input (None for values without one): rows of the sequences keep what
    the sequence's layout says, rows of the state buffers are C-contiguous if the outer
    buffer is (the loop's own buffers are; the outer buffer may be used instead), the
    rest is passed through."""
    from pytensor.link.llvm.layout import itemsize_of, shape_of

    info = op.info
    outer = list(zip(node.inputs, in_layouts, strict=True))

    def layout_of(var, layout):
        if is_absent(var.type):
            return None
        return getattr(repr_of(var.type, layout), "layout", None)  # (scalars: none)

    layouts = []
    for var, layout in op.outer_seqs(outer):
        if var.type.ndim < 1:
            raise NotImplementedError("a 0-d sequence")
        layouts.append(
            remove_axis(layout_of(var, layout), 0, shape_of(var), itemsize_of(var))
        )
    states = [*op.outer_mitmot(outer), *op.outer_mitsot(outer), *op.outer_sitsot(outer)]
    read_taps = [
        *info.mit_mot_in_slices,
        *info.mit_sot_in_slices,
        *info.sit_sot_in_slices,
    ]
    for (var, layout), taps in zip(states, read_taps, strict=True):
        ndim = var.type.ndim
        if ndim < 1:
            raise NotImplementedError("Scan: state buffers need at least one dimension")
        outer_layout = layout_of(var, layout)
        c = outer_layout is not None and outer_layout.is_c(shape_of(var))
        layouts += [Layout.c(ndim - 1) if c else Layout.dynamic(ndim - 1)] * len(taps)
    for var, _ in op.outer_untraced_sit_sot(outer):
        layouts.append(None if is_absent(var.type) else Layout.c(var.type.ndim))
    for var, layout in op.outer_non_seqs(outer):
        layouts.append(layout_of(var, layout))
    return layouts


def emit_scan(op, node, cg, inputs, outputs):
    b = cg.b
    info = op.info
    body = cg.body
    in_vids, out_vids = body.in_ids, body.out_ids

    n_steps = b.smax(_as_i64(b, inputs[0], node.inputs[0].type.dtype), 0)
    sequences = op.outer_seqs(inputs)
    for k, seq in enumerate(sequences):
        if seq.ndim < 1:
            raise NotImplementedError("a 0-d sequence")
        cg.check(
            b.sge(seq.shape[0], n_steps), f"Scan: sequence {k} is shorter than n_steps"
        )
    non_sequences = op.outer_non_seqs(inputs)
    states = _allocate_state_buffers(cg, op, inputs, outputs)
    nit_sots = _allocate_nit_sot_buffers(cg, op, node, inputs, outputs)
    untraced = _copy_untraced_states(cg, op, inputs, outputs)

    # the inner inputs and outputs, per kind (in the order of `states`' taps)
    state_read_vids = [
        *op.inner_mitmot(in_vids),
        *op.inner_mitsot(in_vids),
        *op.inner_sitsot(in_vids),
    ]
    state_reads = [(s, tap) for s in states for tap in s.read_taps]
    state_write_vids = [
        *op.inner_mitmot_outs(out_vids),
        *op.inner_mitsot_outs(out_vids),
        *op.inner_sitsot_outs(out_vids),
    ]
    state_writes = [(s, k) for s in states for k in range(len(s.write_taps))]
    untraced_in_vids = op.inner_untraced_sit_sot(in_vids)
    untraced_out_vids = op.inner_untraced_sit_sot_outs(out_vids)
    nit_sot_vids = op.inner_nitsot_outs(out_vids)
    may_view_states = _may_view_states(body, [*state_read_vids, *untraced_in_vids])

    # whether the body may produce each write tap straight into its row, decided once.
    # If only the run time decides, it writes into `aside` when it may not.
    write_plans = []  # (in_place i1, aside Array or None)
    for s, k in state_writes:
        in_place = s.write_in_place(b, k)
        aside = None
        if in_place.const_int is None:
            if s.buffer.layout.is_c(s.buffer.static_shape()):
                aside = cg.temporary(s.buffer.dtype, list(s.buffer.shape[1:]))
            else:
                in_place = b.const_bool(False)
        write_plans.append((in_place, aside))

    counters = _RowCounters()
    for s in states:
        s.add_rows(b, counters)
    for nit in nit_sots:
        counters.add(b, nit.row_key(), nit.stored_rows, 0)
    # carried through the loop: (the steps taken, for a while-loop), the rows
    n_counts = 1 if info.as_while else 0

    constants = {}
    bind_constants(cg, body.constants, constants)

    # scratch memory the body takes is released after every iteration (to a mark
    # taken anew each time: a release ends its mark)
    scratch_mark = b.alloca(ir.i64)
    b.store(b.rt_scratch_mark(cg.ctx), scratch_mark)

    def step(i, *carried):
        rows = carried[n_counts:]
        uses_scratch, cg.uses_scratch = cg.uses_scratch, False
        vals = dict(constants)

        # the inner inputs
        for vid, seq in zip(op.inner_seqs(in_vids), sequences, strict=True):
            vals[vid] = seq.at(b, i)
        for vid, (s, tap) in zip(state_read_vids, state_reads, strict=True):
            vals[vid] = s.buffer.at(b, counters.row(rows, s.row_key(tap)))
        untraced_now = []
        for vid, out in zip(untraced_in_vids, untraced, strict=True):
            current = None if out is None else out.read()
            untraced_now.append(current)
            if current is not None:
                vals[vid] = current
        vals.update(
            (vid, a)
            for vid, a in zip(op.inner_non_seqs(in_vids), non_sequences, strict=True)
            if a is not None
        )

        # the rows the outputs go to; where they may be produced directly
        out_rows = []  # (vid, row, target or None, in_place)
        targets = {}
        for vid, (s, k), (in_place, aside) in zip(
            state_write_vids, state_writes, write_plans, strict=True
        ):
            row = s.buffer.at(b, counters.row(rows, s.row_key(s.write_taps[k])))
            target = None
            if in_place.const_int != 0:
                target = row if aside is None else _select(b, in_place, row, aside)
                targets.setdefault(vid, Destination(array=target, layout=row.layout))
            out_rows.append((vid, row, target, in_place))
        for vid, nit in zip(nit_sot_vids, nit_sots, strict=True):
            if nit.buffer is not None:
                row = nit.buffer.at(b, counters.row(rows, nit.row_key()))
                out_rows.append((vid, row, row, None))
                targets.setdefault(vid, Destination(array=row, layout=row.layout))

        written, owned = emit_items(
            cg, cg.module, f"{cg.name}_body", body.items, vals, targets
        )

        # outputs that may be views of the states are copied aside before any state
        # memory is overwritten
        results = {}
        for vid in out_vids:
            if vid in results:
                continue
            v = vals[vid]
            if vid in may_view_states and isinstance(v, Array):
                aside = cg.temporary(v.dtype, list(v.shape))
                copy(cg, v, aside)
                v = aside
            results[vid] = v

        for vid, row, target, in_place in out_rows:
            if not (vid in written and targets[vid].array is target):
                copy(cg, results[vid], row)
            elif target is not row:
                # produced aside where the row was one the iteration read
                with b.if_(b.not_(in_place)):
                    copy(cg, target, row)
        for vid, nit in zip(nit_sot_vids, nit_sots, strict=True):
            if nit.buffer is None:
                row = counters.row(rows, nit.row_key())
                _store_nit_sot_row_of_dynamic_shape(cg, nit, i, row, results[vid])
        for vid, out, current in zip(
            untraced_out_vids, untraced, untraced_now, strict=True
        ):
            if out is not None:
                _replace_untraced_state(cg, out, current, results[vid])

        if info.as_while:
            cond = results[out_vids[-1]]
            if isinstance(cond, Array):
                cond = load(b, cond.dtype, cond.data)
            dtype = op.fgraph.outputs[-1].type.dtype
            stop = b.ne(cast(b, cond, dtype, "int8"), 0)
        release(cg, owned)
        if cg.uses_scratch:
            b.rt_scratch_release(cg.ctx, b.load(ir.i64, scratch_mark))
            b.store(b.rt_scratch_mark(cg.ctx), scratch_mark)
        cg.uses_scratch |= uses_scratch
        next_rows = counters.advance(b, rows)
        if info.as_while:
            return [b.add(carried[0], 1), *next_rows, stop]
        return next_rows

    if info.as_while:
        init = [b.i64(0), *counters.initial]
        steps_run = b.fold(n_steps, init, step, until=True)[0]
    else:
        b.fold(n_steps, counters.initial, step)
        steps_run = n_steps
    b.rt_scratch_release(cg.ctx, b.load(ir.i64, scratch_mark))

    for s in states:
        if not s.is_mit_mot:
            trace_len = b.add(steps_run, s.first_write_row)
            _rotate_and_trim(cg, s.output, s.buffer, trace_len, info.as_while)
    for nit in nit_sots:
        buffer = nit.buffer
        if buffer is None:
            # allocated by the first iteration; if none ran: an empty array
            with b.if_(b.eq(nit.output.data(), b.null())):
                nit.output.allocate([nit.n_rows, *(b.i64(0),) * nit.row_ndim])
            buffer = nit.output.read()
        _rotate_and_trim(cg, nit.output, buffer, steps_run, info.as_while)
        with b.if_(b.eq(nit.n_rows, 0)):
            nit.output.set_length(0, nit.n_rows)


__all__ = ["emit_scan", "register", "scan_body_layouts"]
