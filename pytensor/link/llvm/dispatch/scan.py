"""Scan as generated code: the loop, its buffers and its body in one function.

The conventions are PyTensor's. Every traced state (mit-mot, mit-sot, sit-sot) has a
buffer whose first `lookback` rows hold the initial values; iteration `i` reads tap `t`
at row `(i + lookback + t) % len` and writes its output taps the same way. nit-sot outputs go to row `i % len`. After the loop,
buffers shorter than the trace are rotated (in place) so the last value is at the end,
while-loops are truncated to the iterations run, and unused rows of regular loops are
zeroed. nit-sot buffers whose rows have no static shape are allocated by the first
iteration.

A state's buffer is the outer buffer itself when the Scan may destroy it (`inplace`,
not for while-loops); else a new one, into which only the initial rows are copied.

Every iteration runs the inner graph, the loop body (`Codegen.graph_function`). Nodes
we generate are inlined, the others are calls of their Rust or numba kernels. The body
is specialized for the layouts of what the loop passes it: rows of the sequences, rows
of the buffers, and the non-sequences.

Each output of the body has an `OutputDestination` that says where it goes: `_StateRow` (a row
of a state's buffer), `_NitSotRow` (a row of a nit-sot buffer) or `_UntracedState`.
An output is written directly into its row, unless the iteration also reads that row.
That only happens in short circular buffers, and is decided once, before the loop. Then
the output is written to memory aside and copied into the row when the iteration is
done. So nothing the iteration reads is overwritten before the iteration is done.

Memory that kernels allocate in the body, and scratch memory the body takes, are
released after every iteration. The rows the loop visits are counters carried from one
iteration to the next (`_RowCounters`).
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.analysis.layout import Layout, remove_axis
from pytensor.link.llvm.codegen.destination import (
    OutputDestination,
    ScalarDestination,
    SlotDestination,
)
from pytensor.link.llvm.codegen.emit import (
    Array,
    Codegen,
    advance,
    cast,
    constant,
    copy,
    for_each,
    load,
    store,
)
from pytensor.link.llvm.codegen.interface import (
    destinations,
    llvm_emit,
    llvm_layout,
)
from pytensor.link.llvm.codegen.types import ir, is_absent, itemsize, value_type


def register():
    from pytensor.scan.op import Scan

    @llvm_layout.register(Scan)
    def _(op, node, input_layouts):
        # buffers allocated by the loop, C-contiguous (as the Rust kernel's)
        return [
            None if is_absent(output.type) else Layout.c(output.type.ndim)
            for output in node.outputs
        ]

    @llvm_emit.register(Scan)
    def _(op, node, cg, inputs, outputs):
        emit_scan(op, node, cg, inputs, outputs)


# ------------------------------------------------------------------ the buffers


@dataclass
class _StateBuffer:
    """The buffer of a traced state (mit-mot, mit-sot, sit-sot): the outer output."""

    output: OutputDestination
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

    output: SlotDestination
    #: the length the output has (i64, >= 0)
    n_rows: ir.Value
    #: the rows the buffer has: `max(n_rows, 1)`; iteration `i` writes row
    #: `i % stored_rows`
    stored_rows: ir.Value
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
        # the outer buffer itself, if the Scan may destroy it and the planner lets it
        buffer = out.buffer(init)
        if not out.reuses(init):
            if is_mit_mot:
                copy(cg, init, buffer)
            else:
                # the loop only reads the initial rows before it writes a row; the
                # rows it doesn't write are zeroed after it
                n = b.smin(init.shape[0], lookback)
                copy(cg, _row_range(b, init, 0, n), _row_range(b, buffer, 0, n))
        states.append(
            _StateBuffer(
                output=out.destination,
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


def _copy_untraced_states(cg, op, inputs, outputs) -> list[SlotDestination | None]:
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


# ------------------------------------------------------------------ the body's outputs


def _check_shape(cg, shape, row_shape, message):
    for n, m in zip(shape, row_shape, strict=True):
        cg.check(cg.b.eq(n, m), message)


@dataclass
class _StateRow(OutputDestination):
    """Where the body writes a new value of a traced state (one of its write taps).

    The value goes into its row of the state's buffer. If the iteration may also read
    that row (`_StateBuffer.write_in_place`), the value is written to memory aside
    instead, and copied into the row when the iteration is done (`finish`)."""

    codegen: Codegen
    row: Array
    #: i1: may the value be written directly into the row?
    in_place: ir.Value
    #: memory aside (the row's shape, C-contiguous), for when only the run time knows
    #: `in_place`
    aside: Array | None
    target: Array | None = None

    @property
    def dtype(self):
        return self.row.dtype

    def allocate(self, shape: list) -> Array:
        cg = self.codegen
        message = "Scan: a body output has another shape than its rows"
        _check_shape(cg, shape, self.row.shape, message)
        known = self.in_place.const_int
        if known == 1:
            self.target = self.row
        elif known is None and self.aside is not None:
            self.target = _select(cg.b, self.in_place, self.row, self.aside)
        else:
            self.target = cg.temporary(self.row.dtype, list(self.row.shape))
        return self.target

    def finish(self) -> None:
        if self.target is None:
            raise ValueError("an output was never set")
        if self.target is self.row:
            return
        b = self.codegen.b
        if self.in_place.const_int == 0:
            copy(self.codegen, self.target, self.row)
            return
        with b.if_(b.not_(self.in_place)):
            copy(self.codegen, self.target, self.row)


@dataclass
class _NitSotRow(OutputDestination):
    """Where the body writes a nit-sot output: its row of the nit-sot buffer.

    If the rows have no static shape, the buffer doesn't exist before the loop: the
    first iteration allocates it, with rows of the shape of its output. The outputs of
    all later iterations must have the same shape."""

    codegen: Codegen
    nit: _NitSotBuffer
    #: the iteration, and the row it writes to
    i: ir.Value
    row_index: ir.Value
    row: Array | None = None

    @property
    def dtype(self):
        return self.nit.output.dtype

    def allocate(self, shape: list) -> Array:
        b, nit = self.codegen.b, self.nit
        buffer = nit.buffer
        if buffer is None:
            with b.if_(b.eq(self.i, 0)):
                nit.output.allocate([nit.stored_rows, *shape])
            buffer = nit.output.read()
        _check_shape(
            self.codegen,
            shape,
            buffer.shape[1:],
            "Scan: nit-sot body outputs have different shapes",
        )
        self.row = buffer.at(b, self.row_index)
        return self.row

    def finish(self) -> None:
        if self.row is None:
            raise ValueError("an output was never set")


@dataclass
class _UntracedState(OutputDestination):
    """Where the body writes the next value of an untraced state.

    The value goes into temporary memory. It replaces the state when the iteration is
    done (`finish`): until then the iteration may still read the current value."""

    codegen: Codegen
    out: SlotDestination
    current: Array
    value: Array | None = None

    @property
    def dtype(self):
        return self.out.dtype

    def allocate(self, shape: list) -> Array:
        self.value = self.codegen.temporary(self.out.dtype, shape)
        return self.value

    def finish(self) -> None:
        if self.value is None:
            raise ValueError("an output was never set")
        _replace_untraced_state(self.codegen, self.out, self.current, self.value)


def _replace_untraced_state(cg, out: SlotDestination, current: Array, src: Array):
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
        with b.if_(b.sgt(shift, 0)), cg.scratch_scope():
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


def _row_layout(a: Array) -> Layout:
    """The layout of one row of `a` (what `a.at(b, i)` has)."""
    return remove_axis(a.layout, 0, a.static_shape(), itemsize(a.dtype))


def emit_scan(op, node, cg, inputs, outputs):
    b = cg.b
    info = op.info
    fgraph = op.fgraph
    # positions of the inner inputs and outputs
    ins = list(range(len(fgraph.inputs)))
    outs = list(range(len(fgraph.outputs)))

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
    # (nit-sot outputs and untraced states are worked on in their slots)
    nit_sots = _allocate_nit_sot_buffers(cg, op, node, inputs, destinations(outputs))
    untraced = _copy_untraced_states(cg, op, inputs, destinations(outputs))

    # the inner inputs and outputs, per kind (in the order of `states`' taps)
    state_reads_at = [
        *op.inner_mitmot(ins),
        *op.inner_mitsot(ins),
        *op.inner_sitsot(ins),
    ]
    state_reads = [(s, tap) for s in states for tap in s.read_taps]
    state_writes_at = [
        *op.inner_mitmot_outs(outs),
        *op.inner_mitsot_outs(outs),
        *op.inner_sitsot_outs(outs),
    ]
    state_writes = [(s, k) for s in states for k in range(len(s.write_taps))]
    untraced_in_at = op.inner_untraced_sit_sot(ins)
    untraced_out_at = op.inner_untraced_sit_sot_outs(outs)
    nit_sot_at = op.inner_nitsot_outs(outs)

    # the body, for the layouts of what the loop passes it
    body_layouts: list = [None] * len(ins)
    for k, seq in zip(op.inner_seqs(ins), sequences, strict=True):
        body_layouts[k] = _row_layout(seq)
    for k, (s, _) in zip(state_reads_at, state_reads, strict=True):
        body_layouts[k] = _row_layout(s.buffer)
    for k, out in zip(untraced_in_at, untraced, strict=True):
        if out is not None:
            body_layouts[k] = out.allocation_layout or Layout.c(out.ndim)
    for k, a in zip(op.inner_non_seqs(ins), non_sequences, strict=True):
        if isinstance(a, Array):
            body_layouts[k] = a.layout
    body = cg.graph_function(fgraph.inputs, fgraph.outputs, body_layouts, name="body")

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
    cond_dtype = fgraph.outputs[-1].type.dtype

    def step(i, *carried):
        rows = carried[n_counts:]
        # the body's scratch memory and buffers are given up after every iteration
        with cg.scratch_scope():
            args: list = [None] * len(ins)
            for k, seq in zip(op.inner_seqs(ins), sequences, strict=True):
                args[k] = seq.at(b, i)
            for k, (s, tap) in zip(state_reads_at, state_reads, strict=True):
                args[k] = s.buffer.at(b, counters.row(rows, s.row_key(tap)))
            for k, a in zip(op.inner_non_seqs(ins), non_sequences, strict=True):
                args[k] = a

            results: list = [None] * len(outs)
            for k, (s, j), (in_place, aside) in zip(
                state_writes_at, state_writes, write_plans, strict=True
            ):
                row = s.buffer.at(b, counters.row(rows, s.row_key(s.write_taps[j])))
                results[k] = _StateRow(cg, row, in_place, aside)
            for k, nit in zip(nit_sot_at, nit_sots, strict=True):
                results[k] = _NitSotRow(cg, nit, i, counters.row(rows, nit.row_key()))
            for k_in, k_out, out in zip(
                untraced_in_at, untraced_out_at, untraced, strict=True
            ):
                if out is not None:
                    current = out.read()
                    args[k_in] = current
                    results[k_out] = _UntracedState(cg, out, current)
            if info.as_while:
                cond = ScalarDestination(
                    cg, b.alloca(value_type(cond_dtype)), cond_dtype
                )
                results[-1] = cond

            body.emit_call(cg, args, results)

            if info.as_while:
                value = b.load(value_type(cond_dtype), cond.pointer)
                stop = b.ne(cast(b, value, cond_dtype, "int8"), 0)
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


__all__ = ["emit_scan", "register"]
