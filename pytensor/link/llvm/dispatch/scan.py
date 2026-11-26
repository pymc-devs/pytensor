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
of a state's buffer), `_NitSotRow` (a row of a nit-sot buffer) or `_UntracedState`; a while-loop's
condition is `Captured`.
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
from typing import TYPE_CHECKING

import pytensor.scalar as ps
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, advance
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.destination import Captured, OutputDestination
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
    EmitterOutput,
    TensorOutput,
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.num import cast, constant, load, store
from pytensor.link.llvm.codegen.signature import overwritten_layout
from pytensor.link.llvm.codegen.types import element_type, ir, is_absent


if TYPE_CHECKING:
    from pytensor.graph.basic import Apply
    from pytensor.scan.op import Scan


def register():
    from pytensor.scan.op import Scan

    @llvm_layout.register(Scan)
    def _(op, node, input_layouts):
        # buffers allocated by the loop, C-contiguous (as the Rust kernel's), or the
        # initial values they can be written into
        return [
            None
            if is_absent(output.type)
            else overwritten_layout(node, k, input_layouts)
            or Layout.c(output.type.ndim)
            for k, output in enumerate(node.outputs)
        ]

    @llvm_supports.register(Scan)
    def _(op, node) -> bool:
        # sequences and the buffers of traced states have rows
        inputs = node.inputs
        with_rows = [
            *op.outer_seqs(inputs),
            *op.outer_mitmot(inputs),
            *op.outer_mitsot(inputs),
            *op.outer_sitsot(inputs),
        ]
        return supported_types(node) and all(var.type.ndim >= 1 for var in with_rows)

    @llvm_emit.register(Scan)
    def _(op, node, cg, inputs, outputs):
        emit_scan(op, node, cg, inputs, outputs)


# ------------------------------------------------------------------ the buffers


@dataclass
class _StateBuffer:
    """The buffer of a traced state (mit-mot, mit-sot, sit-sot): the outer output."""

    output: TensorOutput
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
        length = self.buffer.shape[0]
        n = b.smax(length, 1) if length.const_int is None else max(length.const_int, 1)
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

    output: TensorOutput
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
        for row, n in zip(rows, self.moduli, strict=True):
            row = b.add(row, 1)
            next_rows.append(b.select(b.eq(row, n), b.i64(0), row))
        return next_rows


def _as_i64(b, a, scalar_type):
    """An integer 0-d array (or scalar value) as i64."""
    v = load(b, scalar_type, a.data) if isinstance(a, Array) else a
    return cast(b, v, scalar_type, ps.int64)


def _row_range(b, buf: Array, start, count) -> Array:
    """Rows `start .. start + count` of `buf`."""
    p = advance(b, buf.data, start, buf.strides[0], buf.scalar_type)
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
    zero = constant(b, a.scalar_type, 0)
    cg.for_each(a.shape, [a], lambda p: store(b, a.scalar_type, zero, p[0]))


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
        reads = [int(t) for t in reads]
        lookback = -min(min(reads, default=0), 0)
        is_mit_mot = k < info.n_mit_mot
        if out.may_overwrite(init):
            # the outer buffer itself
            buffer = out.overwrite(init)
        else:
            buffer = out.allocate(list(init.shape))
            if is_mit_mot:
                buffer.copy_from(cg, init)
            else:
                # the loop only reads the initial rows before it writes a row; the
                # rows it doesn't write are zeroed after it
                n = b.smin(init.shape[0], lookback)
                source = _row_range(b, init, 0, n)
                _row_range(b, buffer, 0, n).copy_from(cg, source)
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
        n_rows = b.smax(_as_i64(b, length, element_type(length_var.type)), 0)
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


def _copy_untraced_states(cg, op, inputs, outputs) -> list[TensorOutput | None]:
    """The untraced states, copied into their outputs (None: no data, a random
    generator). Each iteration's output replaces them; an output of another shape is
    reallocated, so they are read (`current`) in every iteration."""
    untraced = []
    for init, out in zip(
        op.outer_untraced_sit_sot(inputs),
        op.outer_untraced_sit_sot_outs(outputs),
        strict=True,
    ):
        if out is not None:
            out.allocate(list(init.shape)).copy_from(cg, init)
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

    cg: Codegen
    row: Array
    #: i1: may the value be written directly into the row?
    in_place: ir.Value
    #: memory aside (the row's shape, C-contiguous), for when only the run time knows
    #: `in_place`
    aside: Array | None
    target: Array | None = None

    @property
    def scalar_type(self):
        return self.row.scalar_type

    def allocate(self, shape: list) -> Array:
        cg = self.cg
        message = "Scan: a body output has another shape than its rows"
        _check_shape(cg, shape, self.row.shape, message)
        known = self.in_place.const_int
        if known == 1:
            self.target = self.row
        elif known is None and self.aside is not None:
            self.target = _select(cg.b, self.in_place, self.row, self.aside)
        else:
            self.target = cg.temporary(self.row.scalar_type, list(self.row.shape))
        return self.target

    def finish(self) -> None:
        if self.target is None:
            raise ValueError("an output was never set")
        if self.target is self.row:
            return
        b = self.cg.b
        if self.in_place.const_int == 0:
            self.row.copy_from(self.cg, self.target)
            return
        with b.if_(b.not_(self.in_place)):
            self.row.copy_from(self.cg, self.target)


@dataclass
class _NitSotRow(OutputDestination):
    """Where the body writes a nit-sot output: its row of the nit-sot buffer.

    If the rows have no static shape, the buffer doesn't exist before the loop: the
    first iteration allocates it, with rows of the shape of its output. The outputs of
    all later iterations must have the same shape."""

    cg: Codegen
    nit: _NitSotBuffer
    #: the row it writes to
    row_index: ir.Value
    row: Array | None = None

    @property
    def scalar_type(self):
        return self.nit.output.scalar_type

    def allocate(self, shape: list) -> Array:
        b, nit = self.cg.b, self.nit
        buffer = nit.buffer
        if buffer is None:
            buffer = nit.output.allocate_once([nit.stored_rows, *shape])
        _check_shape(
            self.cg,
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

    cg: Codegen
    out: TensorOutput
    value: Array | None = None

    @property
    def scalar_type(self):
        return self.out.scalar_type

    def allocate(self, shape: list) -> Array:
        self.value = self.cg.temporary(self.out.scalar_type, shape)
        return self.value

    def finish(self) -> None:
        if self.value is None:
            raise ValueError("an output was never set")
        # (in new memory if its shape changed)
        self.out.reallocate(list(self.value.shape)).copy_from(self.cg, self.value)


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
            aside = cg.temporary(buffer.scalar_type, list(buffer.shape))
            aside.copy_from(cg, buffer)
            rest = b.sub(n, shift)
            tail = _row_range(b, aside, shift, rest)
            _row_range(b, buffer, 0, rest).copy_from(cg, tail)
            head = _row_range(b, aside, 0, shift)
            _row_range(b, buffer, rest, shift).copy_from(cg, head)
    with b.if_(b.and_(b.not_(circular), b.sgt(n, trace_len))):
        if is_while:
            out.truncate(trace_len)
        else:
            _fill_zero(cg, _row_range(b, buffer, trace_len, b.sub(n, trace_len)))


# ------------------------------------------------------------------ the scan


def _row_layout(a: Array) -> Layout:
    """The layout of one row of `a` (what `a.at(b, i)` has)."""
    return a.layout.remove_axis(0, a.static_shape(), a.itemsize)


def emit_scan(
    op: Scan,
    node: Apply,
    cg: Codegen,
    inputs: list[EmitterInput],
    outputs: list[EmitterOutput | None],
) -> None:
    b = cg.b
    info = op.info
    fgraph = op.fgraph
    # positions of the inner inputs and outputs
    ins = list(range(len(fgraph.inputs)))
    outs = list(range(len(fgraph.outputs)))

    n_steps = b.smax(_as_i64(b, inputs[0], element_type(node.inputs[0].type)), 0)
    sequences = op.outer_seqs(inputs)
    for k, seq in enumerate(sequences):
        cg.check(
            b.sge(seq.shape[0], n_steps), f"Scan: sequence {k} is shorter than n_steps"
        )
    non_sequences = op.outer_non_seqs(inputs)
    states = _allocate_state_buffers(cg, op, inputs, outputs)
    nit_sots = _allocate_nit_sot_buffers(cg, op, node, inputs, outputs)
    untraced = _copy_untraced_states(cg, op, inputs, outputs)

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
            body_layouts[k] = out.allocated_layout
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
                aside = cg.temporary(s.buffer.scalar_type, list(s.buffer.shape[1:]))
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
    cond_type = element_type(fgraph.outputs[-1].type) if info.as_while else None

    def step(i: ir.Value, *carried: ir.Value) -> list[ir.Value]:
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
                results[k] = _NitSotRow(cg, nit, counters.row(rows, nit.row_key()))
            for k_in, k_out, out in zip(
                untraced_in_at, untraced_out_at, untraced, strict=True
            ):
                if out is not None:
                    args[k_in] = out.current()
                    results[k_out] = _UntracedState(cg, out)
            # cond_type is only set for while-loops.
            if cond_type is not None:
                cond = Captured(cg, cond_type)
                results[-1] = cond

            body.emit_call(cg, args, results)

            if cond_type is not None:
                value = cond.scalar()
                stop = b.ne(cast(b, value, cond_type, ps.int8), 0)
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
            buffer = nit.output.allocate_once([nit.n_rows, *(b.i64(0),) * nit.row_ndim])
        _rotate_and_trim(cg, nit.output, buffer, steps_run, info.as_while)
        with b.if_(b.eq(nit.n_rows, 0)):
            nit.output.truncate(nit.n_rows)


__all__ = ["emit_scan", "register"]
