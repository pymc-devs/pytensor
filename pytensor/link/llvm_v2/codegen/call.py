"""Calls from generated code to functions with the C ABI (DESIGN.md, section 2a).

Rust kernels, numba implementations and nested programs all have the C ABI the
interpreter calls (section 6). Generated code calls them the same way: it packs the
arguments into `PtArray`s, calls the function through its handle and reads back what
the callee wrote. A node implemented by such a function then has a node function like
any other (`build_node_function(..., callee=...)`), so units and generated loops can
contain it.

The handles (function address and the callee's env) aren't part of the IR, which must
not depend on the process (the compile cache is keyed on it): they are in the calling
function's `env`, two pointers per callee, in the order of a `Closure`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pytensor.link.llvm_v2.codegen.types import PTARRAY_SIZE, ir, memory_type, ptarray_type, slot_type


@dataclass
class Callee:
    """A function with the C ABI that generated code calls for a node."""

    address: int
    env: int
    #: "standard" (ctx, args, env, ws) or "foreign" (ctx, args, env)
    abi: str
    #: what it produces, per node output (`Output`s, `view_of` counting node inputs)
    outputs: list
    description: str = ""
    keepalive: object = None


@dataclass
class Closure:
    """What a generated function finds in its `env`: pointer-sized slots holding the
    handles of its callees (function, env: two slots) and the data of array constants
    (one slot). The IR only refers to slot positions."""

    slots: list = field(default_factory=list)
    _keep: list = field(default_factory=list)
    _table: object = None

    def add(self, callee: Callee) -> int:
        """Slot of `callee`'s handle (its env follows in the next slot)."""
        self.slots += [callee.address, callee.env]
        self._keep.append(callee.keepalive)
        return len(self.slots) - 2

    def add_data(self, array: np.ndarray) -> int:
        """Slot holding the address of `array`'s data (kept alive with the closure)."""
        self.slots.append(array.ctypes.data)
        self._keep.append(array)
        return len(self.slots) - 1

    @property
    def table(self):
        """The env: the slots as a uint64 array (None if empty). Keep it (and
        `keepalive`) alive while the code runs."""
        if not self.slots:
            return None
        if self._table is None or len(self._table) != len(self.slots):
            self._table = np.array(self.slots, dtype=np.uint64)
        return self._table

    @property
    def env(self) -> int:
        t = self.table
        return 0 if t is None else t.ctypes.data

    @property
    def keepalive(self):
        return (self.table, self._keep)


def load_slot(cx, index: int):
    """The pointer in slot `index` of the env."""
    b, m = cx.b, ir()
    return b.load(m.ptr, b.gep_bytes(cx.env, b.const_int(m.i64, 8 * index)))


def emit_call(cx, callee: Callee, index: int, inputs: list, outputs: list) -> None:
    """Call `callee` (handle in env slot `index`, see `Closure.add`) with `inputs` ((Array or scalar value
    or None for absent ones, dtype) pairs) and `outputs` (ProvidedOutput, SlotOutput,
    or None).

    Allocated and view outputs land in their slots, owner included, also when the call
    fails (the caller releases them); a failure returns its code (the callee set the
    message). The callee's scratch is released after the call."""
    from pytensor.link.llvm_v2.codegen.emit import Array, ProvidedOutput, SlotOutput, store

    b, m = cx.b, ir()
    pa = ptarray_type()
    i64 = m.i64
    entries = []  # (data, shape ptr, strides ptr, ndim, slot output or None)

    def storage(values):
        p = b.alloca(m.Type.array(i64, max(len(values), 1)))
        for k, v in enumerate(values):
            b.store(v, b.gep(i64, p, b.const_int(i64, k)))
        return p

    for a, dtype in inputs:
        if a is None:
            continue
        if isinstance(a, Array):
            entries.append((a.data, storage(a.shape), storage(a.strides), len(a.shape), None))
        else:  # a scalar: a 0-d array
            p = b.alloca(memory_type(dtype))
            store(b, dtype, a, p)
            entries.append((p, storage([]), storage([]), 0, None))
    for o in outputs:
        if o is None:
            continue
        if isinstance(o, ProvidedOutput):
            a = o.array
            entries.append((a.data, storage(a.shape), storage(a.strides), len(a.shape), None))
        elif isinstance(o, SlotOutput):
            st = slot_type(o.ndim)
            entries.append((b.null(), b.field_ptr(st, o.slot, 1), b.field_ptr(st, o.slot, 2), o.ndim, o))
        else:
            raise NotImplementedError(f"{callee.description}: a {type(o).__name__} output")

    args = b.alloca(m.Type.array(pa, max(len(entries), 1)))
    ptrs = []
    for k, (data, shape, strides, nd, _) in enumerate(entries):
        p = b.gep_bytes(args, b.const_int(i64, k * PTARRAY_SIZE))
        b.store(data, b.field_ptr(pa, p, 0))
        b.store(shape, b.field_ptr(pa, p, 1))
        b.store(strides, b.field_ptr(pa, p, 2))
        b.store(b.null(), b.field_ptr(pa, p, 3))
        b.store(b.const_int(i64, nd), b.field_ptr(pa, p, 4))
        ptrs.append(p)

    fn = load_slot(cx, index)
    fenv = load_slot(cx, index + 1)
    mark = b.rt_scratch_mark(cx.ctx)
    call_args = [cx.ctx, args, fenv] + ([b.null()] if callee.abi == "standard" else [])
    rc = b.call_ptr(fn, call_args, m.i32)
    b.rt_scratch_release(cx.ctx, mark)

    for p, (_, _, _, _, o) in zip(ptrs, entries, strict=True):
        if o is None:
            continue
        st = slot_type(o.ndim)
        b.store(b.load(m.ptr, b.field_ptr(pa, p, 0)), b.field_ptr(st, o.slot, 0))
        b.store(b.load(m.ptr, b.field_ptr(pa, p, 3)), b.field_ptr(st, o.slot, 3))
        o.done = True
    with b.if_(b.icmp(m.ICmp.NE, rc, b.const_int(m.i32, 0))):
        b.ret(rc)


__all__ = ["Callee", "Closure", "emit_call", "load_slot"]
