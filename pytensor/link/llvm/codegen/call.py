"""Calls from generated code to other kernels (DESIGN.md, section 2a).

Rust kernels, numba kernels and subprograms all have the kernel ABI the interpreter
calls (section 6). Generated code calls them the same way: it packs the arguments into
`PtArray`s, calls the kernel through its entry in the link table and reads back what
the kernel wrote. A node implemented by such a kernel then has a node function like
any other (`build_node_function(..., kernel=...)`), so units and generated loops can
contain it.

The addresses (of the kernel and of its config) aren't part of the IR, which must not
depend on the process (the compile cache is keyed on it): they are entries of the
calling kernel's link table, which is its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pytensor.link.llvm.codegen.types import ir, memory_type, PtArray, PTARRAY_SIZE, ptarray_type, Slot, slot_type


@dataclass
class CalledKernel:
    """A kernel that generated code calls for a node: a Rust kernel, a numba kernel or a
    subprogram."""

    address: int
    #: the kernel's config (`const void* config` of the kernel ABI)
    config: int
    #: what it produces, per node output (`Output`s, `aliases` counting node inputs)
    outputs: list
    description: str = ""
    keepalive: object = None


@dataclass
class LinkTable:
    """The config of a generated kernel: pointer-sized entries holding the kernels it
    calls (address, then their config: two entries) and the contents of its array
    constants (one entry). The IR only refers to entry positions."""

    entries: list = field(default_factory=list)
    _keep: list = field(default_factory=list)
    _table: object = None

    def link_kernel(self, kernel: CalledKernel) -> int:
        """The entry of `kernel`'s address (its config is in the next entry)."""
        self.entries += [kernel.address, kernel.config]
        self._keep.append(kernel.keepalive)
        return len(self.entries) - 2

    def link_array(self, array: np.ndarray) -> int:
        """The entry holding the address of `array`'s contents (kept alive with the
        table)."""
        self.entries.append(array.ctypes.data)
        self._keep.append(array)
        return len(self.entries) - 1

    @property
    def table(self):
        """The entries as a uint64 array (None if empty). Keep it (and `keepalive`)
        alive while the code runs."""
        if not self.entries:
            return None
        if self._table is None or len(self._table) != len(self.entries):
            self._table = np.array(self.entries, dtype=np.uint64)
        return self._table

    @property
    def address(self) -> int:
        """The table's address: the config of the kernel it belongs to (0 if empty)."""
        t = self.table
        return 0 if t is None else t.ctypes.data

    @property
    def keepalive(self):
        return (self.table, self._keep)


def linked(cg, index: int):
    """The pointer in entry `index` of the link table."""
    b = cg.b
    return b.load(ir.ptr, b.gep_bytes(cg.config, b.i64(8 * index)))


def call_kernel(cg, kernel: CalledKernel, index: int, inputs: list, outputs: list) -> None:
    """Call `kernel` (at entry `index` of the link table, see `LinkTable.link_kernel`)
    with `inputs` ((Array, scalar value or None for absent ones, dtype) pairs) and
    `outputs` (ProvidedOutput, SlotOutput, or None).

    Allocated and view outputs land in their output slots, owner included, also when the
    call fails (the caller releases them); a failure returns its code (the kernel set
    the message). The kernel's scratch is released after the call."""
    from pytensor.link.llvm.codegen.emit import Array, ProvidedOutput, SlotOutput, store

    b = cg.b
    pa = ptarray_type()
    i64 = ir.i64
    args_of = []  # per argument: (data, shape ptr, strides ptr, ndim, SlotOutput or None)

    def storage(values):
        p = b.alloca(ir.Type.array(i64, max(len(values), 1)))
        for k, v in enumerate(values):
            b.store(v, b.gep(i64, p, b.i64(k)))
        return p

    for a, dtype in inputs:
        if a is None:
            continue
        if isinstance(a, Array):
            args_of.append((a.data, storage(a.shape), storage(a.strides), len(a.shape), None))
        else:  # a scalar: a 0-d array
            p = b.alloca(memory_type(dtype))
            store(b, dtype, a, p)
            args_of.append((p, storage([]), storage([]), 0, None))
    for o in outputs:
        if o is None:
            continue
        if isinstance(o, ProvidedOutput):
            a = o.array
            args_of.append((a.data, storage(a.shape), storage(a.strides), len(a.shape), None))
        elif isinstance(o, SlotOutput):
            st = slot_type(o.ndim)
            args_of.append((b.null(), b.field_ptr(st, o.slot, Slot.LENGTHS), b.field_ptr(st, o.slot, Slot.STRIDES), o.ndim, o))
        else:
            raise NotImplementedError(f"{kernel.description}: a {type(o).__name__} output")

    args = b.alloca(ir.Type.array(pa, max(len(args_of), 1)))
    ptrs = []
    for k, (data, shape, strides, nd, _) in enumerate(args_of):
        p = b.gep_bytes(args, b.i64(k * PTARRAY_SIZE))
        b.store(data, b.field_ptr(pa, p, PtArray.DATA))
        b.store(shape, b.field_ptr(pa, p, PtArray.SHAPE))
        b.store(strides, b.field_ptr(pa, p, PtArray.STRIDES))
        b.store(b.null(), b.field_ptr(pa, p, PtArray.OWNER))
        b.store(b.i64(nd), b.field_ptr(pa, p, PtArray.NDIM))
        ptrs.append(p)

    fn = linked(cg, index)
    config = linked(cg, index + 1)
    mark = b.rt_scratch_mark(cg.ctx)
    rc = b.call_ptr(fn, [cg.ctx, args, config], ir.i32)
    b.rt_scratch_release(cg.ctx, mark)

    for p, (_, _, _, _, o) in zip(ptrs, args_of, strict=True):
        if o is None:
            continue
        st = slot_type(o.ndim)
        b.store(b.load(ir.ptr, b.field_ptr(pa, p, PtArray.DATA)), b.field_ptr(st, o.slot, Slot.DATA))
        b.store(b.load(ir.ptr, b.field_ptr(pa, p, PtArray.OWNER)), b.field_ptr(st, o.slot, Slot.OWNER))
        o.done = True
    with b.if_(b.ne(rc, b.i32(0))):
        b.ret(rc)


__all__ = ["CalledKernel", "LinkTable", "call_kernel", "linked"]
