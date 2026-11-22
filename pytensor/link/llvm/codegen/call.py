"""Calls from generated code to other kernels.

Rust kernels, numba kernels and separately compiled code all have the kernel ABI. Generated code calls them all the same way: it packs the arguments into
`PtArray`s (with the number of axes and the dtype of each), calls the kernel through its
entry in the link table and reads back what the kernel wrote. A node implemented by such
a kernel then has a node function like any other (`build_node_function(...,
kernel=...)`), so units, programs and generated loops can contain it; what isn't a node
(a compiled unit called by a program) gets one from its signature
(`build_call_function`).

The addresses (of the kernel and of its config) aren't part of the IR: they are entries
of the calling kernel's link table, which is its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pytensor.link.llvm.codegen.types import (
    PTARRAY_SIZE,
    PtArray,
    dtype_code,
    ir,
    memory_type,
    ptarray_type,
)
from pytensor.scalar.basic import ScalarType


@dataclass
class Kernel:
    """A kernel that generated code calls for a node: a Rust kernel, a numba kernel, or
    code compiled on its own."""

    address: int
    #: the kernel's config (`const void* config` of the kernel ABI)
    config: int
    #: what it produces, per node output (`Output`s)
    outputs: list
    description: str = ""
    keepalive: object = None
    #: node inputs the kernel overwrites (a numba implementation of an op that destroys
    #: them): generated code calling it passes copies
    destroys: frozenset = frozenset()


@dataclass
class LinkTable:
    """The config of a generated kernel: pointer-sized entries holding the kernels it
    calls (address, then their config: two entries) and the contents of its array
    constants (one entry). The IR only refers to entry positions."""

    entries: list = field(default_factory=list)
    _keep: list = field(default_factory=list)
    _table: np.ndarray | None = None
    #: (entry, kernel) of every kernel linked, for `relink`
    _kernels: list = field(default_factory=list)
    #: contents -> entry of the arrays linked (equal ones share one)
    _arrays: dict = field(default_factory=dict)

    def link_kernel(self, kernel: Kernel) -> int:
        """The entry of `kernel`'s address (its config is in the next entry)."""
        self.entries += [kernel.address, kernel.config]
        self._keep.append(kernel)
        self._kernels.append((len(self.entries) - 2, kernel))
        return len(self.entries) - 2

    def relink(self):
        """Update the kernels' entries: for kernels whose address (or config) was set
        after they were linked, once compiled. Before the table is used."""
        for k, kernel in self._kernels:
            self.entries[k : k + 2] = [kernel.address, kernel.config]
        self._table = None

    def link_array(self, array: np.ndarray, share: bool = True) -> int:
        """The entry holding the address of the contents of `array`, a C-contiguous
        buffer (kept alive with the table). With `share`, equal buffers (read only)
        share an entry (e.g. a matrix and its transpose, each packed in its own order)."""
        if not array.flags.c_contiguous:
            raise ValueError("link_array takes C-contiguous arrays")
        key = (array.dtype.itemsize, array.tobytes()) if share else None
        if key is None or key not in self._arrays:
            self.entries.append(array.ctypes.data)
            self._keep.append(array)
            if key is None:
                return len(self.entries) - 1
            self._arrays[key] = len(self.entries) - 1
        return self._arrays[key]

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
    return b.load(ir.ptr, b.gep_bytes(cg.config, 8 * index))


def call_kernel(cg, kernel: Kernel, index: int, inputs: list, outputs: list) -> None:
    """Call `kernel` (at entry `index` of the link table, see `LinkTable.link_kernel`)
    with `inputs` ((Array, scalar value or None for absent ones, `ScalarType`) pairs) and
    `outputs` (`ProvidedDestination`, `SlotDestination`, or None).

    Allocated and view outputs land in their output slots, owner included, also when the
    call fails (the caller releases them); a failure returns its code (the kernel set
    the message). The kernel's scratch is released after the call."""
    from pytensor.link.llvm.codegen.array import Array, Slot
    from pytensor.link.llvm.codegen.destination import (
        ProvidedDestination,
        SlotDestination,
    )
    from pytensor.link.llvm.codegen.num import store

    b = cg.b
    ptarray = ptarray_type()
    i64 = ir.i64
    # per argument: (data, shape ptr, strides ptr, ndim, dtype, slot or None)
    args_of: list[
        tuple[object, object, object, int, ScalarType, SlotDestination | None]
    ] = []
    n_inputs = 0

    def storage(values):
        p = b.alloca(ir.Type.array(i64, max(len(values), 1)))
        for k, v in enumerate(values):
            b.store(v, b.gep(i64, p, k))
        return p

    for a, scalar_type in inputs:
        if a is None:
            continue
        n_inputs += 1
        if isinstance(a, Array):
            args_of.append(
                (
                    a.data,
                    storage(a.shape),
                    storage(a.strides),
                    len(a.shape),
                    scalar_type,
                    None,
                )
            )
        else:  # a scalar: a 0-d array
            p = b.alloca(memory_type(scalar_type))
            store(b, scalar_type, a, p)
            args_of.append((p, storage([]), storage([]), 0, scalar_type, None))
    for destination in outputs:
        if destination is None:
            continue
        if isinstance(destination, ProvidedDestination):
            a = destination.array
            args_of.append(
                (
                    a.data,
                    storage(a.shape),
                    storage(a.strides),
                    len(a.shape),
                    a.scalar_type,
                    None,
                )
            )
        elif isinstance(destination, SlotDestination):
            args_of.append(
                (
                    b.null(),
                    destination.slot.field(b, Slot.LENGTHS),
                    destination.slot.field(b, Slot.STRIDES),
                    destination.ndim,
                    destination.scalar_type,
                    destination,
                )
            )
        else:
            raise NotImplementedError(
                f"{kernel.description}: a {type(destination).__name__} output"
            )

    args = b.alloca(ir.Type.array(ptarray, max(len(args_of), 1)))
    ptrs = []
    for k, (data, shape, strides, nd, scalar_type, _) in enumerate(args_of):
        p = b.gep_bytes(args, k * PTARRAY_SIZE)
        b.store(data, b.field_ptr(ptarray, p, PtArray.DATA))
        b.store(shape, b.field_ptr(ptarray, p, PtArray.SHAPE))
        b.store(strides, b.field_ptr(ptarray, p, PtArray.STRIDES))
        b.store(b.null(), b.field_ptr(ptarray, p, PtArray.OWNER))
        b.store(b.i32(nd), b.field_ptr(ptarray, p, PtArray.NDIM))
        b.store(b.i32(dtype_code(scalar_type)), b.field_ptr(ptarray, p, PtArray.DTYPE))
        ptrs.append(p)

    fn = linked(cg, index)
    config = linked(cg, index + 1)
    mark = b.rt_scratch_mark(cg.ctx)
    counts = [b.i32(n_inputs), b.i32(len(args_of) - n_inputs)]
    rc = b.call_ptr(fn, [cg.ctx, args, *counts, config], ir.i32)
    b.rt_scratch_release(cg.ctx, mark)

    for p, (*_, destination) in zip(ptrs, args_of, strict=True):
        if destination is None:
            continue
        b.store(
            b.load(ir.ptr, b.field_ptr(ptarray, p, PtArray.DATA)),
            destination.slot.field(b, Slot.DATA),
        )
        b.store(
            b.load(ir.ptr, b.field_ptr(ptarray, p, PtArray.OWNER)),
            destination.slot.field(b, Slot.OWNER),
        )
        destination.done = True
    with b.if_(b.ne(rc, 0)):
        b.ret(rc)


def call_destroying(cg, kernel: Kernel, index: int, inputs: list, outputs: list):
    """`call_kernel`, with copies of the inputs the kernel overwrites
    (`Kernel.destroys`, positions in `inputs`): it may only write its outputs."""
    from pytensor.link.llvm.codegen.array import Array

    ins = []
    for pos, (a, scalar_type) in enumerate(inputs):
        if pos in kernel.destroys and isinstance(a, Array):
            mine = cg.temporary(a.scalar_type, a.shape)
            mine.copy_from(cg, a)
            a = mine
        ins.append((a, scalar_type))
    call_kernel(cg, kernel, index, ins, outputs)


def build_call_function(
    module, name: str, signature, kernel: Kernel, links: LinkTable, index=None
):
    """A function (always inlined) with `signature` that calls `kernel`: how generated
    code runs a kernel that isn't a node's (a compiled unit, a separately compiled
    node), and nodes the planner gives a kernel. `index`: the kernel's entry
    in `links` (default: linked here). Returns a `NodeFunction`."""
    from pytensor.link.llvm.codegen.array import Array, Slot
    from pytensor.link.llvm.codegen.context import Codegen
    from pytensor.link.llvm.codegen.destination import (
        ProvidedDestination,
        SlotDestination,
    )
    from pytensor.link.llvm.codegen.node import NodeFunction
    from pytensor.link.llvm.codegen.signature import Passing

    fn = module.function(
        name,
        signature.param_types(),
        ir.i32,
        inline=ir.Inline.Always,
        debug_name=kernel.description[:200],
    )
    b = fn.builder()
    params = iter(fn.params)
    ctx, config = next(params), next(params)
    cg = Codegen(b, ctx, config, links=links, module=module, name=name)
    inputs = [
        Array.unpack(b, representation, next(params))
        for representation in signature.inputs
    ]
    for pos, a in enumerate(inputs):
        a.base, a.writable = pos, False
    outputs: list = []
    for output in signature.outputs:
        if output.is_absent:
            outputs.append(None)
        elif output.passing is Passing.CALLER_BUFFER:
            array = Array.unpack(b, output.representation, next(params))
            outputs.append(ProvidedDestination(cg, array))
        else:
            outputs.append(
                SlotDestination(
                    cg,
                    Slot(next(params), output.representation.ndim),
                    output.representation.scalar_type,
                    signature.alias_arguments(output),
                )
            )
    if index is None:
        index = links.link_kernel(kernel)
    with cg.scratch_scope(function=True):
        call_destroying(
            cg, kernel, index, [(a, a.scalar_type) for a in inputs], outputs
        )
    b.ret(0)
    return NodeFunction(fn, signature, links)


__all__ = [
    "Kernel",
    "LinkTable",
    "build_call_function",
    "call_destroying",
    "call_kernel",
    "linked",
]
