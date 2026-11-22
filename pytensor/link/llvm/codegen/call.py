"""Calls from generated code to other kernels.

Rust kernels, numba kernels and separately compiled code all have the kernel ABI.
Generated code calls them all the same way (`call_kernel`): it packs the arguments into
`PtArray`s (with the number of axes and the dtype of each), calls the kernel through its
entry in the link table and reads back what the kernel wrote. A node run by such a kernel
has a node function like any other, whose body is the call (`implementation.RustKernel`,
`NumbaKernel`, `OwnModule`), so units, programs and generated loops can contain it.

The addresses (of the kernel and of its config) aren't part of the IR: they are entries
of the calling kernel's link table, which is its config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

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


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.array import Array
    from pytensor.link.llvm.codegen.context import Codegen
    from pytensor.link.llvm.codegen.destination import (
        OutputDestination,
        SlotDestination,
    )


#: An input of a kernel call: the value (None for one without data) and its type.
type KernelInput = tuple[Array | ir.Value | None, ScalarType | None]


@dataclass
class Kernel:
    """Something generated code can call through the kernel ABI: a Rust kernel, a numba
    kernel, or generated code compiled on its own."""

    address: int
    #: the kernel's config (`const void* config` of the kernel ABI)
    config: int
    description: str = ""
    keepalive: object = None


@dataclass
class LinkTable:
    """The config of a generated kernel: pointer-sized entries holding the kernels it
    calls (address, then their config: two entries) and the contents of its array
    constants (one entry). The IR only refers to entry positions."""

    entries: list[int] = field(default_factory=list)
    _keep: list[object] = field(default_factory=list)
    _table: np.ndarray | None = None
    #: (entry, kernel) of every kernel linked, for `relink`
    _kernels: list[tuple[int, Kernel]] = field(default_factory=list)
    #: (itemsize, contents) -> entry of the arrays linked (equal ones share one)
    _arrays: dict[tuple[int, bytes], int] = field(default_factory=dict)

    def link_kernel(self, kernel: Kernel) -> int:
        """The entry of `kernel`'s address (its config is in the next entry)."""
        self.entries += [kernel.address, kernel.config]
        self._keep.append(kernel)
        self._kernels.append((len(self.entries) - 2, kernel))
        return len(self.entries) - 2

    def relink(self) -> None:
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
    def table(self) -> np.ndarray | None:
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
        return 0 if t is None else int(t.ctypes.data)

    @property
    def keepalive(self) -> tuple[np.ndarray | None, list[object]]:
        return (self.table, self._keep)


def linked(cg: Codegen, index: int) -> ir.Value:
    """The pointer in entry `index` of the link table."""
    b = cg.b
    return b.load(ir.ptr, b.gep_bytes(cg.config, 8 * index))


class _Argument(NamedTuple):
    """One `PtArray` of a kernel call: what goes into its fields, and the slot to copy
    an output back into (None for inputs and provided memory)."""

    data: ir.Value
    shape: ir.Value
    strides: ir.Value
    ndim: int
    scalar_type: ScalarType
    slot_destination: SlotDestination | None = None


def call_kernel(
    cg: Codegen,
    kernel: Kernel,
    inputs: Sequence[KernelInput],
    outputs: Sequence[OutputDestination | None],
) -> None:
    """Call `kernel` (linked into the module's link table, `LinkTable.link_kernel`)
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
    args_of: list[_Argument] = []
    n_inputs = 0

    def storage(values: Sequence[ir.Value]) -> ir.Value:
        p = b.alloca(ir.Type.array(i64, max(len(values), 1)))
        for k, v in enumerate(values):
            b.store(v, b.gep(i64, p, k))
        return p

    for a, scalar_type in inputs:
        if a is None or scalar_type is None:
            continue
        n_inputs += 1
        if isinstance(a, Array):
            args_of.append(
                _Argument(
                    a.data,
                    storage(a.shape),
                    storage(a.strides),
                    len(a.shape),
                    scalar_type,
                )
            )
        else:  # a scalar: a 0-d array
            p = b.alloca(memory_type(scalar_type))
            store(b, scalar_type, a, p)
            args_of.append(_Argument(p, storage([]), storage([]), 0, scalar_type))
    for destination in outputs:
        if destination is None:
            continue
        if isinstance(destination, ProvidedDestination):
            array = destination.array
            args_of.append(
                _Argument(
                    array.data,
                    storage(array.shape),
                    storage(array.strides),
                    len(array.shape),
                    array.scalar_type,
                )
            )
        elif isinstance(destination, SlotDestination):
            args_of.append(
                _Argument(
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
    ptrs: list[ir.Value] = []
    for k, (data, shape, strides, nd, scalar_type, _) in enumerate(args_of):
        p = b.gep_bytes(args, k * PTARRAY_SIZE)
        b.store(data, b.field_ptr(ptarray, p, PtArray.DATA))
        b.store(shape, b.field_ptr(ptarray, p, PtArray.SHAPE))
        b.store(strides, b.field_ptr(ptarray, p, PtArray.STRIDES))
        b.store(b.null(), b.field_ptr(ptarray, p, PtArray.OWNER))
        b.store(b.i32(nd), b.field_ptr(ptarray, p, PtArray.NDIM))
        b.store(b.i32(dtype_code(scalar_type)), b.field_ptr(ptarray, p, PtArray.DTYPE))
        ptrs.append(p)

    index = cg.links.link_kernel(kernel)
    fn = linked(cg, index)
    config = linked(cg, index + 1)
    mark = b.rt_scratch_mark(cg.ctx)
    counts = [b.i32(n_inputs), b.i32(len(args_of) - n_inputs)]
    rc = b.call_ptr(fn, [cg.ctx, args, *counts, config], ir.i32)
    assert rc is not None  # (it returns an i32)
    b.rt_scratch_release(cg.ctx, mark)

    for p, argument in zip(ptrs, args_of, strict=True):
        destination = argument.slot_destination
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


__all__ = [
    "Kernel",
    "LinkTable",
]
