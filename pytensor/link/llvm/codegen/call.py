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
        ScalarDestination,
        SlotDestination,
    )
    from pytensor.link.llvm.codegen.kernel_module import KernelModule


#: An input of a kernel call: the value (None for one without data) and its type.
type KernelInput = tuple[Array | ir.Value | None, ScalarType | None]


@dataclass(frozen=True)
class Kernel:
    """A kernel generated code can call through the kernel ABI, available now: a Rust
    kernel or a numba kernel."""

    address: int
    #: the kernel's config (`const void* config` of the kernel ABI)
    config: int
    description: str = ""
    keepalive: object = None
    #: may it run work on the runtime's thread pool?
    uses_pool: bool = False


#: What generated code calls: a kernel available now, or generated code in a module of
#: its own (`KernelModule`), compiled with the code calling it and linked then (`link`).
type Callee = Kernel | KernelModule


@dataclass
class LinkTable:
    """The config of a generated kernel: pointer-sized entries holding the kernels it
    calls (address, then their config: two entries) and the contents of its array
    constants (one entry). The IR only refers to entry positions.

    It is filled while code is generated, then fixed when its module is linked
    (`kernel_module.link`): its address is known from then on, and the entries of
    modules it calls are written then."""

    entries: list[int] = field(default_factory=list)
    _keep: list[object] = field(default_factory=list)
    _table: np.ndarray | None = None
    #: (entry, module) of the modules linked, whose entries `link` writes
    _modules: list[tuple[int, KernelModule]] = field(default_factory=list)
    #: (itemsize, contents) -> entry of the arrays linked (equal ones share one)
    _arrays: dict[tuple[int, bytes], int] = field(default_factory=dict)

    def link_kernel(self, callee: Callee) -> int:
        """The entry of `callee`'s address (its config is in the next entry)."""
        self._check_open()
        k = len(self.entries)
        if isinstance(callee, Kernel):
            self.entries += [callee.address, callee.config]
        else:
            self.entries += [0, 0]
            self._modules.append((k, callee))
        self._keep.append(callee)
        return k

    def link_array(self, array: np.ndarray, share: bool = True) -> int:
        """The entry holding the address of the contents of `array`, a C-contiguous
        buffer (kept alive with the table). With `share`, equal buffers (read only)
        share an entry (e.g. a matrix and its transpose, each packed in its own order)."""
        self._check_open()
        if not array.flags.c_contiguous:
            raise ValueError("link_array takes C-contiguous arrays")
        key = (array.dtype.itemsize, array.tobytes()) if share else None
        if key is not None and key in self._arrays:
            return self._arrays[key]
        self.entries.append(array.ctypes.data)
        self._keep.append(array)
        if key is not None:
            self._arrays[key] = len(self.entries) - 1
        return len(self.entries) - 1

    def fix(self) -> None:
        """Fix the table (`link`): no entries can be added, its address is final."""
        if self._table is None:
            self._table = np.array(self.entries or [0], dtype=np.uint64)

    def write_module_entries(self) -> None:
        """Write the address and config of every module linked (once all are fixed and
        compiled: `link`)."""
        table = self._fixed()
        for k, module in self._modules:
            table[k : k + 2] = [module.address, module.config]
            self.entries[k : k + 2] = [module.address, module.config]

    @property
    def address(self) -> int:
        """The table's address: the config of the kernel it belongs to."""
        return int(self._fixed().ctypes.data)

    @property
    def keepalive(self) -> tuple[np.ndarray, list[object]]:
        return (self._fixed(), self._keep)

    def _fixed(self) -> np.ndarray:
        if self._table is None:
            raise ValueError("the link table isn't fixed yet (its module isn't linked)")
        return self._table

    def _check_open(self) -> None:
        if self._table is not None:
            raise ValueError("the link table is fixed: its module is linked")


def linked(cg: Codegen, index: int) -> ir.Value:
    """The pointer in entry `index` of the link table."""
    b = cg.b
    return b.load(ir.ptr, b.gep_bytes(cg.config, 8 * index))


class _Argument(NamedTuple):
    """One `PtArray` of a kernel call: what goes into its fields, and the slot to copy
    an output back into, or the ScalarType output to load it into (None for inputs and
    provided memory)."""

    data: ir.Value
    shape: ir.Value
    strides: ir.Value
    ndim: int
    scalar_type: ScalarType
    slot_destination: SlotDestination | None = None
    scalar_destination: ScalarDestination | None = None


def call_kernel(
    cg: Codegen,
    kernel: Callee,
    inputs: Sequence[KernelInput],
    outputs: Sequence[OutputDestination | None],
) -> None:
    """Call `kernel` (linked into the module's link table, `LinkTable.link_kernel`)
    with `inputs` ((Array, scalar value or None for absent ones, `ScalarType`) pairs) and
    `outputs` (`ProvidedDestination`, `SlotDestination`, `ScalarDestination`, or None).
    A ScalarType output is a 0-d array the kernel allocates (as numba kernels do): its
    value is loaded into the destination, and the array given up.

    Allocated and view outputs land in their output slots, owner included, also when the
    call fails (the caller releases them); a failure returns its code (the kernel set
    the message). The kernel's scratch is released after the call."""
    from pytensor.link.llvm.codegen.array import Array, Slot
    from pytensor.link.llvm.codegen.destination import (
        ProvidedDestination,
        ScalarDestination,
        SlotDestination,
    )
    from pytensor.link.llvm.codegen.num import load, store

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
        elif isinstance(destination, ScalarDestination):
            args_of.append(
                _Argument(
                    b.null(),
                    storage([]),
                    storage([]),
                    0,
                    destination.scalar_type,
                    scalar_destination=destination,
                )
            )
        else:
            raise NotImplementedError(
                f"{_name(kernel)}: a {type(destination).__name__} output"
            )

    args = b.alloca(ir.Type.array(ptarray, max(len(args_of), 1)))
    ptrs: list[ir.Value] = []
    for k, (data, shape, strides, nd, scalar_type, *_) in enumerate(args_of):
        p = b.gep_bytes(args, k * PTARRAY_SIZE)
        b.store(data, b.field_ptr(ptarray, p, PtArray.DATA))
        b.store(shape, b.field_ptr(ptarray, p, PtArray.SHAPE))
        b.store(strides, b.field_ptr(ptarray, p, PtArray.STRIDES))
        b.store(b.null(), b.field_ptr(ptarray, p, PtArray.OWNER))
        b.store(b.i32(nd), b.field_ptr(ptarray, p, PtArray.NDIM))
        b.store(b.i32(dtype_code(scalar_type)), b.field_ptr(ptarray, p, PtArray.DTYPE))
        ptrs.append(p)

    index = cg.links.link_kernel(kernel)
    if isinstance(kernel, Kernel) and kernel.uses_pool:
        cg.unit.uses_pool = True
    fn = linked(cg, index)
    config = linked(cg, index + 1)
    mark = b.rt_scratch_mark(cg.ctx)
    counts = [b.i32(n_inputs), b.i32(len(args_of) - n_inputs)]
    rc = b.call_ptr(fn, [cg.ctx, args, *counts, config], ir.i32)
    assert rc is not None  # (it returns an i32)
    b.rt_scratch_release(cg.ctx, mark)

    scalars = []  # (destination, data, owner) of the ScalarType outputs
    for p, argument in zip(ptrs, args_of, strict=True):
        if argument.scalar_destination is not None:
            scalars.append(
                (
                    argument.scalar_destination,
                    b.load(ir.ptr, b.field_ptr(ptarray, p, PtArray.DATA)),
                    b.load(ir.ptr, b.field_ptr(ptarray, p, PtArray.OWNER)),
                )
            )
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
        for _, _, owner in scalars:
            b.rt_decref(cg.ctx, owner)
        b.ret(rc)
    for destination, data, owner in scalars:
        destination.set(load(b, destination.scalar_type, data))
        b.rt_decref(cg.ctx, owner)


def _name(callee: Callee) -> str:
    return callee.description if isinstance(callee, Kernel) else callee.module.name


__all__ = [
    "Callee",
    "Kernel",
    "LinkTable",
]
