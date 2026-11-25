"""Where a computed value goes: output destinations.

An `OutputDestination` is the place a result ends up in: memory the caller provides
(`ProvidedDestination`), an output slot the caller reads back (`SlotDestination`), or a
pointer to a scalar (`ScalarDestination`). Destinations are the low-level side of
outputs. Op emitters don't use them directly: they get the handles of `interface.py`,
which wrap a destination and only offer what the op declared.

Code that runs a whole graph (`GraphFunction.emit_call`) works with destinations. An op
that produces its outputs by running an inner graph (Scan, Blockwise, compiled units)
hands its output handles on, or subclasses `OutputDestination` for results that go
somewhere of its own (a row of a Scan's buffer), or keeps a result itself (`Captured`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, Slot, size
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.num import load
from pytensor.link.llvm.codegen.types import (
    TensorRepr,
    ir,
)
from pytensor.scalar.basic import ScalarType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.signature import Output


class OutputDestination:
    """Where one result goes: the protocol a producer (`GraphFunction.emit_call`) uses.

    A producer either asks for memory to write the result into (`allocate`), hands
    over an array that already holds the result (`assign`), or, for a ScalarType
    value, sets it (`set`). `finish` is called once the producer is done.

    Subclass it for a result that goes somewhere of your own: implement `scalar_type`
    and `allocate` (and `finish`, to act once the result is there). The rest of the
    methods are what `TensorOutput` offers ops for the destinations it wraps.
    """

    cg: Codegen
    scalar_type: ScalarType

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return memory of `shape` to write the result into."""
        raise NotImplementedError

    def assign(self, array: Array) -> Array:
        """Make `array` the result.

        The array is kept as a view where this destination accepts one, and copied
        otherwise. Return the array the result now lives in."""
        result = self.allocate(array.shape)
        result.copy_from(self.cg, array)
        return result

    def set(self, value: ir.Value) -> None:
        raise TypeError("set() is for ScalarType outputs; tensors are allocated")

    def finish(self) -> None:
        """Called when the producer is done."""

    def slot_for_new_memory(self, representation: TensorRepr) -> Slot | None:
        """Return the slot a producer may allocate the result into directly, or None
        (the default).

        `GraphFunction.emit_call` asks this for a result a node allocates anyway (a
        kernel's, or one whose shape is only known at run time), described by
        `representation`. If the slot is returned, the result counts as produced: the
        producer must allocate it there."""
        return None

    # ---- behind `TensorOutput`'s methods for ops that run an inner graph

    def allocate_once(self, shape: list[ir.Value]) -> Array:
        """`TensorOutput.allocate_once`."""
        raise NotImplementedError(f"{type(self).__name__}: allocate_once")

    def reallocate(self, shape: list[ir.Value]) -> Array:
        """`TensorOutput.reallocate`."""
        raise NotImplementedError(f"{type(self).__name__}: reallocate")

    def truncate(self, length: ir.Value) -> None:
        """`TensorOutput.truncate`."""
        raise NotImplementedError(f"{type(self).__name__}: truncate")

    def current(self) -> Array:
        """`TensorOutput.current`."""
        raise NotImplementedError(f"{type(self).__name__}: current")

    @property
    def allocated_layout(self) -> Layout:
        """`TensorOutput.allocated_layout`."""
        raise NotImplementedError(f"{type(self).__name__}: allocated_layout")


@dataclass
class ProvidedDestination(OutputDestination):
    """Memory the caller provides, with the caller's strides."""

    cg: Codegen
    array: Array

    @property
    def scalar_type(self) -> ScalarType:  # type: ignore[override]
        return self.array.scalar_type

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return the provided memory, which must have `shape`.

        A length that differs is an error while generating if both are constants, and
        else checked at run time with the `check_layouts` option."""
        cg = self.cg
        if len(shape) != self.array.ndim:
            raise ValueError(
                f"allocate: {len(shape)} axes, but the provided memory has "
                f"{self.array.ndim}"
            )
        for axis, (wanted, provided) in enumerate(
            zip(shape, self.array.shape, strict=True)
        ):
            wanted_length, provided_length = wanted.const_int, provided.const_int
            if wanted_length is not None and provided_length is not None:
                if wanted_length != provided_length:
                    raise ValueError(
                        f"allocate: axis {axis} has length {wanted_length}, but the "
                        f"provided memory has {provided_length}"
                    )
            elif cg.check_layouts:
                cg.check(
                    cg.b.eq(wanted, provided),
                    f"allocate: axis {axis} doesn't have the length of the provided "
                    "memory",
                    RuntimeError,
                )
        return self.array

    def allocate_once(self, shape: list[ir.Value]) -> Array:
        return self.allocate(shape)

    def reallocate(self, shape: list[ir.Value]) -> Array:
        """The caller's memory can't be replaced: the shape must stay the same."""
        b = self.cg.b
        for axis, (wanted, provided) in enumerate(
            zip(shape, self.array.shape, strict=True)
        ):
            self.cg.check(
                b.eq(wanted, provided),
                f"axis {axis} of the result changed its length, but the result is in "
                "memory the caller provides",
            )
        return self.array

    def truncate(self, length: ir.Value) -> None:
        raise NotImplementedError("the memory a caller provides can't be shortened")

    def current(self) -> Array:
        return self.array

    @property
    def allocated_layout(self) -> Layout:
        return self.array.layout


@dataclass
class SlotDestination(OutputDestination):
    """An output slot the caller reads back: new memory, or a view of an input."""

    cg: Codegen
    slot: Slot
    scalar_type: ScalarType
    #: Positions of the node inputs the result may be a view of.
    aliased_inputs: tuple[int, ...] = ()
    #: Allocate in the scratch memory of the surrounding compiled unit instead of a
    #: runtime buffer. The slot's owner is then that scratch memory, which the unit
    #: frees (`_emit_steps`).
    in_scratch_memory: bool = False
    #: The packed layout `allocate` produces. None: C-contiguous.
    allocation_layout: Layout | None = None
    #: The layout the result must have, a view or new memory.
    required_layout: Layout | None = None
    #: The alias scope of the memory `allocate` returns, and the scopes it doesn't
    #: alias (`Builder.mark_array`).
    alias_scopes: tuple[str, list[str]] | None = None
    done: bool = False

    @classmethod
    def for_output(
        cls,
        cg: Codegen,
        pointer: ir.Value,
        output: Output,
        aliased_inputs: tuple[int, ...] = (),
        in_scratch_memory: bool = False,
        alias_scopes: tuple[str, list[str]] | None = None,
    ) -> SlotDestination:
        """Create the destination of `output`, a signature `Output` passed in the slot
        at `pointer`.

        New memory is allocated in the output's layout if that layout fixes every
        stride (it is packed). Otherwise it is allocated C-contiguous, which then
        implies the output's layout."""
        representation = output.tensor
        layout = representation.layout
        packed = layout.fully_packed(representation.shape)
        return cls(
            cg,
            Slot(pointer, representation.ndim),
            representation.scalar_type,
            aliased_inputs=aliased_inputs,
            in_scratch_memory=in_scratch_memory,
            allocation_layout=layout if packed else None,
            required_layout=layout,
            alias_scopes=alias_scopes,
        )

    def slot_for_new_memory(self, representation: TensorRepr) -> Slot | None:
        # Memory in scratch is freed by the surrounding unit; memory a kernel allocated
        # into the slot would not be. And the new memory must have the required layout.
        if self.in_scratch_memory or self.required_layout is None:
            return None
        shape, size = (
            representation.shape,
            np.dtype(representation.scalar_type.dtype).itemsize,
        )
        if not representation.layout.implies(self.required_layout, shape, size):
            return None
        self.done = True
        return self.slot

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Allocate a buffer of `shape` in `allocation_layout`, owned by the slot.

        A result that may be a view can still be new memory (a reshape of a
        non-contiguous array): the slot's owner tells the caller which it is."""
        cg, b = self.cg, self.cg.b
        if self.in_scratch_memory:
            # The unit reads it after this node's function returned, and frees it.
            array = cg.temporary(
                self.scalar_type, shape, self.allocation_layout, escapes=True
            )
            self._write(array.data, array.shape, array.strides)
            b.store(array.data, self.slot.field(b, Slot.OWNER))
        else:
            nbytes = b.mul(
                size(b, shape), np.dtype(self.scalar_type.dtype).itemsize, nsw=True
            )
            data = b.rt_alloc(cg.ctx, nbytes, self.slot.field(b, Slot.OWNER))
            cg.check(b.ne(data, b.null()), None)
            array = Array.packed(
                b, self.scalar_type, data, shape, self.allocation_layout
            )
            self._write(data, shape, array.strides)
        if self.alias_scopes is not None:
            b.mark_array(array.data, *self.alias_scopes)
        self.done = True
        return array

    def assign(self, array: Array) -> Array:
        if (
            array.base not in self.aliased_inputs
            or array.scalar_type != self.scalar_type
        ):
            return super().assign(array)
        if self.required_layout is not None:
            shape = array.static_shape()
            if not array.layout.implies(
                self.required_layout, shape, np.dtype(self.scalar_type.dtype).itemsize
            ):
                raise ValueError(
                    f"the view has layout {array.layout.describe(shape)}, but the "
                    f"output is declared {self.required_layout.describe(shape)}"
                )
        if array.base_if is None:
            self.set_view(array)
            return array

        # Only the run time knows whether `array` lies in the input's memory: pass it
        # on as a view if it does, else copy it into new memory.
        def copied() -> None:
            OutputDestination.assign(self, array)

        self.cg.b.if_else(array.base_if, lambda: self.set_view(array), copied)
        # (the view and the copy both have the required layout)
        layout = self.required_layout or Layout.dynamic(self.ndim)
        return self.slot.read(self.cg.b, self.scalar_type, array.static_shape(), layout)

    def set_view(self, array: Array) -> None:
        """Put `array` into the slot as it is, for a caller that knows the result may
        be this view (a compiled unit, which checks at run time)."""
        self._write(array.data, array.shape, array.strides)
        self.done = True

    def finish(self) -> None:
        if not self.done:
            raise ValueError("an output was never set")

    def allocate_once(self, shape: list[ir.Value]) -> Array:
        b = self.cg.b
        with b.if_(b.eq(self.slot.data(b), b.null())):
            self.allocate(shape)
        return self.current()

    def reallocate(self, shape: list[ir.Value]) -> Array:
        cg, b = self.cg, self.cg.b
        current = self.current()
        differs = b.const_bool(False)
        for wanted, length in zip(shape, current.shape, strict=True):
            differs = b.or_(differs, b.ne(wanted, length))
        if differs.const_int == 0:
            return current
        old_owner = b.alloca(ir.ptr)
        b.store(b.null(), old_owner)
        with b.if_(differs):
            b.store(self.slot.owner(b), old_owner)
            self.allocate(shape)
        cg.release_later(old_owner)
        return self.current()

    def truncate(self, length: ir.Value) -> None:
        b = self.cg.b
        b.store(length, b.gep(ir.i64, self.slot.field(b, Slot.LENGTHS), 0))

    def current(self) -> Array:
        return self.slot.read(
            self.cg.b, self.scalar_type, [None] * self.ndim, self.allocated_layout
        )

    @property
    def allocated_layout(self) -> Layout:
        return self.allocation_layout or Layout.c(self.ndim)

    @property
    def ndim(self) -> int:
        return self.slot.ndim

    def _write(
        self, data: ir.Value, shape: Sequence[ir.Value], strides: Sequence[ir.Value]
    ) -> None:
        self.slot.write(self.cg.b, data, shape, strides)


@dataclass
class ScalarDestination(OutputDestination):
    """A ScalarType result, stored through a pointer: `set` it, or compute it into the
    0-d array `allocate([])` returns."""

    cg: Codegen
    pointer: ir.Value
    scalar_type: ScalarType
    array: Array | None = None
    done: bool = False

    def set(self, value: ir.Value) -> None:
        self.cg.b.store(value, self.pointer)
        self.done = True

    def allocate(self, shape: list[ir.Value]) -> Array:
        if shape:
            raise ValueError("a ScalarType output has no axes")
        self.array = self.cg.temporary(self.scalar_type, [])
        return self.array

    def assign(self, array: Array) -> Array:
        self.set(load(self.cg.b, self.scalar_type, array.data))
        return array

    def finish(self) -> None:
        if self.array is not None:
            self.set(load(self.cg.b, self.scalar_type, self.array.data))
        if not self.done:
            raise ValueError("an output was never set")


@dataclass
class Captured(OutputDestination):
    """A result the code running a graph keeps for itself (a Scan's while-condition):
    after `GraphFunction.emit_call`, `value` is the result, an `Array` for a tensor or
    an IR value for a ScalarType."""

    cg: Codegen
    scalar_type: ScalarType
    value: Array | ir.Value | None = None

    def allocate(self, shape: list[ir.Value]) -> Array:
        array = self.cg.temporary(self.scalar_type, shape)
        self.value = array
        return array

    def assign(self, array: Array) -> Array:
        self.value = array
        return array

    def set(self, value: ir.Value) -> None:
        self.value = value

    def finish(self) -> None:
        if self.value is None:
            raise ValueError("an output was never set")

    def scalar(self) -> ir.Value:
        """The result as a value: a ScalarType's, or the element of a 0-d tensor."""
        if isinstance(self.value, Array):
            if self.value.ndim != 0:
                raise ValueError("the result isn't a scalar")
            return load(self.cg.b, self.scalar_type, self.value.data)
        if self.value is None:
            raise ValueError("no result yet")
        return self.value


__all__ = ["Captured", "OutputDestination"]
