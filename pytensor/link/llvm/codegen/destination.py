"""Where a computed value goes: output destinations.

An `OutputDestination` is the place a result ends up in: memory the caller provides, an
output slot the caller reads back, or a pointer to a scalar. Destinations are the
low-level side of outputs. Op emitters don't use them directly: they get the handles
of `interface.py`, which wrap a destination and only offer what the op declared.

Code that runs a whole graph (`GraphFunction.emit_call`) works with destinations, and
ops that produce their outputs by running an inner graph (Scan, Blockwise, compiled
units) pass theirs on, or define their own (a row of a Scan's buffer).
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.emit import (
    Array,
    Codegen,
    copy,
    load,
    packed_array,
    read_slot,
    size,
)
from pytensor.link.llvm.codegen.types import Slot, ir, itemsize, slot_type


class OutputDestination:
    """Where one result goes.

    A producer either asks for memory to write the result into (`allocate`), hands
    over an array that already holds the result (`assign`), or, for a ScalarType
    value, sets it (`set`). `finish` is called once the producer is done.
    """

    codegen: Codegen
    dtype: str

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Return memory of `shape` to write the result into."""
        raise NotImplementedError

    def assign(self, array: Array) -> Array:
        """Make `array` the result.

        The array is kept as a view where this destination accepts one, and copied
        otherwise. Return the array the result now lives in."""
        result = self.allocate(array.shape)
        copy(self.codegen, array, result)
        return result

    def set(self, value: ir.Value) -> None:
        raise TypeError("set() is for ScalarType outputs; tensors are allocated")

    def finish(self) -> None:
        """Called when the producer is done."""

    def offer_slot(self) -> tuple[ir.Value, Layout] | None:
        """Offer the slot a producer may allocate the result into directly.

        `GraphFunction.emit_call` asks this. A node that allocates new memory anyway
        (a kernel, or an output whose shape is only known at run time) can then
        allocate it straight into this slot instead of allocating its own and copying.

        Return `(slot, layout)`, where the new memory must have a layout implying
        `layout`, or None if this destination doesn't accept that (the default). When
        the slot was used, the caller calls `slot_filled()`."""
        return None

    def slot_filled(self) -> None:
        """Record that the result was allocated into the slot from `offer_slot()`."""


@dataclass
class ProvidedDestination(OutputDestination):
    """Memory the caller provides, with the caller's strides."""

    codegen: Codegen
    array: Array

    @property
    def dtype(self) -> str:  # type: ignore[override]
        return self.array.dtype

    def allocate(self, shape: list[ir.Value]) -> Array:
        return self.array


@dataclass
class SlotDestination(OutputDestination):
    """An output slot the caller reads back: new memory, or a view of an input."""

    codegen: Codegen
    slot: ir.Value
    dtype: str
    ndim: int
    #: Positions of the node inputs the result may be a view of.
    aliased_inputs: tuple[int, ...] = ()
    #: Allocate in the scratch memory of the surrounding compiled unit instead of a
    #: runtime buffer. The slot's owner is then that scratch memory, which the unit
    #: frees (`emit_steps`).
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
        codegen: Codegen,
        slot: ir.Value,
        output,
        aliased_inputs: tuple[int, ...] = (),
        **kwargs,
    ) -> SlotDestination:
        """Create the destination of `output`, a signature `Output` passed in a slot.

        New memory is allocated in the output's layout if that layout fixes every
        stride (it is packed). Otherwise it is allocated C-contiguous, which then
        implies the output's layout."""
        representation = output.representation
        packed = output.layout.fully_packed(representation.shape)
        return cls(
            codegen,
            slot,
            representation.dtype,
            representation.ndim,
            aliased_inputs=aliased_inputs,
            allocation_layout=output.layout if packed else None,
            required_layout=output.layout,
            **kwargs,
        )

    def offer_slot(self) -> tuple[ir.Value, Layout] | None:
        # Memory in scratch is freed by the surrounding unit; memory a kernel allocated
        # into the slot would not be.
        if self.in_scratch_memory or self.required_layout is None:
            return None
        return self.slot, self.required_layout

    def slot_filled(self) -> None:
        self.done = True

    def allocate(self, shape: list[ir.Value]) -> Array:
        """Allocate a buffer of `shape` in `allocation_layout`, owned by the slot.

        A result that may be a view can still be new memory (a reshape of a
        non-contiguous array): the slot's owner tells the caller which it is."""
        codegen, builder = self.codegen, self.codegen.b
        if self.in_scratch_memory:
            # The unit reads it after this node's function returned, and frees it.
            array = codegen.temporary(
                self.dtype, shape, self.allocation_layout, escapes=True
            )
            self._write(array.data, array.shape, array.strides)
            builder.store(array.data, self._field(Slot.OWNER))
        else:
            nbytes = builder.mul(size(builder, shape), itemsize(self.dtype), nsw=True)
            data = builder.rt_alloc(codegen.ctx, nbytes, self._field(Slot.OWNER))
            codegen.check(builder.ne(data, builder.null()), None)
            array = packed_array(
                builder, self.dtype, data, shape, self.allocation_layout
            )
            self._write(data, shape, array.strides)
        if self.alias_scopes is not None:
            builder.mark_array(array.data, *self.alias_scopes)
        self.done = True
        return array

    def assign(self, array: Array) -> Array:
        if array.base not in self.aliased_inputs or array.dtype != self.dtype:
            return super().assign(array)
        if self.required_layout is not None:
            shape = array.static_shape()
            if not array.layout.implies(
                self.required_layout, shape, itemsize(self.dtype)
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

        self.codegen.b.if_else(array.base_if, lambda: self.set_view(array), copied)
        # (the view and the copy both have the required layout)
        layout = self.required_layout or Layout.dynamic(self.ndim)
        return read_slot(
            self.codegen.b, self.slot, self.dtype, array.static_shape(), layout
        )

    def set_view(self, array: Array) -> None:
        """Put `array` into the slot as it is, for a caller that knows the result may
        be this view (a compiled unit, which checks at run time)."""
        self._write(array.data, array.shape, array.strides)
        self.done = True

    def finish(self) -> None:
        if not self.done:
            raise ValueError("an output was never set")

    # ---- for code that works on the slot itself (Scan, Blockwise)

    def data(self) -> ir.Value:
        """Return the data pointer in the slot (null until set)."""
        return self.codegen.b.load(ir.ptr, self._field(Slot.DATA))

    def owner(self) -> ir.Value:
        """Return the buffer reference in the slot (null for a view)."""
        return self.codegen.b.load(ir.ptr, self._field(Slot.OWNER))

    def read(self) -> Array:
        """Return the array in the slot (after `allocate`), in `allocation_layout`."""
        layout = self.allocation_layout or Layout.c(self.ndim)
        return read_slot(
            self.codegen.b, self.slot, self.dtype, [None] * self.ndim, layout
        )

    def set_length(self, axis: int, length: ir.Value) -> None:
        """Shorten `axis` of the array in the slot to `length`, keeping its strides."""
        builder = self.codegen.b
        builder.store(length, builder.gep(ir.i64, self._field(Slot.LENGTHS), axis))

    def _field(self, field: int) -> ir.Value:
        return self.codegen.b.field_ptr(slot_type(self.ndim), self.slot, field)

    def _write(self, data, shape, strides) -> None:
        builder = self.codegen.b
        builder.store(data, self._field(Slot.DATA))
        lengths, stride_fields = self._field(Slot.LENGTHS), self._field(Slot.STRIDES)
        for axis in range(self.ndim):
            builder.store(shape[axis], builder.gep(ir.i64, lengths, axis))
            builder.store(strides[axis], builder.gep(ir.i64, stride_fields, axis))


@dataclass
class ScalarDestination(OutputDestination):
    """A ScalarType result, stored through a pointer: `set` it, or compute it into the
    0-d array `allocate([])` returns."""

    codegen: Codegen
    pointer: ir.Value
    dtype: str
    array: Array | None = None
    done: bool = False

    def set(self, value: ir.Value) -> None:
        self.codegen.b.store(value, self.pointer)
        self.done = True

    def allocate(self, shape: list[ir.Value]) -> Array:
        if shape:
            raise ValueError("a ScalarType output has no axes")
        self.array = self.codegen.temporary(self.dtype, [])
        return self.array

    def assign(self, array: Array) -> Array:
        self.set(load(self.codegen.b, self.dtype, array.data))
        return array

    def finish(self) -> None:
        if self.array is not None:
            self.set(load(self.codegen.b, self.dtype, self.array.data))
        if not self.done:
            raise ValueError("an output was never set")


__all__ = [
    "OutputDestination",
    "ProvidedDestination",
    "ScalarDestination",
    "SlotDestination",
]
