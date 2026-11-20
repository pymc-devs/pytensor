"""What an op registers to have its code generated, and what its emitter gets.

An op registers

* `llvm_emit(op, node, cg, inputs, outputs)`: emits the code of a node (required);
* `llvm_layout(op, node, in_layouts)`: the layouts of its outputs, where the default
  isn't right (see there).

The emitter gets

* `cg`: the `Codegen` of the function being built (`cg.b` is the IR builder);
* `inputs`, one per node input:

  - an `Array` for a tensor: the caller's memory, read-only, unless the op destroys
    that input (its `destroy_map`) and the planner lets it: then `writable` is set, and
    the emitter may write it and return it (or a view of it) for an output the op
    destroys it for (`overwrite_or_copy`);
  - an `ir.Value` for a ScalarType value;
  - None for a value without data (a random generator).

  Constants the planner folds into the code arrive the same way;
* `outputs`, one per node output: a `NodeOutput` (None for a value without data). The
  emitter produces each of them, with

  - `out.allocate(shape)`: memory to write the result into (in the output's layout);
  - `out.view(a)`: the result is `a`, which lies in an input's memory: a view where
    the output may be one (`view_map`, or the writable input the op destroys for it),
    else a copy (it returns which);
  - `out.set(value)`: the value of a ScalarType output.

  Where the memory comes from (the caller's, a new buffer, a unit's temporary) is not
  the emitter's business.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

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
from pytensor.link.llvm.layout import Layout


@singledispatch
def llvm_emit(op, node, cg: Codegen, inputs: list, outputs: list) -> None:
    """Emit the code of `node` (see the module's docstring). Raise NotImplementedError
    for a node it can't handle: something else runs it then."""
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@singledispatch
def llvm_layout(op, node, in_layouts) -> list:
    """The layout (layout.py) of each output of `node`, given the inputs' layouts (None
    for inputs that aren't tensors). None for an output: the default.

    The default is C-contiguous for an output in new memory, and nothing known
    (DYNAMIC) for one that may be a view. Ops register their own to say what a view's
    layout is (DimShuffle, Subtensor, ...) or in which memory order they want their
    output (Elemwise: its inputs').

    A layout is binding: whatever runs the node gives the output this layout. New memory
    is provided or allocated in it (it must then be packed; one that isn't means "any
    strides": C-contiguous memory is used). A view must have it (`NodeOutput.view`
    checks), and so must memory a may-be-view output allocates instead (a Reshape that
    copies declares C for both). An output the op may write over an input has that
    input's layout instead (`signature.node_outputs`)."""
    return [None] * len(node.outputs)


# ------------------------------------------------------------------ outputs


class NodeOutput:
    """An output of the node being emitted (see the module's docstring)."""

    cg: Codegen
    dtype: str

    def allocate(self, shape: list) -> Array:
        """Memory of `shape` for the result, in the output's layout."""
        raise NotImplementedError

    def view(self, a: Array) -> Array:
        """The result is `a`, which lies in an input's memory: a view of it where this
        output may be one, else a copy. Returns the output's array: `a`, or the copy."""
        mine = self.allocate(a.shape)
        copy(self.cg, a, mine)
        return mine

    def set(self, value) -> None:
        raise TypeError("set() is for ScalarType outputs; tensors are allocated")

    def finish(self) -> None:
        """Called when the emitter is done."""


@dataclass
class ProvidedOutput(NodeOutput):
    """Memory the caller provides, with the caller's strides."""

    cg: Codegen
    array: Array

    @property
    def dtype(self):
        return self.array.dtype

    def allocate(self, shape: list) -> Array:
        return self.array


@dataclass
class SlotOutput(NodeOutput):
    """An output returned through a slot: memory it allocates, or a view of an input it
    may alias."""

    cg: Codegen
    slot: ir.Value
    dtype: str
    ndim: int
    #: the node inputs it may be a view of (the op's view_map; an input it may
    #: overwrite)
    aliases: tuple = ()
    #: allocate in temporary memory instead of a runtime buffer (a temporary inside a
    #: compiled unit): the slot's owner is then the scratch memory, which the unit
    #: frees (`emit_items`; null for a view)
    temporary: bool = False
    #: the (packed) layout `allocate` produces; default C-contiguous
    layout: Layout | None = None
    #: for a view: the layout it must have (`view` checks)
    declared: Layout | None = None
    #: (scope, noalias scopes) for the memory `allocate` returns (`Builder.mark_array`)
    alias: tuple | None = None
    done: bool = False

    def allocate(self, shape: list) -> Array:
        """A fresh buffer of `shape` in the slot's layout (packed; default C); the slot
        owns it. An output that may be a view can always allocate instead (a reshape of
        a non-contiguous array does): the owner it sets tells the caller."""
        cg, b = self.cg, self.cg.b
        if self.temporary:
            # read by the unit after this node's function returned, which frees it
            a = cg.temporary(self.dtype, shape, self.layout, escapes=True)
            self._write(a.data, a.shape, a.strides)
            b.store(a.data, b.field_ptr(slot_type(self.ndim), self.slot, Slot.OWNER))
        else:
            nbytes = b.mul(size(b, shape), itemsize(self.dtype), nsw=True)
            owner = b.field_ptr(slot_type(self.ndim), self.slot, Slot.OWNER)
            data = b.rt_alloc(cg.ctx, nbytes, owner)
            cg.check(b.ne(data, b.null()), None)
            a = packed_array(b, self.dtype, data, shape, self.layout)
            self._write(data, shape, a.strides)
        if self.alias is not None:
            b.mark_array(a.data, *self.alias)
        self.done = True
        return a

    def view(self, a: Array) -> Array:
        if a.base not in self.aliases or a.dtype != self.dtype:
            return super().view(a)
        if self.declared is not None:
            shape = a.static_shape()
            if not a.layout.implies(self.declared, shape, itemsize(self.dtype)):
                raise ValueError(
                    f"the view has layout {a.layout.describe(shape)}, but the output "
                    f"is declared {self.declared.describe(shape)} (llvm_layout)"
                )
        self.set_view(a)
        return a

    def set_view(self, a: Array) -> None:
        """Put `a` into the slot as it is: the caller knows it is a view the output may
        be (a unit, which checks at run time)."""
        self._write(a.data, a.shape, a.strides)
        self.done = True

    def finish(self) -> None:
        if not self.done:
            raise ValueError("an output was never set")

    # ---- for code that works on the slot itself (Scan)

    def data(self):
        """The data pointer in the slot (null until set)."""
        b = self.cg.b
        return b.load(ir.ptr, b.field_ptr(slot_type(self.ndim), self.slot, Slot.DATA))

    def owner(self):
        """The buffer reference in the slot (null for views)."""
        b = self.cg.b
        return b.load(ir.ptr, b.field_ptr(slot_type(self.ndim), self.slot, Slot.OWNER))

    def read(self) -> Array:
        """The array in the slot (after `allocate`), in the slot's layout."""
        lay = self.layout if self.layout is not None else Layout.c(self.ndim)
        return read_slot(self.cg.b, self.slot, self.dtype, [None] * self.ndim, lay)

    def set_length(self, axis, n):
        """Shorten axis `axis` of the array in the slot to `n` (i64) -- a view of the
        same memory with the same strides."""
        b = self.cg.b
        lp = b.field_ptr(slot_type(self.ndim), self.slot, Slot.LENGTHS)
        b.store(n, b.gep(ir.i64, lp, axis))

    def _write(self, data, shape, strides):
        b = self.cg.b
        t = slot_type(self.ndim)
        b.store(data, b.field_ptr(t, self.slot, Slot.DATA))
        lp = b.field_ptr(t, self.slot, Slot.LENGTHS)
        sp = b.field_ptr(t, self.slot, Slot.STRIDES)
        for a in range(self.ndim):
            b.store(shape[a], b.gep(ir.i64, lp, a))
            b.store(strides[a], b.gep(ir.i64, sp, a))


@dataclass
class ScalarOutput(NodeOutput):
    """A ScalarType output, stored through a pointer: `set` it, or compute it into
    the 0-d array `allocate([])` gives."""

    cg: Codegen
    ptr: ir.Value
    dtype: str
    array: Array | None = None
    done: bool = False

    def set(self, value) -> None:
        self.cg.b.store(value, self.ptr)
        self.done = True

    def allocate(self, shape: list) -> Array:
        if shape:
            raise ValueError("a ScalarType output has no axes")
        self.array = self.cg.temporary(self.dtype, [])
        return self.array

    def view(self, a: Array) -> Array:
        self.set(load(self.cg.b, self.dtype, a.data))
        return a

    def finish(self) -> None:
        if self.array is not None:
            self.set(load(self.cg.b, self.dtype, self.array.data))
        if not self.done:
            raise ValueError("an output was never set")


def overwrite_or_copy(cg, out: NodeOutput, x: Array) -> Array:
    """Memory holding the values of `x` for the result of `out`, to update: `x` itself
    where the op may destroy it (the output is then `x`), else a copy of it in the
    output's memory."""
    if x.writable:
        return out.view(x)
    a = out.allocate(x.shape)
    copy(cg, x, a)
    return a


__all__ = [
    "NodeOutput",
    "ProvidedOutput",
    "ScalarOutput",
    "SlotOutput",
    "llvm_emit",
    "llvm_layout",
    "overwrite_or_copy",
]
