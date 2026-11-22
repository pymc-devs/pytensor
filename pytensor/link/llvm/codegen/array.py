"""Arrays in generated code (`Array`) and the output slots nodes fill (`Slot`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from pytensor.link.llvm.analysis.layout import DYNAMIC, PACKED, Layout, StrideKind
from pytensor.link.llvm.codegen.num import Num, cast, constant, load, store
from pytensor.link.llvm.codegen.types import TensorRepr, ir, memory_type
from pytensor.scalar.basic import ScalarType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.context import Codegen


@dataclass
class Array:
    """An array inside generated code: data pointer, one i64 per length and byte stride.

    Lengths and strides are constants where the type, the layout or an allocation fixes
    them. `layout` is what is known about the strides when generating the code (default:
    nothing); code uses it to pick loops (analysis/layout.py).
    """

    #: the type of the elements
    scalar_type: ScalarType
    data: ir.Value
    #: one i64 per axis
    shape: list[ir.Value]
    #: one i64 per axis, in bytes
    strides: list[ir.Value]
    layout: Layout
    #: the node input this memory belongs to (inputs and their views), if known
    base: int | None = None
    #: may code write it? Inputs may not (`store` refuses)
    writable: bool = True
    #: Some nodes return either a view of an input or new memory, and only the run
    #: time knows which. For such a value, `base` is the input it may be a view of, and
    #: `base_if` is an i1 that is true when it really is one. None means it certainly
    #: is (if `base` is set).
    base_if: ir.Value | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, ScalarType):
            raise TypeError(f"expected a ScalarType, got {self.scalar_type!r}")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def itemsize(self) -> int:
        return np.dtype(self.scalar_type.dtype).itemsize

    def static_shape(self) -> list[int | None]:
        """The lengths as ints, None where not constant."""
        return [s.const_int for s in self.shape]

    def with_layout(self, layout: Layout) -> Array:
        """Return the same array, claiming `layout` for its strides."""
        return replace(self, layout=layout)

    def view(
        self,
        data: ir.Value,
        shape: Sequence[ir.Value],
        strides: Sequence[ir.Value],
        layout: Layout | None = None,
    ) -> Array:
        """An array in the same memory: `data` points into it. It belongs to the same
        input and is as writable as this one. `layout`: default nothing known."""
        return Array(
            self.scalar_type,
            data,
            list(shape),
            list(strides),
            unknown_layout(shape) if layout is None else layout,
            self.base,
            self.writable,
            self.base_if,
        )

    # ---- elements (indices are i64 values or Python ints; nothing is bounds-checked)

    def ptr(self, b: ir.Builder, *index: ir.Value | int) -> ir.Value:
        """Address of the element at `index` (one index per axis)."""
        if len(index) != self.ndim:
            raise ValueError(f"{len(index)} indices for an array with {self.ndim} axes")
        p = self.data
        for i, stride in zip(index, self.strides, strict=True):
            p = advance(b, p, i, stride, self.scalar_type)
        return p

    def load(self, b: ir.Builder, *index: ir.Value | int) -> Num:
        """The element at `index`, as a `Num` of the array's dtype."""
        return Num(b, load(b, self.scalar_type, self.ptr(b, *index)), self.scalar_type)

    def store(
        self, b: ir.Builder, value: Num | ir.Value | float, *index: ir.Value | int
    ) -> None:
        """Write `value` to the element at `index`: a `Num` of the array's dtype, an IR
        value of its type, or a Python number."""
        if isinstance(value, Num):
            if value.scalar_type != self.scalar_type:
                raise TypeError(
                    f"storing a {value.scalar_type} into an array of {self.scalar_type}: convert with astype"
                )
            value = value.value
        elif isinstance(value, bool | int | float):
            value = constant(b, self.scalar_type, value)
        if not self.writable:
            raise ValueError(
                "storing into an input: inputs are only read, unless the op may "
                "destroy them (`writable`). Write the output (`out.allocate(shape)`)"
            )
        store(b, self.scalar_type, value, self.ptr(b, *index))

    def at(self, b: ir.Builder, i: ir.Value | int, axis: int = 0) -> Array:
        """The sub-array at index `i` of `axis` (`a[i]`, `a[:, i]`, ...): a view with
        one axis less, with what its layout still says."""
        from pytensor.link.llvm.analysis.layout import remove_axis

        axis = range(self.ndim)[axis]
        keep = [a for a in range(self.ndim) if a != axis]
        return self.view(
            advance(b, self.data, i, self.strides[axis], self.scalar_type),
            [self.shape[a] for a in keep],
            [self.strides[a] for a in keep],
            remove_axis(
                self.layout,
                axis,
                self.static_shape(),
                self.itemsize,
            ),
        )

    # ---- construction and conversion

    @classmethod
    def c(
        cls,
        scalar_type: ScalarType,
        data: ir.Value,
        shape: Sequence[ir.Value],
        strides: Sequence[ir.Value],
    ) -> Array:
        """A C-contiguous array (strides from `c_strides`)."""
        return cls(scalar_type, data, list(shape), list(strides), Layout.c(len(shape)))

    @classmethod
    def packed(
        cls,
        b: ir.Builder,
        scalar_type: ScalarType,
        data: ir.Value,
        shape: Sequence[ir.Value],
        layout: Layout | None = None,
    ) -> Array:
        """An array at `data` packed in `layout` (all axes packed; default C-contiguous)."""
        nd = len(shape)
        static = [n.const_int for n in shape]
        if layout is None:
            layout = Layout.c(nd)
        layout = Layout.make(static, layout.order, layout.strides)
        if not layout.fully_packed(static):
            raise ValueError(f"can't allocate an array of layout {layout}")
        return cls(
            scalar_type,
            data,
            list(shape),
            layout.stride_values(b, np.dtype(scalar_type.dtype).itemsize, shape, {}),
            layout,
        )

    @classmethod
    def unpack(
        cls, b: ir.Builder, representation: TensorRepr, value: ir.Value
    ) -> Array:
        """The Array of a tensor value in its representation."""
        if representation.ndim == 0:
            return cls(representation.scalar_type, value, [], [], Layout.c(0))
        data = b.extract(value, 0)
        lengths = b.extract(value, 1)
        shape: list[ir.Value] = []
        k = 0
        for s in representation.shape:
            if s is None:
                shape.append(b.extract(lengths, k))
                k += 1
            else:
                shape.append(b.i64(s))
        passed: dict[int, ir.Value] = {}
        if representation.stride_axes:
            st = b.extract(value, 2)
            passed = {
                a: b.extract(st, k) for k, a in enumerate(representation.stride_axes)
            }
        strides = representation.layout.stride_values(
            b, np.dtype(representation.scalar_type.dtype).itemsize, shape, passed
        )
        return cls(
            representation.scalar_type, data, shape, strides, representation.layout
        )

    def pack(self, b: ir.Builder, representation: TensorRepr) -> ir.Value:
        """A tensor value in its representation."""
        if representation.ndim == 0:
            return self.data
        lengths = b.aggregate(
            ir.Type.array(ir.i64, len(representation.dynamic_axes)),
            [self.shape[k] for k in representation.dynamic_axes],
        )
        strides = b.aggregate(
            ir.Type.array(ir.i64, len(representation.stride_axes)),
            [self.strides[k] for k in representation.stride_axes],
        )
        return b.aggregate(representation.ir_type, [self.data, lengths, strides])

    # ---- views and layouts

    def view_layout(
        self,
        shape: Sequence[ir.Value],
        strides: Sequence[ir.Value],
        source: Sequence[int | None],
    ) -> Layout:
        """The layout of a view of this array with `shape` and `strides` (i64 values) whose axis k
        is axis `source[k]` of this one (None: an axis of its own, e.g. a broadcast one).

        An axis that keeps its stride keeps its kind, but PACKED only while the packed
        axes inside it (in this array's memory order) keep their strides and lengths: a packed
        stride is the product of those lengths. A zero stride is 0; any other is DYNAMIC."""
        kept = {
            k: s
            for k, s in enumerate(source)
            if s is not None and strides[k] is self.strides[s]
        }
        view_axis = {s: k for k, s in kept.items()}
        static = self.static_shape()
        still_packed: set[int] = set()
        for ax in reversed(self.layout.order):
            if self.layout.strides[ax] is not PACKED:
                break
            if ax not in view_axis:
                if static[ax] == 1:  # (contributes nothing to the products)
                    continue
                break
            still_packed.add(ax)
            if not _same_length(shape[view_axis[ax]], self.shape[ax]):
                break
        kinds: list[StrideKind | int] = []
        for k, st in enumerate(strides):
            s = kept.get(k)
            kind = DYNAMIC if s is None else self.layout.strides[s]
            if kind is PACKED and s not in still_packed:
                kind = DYNAMIC
            if kind is DYNAMIC and st.const_int == 0:
                kind = 0
            kinds.append(kind)
        # own axes first; this array's axes in its memory order (which orders the packed ones)
        pos = {ax: i for i, ax in enumerate(self.layout.order)}
        order = sorted(
            range(len(shape)),
            key=lambda k: (k in kept, pos[kept[k]] if k in kept else k),
        )
        return Layout.make([n.const_int for n in shape], order, kinds)

    def loop_view(
        self, b: ir.Builder, shape: Sequence[ir.Value], zero_axes: Sequence[int] = ()
    ) -> Array:
        """This array iterated over the loop `shape`: axes of static length 1 here (broadcast)
        and `zero_axes` get stride 0; the layout records that."""
        zero = b.i64(0)
        static = self.static_shape()
        bcast = {
            ax
            for ax in range(self.ndim)
            if ax in zero_axes or (static[ax] == 1 and shape[ax].const_int != 1)
        }
        strides = [zero if ax in bcast else st for ax, st in enumerate(self.strides)]
        kinds = [0 if ax in bcast else k for ax, k in enumerate(self.layout.strides)]
        loop_static = [n.const_int for n in shape]
        return Array(
            self.scalar_type,
            self.data,
            list(shape),
            strides,
            Layout.make(loop_static, self.layout.order, kinds),
        )

    def is_contiguous(
        self,
        b: ir.Builder,
        shape: Sequence[ir.Value],
        order: Sequence[int] | None = None,
    ) -> ir.Value:
        """i1: is this array packed over `shape` in axis `order` (outermost first; default C)? A
        constant when the layout settles it; else checked at run time."""
        nd = len(shape)
        order = list(range(nd)) if order is None else list(order)
        static = [n.const_int for n in shape]
        layout = self.layout
        same_order = [ax for ax in layout.order if static[ax] != 1] == [
            ax for ax in order if static[ax] != 1
        ]
        if layout.fully_packed(static) and same_order:
            return b.const_bool(True)
        if not layout.dynamic_axes and not (layout.fully_packed(static) and same_order):
            # the layout is fully known and isn't this one (degenerate lengths at run time
            # just take the general path)
            if not all(st.const_int is not None for st in self.strides):
                return b.const_bool(False)
        ok = b.const_bool(True)
        expect = b.i64(self.itemsize)
        for ax in reversed(order):
            if static[ax] == 1:
                continue
            st = self.strides[ax]
            if st.const_int is not None and expect.const_int is not None:
                same = b.const_bool(st.const_int == expect.const_int)
            else:
                eq = b.eq(st, expect)
                one = b.eq(shape[ax], 1)
                same = b.or_(eq, one)
            ok = b.and_(ok, same)
            expect = b.mul(expect, shape[ax], nsw=True)
        return ok

    def check_strides(self, cg: Codegen, layout: Layout, what: str) -> None:
        """Fail at run time unless the strides satisfy `layout`.

        Length-1 axes and empty arrays don't count, as in numpy's contiguity rules. `what`
        names this array in the error message."""
        b = cg.b
        expected = layout.stride_values(
            b,
            self.itemsize,
            self.shape,
            dict(enumerate(self.strides)),
        )
        matches = b.const_bool(True)
        empty = b.const_bool(False)
        for length, actual, wanted in zip(
            self.shape, self.strides, expected, strict=True
        ):
            irrelevant = b.eq(length, 1)
            matches = b.and_(matches, b.or_(irrelevant, b.eq(actual, wanted)))
            empty = b.or_(empty, b.eq(length, 0))
        cg.check(
            b.or_(matches, empty),
            f"layout check: {what} doesn't have layout {layout}",
        )

    # ---- copying

    def copy_from(self, cg: Codegen, src: Array) -> None:
        """Copy (and convert) `src`, of the same shape, into this array."""
        b = cg.b

        def body(p: list[ir.Value]) -> None:
            store(
                b,
                self.scalar_type,
                cast(
                    b, load(b, src.scalar_type, p[0]), src.scalar_type, self.scalar_type
                ),
                p[1],
            )

        cg.for_each(self.shape, [src, self], body, parallel_cost=1.0)


def unknown_layout(shape: Sequence[ir.Value]) -> Layout:
    """`Layout.unknown` for the lengths `shape` (i64 values)."""
    return Layout.unknown([n.const_int for n in shape])


def _same_length(m: ir.Value, n: ir.Value) -> bool:
    return m is n or (m.const_int is not None and m.const_int == n.const_int)


def c_strides(
    b: ir.Builder, scalar_type: ScalarType, shape: Sequence[ir.Value]
) -> list[ir.Value]:
    """C-contiguous byte strides for `shape` (i64 values)."""
    strides: list[ir.Value] = []
    acc = b.i64(np.dtype(scalar_type.dtype).itemsize)
    for length in reversed(shape):
        strides.append(acc)
        acc = b.mul(acc, length, nsw=True)
    return strides[::-1]


def size(b: ir.Builder, shape: Sequence[ir.Value]) -> ir.Value:
    """The number of elements of `shape` (i64 values)."""
    n = b.i64(1)
    for s in shape:
        n = b.mul(n, s, nsw=True)
    return n


def advance(
    b: ir.Builder,
    p: ir.Value,
    i: ir.Value | int,
    stride: ir.Value,
    scalar_type: ScalarType,
) -> ir.Value:
    """`p + i * stride` (`stride` in bytes), addressed in elements of `scalar_type`.

    Loops address their elements with this: LLVM's vectorizer versions a loop on a
    stride only known at run time by speculating that it is one *element*. With byte
    offsets it speculates a byte stride of 1, which never holds, and the vectorized loop
    never runs."""
    isz = np.dtype(scalar_type.dtype).itemsize
    if isinstance(i, int):
        i = b.i64(i)
    c = stride.const_int
    if c is not None:
        if c % isz:
            return b.gep_bytes(p, b.mul(i, stride, nsw=True))
        es: ir.Value | int = c // isz
    elif isz == 1:
        es = stride
    else:
        # strides of arrays are multiples of the itemsize (validated at the boundary)
        es = b.ashr(stride, isz.bit_length() - 1)
    return b.gep(memory_type(scalar_type), p, b.mul(i, es, nsw=True))


# ------------------------------------------------------------------ output slots


@cache
def _slot_ir_type(ndim: int) -> ir.Type:
    return ir.Type.struct_(
        [ir.ptr, ir.Type.array(ir.i64, ndim), ir.Type.array(ir.i64, ndim), ir.ptr]
    )


@dataclass(frozen=True)
class Slot:
    """An output slot, which a node fills with its result:
    `{ptr data, [ndim x i64] lengths, [ndim x i64] strides, ptr owner}`.

    The owner is the buffer reference of new memory, and null for a view. Whoever
    created the slot releases the owner once the result is no longer needed."""

    pointer: ir.Value
    ndim: int

    DATA: ClassVar[int] = 0
    LENGTHS: ClassVar[int] = 1
    STRIDES: ClassVar[int] = 2
    OWNER: ClassVar[int] = 3

    @staticmethod
    def ir_type(ndim: int) -> ir.Type:
        return _slot_ir_type(ndim)

    @classmethod
    def alloca(cls, b: ir.Builder, ndim: int) -> Slot:
        """Return an empty slot (all fields null) on the stack."""
        t = _slot_ir_type(ndim)
        pointer = b.alloca(t)
        b.store(b.zero(t), pointer)
        return cls(pointer, ndim)

    def field(self, b: ir.Builder, field: int) -> ir.Value:
        """Return a pointer to `field` (`Slot.DATA`, ...)."""
        return b.field_ptr(_slot_ir_type(self.ndim), self.pointer, field)

    def data(self, b: ir.Builder) -> ir.Value:
        return b.load(ir.ptr, self.field(b, Slot.DATA))

    def owner(self, b: ir.Builder) -> ir.Value:
        return b.load(ir.ptr, self.field(b, Slot.OWNER))

    def write(
        self,
        b: ir.Builder,
        data: ir.Value,
        shape: Sequence[ir.Value],
        strides: Sequence[ir.Value],
    ) -> None:
        """Store an array's data, lengths and strides (not its owner)."""
        b.store(data, self.field(b, Slot.DATA))
        lengths, stride_fields = (
            self.field(b, Slot.LENGTHS),
            self.field(b, Slot.STRIDES),
        )
        for axis in range(self.ndim):
            b.store(shape[axis], b.gep(ir.i64, lengths, axis))
            b.store(strides[axis], b.gep(ir.i64, stride_fields, axis))

    def read(
        self,
        b: ir.Builder,
        scalar_type: ScalarType,
        static_shape: Sequence[int | None],
        layout: Layout | None = None,
    ) -> Array:
        """Return the array in the slot. Static lengths stay constants; with a known
        `layout`, only its dynamic strides are read."""
        data = self.data(b)
        lp, sp = self.field(b, Slot.LENGTHS), self.field(b, Slot.STRIDES)
        shape = [
            b.i64(s) if s is not None else b.load(ir.i64, b.gep(ir.i64, lp, a))
            for a, s in enumerate(static_shape)
        ]
        if layout is None:
            layout = Layout.unknown(static_shape)
        else:
            layout = Layout.make(static_shape, layout.order, layout.strides)
        passed = {a: b.load(ir.i64, b.gep(ir.i64, sp, a)) for a in layout.dynamic_axes}
        strides = layout.stride_values(
            b, np.dtype(scalar_type.dtype).itemsize, shape, passed
        )
        return Array(scalar_type, data, shape, strides, layout)

    def release(self, b: ir.Builder, ctx: ir.Value) -> None:
        """Give up the owner (a no-op for null)."""
        b.rt_decref(ctx, self.owner(b))


__all__ = [
    "Array",
    "advance",
    "c_strides",
    "size",
]
