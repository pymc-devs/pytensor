"""Helpers for generating code: arrays, outputs, errors, workspace, casts, loops."""

from __future__ import annotations

from dataclasses import dataclass, field

from pytensor.link.llvm_v2.layout import DYNAMIC, PACKED, Layout
from pytensor.link.llvm_v2.codegen.types import (
    ScalarRepr,
    TensorRepr,
    ir,
    is_float,
    is_signed,
    itemsize,
    memory_type,
    value_type,
)


# ------------------------------------------------------------------ arrays


@dataclass
class Array:
    """An array inside generated code: data pointer, one i64 per length and byte stride.

    Lengths and strides are constants where the type, the layout or an allocation fixes
    them. `layout` is what is known about the strides when generating the code (default:
    nothing); code uses it to pick loops (layout.py).
    """

    dtype: str
    data: object
    shape: list
    strides: list
    layout: Layout = None

    def __post_init__(self):
        if self.layout is None:
            nd = len(self.shape)
            self.layout = Layout.make(self.static_shape(), range(nd), (DYNAMIC,) * nd)

    @property
    def ndim(self):
        return len(self.shape)

    def static_shape(self):
        """The lengths as ints, None where not constant."""
        return [s.const_int for s in self.shape]

    def with_strides(self, shape, strides, layout=None):
        """The same memory with other lengths/strides (e.g. a broadcast view)."""
        return Array(self.dtype, self.data, list(shape), list(strides), layout)


def c_array(dtype, data, shape, strides) -> Array:
    """A C-contiguous array (strides from `c_strides`)."""
    return Array(dtype, data, list(shape), list(strides), Layout.c(len(shape)))


def packed_array(b, dtype, data, shape, layout: Layout | None = None) -> Array:
    """An array at `data` packed in `layout` (all axes packed; default C-contiguous)."""
    nd = len(shape)
    static = [n.const_int for n in shape]
    if layout is None:
        layout = Layout.c(nd)
    layout = Layout.make(static, layout.order, layout.strides)
    if not layout.fully_packed(static):
        raise ValueError(f"can't allocate an array of layout {layout}")
    return Array(dtype, data, list(shape), layout.stride_values(b, itemsize(dtype), shape, {}), layout)


def loop_view(b, a: Array, shape, zero_axes=()) -> Array:
    """`a` iterated over the loop `shape`: axes of static length 1 in `a` (broadcast)
    and `zero_axes` get stride 0; the layout records that."""
    m = ir()
    zero = b.const_int(m.i64, 0)
    static = a.static_shape()
    bcast = {ax for ax in range(a.ndim) if ax in zero_axes or (static[ax] == 1 and shape[ax].const_int != 1)}
    strides = [zero if ax in bcast else st for ax, st in enumerate(a.strides)]
    kinds = [0 if ax in bcast else k for ax, k in enumerate(a.layout.strides)]
    loop_static = [n.const_int for n in shape]
    return Array(a.dtype, a.data, list(shape), strides, Layout.make(loop_static, a.layout.order, kinds))


def c_strides(b, dtype, shape):
    """C-contiguous byte strides for `shape` (i64 values)."""
    m = ir()
    strides = [None] * len(shape)
    acc = b.const_int(m.i64, itemsize(dtype))
    for a in reversed(range(len(shape))):
        strides[a] = acc
        acc = mul(b, acc, shape[a])
    return strides


def mul(b, x, y):
    """i64 product, folded when both are constants."""
    m = ir()
    cx, cy = x.const_int, y.const_int
    if cx is not None and cy is not None:
        return b.const_int(m.i64, cx * cy)
    if cx == 1:
        return y
    if cy == 1:
        return x
    return b.mul(x, y, nsw=True)


def size(b, shape):
    m = ir()
    n = b.const_int(m.i64, 1)
    for s in shape:
        n = mul(b, n, s)
    return n


def unpack(b, r: TensorRepr, value) -> Array:
    """The Array of a tensor value in its representation."""
    m = ir()
    if r.ndim == 0:
        return Array(r.dtype, value, [], [], Layout.c(0))
    data = b.extract(value, 0)
    lengths = b.extract(value, 1)
    shape, k = [], 0
    for s in r.shape:
        if s is None:
            shape.append(b.extract(lengths, k))
            k += 1
        else:
            shape.append(b.const_int(m.i64, s))
    passed = {}
    if r.stride_axes:
        st = b.extract(value, 2)
        passed = {a: b.extract(st, k) for k, a in enumerate(r.stride_axes)}
    strides = r.layout.stride_values(b, itemsize(r.dtype), shape, passed)
    return Array(r.dtype, data, shape, strides, r.layout)


def pack(b, r: TensorRepr, a: Array):
    """A tensor value in its representation."""
    m = ir()
    if r.ndim == 0:
        return a.data
    lengths = b.aggregate(m.Type.array(m.i64, len(r.dynamic_axes)), [a.shape[k] for k in r.dynamic_axes])
    strides = b.aggregate(m.Type.array(m.i64, len(r.stride_axes)), [a.strides[k] for k in r.stride_axes])
    return b.aggregate(r.ir_type, [a.data, lengths, strides])


def load(b, dtype, ptr):
    """Load an element; bools become i1."""
    m = ir()
    v = b.load(memory_type(dtype), ptr)
    if dtype == "bool":
        return b.icmp(m.ICmp.NE, v, b.const_int(m.i8, 0))
    return v


def store(b, dtype, value, ptr):
    m = ir()
    if dtype == "bool":
        value = b.zext(value, m.i8)
    b.store(value, ptr)


def read_slot(b, slot, dtype, static_shape, layout: Layout | None = None) -> Array:
    """The array a callee wrote into an output slot (static lengths stay constants;
    with a known `layout`, only its dynamic strides are read)."""
    m = ir()
    nd = len(static_shape)
    t = _slot_type(nd)
    data = b.load(m.ptr, b.field_ptr(t, slot, 0))
    lp, sp = b.field_ptr(t, slot, 1), b.field_ptr(t, slot, 2)
    shape = [
        b.const_int(m.i64, s) if s is not None else b.load(m.i64, b.gep(m.i64, lp, b.const_int(m.i64, a)))
        for a, s in enumerate(static_shape)
    ]
    if layout is None:
        layout = Layout.make(static_shape, range(nd), (DYNAMIC,) * nd)
    else:
        layout = Layout.make(static_shape, layout.order, layout.strides)
    passed = {a: b.load(m.i64, b.gep(m.i64, sp, b.const_int(m.i64, a))) for a in layout.dynamic_axes}
    strides = layout.stride_values(b, itemsize(dtype), shape, passed)
    return Array(dtype, data, shape, strides, layout)


def element_ptr(b, a: Array, index: list):
    """Address of the element at `index` (one i64 per axis)."""
    p = a.data
    for i, st in zip(index, a.strides, strict=True):
        p = b.gep_bytes(p, mul(b, i, st))
    return p


# ------------------------------------------------------------------ casts


def constant(b, dtype, value):
    m = ir()
    if is_float(dtype):
        return b.const_float(value_type(dtype), float(value))
    if dtype == "bool":
        return b.const_bool(bool(value))
    return b.const_int(value_type(dtype), int(value))


def cast(b, v, src: str, dst: str):
    """Convert a value like numpy's astype."""
    m = ir()
    if src == dst:
        return v
    st, dt = value_type(src), value_type(dst)
    if dst == "bool":
        if is_float(src):
            return b.fcmp(m.FCmp.UNE, v, b.const_float(st, 0.0))
        return b.icmp(m.ICmp.NE, v, b.const_int(st, 0))
    if is_float(src) and is_float(dst):
        return b.fpext(v, dt) if dt.bits > st.bits else b.fptrunc(v, dt)
    if is_float(dst):
        return b.sitofp(v, dt) if is_signed(src) else b.uitofp(v, dt)
    if is_float(src):
        return b.fptosi(v, dt) if is_signed(dst) else b.fptoui(v, dt)
    # integer to integer (bool is i1, unsigned)
    if dt.bits > st.bits:
        return b.sext(v, dt) if is_signed(src) else b.zext(v, dt)
    if dt.bits < st.bits:
        return b.trunc(v, dt)
    return v  # int8 <-> uint8


# ------------------------------------------------------------------ outputs


@dataclass
class ProvidedOutput:
    """The caller's memory, with the caller's strides."""

    array: Array


@dataclass
class SlotOutput:
    """An output the callee sets through a slot (allocated, or a view of input `view_of`)."""

    cx: "EmitContext"
    slot: object
    dtype: str
    ndim: int
    view_of: int | None = None
    #: allocate in the workspace instead of a runtime buffer (a temporary inside a
    #: compiled unit): the slot gets no owner
    temporary: bool = False
    #: the (packed) layout `allocate` produces; default C-contiguous
    layout: Layout = None
    done: bool = False

    def _write(self, data, shape, strides):
        b = self.cx.b
        t = _slot_type(self.ndim)
        b.store(data, b.field_ptr(t, self.slot, 0))
        lp, sp = b.field_ptr(t, self.slot, 1), b.field_ptr(t, self.slot, 2)
        m = ir()
        for a in range(self.ndim):
            b.store(shape[a], b.gep(m.i64, lp, b.const_int(m.i64, a)))
            b.store(strides[a], b.gep(m.i64, sp, b.const_int(m.i64, a)))

    def allocate(self, shape: list) -> Array:
        """A fresh buffer of `shape` in the slot's layout (packed; default C); the slot
        owns it. (A view output may allocate too, e.g. a reshape of a non-contiguous
        array: the owner it sets then takes precedence over the viewed input's.)"""
        cx, b, m = self.cx, self.cx.b, ir()
        if self.temporary:
            a = cx.temporary(self.dtype, shape, self.layout)
            self._write(a.data, a.shape, a.strides)
            self.done = True
            return a
        nbytes = mul(b, size(b, shape), b.const_int(m.i64, itemsize(self.dtype)))
        owner = b.field_ptr(_slot_type(self.ndim), self.slot, 3)
        data = b.rt_alloc(cx.ctx, nbytes, owner)
        cx.check(b.icmp(m.ICmp.NE, data, b.null()), None)
        a = packed_array(b, self.dtype, data, shape, self.layout)
        self._write(data, shape, a.strides)
        self.done = True
        return a

    def set_view(self, a: Array):
        """The output is `a`, which lies in the memory of input `view_of` (the caller
        sets the owner)."""
        if self.view_of is None:
            raise ValueError("not a view output")
        self._write(a.data, a.shape, a.strides)
        self.done = True


@dataclass
class ScalarOutput:
    ptr: object
    dtype: str
    done: bool = False

    def set(self, b, value):
        b.store(value, self.ptr)
        self.done = True


def _slot_type(ndim):
    from pytensor.link.llvm_v2.codegen.types import slot_type

    return slot_type(ndim)


# ------------------------------------------------------------------ the context


@dataclass
class EmitContext:
    """What generated code needs besides the builder: the runtime context, the
    environment, the workspace."""

    b: object
    ctx: object
    env: object
    ws: object
    #: bytes of static workspace used so far
    ws_static: int = 0
    _checks: int = field(default=0)

    def fail(self, message: str | None):
        """Report `message` (unless the callee already did) and return 1."""
        m = ir()
        if message is not None:
            self.b.rt_fail(self.ctx, message)
        self.b.ret(self.b.const_int(m.i32, 1))

    def check(self, ok, message: str | None):
        """Fail with `message` unless `ok` (i1). Constant conditions fold."""
        m = ir()
        c = ok.const_int
        if c == 1:
            return
        not_ok = self.b.xor(ok, self.b.const_bool(True))
        with self.b.if_(not_ok):
            self.fail(message)

    def workspace(self, nbytes):
        """Temporary memory of `nbytes` (i64), 64-byte aligned, valid during this call."""
        m = ir()
        b = self.b
        n = nbytes.const_int
        if n is not None:
            off = self.ws_static
            self.ws_static += max(64, -(-n // 64) * 64)
            return b.gep_bytes(self.ws, b.const_int(m.i64, off))
        p = b.rt_ws_alloc(self.ctx, nbytes)
        self.check(b.icmp(m.ICmp.NE, p, b.null()), None)
        return p

    def temporary(self, dtype, shape, layout: Layout | None = None) -> Array:
        """A temporary array in the workspace, packed in `layout` (default C)."""
        b, m = self.b, ir()
        nbytes = mul(b, size(b, shape), b.const_int(m.i64, itemsize(dtype)))
        data = self.workspace(nbytes)
        return packed_array(b, dtype, data, shape, layout)


# ------------------------------------------------------------------ loops


def is_contiguous(b, shape, a: Array, order=None):
    """i1: is `a` packed over `shape` in axis `order` (outermost first; default C)? A
    constant when the layout settles it; else checked at run time."""
    m = ir()
    nd = len(shape)
    order = list(range(nd)) if order is None else list(order)
    static = [n.const_int for n in shape]
    lay = a.layout
    same_order = [ax for ax in lay.order if static[ax] != 1] == [ax for ax in order if static[ax] != 1]
    if lay.fully_packed(static) and same_order:
        return b.const_bool(True)
    if not lay.dynamic_axes and not (lay.fully_packed(static) and same_order):
        # the layout is fully known and isn't this one (degenerate lengths at run time
        # just take the general path)
        if not all(st.const_int is not None for st in a.strides):
            return b.const_bool(False)
    ok = b.const_bool(True)
    expect = b.const_int(m.i64, itemsize(a.dtype))
    for ax in reversed(order):
        if static[ax] == 1:
            continue
        st = a.strides[ax]
        if st.const_int is not None and expect.const_int is not None:
            same = b.const_bool(st.const_int == expect.const_int)
        else:
            eq = b.icmp(m.ICmp.EQ, st, expect)
            one = b.icmp(m.ICmp.EQ, shape[ax], b.const_int(m.i64, 1))
            same = b.or_(eq, one)
        ok = _and(b, ok, same)
        expect = mul(b, expect, shape[ax])
    return ok


def _and(b, x, y):
    if x.const_int == 1:
        return y
    if y.const_int == 1:
        return x
    if x.const_int == 0 or y.const_int == 0:
        return b.const_bool(False)
    return b.and_(x, y)


def loop_order(shape, arrays) -> list[int]:
    """The axis order (outermost first) to loop in: the memory order most arrays share
    (by their packed axes; on a tie, the one of the later array, usually an output).
    Identity when nothing is known."""
    static = [n.const_int for n in shape]
    votes = {}  # packed block (innermost first) -> (count, last position)
    for pos, a in enumerate(arrays):
        block = tuple(a.layout.packed_block(static))
        if block:
            count, _ = votes.get(block, (0, -1))
            votes[block] = (count + len(block), pos)
    if not votes:
        return list(range(len(shape)))
    block = max(votes, key=lambda k: votes[k])
    rest = [ax for ax in range(len(shape)) if ax not in block]
    return rest + list(reversed(block))


def for_each(b, shape, arrays, body):
    """For every index of `shape`: `body(ptrs)` with the element pointers of `arrays`.

    `arrays` have strides over `shape` (0 where broadcast). The loops follow memory order
    (`loop_order`). When every array is packed in that order this is one flat loop over
    element indices (what the vectorizer likes). The layouts decide that when they can;
    only arrays whose layout leaves it open are checked at run time, and then both the
    flat loop and the nest are generated.
    """
    order = loop_order(shape, arrays)
    contig = b.const_bool(True)
    for a in arrays:
        contig = _and(b, contig, is_contiguous(b, shape, a, order))
    c = contig.const_int
    if c == 1:
        _flat(b, shape, arrays, body)
    elif c == 0:
        _strided(b, shape, arrays, body, order)
    else:
        b.if_else(
            contig, lambda: _flat(b, shape, arrays, body) or [], lambda: _strided(b, shape, arrays, body, order) or []
        )


def _flat(b, shape, arrays, body):
    n = size(b, shape)
    with b.loop(n) as i:
        body([b.gep(memory_type(a.dtype), a.data, i) for a in arrays])


def _strided(b, shape, arrays, body, order, level=0, base=None):
    """Loops over the axes in `order` (outermost first; may be a subset of the axes)."""
    if base is None:
        base = [a.data for a in arrays]
    if level == len(order):
        body(base)
        return
    axis = order[level]
    with b.loop(shape[axis]) as i:
        ptrs = [b.gep_bytes(p, mul(b, i, a.strides[axis])) for p, a in zip(base, arrays, strict=True)]
        _strided(b, shape, arrays, body, order, level + 1, ptrs)


def copy(cx, src: Array, dst: Array):
    """Copy (and convert) `src` into `dst` of the same shape."""
    b = cx.b

    def body(p):
        store(b, dst.dtype, cast(b, load(b, src.dtype, p[0]), src.dtype, dst.dtype), p[1])

    for_each(b, dst.shape, [src, dst], body)


__all__ = [
    "Array",
    "c_array",
    "packed_array",
    "loop_order",
    "loop_view",
    "EmitContext",
    "ProvidedOutput",
    "ScalarOutput",
    "ScalarRepr",
    "SlotOutput",
    "TensorRepr",
    "cast",
    "constant",
    "copy",
    "for_each",
    "load",
    "store",
]
