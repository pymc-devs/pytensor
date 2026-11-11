"""Helpers for generating code: arrays, outputs, errors, temporary memory, casts, loops."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pytensor.link.llvm.layout import DYNAMIC, Layout
from pytensor.link.llvm.codegen.types import (
    ir,
    is_float,
    is_signed,
    itemsize,
    memory_type,
    ScalarRepr,
    Slot,
    TensorRepr,
    value_type,
)


#: Temporaries of a static size up to this many bytes are stack memory (`temp_memory`),
#: as long as one generated function has no more than the budget of them.
STACK_TEMPORARY_BYTES = 1024
STACK_BUDGET_BYTES = 16 * 1024

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

    # ---- elements (indices are i64 values or Python ints; nothing is bounds-checked)

    def ptr(self, b, *index):
        """Address of the element at `index` (one index per axis)."""
        if len(index) != self.ndim:
            raise ValueError(f"{len(index)} indices for an array with {self.ndim} axes")
        p = self.data
        for i, stride in zip(index, self.strides, strict=True):
            p = advance(b, p, i, stride, self.dtype)
        return p

    def load(self, b, *index):
        """The element at `index`, as a `Num` of the array's dtype."""
        from pytensor.link.llvm.codegen.num import Num

        return Num(b, load(b, self.dtype, self.ptr(b, *index)), self.dtype)

    def store(self, b, value, *index):
        """Write `value` to the element at `index`: a `Num` of the array's dtype, an IR
        value of its type, or a Python number."""
        from pytensor.link.llvm.codegen.num import Num

        if isinstance(value, Num):
            if value.dtype != self.dtype:
                raise TypeError(f"storing a {value.dtype} into an array of {self.dtype}: convert with astype")
            value = value.value
        elif isinstance(value, bool | int | float):
            value = constant(b, self.dtype, value)
        store(b, self.dtype, value, self.ptr(b, *index))

    def at(self, b, i, axis: int = 0) -> Array:
        """The sub-array at index `i` of `axis` (`a[i]`, `a[:, i]`, ...): a view with
        one axis less, with what its layout still says."""
        from pytensor.link.llvm.layout import remove_axis

        axis = range(self.ndim)[axis]
        keep = [a for a in range(self.ndim) if a != axis]
        return Array(
            self.dtype,
            advance(b, self.data, i, self.strides[axis], self.dtype),
            [self.shape[a] for a in keep],
            [self.strides[a] for a in keep],
            remove_axis(self.layout, axis, self.static_shape(), itemsize(self.dtype)),
        )


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
    zero = b.i64(0)
    static = a.static_shape()
    bcast = {ax for ax in range(a.ndim) if ax in zero_axes or (static[ax] == 1 and shape[ax].const_int != 1)}
    strides = [zero if ax in bcast else st for ax, st in enumerate(a.strides)]
    kinds = [0 if ax in bcast else k for ax, k in enumerate(a.layout.strides)]
    loop_static = [n.const_int for n in shape]
    return Array(a.dtype, a.data, list(shape), strides, Layout.make(loop_static, a.layout.order, kinds))


def c_strides(b, dtype, shape):
    """C-contiguous byte strides for `shape` (i64 values)."""
    strides = [None] * len(shape)
    acc = b.i64(itemsize(dtype))
    for a in reversed(range(len(shape))):
        strides[a] = acc
        acc = b.mul(acc, shape[a], nsw=True)
    return strides


def size(b, shape):
    n = b.i64(1)
    for s in shape:
        n = b.mul(n, s, nsw=True)
    return n


def unpack(b, r: TensorRepr, value) -> Array:
    """The Array of a tensor value in its representation."""
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
            shape.append(b.i64(s))
    passed = {}
    if r.stride_axes:
        st = b.extract(value, 2)
        passed = {a: b.extract(st, k) for k, a in enumerate(r.stride_axes)}
    strides = r.layout.stride_values(b, itemsize(r.dtype), shape, passed)
    return Array(r.dtype, data, shape, strides, r.layout)


def pack(b, r: TensorRepr, a: Array):
    """A tensor value in its representation."""
    if r.ndim == 0:
        return a.data
    lengths = b.aggregate(ir.Type.array(ir.i64, len(r.dynamic_axes)), [a.shape[k] for k in r.dynamic_axes])
    strides = b.aggregate(ir.Type.array(ir.i64, len(r.stride_axes)), [a.strides[k] for k in r.stride_axes])
    return b.aggregate(r.ir_type, [a.data, lengths, strides])


def load(b, dtype, ptr):
    """Load an element; bools become i1."""
    v = b.load(memory_type(dtype), ptr)
    if dtype == "bool":
        return b.ne(v, b.i8(0))
    return v


def store(b, dtype, value, ptr):
    if dtype == "bool":
        value = b.zext(value, ir.i8)
    b.store(value, ptr)


def read_slot(b, slot, dtype, static_shape, layout: Layout | None = None) -> Array:
    """The array a kernel wrote into an output slot (static lengths stay constants;
    with a known `layout`, only its dynamic strides are read)."""
    nd = len(static_shape)
    t = _slot_type(nd)
    data = b.load(ir.ptr, b.field_ptr(t, slot, Slot.DATA))
    lp, sp = b.field_ptr(t, slot, Slot.LENGTHS), b.field_ptr(t, slot, Slot.STRIDES)
    shape = [
        b.i64(s) if s is not None else b.load(ir.i64, b.gep(ir.i64, lp, b.i64(a)))
        for a, s in enumerate(static_shape)
    ]
    if layout is None:
        layout = Layout.make(static_shape, range(nd), (DYNAMIC,) * nd)
    else:
        layout = Layout.make(static_shape, layout.order, layout.strides)
    passed = {a: b.load(ir.i64, b.gep(ir.i64, sp, b.i64(a))) for a in layout.dynamic_axes}
    strides = layout.stride_values(b, itemsize(dtype), shape, passed)
    return Array(dtype, data, shape, strides, layout)


def advance(b, p, i, stride, dtype):
    """`p + i * stride` (`stride` in bytes), addressed in elements of `dtype`.

    Loops address their elements with this: LLVM's vectorizer versions a loop on a
    stride only known at run time by speculating that it is one *element*. With byte
    offsets it speculates a byte stride of 1, which never holds, and the vectorized loop
    never runs."""
    isz = itemsize(dtype)
    if isinstance(i, int):
        i = b.i64(i)
    c = stride.const_int
    if c is not None:
        if c % isz:
            return b.gep_bytes(p, b.mul(i, stride, nsw=True))
        es = b.i64(c // isz)
    elif isz == 1:
        es = stride
    else:
        # strides of arrays are multiples of the itemsize (validated at the boundary)
        es = b.ashr(stride, b.i64(isz.bit_length() - 1))
    return b.gep(memory_type(dtype), p, b.mul(i, es, nsw=True))


# ------------------------------------------------------------------ casts


def constant(b, dtype, value):
    if is_float(dtype):
        return b.const_float(value_type(dtype), float(value))
    if dtype == "bool":
        return b.const_bool(bool(value))
    return b.const_int(value_type(dtype), int(value))


def cast(b, v, src: str, dst: str):
    """Convert a value like numpy's astype."""
    if src == dst:
        return v
    st, dt = value_type(src), value_type(dst)
    if dst == "bool":
        if is_float(src):
            return b.fcmp(ir.FCmp.UNE, v, b.const_float(st, 0.0))
        return b.ne(v, b.const_int(st, 0))
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
    """An output the kernel sets through a slot (allocated, or a view of input `view_of`)."""

    cg: "Codegen"
    slot: object
    dtype: str
    ndim: int
    view_of: int | None = None
    #: allocate in temporary memory instead of a runtime buffer (a temporary inside a
    #: compiled unit): the slot gets no owner
    temporary: bool = False
    #: the (packed) layout `allocate` produces; default C-contiguous
    layout: Layout = None
    #: for a view: the layout `llvm_outputs` declared (`set_view` checks it)
    declared: Layout = None
    #: (scope, noalias scopes) for the memory `allocate` returns (`Builder.mark_array`)
    alias: tuple | None = None
    done: bool = False

    def _write(self, data, shape, strides):
        b = self.cg.b
        t = _slot_type(self.ndim)
        b.store(data, b.field_ptr(t, self.slot, Slot.DATA))
        lp, sp = b.field_ptr(t, self.slot, Slot.LENGTHS), b.field_ptr(t, self.slot, Slot.STRIDES)
        for a in range(self.ndim):
            b.store(shape[a], b.gep(ir.i64, lp, b.i64(a)))
            b.store(strides[a], b.gep(ir.i64, sp, b.i64(a)))

    def allocate(self, shape: list) -> Array:
        """A fresh buffer of `shape` in the slot's layout (packed; default C); the slot
        owns it. (A view output may allocate too, e.g. a reshape of a non-contiguous
        array: the owner it sets then takes precedence over the viewed input's.)"""
        cg, b = self.cg, self.cg.b
        if self.temporary:
            # read by the unit after this node's function returned
            a = cg.temporary(self.dtype, shape, self.layout, escapes=True)
            self._write(a.data, a.shape, a.strides)
            self._mark(a)
            self.done = True
            return a
        nbytes = b.mul(size(b, shape), b.i64(itemsize(self.dtype)), nsw=True)
        owner = b.field_ptr(_slot_type(self.ndim), self.slot, Slot.OWNER)
        data = b.rt_alloc(cg.ctx, nbytes, owner)
        cg.check(b.ne(data, b.null()), None)
        a = packed_array(b, self.dtype, data, shape, self.layout)
        self._write(data, shape, a.strides)
        self._mark(a)
        self.done = True
        return a

    def data(self):
        """The data pointer in the slot (null until set)."""
        b = self.cg.b
        return b.load(ir.ptr, b.field_ptr(_slot_type(self.ndim), self.slot, Slot.DATA))

    def owner(self):
        """The buffer reference in the slot (null for views)."""
        b = self.cg.b
        return b.load(ir.ptr, b.field_ptr(_slot_type(self.ndim), self.slot, Slot.OWNER))

    def read(self) -> Array:
        """The array in the slot (after `allocate`), in the slot's layout."""
        lay = self.layout if self.layout is not None else Layout.c(self.ndim)
        return read_slot(self.cg.b, self.slot, self.dtype, [None] * self.ndim, lay)

    def set_length(self, axis, n):
        """Shorten axis `axis` of the array in the slot to `n` (i64) -- a view of the
        same memory with the same strides."""
        b = self.cg.b
        lp = b.field_ptr(_slot_type(self.ndim), self.slot, Slot.LENGTHS)
        b.store(n, b.gep(ir.i64, lp, b.i64(axis)))

    def _mark(self, a: Array):
        if self.alias is not None:
            self.cg.b.mark_array(a.data, *self.alias)

    def set_view(self, a: Array):
        """The output is `a`, which lies in the memory of input `view_of` (the caller
        sets the owner)."""
        if self.view_of is None:
            raise ValueError("not a view output")
        if self.declared is not None:
            shape = a.static_shape()
            if not a.layout.implies(self.declared, shape, itemsize(self.dtype)):
                raise ValueError(
                    f"the view has layout {a.layout.describe(shape)}, but llvm_outputs declared "
                    f"{self.declared.describe(shape)}"
                )
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
    from pytensor.link.llvm.codegen.types import slot_type

    return slot_type(ndim)


# ------------------------------------------------------------------ the context


@dataclass
class Codegen:
    """The state of generating one kernel's code: the builder and the kernel's
    parameters (`ctx`, its `config` = link table), and what nested code
    needs (link table, planner, module)."""

    b: object
    ctx: object
    #: the kernel's config: its link table (call.py)
    config: object
    #: bytes of stack temporaries so far (`temp_memory`)
    stack_bytes: int = 0
    #: may loops run in parallel (`Options.parallel`)?
    parallel: bool = True
    #: the link table of the kernel being generated (call.py); None: it calls nothing
    links: object = None
    #: the planner (program.py), for code that plans inner graphs (Scan); None: no
    #: inner graphs
    planner: object = None
    #: the module being built, and the name of the function being built in it (nested
    #: functions are named after it)
    module: object = None
    name: str = ""
    _checks: int = field(default=0)

    def par(self, per_element: float) -> ParallelPolicy | None:
        """How loops over elements costing `per_element` each (`work.py`) may run in
        parallel, or None."""
        return ParallelPolicy(self, per_element) if self.parallel and per_element > 0 else None

    def fail(self, message: str | None):
        """Report `message` (unless a called kernel already did) and return 1."""
        if message is not None:
            self.b.rt_fail(self.ctx, message)
        self.b.ret(self.b.i32(1))

    def check(self, ok, message: str | None):
        """Fail with `message` unless `ok` (i1). Constant conditions fold."""
        c = ok.const_int
        if c == 1:
            return
        not_ok = self.b.xor(ok, self.b.const_bool(True))
        with self.b.if_(not_ok):
            self.fail(message)

    def temp_memory(self, nbytes, escapes: bool = False):
        """Temporary memory of `nbytes` (i64), 64-byte aligned, uninitialized.

        Small memory of static size is on the stack of the function being generated:
        LLVM sees all of it, and may keep it in registers. Everything else is scratch
        memory of the context (`rt_scratch`), which is valid until the task ends.

        `escapes`: the memory is used after the function being generated returns (an
        output that is a temporary of the unit around it); then it is never on the stack.

        Each call is a new piece of scratch: ask before a loop, not inside it."""
        b = self.b
        n = nbytes.const_int
        if n is not None and not escapes and n <= STACK_TEMPORARY_BYTES:
            rounded = max(64, -(-n // 64) * 64)
            if self.stack_bytes + rounded <= STACK_BUDGET_BYTES:
                self.stack_bytes += rounded
                return b.alloca(ir.Type.array(ir.i8, max(n, 1)))
        p = b.rt_scratch(self.ctx, nbytes)
        self.check(b.ne(p, b.null()), None)
        return p

    def temporary(self, dtype, shape, layout: Layout | None = None, escapes: bool = False) -> Array:
        """A temporary array (`temp_memory`), packed in `layout` (default C). `shape`:
        i64 values or Python ints."""
        b = self.b
        shape = [b.i64(n) if isinstance(n, int) else n for n in shape]
        nbytes = b.mul(size(b, shape), b.i64(itemsize(dtype)), nsw=True)
        data = self.temp_memory(nbytes, escapes)
        return packed_array(b, dtype, data, shape, layout)


# ------------------------------------------------------------------ loops


def is_contiguous(b, shape, a: Array, order=None):
    """i1: is `a` packed over `shape` in axis `order` (outermost first; default C)? A
    constant when the layout settles it; else checked at run time."""
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
    expect = b.i64(itemsize(a.dtype))
    for ax in reversed(order):
        if static[ax] == 1:
            continue
        st = a.strides[ax]
        if st.const_int is not None and expect.const_int is not None:
            same = b.const_bool(st.const_int == expect.const_int)
        else:
            eq = b.eq(st, expect)
            one = b.eq(shape[ax], b.i64(1))
            same = b.or_(eq, one)
        ok = b.and_(ok, same)
        expect = b.mul(expect, shape[ax], nsw=True)
    return ok


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


# ------------------------------------------------------------------ parallel loops

#: cost (work.py units) of one chunk of a parallel loop: enough to make the overhead
#: of a parallel region (microseconds) small
CHUNK_COST = 50_000
#: loops cheaper than this stay serial (decided when generating the code when the
#: shapes are static; at run time by the runtime otherwise: one chunk runs directly)
MIN_PARALLEL_COST = 2 * CHUNK_COST


@dataclass
class ParallelPolicy:
    """A loop whose iterations may run in parallel: the context, the cost of one item."""

    cg: Codegen
    per_element: float

    def worth_it(self, n_elements) -> bool:
        """False if the loop is statically too small for a parallel region."""
        n = n_elements.const_int
        return n is None or n * self.per_element >= MIN_PARALLEL_COST

    def grain(self, b, elements_per_iteration):
        """Iterations per chunk (i64), for iterations of `elements_per_iteration` items."""
        per = elements_per_iteration.const_int
        if per is not None:
            return b.i64(max(1, math.ceil(CHUNK_COST / (self.per_element * max(per, 1)))))
        target = b.i64(max(1, math.ceil(CHUNK_COST / self.per_element)))
        one = b.i64(1)
        per = b.select(b.sgt(elements_per_iteration, one), elements_per_iteration, one)
        g = b.sdiv(target, per)
        return b.select(b.sgt(g, one), g, one)

    def range(self, b, n, elements_per_iteration, body):
        """`body(i)` for i in 0..n, in chunks that may run in parallel; failures in a
        chunk fail the function."""
        with b.parallel(self.cg.ctx, n, self.grain(b, elements_per_iteration)) as r:
            with b.loop(b.sub(r.end, r.start)) as j:
                body(b.add(r.start, j))
        self.cg.check(b.eq(r.result, b.i32(0)), None)


def for_each(b, shape, arrays, body, par: ParallelPolicy | None = None):
    """For every index of `shape`: `body(ptrs)` with the element pointers of `arrays`.

    `arrays` have strides over `shape` (0 where broadcast). The loops follow memory order
    (`loop_order`). When every array is packed in that order this is one flat loop over
    element indices (what the vectorizer likes). The layouts decide that when they can;
    only arrays whose layout leaves it open are checked at run time, and then both the
    flat loop and the nest are generated.

    With `par`, the loop (the flat one, or the nest's outermost) runs in chunks that may
    run in parallel when there's enough work. Iterations must not depend on each other.
    """
    order = loop_order(shape, arrays)
    if par is not None and not par.worth_it(size(b, shape)):
        par = None
    contig = b.const_bool(True)
    for a in arrays:
        contig = b.and_(contig, is_contiguous(b, shape, a, order))
    c = contig.const_int
    if c == 1:
        _flat(b, shape, arrays, body, par)
    elif c == 0:
        _strided(b, shape, arrays, body, order, par=par)
    else:
        b.if_else(
            contig,
            lambda: _flat(b, shape, arrays, body, par),
            lambda: _strided(b, shape, arrays, body, order, par=par),
        )


def _flat(b, shape, arrays, body, par=None):
    n = size(b, shape)

    def at(i):
        body([b.gep(memory_type(a.dtype), a.data, i) for a in arrays])

    if par is not None:
        par.range(b, n, b.i64(1), at)
        return
    with b.loop(n) as i:
        at(i)


def _strided(b, shape, arrays, body, order, level=0, base=None, par=None):
    """Loops over the axes in `order` (outermost first; may be a subset of the axes).
    With `par`, the outermost loop runs in chunks that may run in parallel."""
    if base is None:
        base = [a.data for a in arrays]
    if level == len(order):
        body(base)
        return
    axis = order[level]

    def at(i):
        ptrs = [advance(b, p, i, a.strides[axis], a.dtype) for p, a in zip(base, arrays, strict=True)]
        _strided(b, shape, arrays, body, order, level + 1, ptrs)

    if par is not None and level == 0:
        inner = size(b, [shape[a] for a in order[1:]])
        par.range(b, shape[axis], inner, at)
        return
    with b.loop(shape[axis]) as i:
        at(i)


def copy(cg, src: Array, dst: Array):
    """Copy (and convert) `src` into `dst` of the same shape."""
    b = cg.b

    def body(p):
        store(b, dst.dtype, cast(b, load(b, src.dtype, p[0]), src.dtype, dst.dtype), p[1])

    for_each(b, dst.shape, [src, dst], body, cg.par(1.0))


__all__ = [
    "Array",
    "advance",
    "c_array",
    "packed_array",
    "loop_order",
    "loop_view",
    "Codegen",
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
