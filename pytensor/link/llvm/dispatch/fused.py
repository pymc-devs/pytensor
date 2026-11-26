"""Code generation for FusedElemwise: an Elemwise together with the indexing and
reductions around it, generated as one loop. The rewrite that creates these nodes is in
pytensor.tensor.rewriting.fused_elemwise.

For example, with vectors `x`, `w`, `t` and an integer vector `idx`:

    y = exp(x[idx]) * w           # the loop reads x[idx[i]], x[idx] is never built
    s = y.sum()                   # the loop adds y[i] to s, y is never built
    u = inc_subtensor(t[idx], y)  # the loop adds y[i] to u[idx[i]], u is a copy of t

The node's inputs are the Elemwise's inputs, with `x` in place of `x[idx]`, then the
index arrays, then the write targets. If a node has a pattern this code can't generate,
`fused_supported` says so, and the planner runs the node's inner graph op by op instead.

The code is organized as follows:

1. `fused_spec` reads from the op which Elemwise inputs are read through an index
   (`Read`), and which outputs are reduced (`Reduce`) or written through an index
   (`Write`).
2. `loop_shape` works out the length of each loop axis.
3. Each Elemwise input and output gets a member object that generates its part of the
   work at one element. For inputs these are `LoopInvariant`, `PlainInput` and
   `IndexedRead`; for outputs `PlainOutput`, `Reduction` and `IndexedWrite`.
4. `Body` generates the work at one element: it asks the input members for their
   values, applies the scalar op, and hands each output member its value.
5. `Codegen.for_each` generates the loops around it. `Body.innermost` generates the
   innermost loop itself, so that outputs can keep values in registers across it.

Keeping values in registers matters for two kinds of outputs. A sum keeps its running
total in a register instead of loading and storing it at every element, which also lets
LLVM vectorize the loop. An increment keeps the element it is adding to in a register
while the index stays the same. An output says whether it does this with
`Output.carries`. The values it keeps are passed into the code for each element and
returned from it, see `Output.element`.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self

import pytensor.scalar as ps
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.link.llvm.analysis.layout import Layout, Stride
from pytensor.link.llvm.codegen.array import Array, advance, size
from pytensor.link.llvm.codegen.context import (
    Codegen,
    Element,
    IndexBounds,
    LoopIndex,
    index_bounds,
)
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
    EmitterOutput,
    TensorOutput,
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.num import cast, constant, load, store
from pytensor.link.llvm.codegen.types import element_type, ir, memory_type
from pytensor.link.llvm.dispatch.elemwise import identity_value
from pytensor.link.llvm.dispatch.scalar import emit_scalar, scalar_supported
from pytensor.scalar.basic import ScalarOp, ScalarType, add
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.subtensor import AdvancedSubtensor


if TYPE_CHECKING:
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise


#: The number of elements in a chunk of `chunked_loop`.
CHUNK = 256

OUT_OF_BOUNDS = "FusedElemwise: index out of bounds ({})"


# ------------------------------------------------------------------ the spec


@dataclass(frozen=True)
class Read:
    """An Elemwise input that is `x[idx]`, where `idx` is a 1-d integer array that
    indexes one axis of `x`. The loop reads `x[idx[i]]` directly, so `x[idx]` is never
    built."""

    #: The position of `idx` among the node's inputs.
    index_input: int
    #: The axis of `x` that `idx` indexes, for example 1 for `x[:, idx]`.
    axis: int


@dataclass(frozen=True)
class Write:
    """An Elemwise output `y` that is only used in `inc_subtensor(t[idx], y)` or
    `set_subtensor(t[idx], y)`. The loop adds or writes `y[i]` into `t[idx[i]]`, in a
    copy of `t`, so `y` is never built."""

    #: The position of `idx` among the node's inputs.
    index_input: int
    #: The axis of `t` that `idx` indexes.
    axis: int
    #: The position of `t` among the node's inputs.
    target_input: int
    #: True for `set_subtensor`, False for `inc_subtensor`.
    set: bool
    #: How many more axes `t` has than `y`. The rewrite puts these axes last, so that
    #: `m[:, idx]` of a matrix `m` becomes `m.T[idx]` with one extra axis. Each `y[i]`
    #: then goes into every element of `t[idx[i]]`.
    extra: int


@dataclass(frozen=True)
class Reduce:
    """An Elemwise output `y` that is only used in a reduction, like `y.sum(axis=0)`.
    The loop combines each element of `y` into the result, so `y` is never built."""

    #: The op that combines two values: add for a sum, mul for a product, and so on.
    scalar_op: ScalarOp
    #: The loop axes that are reduced.
    axes: tuple[int, ...]
    #: The value the result starts from, for example 0 for a sum. `fused_supported`
    #: rejects reductions where it is None.
    identity: bool | float | int | None
    #: The type the values are combined in. It can differ from the output's type.
    acc: ScalarType


@dataclass(frozen=True)
class FusedSpec:
    """What a FusedElemwise node computes, as read from the op by `fused_spec`."""

    #: The Elemwise node of the inner graph.
    elemwise: Apply[Elemwise]
    #: Its scalar op, which the loop applies at every element.
    scalar_op: ScalarOp
    #: The number of Elemwise inputs. They are the node's first inputs.
    n_elem: int
    #: For each Elemwise input, a `Read` if it is read through an index, else None.
    reads: tuple[Read | None, ...]
    #: For each output, a `Reduce` or a `Write`, or None if it is used as it is.
    outs: tuple[Reduce | Write | None, ...]

    @property
    def ndim(self) -> int:
        """The number of axes of the loop."""
        return int(self.elemwise.outputs[0].type.ndim)


def fused_spec(op: FusedElemwise, node: Apply) -> FusedSpec:
    """Read what `node` computes from its op. Raises NotImplementedError for patterns
    this code can't generate."""
    if any(isinstance(v.type, ScalarType) for v in (*node.inputs, *node.outputs)):
        # Only the numba-only `scalarize` rewrite makes inputs or outputs scalars.
        raise NotImplementedError("FusedElemwise: scalar inputs or outputs")
    inner = op.fgraph
    elems = [n for n in inner.apply_nodes if isinstance(n.op, Elemwise)]
    if len(elems) != 1:
        raise NotImplementedError("FusedElemwise: expected one inner Elemwise")
    (E,) = elems
    n_idx = len(op.indexed_inputs)
    if len(op.indexed_outputs) not in (0, n_idx):
        raise NotImplementedError("FusedElemwise: inconsistent index specs")
    written = sorted(int(s) for e in op.indexed_outputs if e is not None for s in e[0])
    n_elem = len(node.inputs) - n_idx - len(written)
    if len(E.inputs) != n_elem or len(E.outputs) != len(node.outputs):
        raise NotImplementedError("FusedElemwise: unexpected inner structure")

    def check_index(position: int) -> None:
        t = node.inputs[position].type
        if t.ndim != 1 or not t.dtype.startswith(("int", "uint")):
            raise NotImplementedError(
                "FusedElemwise: only 1-d integer index arrays are fused"
            )

    reads: list[Read | None] = [None] * n_elem
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, axis = entry
        check_index(n_elem + k)
        for s in sources:
            if reads[s] is not None:
                raise NotImplementedError(
                    "FusedElemwise: a source read through several indices"
                )
            reads[s] = Read(n_elem + k, int(axis))
    for j, e_in in enumerate(E.inputs):
        src = inner.inputs[j]
        if reads[j] is not None:
            if not (
                e_in.owner
                and isinstance(e_in.owner.op, AdvancedSubtensor)
                and e_in.owner.inputs[0] is src
            ):
                raise NotImplementedError("FusedElemwise: unexpected indexed read")
        elif e_in is not src:
            raise NotImplementedError("FusedElemwise: transformed Elemwise input")
    outs: list[Reduce | Write | None] = [None] * len(node.outputs)
    for i, reduced in enumerate(op.reduced_outputs):
        if reduced is None:
            continue
        scalar_op, axes, identity, acc = reduced
        if node.outputs[i].type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError("FusedElemwise: reduction changes the dtype")
        outs[i] = Reduce(
            scalar_op, tuple(int(a) for a in axes), identity, element_type(acc)
        )
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, axis, modes = entry
        check_index(n_elem + k)
        for s, mode in zip(sources, modes, strict=True):
            if outs[s] is not None:
                raise NotImplementedError(
                    "FusedElemwise: an output written through several indices"
                )
            target_input = n_elem + n_idx + written.index(int(s))
            extra = node.inputs[target_input].type.ndim - E.outputs[s].type.ndim
            if extra < 0 or int(axis) >= E.outputs[s].type.ndim:
                raise NotImplementedError(
                    "FusedElemwise: a write target with fewer axes than the loop"
                )
            outs[s] = Write(n_elem + k, int(axis), target_input, mode == "set", extra)
    for i, var in enumerate(node.outputs):
        if var.type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError(
                "FusedElemwise: output dtype differs from the loop's"
            )
    return FusedSpec(E, E.op.scalar_op, n_elem, tuple(reads), tuple(outs))


def fused_supported(op: FusedElemwise, node: Apply) -> bool:
    """Can `emit_fused` generate `node` as one loop? `fused_spec` must accept its
    pattern, and the scalar ops of the Elemwise and of the reductions must be ones the
    LLVM backend can generate."""
    if not supported_types(node):
        return False
    try:
        spec = fused_spec(op, node)
        E = spec.elemwise
        out_types = [element_type(v.type) for v in E.outputs]
        for v in E.inputs:
            element_type(v.type)
    except NotImplementedError:
        return False
    if not scalar_supported(spec.scalar_op, out_types):
        return False
    return all(
        out.identity is not None and scalar_supported(out.scalar_op, [out.acc])
        for out in spec.outs
        if isinstance(out, Reduce)
    )


def fusable(op: Op, node: Apply) -> bool:
    """Is `op` a FusedElemwise that `emit_fused` generates as one loop? The planner
    keeps such nodes whole, and inlines the inner graph of other OpFromGraphs."""
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    return isinstance(op, FusedElemwise) and fused_supported(op, node)


def _register() -> None:
    from pytensor.compile.builders import OpFromGraph
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    llvm_supports.register(FusedElemwise)(fused_supported)

    @llvm_emit.register(FusedElemwise)
    def _(
        op: FusedElemwise,
        node: Apply,
        cg: Codegen,
        inputs: list[EmitterInput],
        outputs: list[EmitterOutput | None],
    ) -> None:
        emit_fused(op, node, cg, inputs, outputs)

    @llvm_layout.register(FusedElemwise)
    def _(
        op: FusedElemwise, node: Apply, input_layouts: list[Layout | None]
    ) -> list[Layout | None]:
        # A node that isn't generated as one loop runs as an OpFromGraph.
        if not fused_supported(op, node):
            return llvm_layout.dispatch(OpFromGraph)(op, node, input_layouts)
        return [None] * len(node.outputs)


# ------------------------------------------------------------------ the loop shape


@dataclass(frozen=True)
class Loop:
    """The loop being generated, which the members need to generate their code. Its
    axes are those of the Elemwise's output, with lengths `shape`."""

    cg: Codegen
    shape: list[ir.Value]

    @property
    def b(self) -> ir.Builder:
        return self.cg.b

    @property
    def ndim(self) -> int:
        return len(self.shape)


def loop_shape(cg: Codegen, spec: FusedSpec, inputs: Sequence[Array]) -> list[ir.Value]:
    """The length of each loop axis.

    Where the Elemwise's output type has a static length, that is used. Otherwise
    the length comes from the first of these that has the axis:

    * an input used as it is, unless it is broadcast along the axis,
    * an input `x[idx]`, which gives the length of `idx` along the indexed axis and
      the length of `x` along the others,
    * a write into `t[idx]`, which gives the length of `idx` along the indexed axis
      and the length of `t` along the others.

    All of them must agree. Where that isn't known from constants, it is checked at run
    time."""
    b = cg.b
    nd = spec.ndim
    candidates: list[list[ir.Value]] = [[] for _ in range(nd)]
    for j in range(spec.n_elem):
        x = inputs[j]
        read = spec.reads[j]
        etype = spec.elemwise.inputs[j].type
        for a in range(nd):
            if read is not None and a == read.axis:
                candidates[a].append(inputs[read.index_input].shape[0])
            elif etype.shape[a] != 1:
                candidates[a].append(x.shape[a])
    for out in spec.outs:
        if isinstance(out, Write):
            target = inputs[out.target_input]
            index = inputs[out.index_input]
            for a in range(nd):
                length = index.shape[0] if a == out.axis else target.shape[a]
                candidates[a].append(length)
    static = spec.elemwise.outputs[0].type.shape
    shape = []
    for a in range(nd):
        if static[a] is not None:
            n = b.i64(static[a])
        elif candidates[a]:
            n = candidates[a][0]
        else:
            n = b.i64(1)
        for c in candidates[a]:
            if not (c.const_int is not None and c.const_int == n.const_int):
                cg.check(b.eq(c, n), f"FusedElemwise: shape mismatch on axis {a}")
        shape.append(n)
    return shape


# ------------------------------------------------------------------ the members


#: For each output, the values it keeps in registers across the innermost loop, or
#: None if it keeps none. See `Output.carries`.
type Carried = list[list[ir.Value] | None]


class AxesView(NamedTuple):
    """Some of the axes of an array, as the shape, strides and layout of an array with
    only those axes. `IndexedWrite` uses it for the extra trailing axes of its
    target."""

    shape: list[ir.Value]
    strides: list[ir.Value]
    layout: Layout


@dataclass(frozen=True)
class IndexArray:
    """An index array `idx` of the node, for an `IndexedRead` or `IndexedWrite`."""

    array: Array
    #: What is known about its values. It can make bounds checks unnecessary.
    bounds: IndexBounds

    @classmethod
    def of(cls, node: Apply, inputs: Sequence[Array], position: int) -> Self:
        """The index array that is the node's input number `position`."""
        return cls(inputs[position], index_bounds(node.inputs[position]))

    def lookup(self, loop: Loop, i: ir.Value, n: ir.Value, what: str) -> ir.Value:
        """Generate `idx[i]`, for use as an index into an axis of length `n`. Negative
        values count from the end, as in numpy, and values out of range raise
        IndexError. For reads, that error is raised after the loop, see `read_checks`.
        `what` is "read" or "write", and goes into the error message."""
        value = self.array.load(loop.b, i).astype(ps.int64).value
        return loop.cg.checked_index(
            value, n, OUT_OF_BOUNDS.format(what), self.bounds, defer=what == "read"
        )


class Member:
    """An input or output of the Elemwise, which generates its part of the work at
    each element.

    `Codegen.for_each` moves a pointer through each member's `operand`, an array with
    strides over the loop's axes. A stride of 0 along an axis means that the member
    stays at the same element while the loop moves along that axis. A member that
    doesn't need memory in the loop has no operand, and gets a null pointer that it
    doesn't use."""

    operand: Array | None

    def fits_flat_loop(self) -> bool:
        """Can this member run in one flat loop over all elements? Members that are
        broadcast or read or written through an index can't, because they need the
        loop index along each axis."""
        return True

    def carries(self, along: frozenset[int]) -> bool:
        """Does this member keep values in registers across an innermost loop over the
        axes `along`? Only outputs can, see `Output`."""
        return False


class Input(Member):
    """An Elemwise input, which generates its value at each element."""

    def value(self, ptr: ir.Value, index: LoopIndex) -> tuple[ir.Value, ScalarType]:
        """Generate the input's value at the current element, and return it with its
        type. `ptr` points to the current element of `operand`. `index` is the loop
        index along each axis. It is empty in the flat loop, which indexed members
        never run in."""
        raise NotImplementedError


@dataclass(frozen=True)
class LoopInvariant(Input):
    """An input that has the same value at every element, because it has length 1
    along every axis. It is loaded once, before the loop."""

    arg: tuple[ir.Value, ScalarType]
    operand: None = None

    @classmethod
    def make(cls, loop: Loop, x: Array, d: ScalarType) -> Self:
        value = load(loop.b, x.scalar_type, x.data)
        return cls((cast(loop.b, value, x.scalar_type, d), d))

    def value(self, ptr: ir.Value, index: LoopIndex) -> tuple[ir.Value, ScalarType]:
        return self.arg


@dataclass(frozen=True)
class PlainInput(Input):
    """An input that is used as it is: its value is loaded at each element. Along axes
    where its type has length 1, it is broadcast, so the loop doesn't move its
    pointer along them."""

    loop: Loop
    operand: Array
    scalar_type: ScalarType
    #: Is it broadcast along an axis that may be longer than 1? Then it can't run in
    #: the flat loop.
    broadcasts: bool

    @classmethod
    def make(cls, loop: Loop, x: Array, var: Variable, d: ScalarType) -> Self:
        static = [n.const_int for n in loop.shape]
        broadcasts = any(
            s == 1 and static[a] != 1 for a, s in enumerate(var.type.shape)
        )
        return cls(loop, x.loop_view(loop.b, loop.shape), d, broadcasts)

    def fits_flat_loop(self) -> bool:
        return not self.broadcasts

    def value(self, ptr: ir.Value, index: LoopIndex) -> tuple[ir.Value, ScalarType]:
        b, source_type = self.loop.b, self.operand.scalar_type
        v = load(b, source_type, ptr)
        return cast(b, v, source_type, self.scalar_type), self.scalar_type


@dataclass(frozen=True)
class IndexedRead(Input):
    """An input `x[idx]`, see `Read`. At loop index `i` along `axis`, its value is
    `x[..., idx[i], ...]`."""

    loop: Loop
    #: `x` with stride 0 along `axis`. The loop moves this pointer along the other
    #: axes, and `idx` picks the element along `axis`.
    operand: Array
    #: `x` itself.
    source: Array
    index: IndexArray
    axis: int
    scalar_type: ScalarType

    @classmethod
    def make(
        cls, loop: Loop, x: Array, index: IndexArray, axis: int, d: ScalarType
    ) -> Self:
        operand = x.loop_view(loop.b, loop.shape, zero_axes=(axis,))
        return cls(loop, operand, x, index, axis, d)

    def fits_flat_loop(self) -> bool:
        return False

    def value(self, ptr: ir.Value, index: LoopIndex) -> tuple[ir.Value, ScalarType]:
        b, x = self.loop.b, self.source
        n = x.shape[self.axis]
        i = self.index.lookup(self.loop, index[self.axis], n, "read")
        p = advance(b, ptr, i, x.strides[self.axis], x.scalar_type)
        v = load(b, x.scalar_type, p)
        return cast(b, v, x.scalar_type, self.scalar_type), self.scalar_type


class Output(Member):
    """An Elemwise output, which handles its value at each element: it stores it,
    adds it somewhere, or combines it into a reduction.

    Some outputs keep values in registers across the innermost loop, instead of loading
    and storing them at every element. A sum keeps its running total, for example.
    `carries` says whether an output does this for a given innermost loop. If it does,
    `Body.innermost` calls `start` before the innermost loop to load the values,
    passes them into `element` at every element and gets the updated values back, and
    calls `stop` after the loop to store them. For outputs that keep no values,
    `element` gets and returns None."""

    #: The type of the values the Elemwise produces for this output.
    scalar_type: ScalarType

    #: Does each element's update depend on the previous one in a way that keeps LLVM
    #: from vectorizing the loop? Then `chunked_loop` generates the innermost loop.
    #: This is True for increments. A sum's running total LLVM can vectorize.
    serial: ClassVar[bool] = False

    def element(
        self,
        ptr: ir.Value,
        index: LoopIndex,
        value: ir.Value,
        carried: list[ir.Value] | None,
    ) -> list[ir.Value] | None:
        """Generate what the output does with `value`, its value at the current element.
        `ptr` and `index` are as in `Input.value`. `carried` are the values it keeps
        in registers, as they were after the previous element. Returns them as they are
        after this element, or None if the output keeps no values."""
        raise NotImplementedError

    def start(self, ptr: ir.Value) -> list[ir.Value]:
        """Generate the loads of the values the output keeps in registers, before an
        innermost loop whose first element is at `ptr`."""
        raise NotImplementedError

    def stop(self, ptr: ir.Value, carried: list[ir.Value]) -> None:
        """Generate the stores of the values the output kept in registers, after an
        innermost loop whose first element is at `ptr`."""
        raise NotImplementedError

    def deferred(self, slot: ir.Value) -> Output:
        """Only for `serial` outputs: a stand-in for this output in the first pass of
        `chunked_loop`, which stores the value at `slot` instead of handling it."""
        raise NotImplementedError

    def finish(self) -> None:
        """Generate what has to happen once after the whole loop, for example
        converting an accumulator to the output's type."""


@dataclass(frozen=True)
class PlainOutput(Output):
    """An output that is used as it is: its value is stored at each element."""

    loop: Loop
    operand: Array
    scalar_type: ScalarType

    @classmethod
    def make(cls, loop: Loop, out: TensorOutput, d: ScalarType) -> Self:
        operand = out.allocate(loop.shape).loop_view(loop.b, loop.shape)
        return cls(loop, operand, d)

    def element(
        self,
        ptr: ir.Value,
        index: LoopIndex,
        value: ir.Value,
        carried: list[ir.Value] | None,
    ) -> None:
        store(self.loop.b, self.scalar_type, value, ptr)


@dataclass(frozen=True)
class Reduction(Output):
    """A reduced output, see `Reduce`. The values are combined into `acc`, which starts
    at the identity. Usually `acc` is the result itself. If the values are combined in
    another type, `acc` is a temporary, and `finish` converts it into the result."""

    loop: Loop
    spec: Reduce
    result: Array
    acc: Array
    #: `acc` with stride 0 along the reduced axes, so that all elements along them are
    #: combined into the same entry.
    operand: Array
    scalar_type: ScalarType

    @classmethod
    def make(cls, loop: Loop, spec: Reduce, out: TensorOutput, d: ScalarType) -> Self:
        b = loop.b
        shape = [n for a, n in enumerate(loop.shape) if a not in spec.axes]
        result = out.allocate(shape)
        acc = result
        if spec.acc != result.scalar_type:
            acc = loop.cg.temporary(spec.acc, shape)
        identity = identity_value(spec.scalar_op, spec.identity, spec.acc)
        init = constant(b, spec.acc, identity)
        loop.cg.for_each(shape, [acc], lambda p: store(b, spec.acc, init, p[0]))
        operand = reduced_view(b, acc, loop.shape, spec.axes)
        return cls(loop, spec, result, acc, operand, d)

    def fits_flat_loop(self) -> bool:
        return self.carries(frozenset(range(self.loop.ndim)))

    def carries(self, along: frozenset[int]) -> bool:
        return along <= set(self.spec.axes)

    def start(self, ptr: ir.Value) -> list[ir.Value]:
        return [load(self.loop.b, self.spec.acc, ptr)]

    def element(
        self,
        ptr: ir.Value,
        index: LoopIndex,
        value: ir.Value,
        carried: list[ir.Value] | None,
    ) -> list[ir.Value] | None:
        b, acc = self.loop.b, self.spec.acc
        current = load(b, acc, ptr) if carried is None else carried[0]
        args = [(current, acc), (cast(b, value, self.scalar_type, acc), acc)]
        new = emit_scalar(self.spec.scalar_op, b, args, [acc])[0]
        if carried is None:
            store(b, acc, new, ptr)
            return None
        return [new]

    def stop(self, ptr: ir.Value, carried: list[ir.Value]) -> None:
        store(self.loop.b, self.spec.acc, carried[0], ptr)

    def finish(self) -> None:
        if self.acc is not self.result:
            self.result.copy_from(self.loop.cg, self.acc)


@dataclass(frozen=True)
class IndexedWrite(Output):
    """An output that is written through an index, see `Write`. At loop index `i`
    along `axis`, its value `y[i]` is set at, or added to, `result[..., idx[i], ...]`,
    where `result` starts as a copy of the target `t`. The elements are handled in loop
    order, so if an index repeats, the last set wins and all increments add up.

    If `t` has more axes than the loop, each `y[i]` goes into every element along the
    extra axes, which `block` describes."""

    loop: Loop
    spec: Write
    result: Array
    #: `result` over the loop's axes, with stride 0 along `axis`. The loop moves this
    #: pointer along the other axes, and `idx` picks the element along `axis`.
    operand: Array
    index: IndexArray
    #: The extra trailing axes of `t`, or None if it has none.
    block: AxesView | None
    scalar_type: ScalarType
    serial: ClassVar[bool] = True

    @classmethod
    def make(
        cls,
        loop: Loop,
        spec: Write,
        target: Array,
        index: IndexArray,
        out: TensorOutput,
        d: ScalarType,
    ) -> Self:
        result = out.allocate(target.shape)
        result.copy_from(loop.cg, target)
        nd = loop.ndim
        block = None
        stepped = result
        if spec.extra:
            block = _sub_axes(result, range(nd, result.ndim))
            stepped = result.view(result.data, *_sub_axes(result, range(nd)))
        operand = stepped.loop_view(loop.b, loop.shape, zero_axes=(spec.axis,))
        return cls(loop, spec, result, operand, index, block, d)

    def fits_flat_loop(self) -> bool:
        return False

    def carries(self, along: frozenset[int]) -> bool:
        # An increment along the innermost axis keeps the element it adds to in a
        # register while the index repeats, see `_increment_run`.
        spec = self.spec
        return not spec.set and not spec.extra and along == {spec.axis}

    def start(self, ptr: ir.Value) -> list[ir.Value]:
        # There is no current element yet, which index -1 stands for. The value isn't
        # used before an element is loaded.
        return [self.loop.b.i64(-1), constant(self.loop.b, self.scalar_type, 0)]

    def element(
        self,
        ptr: ir.Value,
        index: LoopIndex,
        value: ir.Value,
        carried: list[ir.Value] | None,
    ) -> list[ir.Value] | None:
        n = self.result.shape[self.spec.axis]
        i = self.index.lookup(self.loop, index[self.spec.axis], n, "write")
        if carried is not None:
            return self._increment_run(ptr, i, value, *carried)
        p = self._at(ptr, i)
        if self.block is None:
            self._write(p, value)
        else:
            block = Array(self.scalar_type, p, *self.block)
            self.loop.cg.for_each(
                block.shape, [block], lambda q: self._write(q[0], value)
            )
        return None

    def stop(self, ptr: ir.Value, carried: list[ir.Value]) -> None:
        previous, current = carried
        b = self.loop.b
        with b.if_(b.sge(previous, 0)):
            store(b, self.scalar_type, current, self._at(ptr, previous))

    def deferred(self, slot: ir.Value) -> Output:
        return Deferred(self.loop, self.operand, self.scalar_type, slot)

    def _at(self, ptr: ir.Value, i: ir.Value) -> ir.Value:
        """The pointer to the element at index `i` along `axis`, from the loop's
        pointer, which is at index 0 along `axis`."""
        r = self.result
        return advance(self.loop.b, ptr, i, r.strides[self.spec.axis], r.scalar_type)

    def _write(self, p: ir.Value, value: ir.Value) -> None:
        """Generate setting `value` at `p`, or adding it to what is there."""
        b, d = self.loop.b, self.scalar_type
        if not self.spec.set:
            value = emit_scalar(add, b, [(load(b, d, p), d), (value, d)], [d])[0]
        store(b, d, value, p)

    def _increment_run(
        self,
        ptr: ir.Value,
        i: ir.Value,
        value: ir.Value,
        previous: ir.Value,
        current: ir.Value,
    ) -> list[ir.Value]:
        """Generate adding `value` to the element at index `i`, while keeping the
        element being added to in a register. `current` is the value of the element at
        index `previous`.

        As long as the index stays the same, only the register changes. When it
        changes, the old element is stored and the new one loaded. This way, a repeated
        index doesn't make each addition wait for the previous one to reach memory. The
        additions happen in the same order as without this."""
        b, d = self.loop.b, self.scalar_type

        def same() -> list[ir.Value]:
            return [previous, emit_scalar(add, b, [(current, d), (value, d)], [d])[0]]

        def other() -> list[ir.Value]:
            self.stop(ptr, [previous, current])
            old = load(b, d, self._at(ptr, i))
            return [i, emit_scalar(add, b, [(old, d), (value, d)], [d])[0]]

        return b.if_else(b.eq(i, previous), same, other)


@dataclass(frozen=True)
class Deferred(Output):
    """The stand-in for an IndexedWrite in the first pass of `chunked_loop`. It stores
    the value at `slot`, a position in a buffer on the stack, and the second pass
    hands it to the write."""

    loop: Loop
    #: The write's operand, so that the loop moves the same pointers as with the write.
    operand: Array
    scalar_type: ScalarType
    slot: ir.Value

    def element(
        self,
        ptr: ir.Value,
        index: LoopIndex,
        value: ir.Value,
        carried: list[ir.Value] | None,
    ) -> None:
        store(self.loop.b, self.scalar_type, value, self.slot)


def make_inputs(
    loop: Loop, spec: FusedSpec, node: Apply, inputs: Sequence[Array]
) -> list[Input]:
    """Create the members for the Elemwise's inputs: an IndexedRead for an input read
    through an index, a LoopInvariant for an input with the same value at every
    element, and a PlainInput for the others."""
    members: list[Input] = []
    for j in range(spec.n_elem):
        x, var = inputs[j], spec.elemwise.inputs[j]
        d = element_type(var.type)
        read = spec.reads[j]
        if read is not None:
            index = IndexArray.of(node, inputs, read.index_input)
            members.append(IndexedRead.make(loop, x, index, read.axis, d))
        elif all(s == 1 for s in var.type.shape):
            members.append(LoopInvariant.make(loop, x, d))
        else:
            members.append(PlainInput.make(loop, x, var, d))
    return members


def make_outputs(
    loop: Loop,
    spec: FusedSpec,
    node: Apply,
    inputs: Sequence[Array],
    outputs: Sequence[TensorOutput],
) -> list[Output]:
    """Create the members for the outputs: a Reduction, an IndexedWrite or a
    PlainOutput. This generates the allocation of the outputs, the copies of the write
    targets, and the initialization of the accumulators with the identity."""
    members: list[Output] = []
    for k, (out_spec, out) in enumerate(zip(spec.outs, outputs, strict=True)):
        d = element_type(spec.elemwise.outputs[k].type)
        if isinstance(out_spec, Reduce):
            members.append(Reduction.make(loop, out_spec, out, d))
        elif isinstance(out_spec, Write):
            target = inputs[out_spec.target_input]
            index = IndexArray.of(node, inputs, out_spec.index_input)
            members.append(IndexedWrite.make(loop, out_spec, target, index, out, d))
        else:
            members.append(PlainOutput.make(loop, out, d))
    return members


# ------------------------------------------------------------------ the element


@dataclass(frozen=True)
class Body:
    """Generates the work at one element: it asks the inputs for their values, applies
    the Elemwise's scalar op, and hands each output its value. `innermost` generates
    the innermost loop around it."""

    loop: Loop
    scalar_op: ScalarOp
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]

    @property
    def members(self) -> tuple[Member, ...]:
        return (*self.inputs, *self.outputs)

    @property
    def operands(self) -> list[Array]:
        """The arrays `Codegen.for_each` moves pointers through: the operands of the
        members that have one, in the order of `members`."""
        return [m.operand for m in self.members if m.operand is not None]

    def __call__(
        self, ptrs: Sequence[ir.Value], index: LoopIndex, carried: Carried
    ) -> Carried:
        """Generate the work at one element. `ptrs` are the pointers to it in
        `operands`, and `index` is the loop index. `carried` are the values the outputs
        keep in registers, as they were after the previous element. Returns them as
        they are after this element."""
        member_ptrs = self._member_pointers(ptrs)
        n_in = len(self.inputs)
        args = [
            x.value(p, index)
            for x, p in zip(self.inputs, member_ptrs[:n_in], strict=True)
        ]
        out_types = [out.scalar_type for out in self.outputs]
        values = emit_scalar(self.scalar_op, self.loop.b, args, out_types)
        return [
            out.element(p, index, v, c)
            for out, p, v, c in zip(
                self.outputs, member_ptrs[n_in:], values, carried, strict=True
            )
        ]

    def innermost(
        self, axes: frozenset[int], n: ir.Value, at: Callable[[ir.Value], Element]
    ) -> None:
        """Generate the innermost loop, for `Codegen.for_each`: `n` elements along the
        loop axes `axes`, where `at(i)` gives the pointers and loop index of element i.

        Outputs that keep values in registers across this loop load them before it and
        store them after it. If one of them is `serial`, `chunked_loop` generates the
        loop, else it is a plain loop over the elements."""
        b = self.loop.b
        start_ptrs, _ = at(b.i64(0))
        out_ptrs = self._member_pointers(start_ptrs)[len(self.inputs) :]
        init: Carried = [
            out.start(p) if out.carries(axes) else None
            for out, p in zip(self.outputs, out_ptrs, strict=True)
        ]
        serial = {
            k
            for k, (out, c) in enumerate(zip(self.outputs, init, strict=True))
            if c is not None and out.serial
        }
        if serial:
            final = chunked_loop(self, n, at, out_ptrs, init, serial)
        else:

            def step(i: ir.Value, *values: ir.Value) -> list[ir.Value]:
                ptrs, index = at(i)
                return _pack(self(ptrs, index, _unpack(values, init)))

            final = _unpack(b.fold(n, _pack(init), step), init)
        for out, p, values in zip(self.outputs, out_ptrs, final, strict=True):
            if values is not None:
                out.stop(p, values)

    def with_outputs(self, replaced: Mapping[int, Output]) -> Body:
        """The same body, with the outputs at some positions replaced."""
        outputs = tuple(replaced.get(k, out) for k, out in enumerate(self.outputs))
        return replace(self, outputs=outputs)

    def _member_pointers(self, ptrs: Sequence[ir.Value]) -> list[ir.Value]:
        """The pointer of each member, from the pointers into `operands`. Members
        without an operand get a null pointer."""
        it = iter(ptrs)
        null = self.loop.b.null()
        return [null if m.operand is None else next(it) for m in self.members]


def _pack(carried: Carried) -> list[ir.Value]:
    """The carried values as one flat list, which is how `fold` carries values."""
    return [v for values in carried if values is not None for v in values]


def _unpack(flat: Sequence[ir.Value], like: Carried) -> Carried:
    """Undo `_pack`: regroup the values in `flat` the way `like` is grouped."""
    it = iter(flat)
    return [None if values is None else [next(it) for _ in values] for values in like]


# ------------------------------------------------------------------ the loops


def chunked_loop(
    body: Body,
    n: ir.Value,
    at: Callable[[ir.Value], Element],
    out_ptrs: Sequence[ir.Value],
    init: Carried,
    serial: set[int],
) -> Carried:
    """Generate the innermost loop when it carries increments, the outputs `serial`.
    Returns the values the outputs keep in registers after the loop.

    Each increment depends on the previous one, which would keep the whole loop from
    being vectorized. So the loop runs in chunks of CHUNK elements, and each chunk in
    two passes:

    1. Compute every element, and store the values of the increments in buffers on the
       stack, see `Deferred`. There are no dependencies between elements in this pass,
       so it can be vectorized.
    2. Add the buffered values to their elements, in order.

    The results are the same as with one loop."""
    b = body.loop.b
    outputs = body.outputs
    buffers = {k: b.alloca(memory_type(outputs[k].scalar_type), CHUNK) for k in serial}
    first_init = [None if k in serial else c for k, c in enumerate(init)]
    second_init = [c if k in serial else None for k, c in enumerate(init)]

    def chunk(c: ir.Value, *values: ir.Value) -> list[ir.Value]:
        carried = _unpack(values, init)
        start = b.mul(c, CHUNK, nsw=True)
        m = b.smin(b.i64(CHUNK), b.sub(n, start))
        first_carried = [None if k in serial else v for k, v in enumerate(carried)]

        def compute(j: ir.Value, *values: ir.Value) -> list[ir.Value]:
            ptrs, index = at(b.add(start, j, nsw=True))
            first = body.with_outputs(
                {
                    k: outputs[k].deferred(
                        b.gep(memory_type(outputs[k].scalar_type), buffers[k], j)
                    )
                    for k in serial
                }
            )
            return _pack(first(ptrs, index, _unpack(values, first_init)))

        computed = _unpack(b.fold(m, _pack(first_carried), compute), first_init)
        runs = [carried[k] if k in serial else None for k in range(len(outputs))]

        def increment(j: ir.Value, *values: ir.Value) -> list[ir.Value]:
            _, index = at(b.add(start, j, nsw=True))
            current = _unpack(values, second_init)
            new: Carried = [None] * len(outputs)
            for k in serial:
                d = outputs[k].scalar_type
                value = load(b, d, b.gep(memory_type(d), buffers[k], j))
                new[k] = outputs[k].element(out_ptrs[k], index, value, current[k])
            return _pack(new)

        added = _unpack(b.fold(m, _pack(runs), increment), second_init)
        merged = [added[k] if k in serial else computed[k] for k in range(len(outputs))]
        return _pack(merged)

    n_chunks = b.udiv(b.add(n, CHUNK - 1, nsw=True), CHUNK)
    return _unpack(b.fold(n_chunks, _pack(init), chunk), init)


def run_loops(body: Body) -> None:
    """Generate the loops over all elements. They run as one flat loop where every
    member allows it and the operands are laid out the same way, see
    `Codegen.for_each`."""
    loop = body.loop
    flat = all(m.fits_flat_loop() for m in body.members)
    loop.cg.for_each(loop.shape, body.operands, inner=body.innermost, flat=flat)


@contextmanager
def read_checks(loop: Loop, inputs: Sequence[Input]) -> Generator[None]:
    """Use around the loop, so that out-of-range indices of `IndexedRead`s raise after
    the loop instead of inside it. Then the loop has no early exit and can be
    vectorized. Inside the loop, such indices are clamped into the axis. That needs an
    axis that isn't empty, so an empty axis that is read through an index raises
    before the loop, unless the loop has no elements."""
    reads = [x for x in inputs if isinstance(x, IndexedRead)]
    if not reads:
        yield
        return
    cg, b = loop.cg, loop.b
    message = OUT_OF_BOUNDS.format("read")
    n = size(b, loop.shape)
    for read in reads:
        length = read.source.shape[read.axis]
        cg.check(b.or_(b.ne(length, 0), b.eq(n, 0)), message, IndexError)
    with cg.deferred_checks([message]):
        yield


def emit_fused(
    op: FusedElemwise,
    node: Apply,
    cg: Codegen,
    inputs: Sequence[EmitterInput],
    outputs: Sequence[EmitterOutput | None],
) -> None:
    """Generate the code of the FusedElemwise `node`."""
    spec = fused_spec(op, node)
    # `fused_spec` rejects scalar inputs and outputs, so all inputs are arrays and
    # all outputs tensors.
    arrays = typing.cast(Sequence[Array], inputs)
    tensor_outputs = typing.cast(Sequence[TensorOutput], outputs)
    loop = Loop(cg, loop_shape(cg, spec, arrays))
    body = Body(
        loop,
        spec.scalar_op,
        tuple(make_inputs(loop, spec, node, arrays)),
        tuple(make_outputs(loop, spec, node, arrays, tensor_outputs)),
    )
    with read_checks(loop, body.inputs):
        run_loops(body)
    for out in body.outputs:
        out.finish()


# ------------------------------------------------------------------ views


def reduced_view(
    b: ir.Builder, acc: Array, shape: Sequence[ir.Value], axes: Sequence[int]
) -> Array:
    """`acc`, which has the shape of a reduction's result, as an array over the full
    loop `shape`. It has stride 0 along the reduced `axes`, so that every element along
    them is the same entry, and `acc`'s own strides and layout along the others."""
    nd = len(shape)
    kept = [a for a in range(nd) if a not in axes]
    strides = [b.i64(0)] * nd
    kinds: list[Stride] = [0] * nd
    for k, a in enumerate(kept):
        strides[a] = acc.strides[k]
        kinds[a] = acc.layout.strides[k]
    order = [a for a in range(nd) if a in axes] + [kept[k] for k in acc.layout.order]
    return Array(
        acc.scalar_type,
        acc.data,
        list(shape),
        strides,
        Layout.make([n.const_int for n in shape], order, kinds),
    )


def _sub_axes(a: Array, axes: Sequence[int]) -> AxesView:
    """The axes `axes` of `a`, as an `AxesView`."""
    shape = [a.shape[k] for k in axes]
    strides = [a.strides[k] for k in axes]
    return AxesView(shape, strides, a.view_layout(shape, strides, list(axes)))


_register()

__all__ = ["fusable", "fused_supported"]
