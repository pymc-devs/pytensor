"""Blockwise as generated code: a loop over the batch, running the core op.

A Blockwise reaches the backend wrapped with its core shapes
(`BlockwiseWithCoreShape`, from `introduce_explicit_core_shape_blockwise`): the graph
computes the lengths of the outputs' core axes, so the outputs are allocated in the
batch shape and the core shape before the loop. A Blockwise whose core shapes the
rewrite can't infer (its core op has no `infer_shape`, or they depend on values that
differ along the batch) isn't generated: numba runs it (`Blockwise.perform`, in object
mode).

Every iteration runs the core graph (the core op's node, and whatever its `make_node`
adds) with `Codegen.graph_function`. A core op we generate is inlined into the loop;
others are calls of their Rust or numba kernels. Every iteration reads the inputs at one
batch index (at index 0 along broadcast batch axes) and writes the outputs' rows at that
index (`_BatchRow`). The iterations are independent and may run in parallel.

`batch_loop` is that loop, also for random variables (dispatch/random.py).
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.traversal import io_toposort
from pytensor.link.llvm.analysis.layout import (
    Layout,
    itemsize_of,
    remove_axis,
    shape_of,
)
from pytensor.link.llvm.analysis.work import llvm_work
from pytensor.link.llvm.codegen.array import Array, size
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.destination import OutputDestination
from pytensor.link.llvm.codegen.interface import (
    TensorOutput,
    llvm_emit,
    llvm_layout,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.types import ir
from pytensor.scalar.basic import ScalarType
from pytensor.scalar.basic import int64 as scalar_int64
from pytensor.tensor.basic import constant
from pytensor.tensor.blockwise import BlockwiseWithCoreShape, OpWithCoreShape
from pytensor.tensor.type import tensor


#: wrapper node -> the node it wraps, on the wrapper's inputs (`unwrapped`)
_unwrapped: weakref.WeakKeyDictionary[Apply, Apply] = weakref.WeakKeyDictionary()


def unwrapped(node) -> Apply:
    """The node an `OpWithCoreShape` node wraps (a Blockwise's or a random
    variable's), on the wrapper's own inputs: the wrapper inlined (constants stay
    visible).

    Every call for the same node returns the same node."""
    # (imported here: it imports the linkers)
    from pytensor.compile.rewriting import inline_ofg_node

    if node not in _unwrapped:
        _unwrapped[node] = inline_ofg_node(node)[0].owner
    return _unwrapped[node]


def wrapped_positions(node) -> list[int]:
    """Per input of the wrapped node, the position of the wrapper's input it is."""
    inputs = node.op.fgraph.inputs
    return [inputs.index(v) for v in _inner_node(node.op).inputs]


def _inner_node(op: OpWithCoreShape) -> Apply:
    [inner] = op.fgraph.apply_nodes
    return inner


@llvm_layout.register(OpWithCoreShape)
def _(op, node, input_layouts):
    # (the default: new memory, C-contiguous; not its inner graph's, it isn't inlined)
    return [None] * len(node.outputs)


#: node -> its core graph (`blockwise_core`)
_cores: weakref.WeakKeyDictionary[Apply, tuple[list, list]] = (
    weakref.WeakKeyDictionary()
)


def blockwise_core(node) -> tuple[list, list]:
    """The core graph of a Blockwise node (the `unwrapped` node of a
    `BlockwiseWithCoreShape`), as (inputs, outputs).

    The core op is applied to new variables of the core types. Its `make_node` may add
    nodes of its own (e.g. a `ScalarFromTensor`), so the result is a graph, not a single
    node. A constant input that isn't batched stays a constant (its core part).

    Every call for the same node returns the same variables: the planner caches the
    graphs it resolves by their variables."""
    if node not in _cores:
        _cores[node] = _make_core(node)
    return _cores[node]


def _make_core(node) -> tuple[list, list]:
    op = node.op
    batch_ndim = op.batch_ndim(node)
    core_inputs: list[Variable] = []
    for inp in node.inputs:
        if isinstance(inp, Constant) and all(inp.type.broadcastable[:batch_ndim]):
            data = inp.data.reshape(inp.data.shape[batch_ndim:])
            core_inputs.append(constant(data, dtype=inp.type.dtype))
        else:
            core_inputs.append(
                tensor(dtype=inp.type.dtype, shape=inp.type.shape[batch_ndim:])
            )
    return core_inputs, op.core_op.make_node(*core_inputs).outputs


def blockwise_core_layouts(node, in_layouts) -> list:
    """The layouts the core node's inputs have in the generated loop: the inputs'
    layouts without the batch axes (`node`: the `unwrapped` Blockwise, `in_layouts`:
    per input of it)."""
    batch_ndim = node.op.batch_ndim(node)
    return [
        core_layout(var, given, batch_ndim)
        for var, given in zip(node.inputs, in_layouts, strict=True)
    ]


def core_layout(var, given: Layout | None, batch_ndim: int) -> Layout:
    """The layout of `var` (laid out as `given`) without its first `batch_ndim` axes."""
    shape, isz = list(shape_of(var)), itemsize_of(var)
    layout = Layout.unknown(shape) if given is None else given.for_shape(shape)
    for _ in range(batch_ndim):
        layout = remove_axis(layout, 0, shape, isz)
        shape = shape[1:]
    return layout


@dataclass(frozen=True)
class Batched:
    """An input of a batch loop: its batch axes are the loop's axes from `first` on,
    except those it broadcasts (`broadcastable`), where it is read at index 0."""

    array: Array
    broadcastable: tuple[bool, ...]
    first: int = 0

    def axes(self) -> list[int | None]:
        """Per batch axis of the input, the loop's axis (None: broadcast)."""
        return [
            None if bc else self.first + j for j, bc in enumerate(self.broadcastable)
        ]


def batch_shape(
    cg: Codegen,
    operands: Sequence[Batched],
    batch_ndim: int,
    given: Sequence | None = None,
    what: str = "Blockwise",
) -> list:
    """The lengths of the loop's axes (i64): `given` (e.g. a random variable's size)
    or else the lengths of the operands that aren't broadcast there. These must agree
    (PyTensor doesn't broadcast at run time)."""
    b = cg.b
    shape = []
    for a in range(batch_ndim):
        lengths = [
            x.array.shape[j]
            for x in operands
            for j, axis in enumerate(x.axes())
            if axis == a
        ]
        n = given[a] if given is not None else None
        if n is None:
            if not lengths:
                shape.append(b.i64(1))
                continue
            # a static length, if any operand has one
            n = next((m for m in lengths if m.const_int is not None), lengths[0])
        for m in lengths:
            if m is not n and not (
                m.const_int is not None and m.const_int == n.const_int
            ):
                cg.check(
                    b.eq(m, n),
                    f"{what}: the batch axes of the inputs have different lengths "
                    "(broadcasting a length of 1 needs a static length of 1)",
                )
        shape.append(n)
    return shape


def core_lengths(cg: Codegen, var, batch_ndim: int, core_shape: Array) -> list:
    """The lengths of the core axes of output `var` (i64): static ones from its type,
    the others from its core shape (an int64 vector the graph computed)."""
    b = cg.b
    return [
        b.i64(static)
        if static is not None
        else core_shape.load(b, a).astype(scalar_int64).value
        for a, static in enumerate(var.type.shape[batch_ndim:])
    ]


def _unravel(b, i, shape):
    """The index into `shape` (C order) of the flat index `i`."""
    index = [None] * len(shape)
    for a in reversed(range(len(shape))):
        if a == 0:
            index[a] = i
        else:
            index[a] = b.urem(i, shape[a])
            i = b.udiv(i, shape[a])
    return index


def _at(b, a, index):
    """`a[index]` for the leading axes."""
    for i in index:
        a = a.at(b, i)
    return a


@dataclass
class _BatchRow(OutputDestination):
    """Where an iteration writes output `k` of the core: the row of the loop's output
    `out` at the current batch index, in `whole`, allocated before the loop."""

    cg: Codegen
    k: int
    out: TensorOutput
    whole: Array
    index: list
    what: str
    row: Array | None = None

    @property
    def scalar_type(self) -> ScalarType:  # type: ignore[override]
        return self.out.scalar_type

    def allocate(self, shape: list) -> Array:
        b = self.cg.b
        self.row = _at(b, self.whole, self.index)
        for m, length in zip(shape, self.row.shape, strict=True):
            if m.const_int is None or m.const_int != length.const_int:
                self.cg.check(
                    b.eq(m, length),
                    f"{self.what}: core output {self.k} doesn't have the shape its "
                    "core shape says",
                )
        return self.row

    def finish(self) -> None:
        if self.row is None:
            raise ValueError("an output was never set")


def batch_loop(
    cg: Codegen,
    core,
    operands: Sequence[Batched | None],
    outputs: Sequence[TensorOutput | None],
    batch: list,
    shapes: Sequence[list],
    parallel: tuple[float, ir.Value] | None = None,
    what: str = "Blockwise",
) -> None:
    """Run the graph function `core` for every index of `batch` (the loop's lengths).

    `operands`: per core input, the batched input it reads (None: a value without
    data). `outputs`: per core output, the loop's output (None: without data), allocated
    as `batch` + its core lengths (`shapes`). `parallel`: (cost per element, elements
    per iteration) if the iterations may run in parallel."""
    b = cg.b
    wholes = [
        None if out is None else out.allocate([*batch, *lengths])
        for out, lengths in zip(outputs, shapes, strict=True)
    ]

    def step(i):
        index = _unravel(b, i, batch)
        # the core's scratch memory and buffers are given up after every iteration
        with cg.scratch_scope():
            args = [
                None
                if x is None
                else _at(b, x.array, [0 if a is None else index[a] for a in x.axes()])
                for x in operands
            ]
            rows = [
                None if whole is None else _BatchRow(cg, k, out, whole, index, what)
                for k, (out, whole) in enumerate(zip(outputs, wholes, strict=True))
            ]
            core.emit_call(cg, args, rows)

    cost, per_iteration = parallel if parallel is not None else (None, None)
    cg.for_range(size(b, batch), step, parallel_cost=cost, per_iteration=per_iteration)


def _iteration_work(cg, core, inputs, batch_ndim):
    """(cost per element, elements per iteration as i64) of the core: `llvm_work` of
    its nodes, a length only known at run time counted as the longest core axis of the
    inputs (rough, as all estimates)."""
    b = cg.b
    longest = b.i64(1)
    for x in inputs:
        for n in x.shape[batch_ndim:]:
            longest = b.smax(longest, n)
    per_element, elements = 0.0, b.i64(1)
    for n in core.nodes:
        w = llvm_work(n.op, n)
        n = b.i64(1)
        for f in w.elements:
            n = b.mul(n, longest if f is None else b.i64(f))
        per_element += w.per_element
        elements = b.smax(elements, n)
    return per_element, elements


@llvm_supports.register(BlockwiseWithCoreShape)
def _(op, node) -> bool:
    # (every node of the core runs in the loop: generated, or a call of its kernel)
    core_inputs, core_outputs = blockwise_core(unwrapped(node))
    return supported_types(node) and all(
        supported_types(n) for n in io_toposort(core_inputs, core_outputs)
    )


@llvm_emit.register(BlockwiseWithCoreShape)
def _(op, node, cg, inputs, outputs):
    inner = unwrapped(node)
    batch_ndim = inner.op.batch_ndim(inner)
    args = [inputs[k] for k in wrapped_positions(node)]
    core_shapes = inputs[len(inputs) - len(node.outputs) :]
    in_layouts = [x.layout if isinstance(x, Array) else None for x in args]
    core = cg.graph_function(
        *blockwise_core(inner), blockwise_core_layouts(inner, in_layouts), name="core"
    )
    operands = [
        Batched(x, var.type.broadcastable[:batch_ndim])
        for x, var in zip(args, inner.inputs, strict=True)
    ]
    batch = batch_shape(cg, operands, batch_ndim)
    shapes = [
        core_lengths(cg, var, batch_ndim, shape)
        for var, shape in zip(node.outputs, core_shapes, strict=True)
    ]
    work = _iteration_work(cg, core, args, batch_ndim)
    batch_loop(cg, core, operands, outputs, batch, shapes, work)


__all__ = [
    "Batched",
    "batch_loop",
    "batch_shape",
    "blockwise_core",
    "blockwise_core_layouts",
    "core_layout",
    "core_lengths",
    "unwrapped",
    "wrapped_positions",
]
