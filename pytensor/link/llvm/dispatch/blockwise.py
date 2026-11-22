"""Blockwise as generated code: a loop over the batch, running the core op.

Every iteration runs the core graph (the core op's node, and whatever its `make_node`
adds) with `Codegen.graph_function`. A core op we generate is inlined into the loop;
others are calls of their Rust or numba kernels. Every iteration reads the inputs at one
batch index (at index 0 along broadcast batch axes) and writes the outputs' rows at that
index (`_BatchRow`).

Before the loop the outputs are allocated in the batch shape and the core shape. A core
length is known from the static shape, or from an input axis of the same name in the
signature (`(m,n),(n)->(m)`). A core length that is neither (`(m,n)->(k)`) is taken from
what the first iteration produces; then the loop is serial and an empty batch fails.
Otherwise the iterations are independent and may run in parallel.
"""

from __future__ import annotations

import dataclasses
import weakref
from dataclasses import dataclass

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.link.llvm.analysis.layout import itemsize_of, remove_axis, shape_of
from pytensor.link.llvm.analysis.work import llvm_work
from pytensor.link.llvm.codegen.array import Array, size
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.destination import OutputDestination, SlotDestination
from pytensor.link.llvm.codegen.interface import llvm_emit
from pytensor.link.llvm.codegen.signature import known_layouts
from pytensor.link.llvm.codegen.types import TensorRepr, repr_of
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import constant
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import tensor


#: node -> its core graph (`blockwise_core`)
_cores: weakref.WeakKeyDictionary[Apply, tuple[list, list]] = (
    weakref.WeakKeyDictionary()
)


def blockwise_core(node) -> tuple[list, list]:
    """The core graph of a Blockwise node, as (inputs, outputs).

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
    layouts without the batch axes."""
    batch_ndim = node.op.batch_ndim(node)
    layouts = []
    for var, layout in zip(node.inputs, known_layouts(node, in_layouts), strict=True):
        representation = repr_of(var.type, layout)
        if not isinstance(representation, TensorRepr):
            raise NotImplementedError(f"Blockwise with a {var.type} input")
        layout = representation.layout
        shape, isz = list(shape_of(var)), itemsize_of(var)
        for _ in range(batch_ndim):
            layout = remove_axis(layout, 0, shape, isz)
            shape = shape[1:]
        layouts.append(layout)
    return layouts


def _batch_shape(cg, node, inputs, batch_ndim):
    """The lengths of the batch axes (i64): those of the inputs that aren't broadcast
    there, which must agree (PyTensor doesn't broadcast at run time)."""
    b = cg.b
    shape = []
    for a in range(batch_ndim):
        lengths = [
            x.shape[a]
            for x, var in zip(inputs, node.inputs, strict=True)
            if not var.type.broadcastable[a]
        ]
        if not lengths:
            shape.append(b.i64(1))
            continue
        # a static length, if any input has one
        n = next((n for n in lengths if n.const_int is not None), lengths[0])
        for m in lengths:
            if m is not n:
                cg.check(
                    b.eq(m, n),
                    "Blockwise: the batch axes of the inputs have different lengths "
                    "(broadcasting a length of 1 needs a static length of 1)",
                )
        shape.append(n)
    return shape


def _core_shapes(op, node, inputs, batch_ndim):
    """Per output, the lengths of its core axes known before the loop (None: only the
    core op knows)."""
    named = {}
    for x, core_signature in zip(inputs, op.inputs_sig, strict=True):
        for name, n in zip(core_signature, x.shape[batch_ndim:], strict=True):
            if name not in named or (
                named[name].const_int is None and n.const_int is not None
            ):
                named[name] = n
    shapes = []
    for var, core_signature in zip(node.outputs, op.outputs_sig, strict=True):
        shape = []
        for name, static in zip(
            core_signature, var.type.shape[batch_ndim:], strict=True
        ):
            if static is not None:
                shape.append(static)
            else:
                shape.append(named.get(name))
        shapes.append(shape)
    return shapes


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
    """Where an iteration writes output `k` of the core: the row of the Blockwise's
    output `out` at the current batch index.

    The whole output is allocated before the loop (`whole`) if its shape is known then.
    Otherwise only the core knows the shape, and the first iteration allocates it."""

    cg: Codegen
    k: int
    out: OutputDestination
    whole: Array | None
    batch: list
    index: list
    #: the iteration
    i: object
    row: Array | None = None

    @property
    def scalar_type(self) -> ScalarType:  # type: ignore[override]
        return self.out.scalar_type

    def allocate(self, shape: list) -> Array:
        b = self.cg.b
        whole = self.whole
        if whole is None:
            # (only an output slot has no shape before the core ran)
            assert isinstance(self.out, SlotDestination)
            with b.if_(b.eq(self.i, 0)):
                self.out.allocate([*self.batch, *shape])
            whole = self.out.read()
        self.row = _at(b, whole, self.index)
        for m, length in zip(shape, self.row.shape, strict=True):
            self.cg.check(
                b.eq(m, length),
                f"Blockwise: core output {self.k} doesn't have the expected shape",
            )
        return self.row

    def finish(self) -> None:
        if self.row is None:
            raise ValueError("an output was never set")


@llvm_emit.register(Blockwise)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    batch_ndim = op.batch_ndim(node)
    in_layouts = [x.layout if isinstance(x, Array) else None for x in inputs]
    core = cg.graph_function(
        *blockwise_core(node), blockwise_core_layouts(node, in_layouts), name="core"
    )
    batch = _batch_shape(cg, node, inputs, batch_ndim)
    n = size(b, batch)

    # the outputs whose shape is known now are allocated now; the others by the first
    # iteration
    known = {}
    for k, shape in enumerate(_core_shapes(op, node, inputs, batch_ndim)):
        if all(c is not None for c in shape):
            known[k] = outputs[k].allocate(
                [*batch, *(b.i64(c) if isinstance(c, int) else c for c in shape)]
            )
    later = [k for k in range(len(outputs)) if k not in known]
    if later:
        cg.check(
            b.ne(n, 0),
            f"Blockwise: the core shape of output {later[0]} isn't known for an empty batch",
        )

    policy = None
    if not later:
        per_element, per_iteration = _iteration_work(cg, core, inputs, batch_ndim)
        policy = cg.parallel_policy(per_element)
        if policy is not None and not policy.worth_it(b.mul(n, per_iteration)):
            policy = None
    # (no nested parallel regions: inside a parallel loop the core runs serially)
    inner = dataclasses.replace(cg, parallel=False) if policy is not None else cg
    broadcast = [var.type.broadcastable[:batch_ndim] for var in node.inputs]

    def step(i):
        index = _unravel(b, i, batch)
        # the core's scratch memory and buffers are given up after every iteration
        # (outputs allocated in iteration 0 are the caller's)
        with inner.scratch_scope():
            args = [
                _at(b, x, [0 if bc else j for bc, j in zip(bcast, index, strict=True)])
                for x, bcast in zip(inputs, broadcast, strict=True)
            ]
            rows = [
                _BatchRow(inner, k, out.destination, known.get(k), batch, index, i)
                for k, out in enumerate(outputs)
            ]
            core.emit_call(inner, args, rows)

    if policy is not None:
        policy.range(b, n, per_iteration, step)
    else:
        with b.loop(n) as i:
            step(i)


__all__ = ["blockwise_core", "blockwise_core_layouts"]
