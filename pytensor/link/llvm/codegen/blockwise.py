"""Blockwise as generated code: a loop over the batch, running the core op.

The core graph (the core op's node, and what its `make_node` adds) is resolved like a
Scan's loop body (`plan.resolve.Resolver.body`): a
core op we generate is inlined into the loop, others are called (Rust kernels, numba,
subprograms). Every iteration reads the inputs at one batch index (broadcast batch axes
at index 0) and writes the outputs' rows at that index.

Before the loop the outputs are allocated in the batch shape and the core shape. A core
length is known from the static shape, or from an input axis of the same name in the
signature (`(m,n),(n)->(m)`). A core length that is neither (`(m,n)->(k)`) is taken from
what the first iteration produces; then the loop is serial and an empty batch fails.
Otherwise the iterations are independent and may run in parallel.
"""

from __future__ import annotations

import dataclasses

from pytensor.graph.basic import Constant, Variable
from pytensor.link.llvm.codegen.emit import copy, size
from pytensor.link.llvm.codegen.function import known_layouts, llvm_emit
from pytensor.link.llvm.codegen.types import repr_of
from pytensor.link.llvm.codegen.unit import (
    Destination,
    bind_constants,
    emit_items,
    release,
)
from pytensor.link.llvm.layout import itemsize_of, remove_axis, shape_of
from pytensor.link.llvm.work import llvm_work
from pytensor.tensor.basic import constant
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import tensor


def blockwise_core(node) -> tuple[list, list]:
    """The core graph of a Blockwise node: (inputs, outputs) of the core op applied to
    variables of the core types. The core op's `make_node` may add nodes of its own (a
    `ScalarFromTensor`), so this is a graph, not one node. Constant inputs that aren't
    batched stay constants (their core)."""
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
        layout = repr_of(var.type, layout).layout
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
    for x, sig in zip(inputs, op.inputs_sig, strict=True):
        for name, n in zip(sig, x.shape[batch_ndim:], strict=True):
            if name not in named or (
                named[name].const_int is None and n.const_int is not None
            ):
                named[name] = n
    shapes = []
    for var, sig in zip(node.outputs, op.outputs_sig, strict=True):
        shape = []
        for name, static in zip(sig, var.type.shape[batch_ndim:], strict=True):
            if static is not None:
                shape.append(static)
            else:
                shape.append(named.get(name))
        shapes.append(shape)
    return shapes


def _iteration_work(cg, body, inputs, batch_ndim):
    """(cost per element, elements per iteration as i64) of the core: `llvm_work` of
    its nodes, a length only known at run time counted as the longest core axis of the
    inputs (rough, as all estimates)."""
    b = cg.b
    longest = b.i64(1)
    for x in inputs:
        for n in x.shape[batch_ndim:]:
            longest = b.smax(longest, n)
    per_element, elements = 0.0, b.i64(1)
    for r in body.items:
        w = llvm_work(r.item.node.op, r.item.node)
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


@llvm_emit.register(Blockwise)
def _(op, node, cg, inputs, outputs):
    body = cg.body
    if body is None or cg.links is None or cg.module is None:
        raise NotImplementedError(f"generated {op}")
    b = cg.b
    batch_ndim = op.batch_ndim(node)
    batch = _batch_shape(cg, node, inputs, batch_ndim)
    n = size(b, batch)

    # the outputs whose shape is known now are allocated now; the others by the first
    # iteration
    known = {}
    for k, core in enumerate(_core_shapes(op, node, inputs, batch_ndim)):
        if all(c is not None for c in core):
            known[k] = outputs[k].allocate(
                [*batch, *(b.i64(c) if isinstance(c, int) else c for c in core)]
            )
    later = [k for k in range(len(outputs)) if k not in known]
    if later:
        cg.check(
            b.ne(n, 0),
            f"Blockwise: the core shape of output {later[0]} isn't known for an empty batch",
        )

    constants = {}
    bind_constants(cg, body.constants, constants)
    par = None
    if not later:
        per_element, per_iteration = _iteration_work(cg, body, inputs, batch_ndim)
        par = cg.par(per_element)
        if par is not None and not par.worth_it(b.mul(n, per_iteration)):
            par = None
    # (no nested parallel regions: inside a parallel loop the core runs serially)
    inner = dataclasses.replace(cg, parallel=False) if par is not None else cg
    broadcast = [var.type.broadcastable[:batch_ndim] for var in node.inputs]

    def step(i):
        index = _unravel(b, i, batch)
        # the core's scratch memory is freed after every iteration (outputs allocated
        # in iteration 0 are the caller's to free)
        with inner.scratch_scope():
            vals = dict(constants)
            for vid, x, bcast in zip(body.in_ids, inputs, broadcast, strict=True):
                if vid not in vals:
                    vals[vid] = _at(
                        b,
                        x,
                        [0 if bc else j for bc, j in zip(bcast, index, strict=True)],
                    )
            rows = {k: _at(b, a, index) for k, a in known.items()}
            targets = {}
            for k, row in rows.items():
                targets.setdefault(
                    body.out_ids[k], Destination(array=row, layout=row.layout)
                )
            written, owned = emit_items(
                inner, cg.module, f"{cg.name}_core", body.items, vals, targets
            )
            for k, vid in enumerate(body.out_ids):
                src = vals[vid]
                if k in rows:
                    row = rows[k]
                    if vid in written and targets[vid].array is row:
                        continue
                else:
                    with b.if_(b.eq(i, 0)):
                        outputs[k].allocate([*batch, *src.shape])
                    row = _at(b, outputs[k].read(), index)
                for m, length in zip(src.shape, row.shape, strict=True):
                    cg.check(
                        b.eq(m, length),
                        f"Blockwise: core output {k} doesn't have the expected shape",
                    )
                copy(inner, src, row)
            release(inner, owned)

    if par is not None:
        par.range(b, n, per_iteration, step)
    else:
        with b.loop(n) as i:
            step(i)


__all__ = ["blockwise_core", "blockwise_core_layouts"]
