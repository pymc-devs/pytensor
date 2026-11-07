"""Translate a rewritten FunctionGraph into the native graph builder of `pytensor_llvm`.

Each Op gets a converter registered with ``llvm_build.register(OpType)``. A converter
receives the node and the builder ids of its inputs and outputs and adds the matching
node(s) to the builder, which validates them immediately. All decisions about buffers,
layouts, in-place updates and kernels are made natively by the planner.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

import pytensor.scalar.basic as ps
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.basic import Constant, Variable
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar.basic import Composite, ScalarOp, ScalarType
from pytensor.tensor.basic import (
    Alloc,
    ARange,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.basic import Join
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.sort import SortOp
from pytensor.tensor.type import TensorType


SUPPORTED_DTYPES = frozenset(
    ["float64", "float32", "int64", "int32", "int16", "int8", "uint8", "bool"]
)


def _native():
    try:
        import pytensor_llvm
    except ImportError as e:
        raise ImportError(
            "The LLVM backend needs the `pytensor_llvm` extension "
            "(build it with rust/pytensor-llvm/build.sh)."
        ) from e
    return pytensor_llvm


class GraphConversion:
    """Maps PyTensor variables to builder ids while a graph is being converted."""

    def __init__(self, numba_fallback=True):
        self.native = _native()
        self.builder = self.native.GraphBuilder()
        self.ids: dict[Variable, int] = {}
        self.numba_fallback = numba_fallback

    def convert_node(self, node):
        """Add `node`, falling back to its numba implementation if needed."""
        ins = [self.var(i) for i in node.inputs]
        outs = [self.var(o) for o in node.outputs]
        try:
            llvm_build(node.op, node, self, ins, outs)
        except NotImplementedError:
            if not self.numba_fallback:
                raise
            from pytensor.link.llvm.numba_fallback import numba_fallback_kernel

            kernel = numba_fallback_kernel(node)
            self.builder.foreign(kernel.address, ins, outs, str(node.op), kernel)

    def inline(self, inner, inputs, outputs):
        """Convert the nodes of an inner graph in place of an OpFromGraph node."""
        for iv, i in zip(inner.inputs, inputs, strict=True):
            self.ids[iv] = i
        copies = []
        for ov, o in zip(inner.outputs, outputs, strict=True):
            if ov.owner is None or ov in self.ids:
                copies.append((ov, o))  # inner input, constant or repeated output
            else:
                self.ids[ov] = o  # the producing inner node writes the outer output
        for n in inner.toposort():
            self.convert_node(n)
        for ov, o in copies:
            self.builder.copy(self.var(ov), o)

    def var(self, v: Variable) -> int:
        if v in self.ids:
            return self.ids[v]
        t = v.type
        if isinstance(t, ScalarType):
            # PyTensor scalars are 0-d tensors here
            bcast, shape = [], []
        elif isinstance(t, TensorType):
            bcast = [bool(b) for b in t.broadcastable]
            shape = [None if s is None else int(s) for s in t.shape]
        else:
            raise NotImplementedError(f"LLVM backend: variables of type {t} are not supported")
        if t.dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: dtype {t.dtype} is not supported")
        if isinstance(v, Constant):
            i = self.builder.constant(np.asarray(v.data, dtype=t.dtype), bcast)
        else:
            i = self.builder.variable(t.dtype, bcast, shape)
        self.ids[v] = i
        return i

    def cast(self, v: Variable, dtype: str) -> int:
        """Builder id of `v` converted to `dtype` (adds a cast Elemwise if needed)."""
        i = self.var(v)
        if v.type.dtype == dtype:
            return i
        if dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: dtype {dtype} is not supported")
        t = v.type
        bcast = [bool(b) for b in t.broadcastable] if isinstance(t, TensorType) else []
        shape = [None if s is None else int(s) for s in t.shape] if isinstance(t, TensorType) else []
        out = self.builder.variable(dtype, bcast, shape)
        sg = self.native.ScalarGraph([t.dtype])
        sg.set_outputs([sg.apply("cast", [sg.input(0)], dtype)])
        self.builder.elemwise(sg, [i], [out])
        return out


# ------------------------------------------------------------------ scalar graphs

_SCALAR_NAME = {"scalarmaximum": "maximum", "scalarminimum": "minimum"}


def _scalar_op_name(op) -> str:
    name = type(op).__name__.lower()
    return _SCALAR_NAME.get(name, name)


def scalar_graph(native, op, input_types):
    """Build a native ScalarGraph for a scalar op (Composite or single op)."""
    sg = native.ScalarGraph([t.dtype for t in input_types])
    inputs = [sg.input(i) for i in range(len(input_types))]
    if isinstance(op, Composite):
        outs = _emit_composite(sg, op, inputs)
    else:
        node = op.make_node(*[t() for t in input_types])
        outs = [_emit_scalar_node(sg, node, inputs)]
    sg.set_outputs(outs)
    return sg


def _scalar_ref(sg, v, refs):
    if v in refs:
        return refs[v]
    if isinstance(v, Constant):
        if v.type.dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: scalar constant of dtype {v.type.dtype}")
        return sg.constant(float(np.asarray(v.data)), v.type.dtype)
    raise NotImplementedError(f"LLVM backend: unresolved scalar variable {v}")


def _emit_composite(sg, comp: Composite, inputs):
    refs = dict(zip(comp.fgraph.inputs, inputs, strict=True))
    for node in comp.fgraph.toposort():
        args = [_scalar_ref(sg, i, refs) for i in node.inputs]
        if isinstance(node.op, Composite):  # nested composites are inlined
            outs = _emit_composite(sg, node.op, args)
        else:
            outs = [_emit_scalar_node(sg, node, args)]
        refs.update(zip(node.outputs, outs, strict=True))
    return [_scalar_ref(sg, o, refs) for o in comp.fgraph.outputs]


def _emit_scalar_node(sg, node, args):
    if len(node.outputs) != 1:
        raise NotImplementedError(f"LLVM backend: multi-output scalar op {node.op}")
    dtype = node.outputs[0].type.dtype
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"LLVM backend: scalar op {node.op} with dtype {dtype}")
    return sg.apply(_scalar_op_name(node.op), args, dtype)


# ------------------------------------------------------------------ op converters


@singledispatch
def llvm_build(op, node, conv: GraphConversion, inputs: list[int], outputs: list[int]):
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@llvm_build.register(Elemwise)
def _(op, node, conv, inputs, outputs):
    types = [ps.get_scalar_type(i.type.dtype) for i in node.inputs]
    sg = scalar_graph(conv.native, op.scalar_op, types)
    conv.builder.elemwise(sg, inputs, outputs)


@llvm_build.register(ScalarOp)
def _(op, node, conv, inputs, outputs):
    # a bare scalar op is a 0-d Elemwise
    sg = scalar_graph(conv.native, op, [i.type for i in node.inputs])
    conv.builder.elemwise(sg, inputs, outputs)


_REDUCE_NAME = {
    ps.Add: "add",
    ps.Mul: "mul",
    ps.Maximum: "max",
    ps.Minimum: "min",
    ps.AND: "and",
    ps.OR: "or",
}


@llvm_build.register(CAReduce)
def _(op, node, conv, inputs, outputs):
    name = _REDUCE_NAME.get(type(op.scalar_op))
    if name is None:
        raise NotImplementedError(f"LLVM backend: CAReduce with {op.scalar_op}")
    ndim = node.inputs[0].type.ndim
    axis = range(ndim) if op.axis is None else [a % ndim for a in op.axis]
    conv.builder.reduce(name, sorted(axis), inputs[0], outputs[0])


@llvm_build.register(DimShuffle)
def _(op, node, conv, inputs, outputs):
    order = [None if o == "x" else int(o) for o in op.new_order]
    conv.builder.dimshuffle(order, inputs[0], outputs[0])


@llvm_build.register(ViewOp)
@llvm_build.register(SpecifyShape)
def _(op, node, conv, inputs, outputs):
    # identity view; SpecifyShape's runtime shape check is not implemented yet
    conv.builder.dimshuffle(list(range(node.inputs[0].type.ndim)), inputs[0], outputs[0])


@llvm_build.register(ScalarFromTensor)
@llvm_build.register(TensorFromScalar)
def _(op, node, conv, inputs, outputs):
    conv.builder.dimshuffle([], inputs[0], outputs[0])


@llvm_build.register(Dot)
def _(op, node, conv, inputs, outputs):
    conv.builder.dot(inputs[0], inputs[1], outputs[0])


@llvm_build.register(SortOp)
def _(op, node, conv, inputs, outputs):
    axis = int(op.axis) % node.inputs[0].type.ndim
    conv.builder.sort(inputs[0], outputs[0], axis, op.kind in ("stable", "mergesort"))


@llvm_build.register(DeepCopyOp)
def _(op, node, conv, inputs, outputs):
    conv.builder.copy(inputs[0], outputs[0])


@llvm_build.register(Shape_i)
def _(op, node, conv, inputs, outputs):
    conv.builder.shape_i(inputs[0], outputs[0], int(op.i))


@llvm_build.register(Shape)
def _(op, node, conv, inputs, outputs):
    conv.builder.shape(inputs[0], outputs[0])


@llvm_build.register(MakeVector)
def _(op, node, conv, inputs, outputs):
    conv.builder.make_vector(inputs, outputs[0])


@llvm_build.register(CheckAndRaise)
def _(op, node, conv, inputs, outputs):
    msg = str(getattr(op, "msg", "Assertion failed"))
    conv.builder.check_and_raise(inputs[0], inputs[1:], outputs[0], msg)


def _idx_list(idx_list):
    """PyTensor's idx_list in the builder's encoding."""
    return [e if isinstance(e, int) else (e.start, e.stop, e.step) for e in idx_list]


def _take_axis(idx_list, index_vars):
    """Axis if the index is a single integer array on one axis with all others whole."""
    ints = [d for d, e in enumerate(idx_list) if isinstance(e, int)]
    if len(ints) != 1 or len(index_vars) != 1:
        return None
    if any(e != slice(None) for d, e in enumerate(idx_list) if d != ints[0]):
        return None
    t = index_vars[0].type
    if not isinstance(t, TensorType) or t.dtype == "bool" or not t.dtype.startswith(("int", "uint")):
        return None
    return ints[0]


@llvm_build.register(Subtensor)
def _(op, node, conv, inputs, outputs):
    conv.builder.subtensor(inputs[0], _idx_list(op.idx_list), inputs[1:], outputs[0])


@llvm_build.register(IncSubtensor)
def _(op, node, conv, inputs, outputs):
    x, y, *idx = node.inputs
    y_id = conv.cast(y, x.type.dtype)
    conv.builder.inc_subtensor(
        inputs[0], y_id, _idx_list(op.idx_list), inputs[2:], outputs[0], op.set_instead_of_inc
    )


@llvm_build.register(AdvancedSubtensor)
def _(op, node, conv, inputs, outputs):
    axis = _take_axis(op.idx_list, node.inputs[1:])
    if axis is None:
        raise NotImplementedError(f"LLVM backend: advanced indexing pattern {op}")
    conv.builder.take(inputs[0], inputs[1], axis, outputs[0])


@llvm_build.register(AdvancedIncSubtensor)
def _(op, node, conv, inputs, outputs):
    axis = _take_axis(op.idx_list, node.inputs[2:])
    if axis is None or (op.ignore_duplicates and not op.set_instead_of_inc):
        # ignore_duplicates increments follow `x[idx] += y` (no accumulation)
        raise NotImplementedError(f"LLVM backend: advanced indexing pattern {op}")
    x, y, _ = node.inputs
    y_id = conv.cast(y, x.type.dtype)
    conv.builder.inc_take(inputs[0], y_id, inputs[2], axis, outputs[0], op.set_instead_of_inc)


@llvm_build.register(Join)
def _(op, node, conv, inputs, outputs):
    dtype = node.outputs[0].type.dtype
    ids = [conv.cast(v, dtype) for v in node.inputs]
    conv.builder.join(ids, int(op.axis), outputs[0])


@llvm_build.register(Reshape)
def _(op, node, conv, inputs, outputs):
    conv.builder.reshape(inputs[0], inputs[1], outputs[0])


def _fused_spec(op, node):
    """Translate a FusedElemwise into the native builder's description, or raise
    NotImplementedError for patterns the native fused kernel doesn't cover."""
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise  # noqa: F401

    inner = op.fgraph
    [E] = [n for n in inner.apply_nodes if isinstance(n.op, Elemwise)]
    ndim = E.outputs[0].type.ndim
    n_idx = len(op.indexed_inputs)
    if len(op.indexed_outputs) not in (0, n_idx):
        raise NotImplementedError("FusedElemwise: inconsistent index specs")
    n_targets = sum(e is not None for e in op.indexed_outputs)
    n_elem = len(node.inputs) - n_idx - n_targets
    if len(E.inputs) != n_elem or len(E.outputs) != len(node.outputs):
        raise NotImplementedError("FusedElemwise: unexpected inner structure")

    def check_index(p):
        t = node.inputs[p].type
        if not isinstance(t, TensorType) or t.ndim != 1 or t.dtype == "bool" or not t.dtype.startswith(("int", "uint")):
            raise NotImplementedError("FusedElemwise: only 1-d integer index arrays are fused")

    reads = [None] * n_elem
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, axis = entry
        check_index(n_elem + k)
        for s in sources:
            reads[s] = (n_elem + k, int(axis))
    # inputs must be the inner graph inputs themselves (or indexed reads of them)
    for j, e_in in enumerate(E.inputs):
        src = inner.inputs[j]
        if reads[j] is not None:
            if not (e_in.owner and isinstance(e_in.owner.op, AdvancedSubtensor) and e_in.owner.inputs[0] is src):
                raise NotImplementedError("FusedElemwise: unexpected indexed read")
        elif e_in is not src and not (isinstance(node.inputs[j].type, ScalarType) and all(e_in.type.broadcastable)):
            raise NotImplementedError("FusedElemwise: transformed Elemwise input")

    outs = [("plain",)] * len(node.outputs)
    for i, spec in enumerate(op.reduced_outputs):
        if spec is None:
            continue
        scalar_op, axes, _identity, acc_dtype = spec
        name = _REDUCE_NAME.get(type(scalar_op))
        if name is None or not (acc_dtype == node.outputs[i].type.dtype == E.outputs[i].type.dtype):
            raise NotImplementedError("FusedElemwise: reduction not supported natively")
        outs[i] = ("reduce", name, [int(a) for a in axes])
    t = 0
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, axis, mode = entry
        check_index(n_elem + k)
        target = n_elem + n_idx + t
        t += 1
        for s in sources:
            outs[s] = ("write", n_elem + k, int(axis), target, mode == "set")
    for i, o in enumerate(node.outputs):
        if o.type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError("FusedElemwise: output dtype differs from the loop's")
    return E, ndim, n_elem, reads, outs


def _register_fused():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    @llvm_build.register(FusedElemwise)
    def _(op, node, conv, inputs, outputs):
        try:
            E, ndim, n_elem, reads, outs = _fused_spec(op, node)
        except NotImplementedError:
            # still native, just not fused
            return conv.inline(op.fgraph, inputs, outputs)
        types = [ps.get_scalar_type(i.type.dtype) for i in E.inputs]
        sg = scalar_graph(conv.native, E.op.scalar_op, types)
        conv.builder.fused_elemwise(sg, inputs, outputs, ndim, n_elem, reads, outs)


_register_fused()


@llvm_build.register(OpFromGraph)
def _(op, node, conv, inputs, outputs):
    # FusedElemwise, ScalarizedElemwise, ScalarCAReduce, user OpFromGraphs, ...
    conv.inline(op.fgraph, inputs, outputs)


@llvm_build.register(ARange)
def _(op, node, conv, inputs, outputs):
    conv.builder.arange(*inputs, outputs[0])


@llvm_build.register(Alloc)
def _(op, node, conv, inputs, outputs):
    conv.builder.alloc(inputs[0], inputs[1:], outputs[0])


# ------------------------------------------------------------------ whole graphs


def fgraph_to_llvm(
    fgraph,
    *,
    fastmath=True,
    opt_level=3,
    cache=True,
    dump_ir=None,
    segments=True,
    segment_max_elems=64,
    numba_fallback=True,
):
    """Convert and compile `fgraph`; returns a native ``CompiledFunction``.

    Ops without an LLVM implementation run through their numba implementation if
    `numba_fallback` is set (see `numba_fallback.py`).
    """
    conv = GraphConversion(numba_fallback=numba_fallback)
    b = conv.builder
    for i in fgraph.inputs:
        b.add_input(conv.var(i))
    for node in fgraph.toposort():
        conv.convert_node(node)
    for o in fgraph.outputs:
        b.add_output(conv.var(o))
    return b.compile(
        fastmath=fastmath,
        opt_level=opt_level,
        cache=cache,
        dump_ir=dump_ir,
        segments=segments,
        segment_max_elems=segment_max_elems,
    )
