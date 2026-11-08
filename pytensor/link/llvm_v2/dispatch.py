"""Translate a rewritten FunctionGraph into a `pytensor_llvm.v2.Graph`.

Each PyTensor Op gets a converter registered with ``llvm_op.register(OpType)``. A
converter returns the native op object for the node (``v2.Dot()``, ``v2.Elemwise(...)``);
the node's inputs, output types and ``view_map`` are taken from the PyTensor node. Ops
without a converter run through their numba implementation.
"""

from __future__ import annotations

from functools import cache, singledispatch

import numpy as np

import pytensor.scalar.basic as ps
import pytensor.scalar.math as psm
from pytensor.compile.builders import OpFromGraph, UnitKind
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.basic import Constant, Variable
from pytensor.scalar.basic import Composite, ScalarType
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.type import TensorType


SUPPORTED_DTYPES = frozenset(["float64", "float32", "int64", "int32", "int16", "int8", "uint8", "bool"])


def native():
    try:
        import pytensor_llvm.v2 as v2
    except ImportError as e:
        raise ImportError(
            "The LLVM backend needs the `pytensor_llvm` extension "
            "(build it with rust/pytensor-llvm/build.sh)."
        ) from e
    return v2


def _type_of(v: Variable):
    """(numpy dtype, static shape) of a variable; scalars are 0-d tensors."""
    t = v.type
    if isinstance(t, ScalarType):
        shape = []
    elif isinstance(t, TensorType):
        shape = [None if s is None else int(s) for s in t.shape]
    else:
        raise NotImplementedError(f"LLVM backend: variables of type {t} are not supported")
    if t.dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"LLVM backend: dtype {t.dtype} is not supported")
    return np.dtype(t.dtype), shape


# ------------------------------------------------------------------ scalar ops

@cache
def _scalar_ops():
    """PyTensor scalar op class -> native scalar op class."""
    v2 = native()
    return {
        ps.Add: v2.Add,
        ps.Mul: v2.Mul,
        ps.Sub: v2.Sub,
        ps.TrueDiv: v2.TrueDiv,
        ps.IntDiv: v2.IntDiv,
        ps.Mod: v2.Mod,
        ps.Pow: v2.Pow,
        ps.Maximum: v2.Maximum,
        ps.Minimum: v2.Minimum,
        ps.ScalarMaximum: v2.Maximum,
        ps.ScalarMinimum: v2.Minimum,
        ps.Clip: v2.Clip,
        ps.Neg: v2.Neg,
        ps.Abs: v2.Abs,
        ps.Sign: v2.Sign,
        ps.Sqr: v2.Sqr,
        ps.Reciprocal: v2.Reciprocal,
        ps.Exp: v2.Exp,
        ps.Log: v2.Log,
        ps.Sqrt: v2.Sqrt,
        ps.Sin: v2.Sin,
        ps.Cos: v2.Cos,
        ps.Floor: v2.Floor,
        ps.Ceil: v2.Ceil,
        ps.Trunc: v2.Trunc,
        ps.RoundHalfToEven: v2.RoundHalfToEven,
        ps.RoundHalfAwayFromZero: v2.RoundHalfAwayFromZero,
        ps.Exp2: v2.Exp2,
        ps.Log2: v2.Log2,
        ps.Log10: v2.Log10,
        ps.Expm1: v2.Expm1,
        ps.Log1p: v2.Log1p,
        ps.Tan: v2.Tan,
        ps.Tanh: v2.Tanh,
        ps.Sinh: v2.Sinh,
        ps.Cosh: v2.Cosh,
        ps.ArcSin: v2.ArcSin,
        ps.ArcCos: v2.ArcCos,
        ps.ArcTan: v2.ArcTan,
        ps.ArcSinh: v2.ArcSinh,
        ps.ArcCosh: v2.ArcCosh,
        ps.ArcTanh: v2.ArcTanh,
        ps.ArcTan2: v2.ArcTan2,
        psm.Erf: v2.Erf,
        psm.Erfc: v2.Erfc,
        psm.GammaLn: v2.GammaLn,
        psm.Gamma: v2.Gamma,
        psm.Psi: v2.Psi,
        psm.Sigmoid: v2.Sigmoid,
        psm.Softplus: v2.Softplus,
        psm.Log1mexp: v2.Log1mexp,
        ps.LT: v2.LT,
        ps.GT: v2.GT,
        ps.LE: v2.LE,
        ps.GE: v2.GE,
        ps.EQ: v2.EQ,
        ps.NEQ: v2.NEQ,
        ps.IsNan: v2.IsNan,
        ps.IsInf: v2.IsInf,
        ps.Switch: v2.Switch,
        ps.AND: v2.AND,
        ps.OR: v2.OR,
        ps.XOR: v2.XOR,
        ps.Invert: v2.Invert,
        ps.Identity: v2.Identity,
        ps.Cast: v2.Cast,
        ps.Second: v2.Second,
    }


def scalar_op(op):
    """The native scalar op for a PyTensor scalar op (Composites become scalar graphs)."""
    if isinstance(op, Composite):
        return _composite(op)
    cls = _scalar_ops().get(type(op))
    if cls is None:
        raise NotImplementedError(f"LLVM backend: scalar op {op} is not supported")
    return cls()


def _composite(comp: Composite):
    v2 = native()
    sg = v2.ScalarGraph([np.dtype(i.type.dtype) for i in comp.fgraph.inputs])
    refs = {i: sg.input(k) for k, i in enumerate(comp.fgraph.inputs)}

    def ref(v):
        if v in refs:
            return refs[v]
        if isinstance(v, Constant):
            if v.type.dtype not in SUPPORTED_DTYPES:
                raise NotImplementedError(f"LLVM backend: scalar constant of dtype {v.type.dtype}")
            return sg.constant(float(np.asarray(v.data)), np.dtype(v.type.dtype))
        raise NotImplementedError(f"LLVM backend: unresolved scalar variable {v}")

    for node in comp.fgraph.toposort():
        for o in node.outputs:
            if o.type.dtype not in SUPPORTED_DTYPES:
                raise NotImplementedError(f"LLVM backend: scalar op {node.op} with dtype {o.type.dtype}")
        outs = sg.apply(scalar_op(node.op), [ref(i) for i in node.inputs], [np.dtype(o.type.dtype) for o in node.outputs])
        refs.update(zip(node.outputs, outs, strict=True))
    sg.set_outputs([ref(o) for o in comp.fgraph.outputs])
    return v2.Composite(sg)


# ------------------------------------------------------------------ graphs


class Conversion:
    """Converts the nodes of one FunctionGraph (or inner graph) into a native Graph."""

    def __init__(self, numba_fallback=True, options=None):
        self.v2 = native()
        self.graph = self.v2.Graph()
        self.values: dict[Variable, object] = {}
        self.numba_fallback = numba_fallback
        self.options = options or {}

    def value(self, v: Variable):
        if v in self.values:
            return self.values[v]
        if isinstance(v, Constant):
            dtype, _ = _type_of(v)
            val = self.graph.add_constant(np.asarray(v.data, dtype=dtype))
            self.values[v] = val
            return val
        raise NotImplementedError(f"LLVM backend: variable {v} used before it is defined")

    def add_input(self, v: Variable):
        dtype, shape = _type_of(v)
        self.values[v] = self.graph.add_input(dtype, shape)

    def convert_node(self, node):
        try:
            op = llvm_op(node.op, node, self)
        except NotImplementedError:
            if not self.numba_fallback:
                raise
            op = self._numba(node)
        if op is INLINED:
            return
        ins = [self.value(i) for i in node.inputs]
        out_types = [_type_of(o) for o in node.outputs]
        view_map = [(o, i) for o, inputs in getattr(node.op, "view_map", {}).items() for i in inputs]
        outs = self.graph.add_node(op, ins, out_types, view_map)
        self.values.update(zip(node.outputs, outs, strict=True))

    def _numba(self, node):
        from pytensor.link.llvm.numba_fallback import numba_fallback_kernel

        kernel = numba_fallback_kernel(node)
        return self.v2.Foreign(kernel.address, str(node.op), kernel)

    def convert_graph(self, inputs, outputs, nodes):
        for i in inputs:
            self.add_input(i)
        for n in nodes:
            self.convert_node(n)
        outs = []
        for o in outputs:
            if o in inputs or isinstance(o, Constant):
                # outputs that are inputs or constants get their own copy
                dtype, shape = _type_of(o)
                (c,) = self.graph.add_node(self.v2.DeepCopy(), [self.value(o)], [(dtype, shape)])
                outs.append(c)
            else:
                outs.append(self.value(o))
        self.graph.set_outputs(outs)
        return self.graph


#: Returned by converters that added the node's outputs themselves.
INLINED = object()


def fgraph_to_llvm(fgraph, *, fastmath=True, opt_level=3, cache=True, numba_fallback=True):
    """Convert and compile `fgraph`; returns a native ``Function`` and timings."""
    import time

    t0 = time.perf_counter()
    options = dict(fastmath=fastmath, cache=cache, opt_level=opt_level)
    conv = Conversion(numba_fallback=numba_fallback, options=options)
    graph = conv.convert_graph(fgraph.inputs, fgraph.outputs, fgraph.toposort())
    t1 = time.perf_counter()
    fn = conv.v2.compile(graph, **options)
    return fn, {"convert_s": t1 - t0, "program_s": time.perf_counter() - t1}


# ------------------------------------------------------------------ converters


@singledispatch
def llvm_op(op, node, cx: Conversion):
    """The native op for `node` (or INLINED if the converter added the outputs)."""
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@llvm_op.register(Elemwise)
def _(op, node, cx):
    return cx.v2.Elemwise(scalar_op(op.scalar_op))


@llvm_op.register(ps.ScalarOp)
def _(op, node, cx):
    # a bare scalar op is a 0-d Elemwise
    return cx.v2.Elemwise(scalar_op(op))


def _identity(op, acc_dtype):
    ident = op.scalar_op.identity
    if ident is None:
        raise NotImplementedError(f"LLVM backend: CAReduce with {op.scalar_op} has no identity")
    ident = np.asarray(ident).item()
    dt = np.dtype(acc_dtype)
    if dt.kind in "iu" and not np.isfinite(ident):
        info = np.iinfo(dt)
        ident = info.min if ident < 0 else info.max
    if dt.kind == "b":
        ident = bool(ident)
    f = float(ident)
    i = int(ident) if np.isfinite(f) else 0
    return f, i


@llvm_op.register(CAReduce)
def _(op, node, cx):
    ndim = node.inputs[0].type.ndim
    axes = sorted(range(ndim) if op.axis is None else [a % ndim for a in op.axis])
    acc = getattr(op, "acc_dtype", None) or node.outputs[0].type.dtype
    if acc not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"LLVM backend: accumulation in {acc}")
    return cx.v2.CAReduce(scalar_op(op.scalar_op), axes, _identity(op, acc), np.dtype(acc))


@llvm_op.register(DimShuffle)
def _(op, node, cx):
    return cx.v2.DimShuffle([None if o == "x" else int(o) for o in op.new_order])


def _identity_view(cx, node):
    """The output is the input, seen as a view."""
    v = cx.value(node.inputs[0])
    dtype, shape = _type_of(node.outputs[0])
    nd = len(shape)
    (out,) = cx.graph.add_node(cx.v2.DimShuffle(list(range(nd))), [v], [(dtype, shape)], [(0, 0)])
    cx.values[node.outputs[0]] = out
    return INLINED


@llvm_op.register(ViewOp)
@llvm_op.register(SpecifyShape)
@llvm_op.register(ScalarFromTensor)
@llvm_op.register(TensorFromScalar)
def _(op, node, cx):
    # SpecifyShape: the static output type is checked when the values are used
    return _identity_view(cx, node)


@llvm_op.register(Dot)
def _(op, node, cx):
    return cx.v2.Dot()


@llvm_op.register(DeepCopyOp)
def _(op, node, cx):
    return cx.v2.DeepCopy()


@llvm_op.register(OpFromGraph)
def _(op, node, cx):
    unit = getattr(op, "unit", None)
    inner = op.fgraph
    if unit is None:
        # inline into the current graph, with its own variable scope: the same inner
        # graph can be shared by several nodes
        outer = cx.values
        cx.values = {iv: outer[ov] if ov in outer else cx_value(cx, outer, ov) for iv, ov in zip(inner.inputs, node.inputs, strict=True)}
        try:
            for n in inner.toposort():
                cx.convert_node(n)
            results = []
            for io, oo in zip(inner.outputs, node.outputs, strict=True):
                if io in inner.inputs or isinstance(io, Constant):
                    dtype, shape = _type_of(oo)
                    (c,) = cx.graph.add_node(cx.v2.DeepCopy(), [cx.value(io)], [(dtype, shape)])
                    results.append(c)
                else:
                    results.append(cx.value(io))
        finally:
            cx.values = outer
        cx.values.update(zip(node.outputs, results, strict=True))
        return INLINED
    sub = Conversion(numba_fallback=cx.numba_fallback, options=cx.options)
    graph = sub.convert_graph(list(inner.inputs), list(inner.outputs), inner.toposort())
    kind = cx.v2.UnitKind.Compiled if unit is UnitKind.COMPILED else cx.v2.UnitKind.Interpreted
    return cx.v2.OpFromGraph(graph, kind, **cx.options)


def cx_value(cx, values, v):
    """`cx.value(v)` resolved in the scope `values`."""
    saved, cx.values = cx.values, values
    try:
        return cx.value(v)
    finally:
        cx.values = saved
