"""Scalarization graph rewriter.

Propagates ``ScalarType`` so values that are logically single scalars are not
boxed into 0-d arrays -- each box is one heap allocation per call, which
dominates the small ``logp_dlogp`` graphs PyMC compiles.

It is one graph rewriter because producer and consumer are decided
independently, then matched. One toposort collects the ``wanted`` set (0-d
values a consumer would take as a scalar, seen through free views like
``DimShuffle``). Each wanted, producible output is *realised* as
``tensor_from_scalar(scalar)`` -- type-preserving, so the scalar is shared and
the graph stays valid -- and then every scalar-capable consumer *bypasses* the
``tensor_from_scalar`` to take the scalar directly. One wanting consumer is
enough: a box that survives is the same allocation the producer had anyway, so
scalarizing is never a loss.

Scalar Ops are ``OpFromGraph``s wrapping the faithful tensor graph, so every
backend still runs them via ``perform``; only Numba has the scalar dispatch.

Only ``CAReduce`` (producer) and ``Join`` (consumer) are handled so far, inlined
below; the day this branching hurts, lift it into a dispatch.
"""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.graph.utils import InconsistencyError
from pytensor.tensor.basic import (
    Join,
    TensorFromScalar,
    join,
    scalar_from_tensor,
    tensor_from_scalar,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle


class ScalarCAReduce(OpFromGraph):
    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, CAReduce)]
        return f"Scalar{node.op}"


class ScalarJoin(OpFromGraph):
    def __str__(self):
        return "ScalarJoin"


class Scalarize(GraphRewriter):
    def apply(self, fgraph):
        toposort = fgraph.toposort()

        # Decide: 0-d values a Join would take as a size-1 entry, peeled back
        # through the free expand_dims to the producing value.
        wanted = set()
        for node in toposort:
            if not (
                isinstance(node.op, Join)
                and node.op.axis == 0
                and node.outputs[0].type.ndim == 1
            ):
                continue
            for inp in node.inputs:
                if inp.owner is not None and isinstance(inp.owner.op, DimShuffle):
                    inp = inp.owner.inputs[0]
                if inp.type.ndim == 0:
                    wanted.add(inp)

        # Realise: hand each wanted full reduction out as a scalar, re-boxed once
        # with tensor_from_scalar so consumers that don't bypass stay valid.
        # Same toposort -- deciding did not touch the graph.
        for node in toposort:
            if node not in fgraph.apply_nodes or node.outputs[0] not in wanted:
                continue
            if not isinstance(node.op, CAReduce):
                continue
            inner = [inp.type() for inp in node.inputs]
            scalar = ScalarCAReduce(inner, [scalar_from_tensor(node.op(*inner))])(
                *node.inputs
            )
            try:
                fgraph.replace_all_validate(
                    [(node.outputs[0], tensor_from_scalar(scalar))], reason="scalarize"
                )
            except InconsistencyError:
                continue

        # Bypass: a Join reads the scalar behind each tensor_from_scalar directly;
        # a box no consumer bypasses DCEs away.
        for node in fgraph.toposort():
            if node not in fgraph.apply_nodes or not isinstance(node.op, Join):
                continue
            outer, inner, entries = [], [], []
            found = False
            for inp in node.inputs:
                src = inp
                if src.owner is not None and isinstance(src.owner.op, DimShuffle):
                    src = src.owner.inputs[0]
                if src.owner is not None and isinstance(src.owner.op, TensorFromScalar):
                    v = src.owner.inputs[0].type()
                    entries.append(tensor_from_scalar(v).dimshuffle("x"))
                    outer.append(src.owner.inputs[0])
                    found = True
                else:
                    v = inp.type()
                    entries.append(v)
                    outer.append(inp)
                inner.append(v)
            if not found:
                continue
            try:
                fgraph.replace_all_validate(
                    [(node.outputs[0], ScalarJoin(inner, [join(0, *entries)])(*outer))],
                    reason="scalarize",
                )
            except InconsistencyError:
                continue


optdb.register(
    "scalarize",
    Scalarize(),
    "numba",
    # After fusion (position 100), so the fused reductions and the raveled
    # gradient Join are in their final form before we scalarize them.
    position=100.5,
)
