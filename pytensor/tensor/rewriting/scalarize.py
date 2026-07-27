"""Scalarization graph rewriter.

Removes heap allocations caused by boxing a logically-single value into a 0-d
array. It works on **boundaries**: ``TensorFromScalar(s)`` is the boundary
between the scalar world (below) and the tensor world (above). The pass is a
single forward toposort that pushes each boundary downstream as far as it goes.
Every op has one rule -- *what happens when a ``TensorFromScalar`` meets me?*:

* a **producer** (``Subtensor`` scalar-index, ``CAReduce`` full-reduction) creates
  a boundary -- a scalar with a ``TensorFromScalar`` just below it;
* a **consumer** (``Join``, and downstream Elemwise/fusion) swallows it;
* ``Elemwise``/``Composite`` is **both**: it swallows its scalar inputs, and if
  it ends up all-scalar it re-emits the ``scalar_op`` -- but only if a client will
  swallow that too, else it keeps the array;
* ``DimShuffle`` (expand/squeeze) is a free view: boundaries pass straight through.

A boundary swallowed all the way cancels (no alloc); one that stops materializes
as the single box the producer always had, so scalarizing is never a loss. Topo
order carries the cascade, so there is no fixpoint.

Runs after Composite fusion and before inplace (an inplace destination is an
array that survives scalarization), and before IndexedFusion -- so it never sees
the ``FusedElemwise``. See ``spec_scalarize.md``.
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
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.subtensor import Subtensor


class ScalarCAReduce(OpFromGraph):
    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, CAReduce)]
        return f"Scalar{node.op}"


class ScalarSubtensor(OpFromGraph):
    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, Subtensor)]
        return f"Scalar{node.op}"


class ScalarJoin(OpFromGraph):
    def __str__(self):
        return "ScalarJoin"


# Op types that can swallow a scalar input (so a producer feeding one is worth
# scalarizing). Elemwise swallows via its scalar_op / the mixed-Elemwise op.
_SCALAR_CONSUMERS = (Join, Elemwise)


def _scalar_behind(var):
    """The scalar behind a ``TensorFromScalar`` (through free views), or None."""
    if var.owner is not None and isinstance(var.owner.op, DimShuffle):
        var = var.owner.inputs[0]
    if var.owner is not None and isinstance(var.owner.op, TensorFromScalar):
        return var.owner.inputs[0]
    return None


def _has_scalar_consumer(fgraph, var):
    """Whether any client of ``var`` (through free views) can swallow a scalar."""
    for client, _ in fgraph.clients[var]:
        if client == "output":
            continue
        if isinstance(client.op, DimShuffle):
            if _has_scalar_consumer(fgraph, client.outputs[0]):
                return True
        elif isinstance(client.op, _SCALAR_CONSUMERS):
            return True
    return False


class Scalarize(GraphRewriter):
    def apply(self, fgraph):
        for node in fgraph.toposort():
            if node not in fgraph.apply_nodes:
                continue
            op = node.op
            replacement = None
            if isinstance(op, Join) and op.axis == 0 and node.outputs[0].type.ndim == 1:
                replacement = self._swallow_join(node)
            elif isinstance(op, Subtensor) and node.outputs[0].type.ndim == 0:
                replacement = self._emit_producer(fgraph, node, ScalarSubtensor)
            elif isinstance(op, Elemwise):
                scalars = [_scalar_behind(inp) for inp in node.inputs]
                # A size-1 all-scalar Elemwise is a pure scalar computation: run
                # the scalar op on the scalars and box only the output.
                if len(node.outputs) == 1 and all(s is not None for s in scalars):
                    self._pass_through_elemwise(fgraph, node, scalars)
            if replacement is None:
                continue
            try:
                fgraph.replace_all_validate(replacement, reason="scalarize")
            except InconsistencyError:
                continue

    @staticmethod
    def _distribute(fgraph, out, scalar):
        # Give each consumer its OWN TensorFromScalar box (the scalar value is
        # shared, the boundary is not), so every downstream fused/elemwise input
        # is single-client and swallows cleanly -- we build the graph we want, a
        # cheap producer feeding N consumers is just N boxes.
        for client, input_idx in list(fgraph.clients[out]):
            if client == "output":
                continue
            box = tensor_from_scalar(scalar)
            if out.type.ndim:
                box = box.dimshuffle(["x"] * out.type.ndim)
            fgraph.change_node_input(client, input_idx, box, reason="scalarize")

    @staticmethod
    def _emit_producer(fgraph, node, scalar_op_type):
        # A producer creates a scalar with a boundary below it, but only if a
        # client will swallow it (otherwise the box survives -- pure churn).
        out = node.outputs[0]
        if not _has_scalar_consumer(fgraph, out):
            return None
        inner = [inp.type() for inp in node.inputs]
        scalar = scalar_op_type(inner, [scalar_from_tensor(node.op(*inner))])(
            *node.inputs
        )
        Scalarize._distribute(fgraph, out, scalar)
        return None

    @staticmethod
    def _pass_through_elemwise(fgraph, node, scalars):
        # size-1 all-scalar Elemwise: swallow every scalar input, re-emit the
        # scalar_op -- but only if a client will swallow it (else box survives).
        out = node.outputs[0]
        if not _has_scalar_consumer(fgraph, out):
            return None
        Scalarize._distribute(fgraph, out, node.op.scalar_op(*scalars))
        return None

    @staticmethod
    def _swallow_join(node):
        outer, inner, entries = [], [], []
        found = False
        for inp in node.inputs:
            scalar = _scalar_behind(inp)
            if scalar is None:
                v = inp.type()
                entries.append(v)
                outer.append(inp)
            else:
                v = scalar.type()
                entries.append(tensor_from_scalar(v).dimshuffle("x"))
                outer.append(scalar)
                found = True
            inner.append(v)
        if not found:
            return None
        return [(node.outputs[0], ScalarJoin(inner, [join(0, *entries)])(*outer))]


optdb.register(
    "scalarize",
    Scalarize(),
    "numba",
    # After Composite fusion (49), before inplace (50.5) -- inplace only touches
    # the arrays that survive scalarization -- and before IndexedFusion (100).
    position=50.4,
)
