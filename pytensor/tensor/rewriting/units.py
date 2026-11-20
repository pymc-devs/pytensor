"""Group small nodes into compiled units (for the LLVM backend).

A late rewrite that wraps runs of small, statically shaped nodes in
``OpFromGraph(execution=Execution.COMPILED)``: the backend generates each group as one
function, so a chain of tiny Elemwise / CAReduce / DimShuffle / DeepCopy nodes doesn't become one
kernel call (and one compiled function) per node.

Groups are convex: no path leaves a group and comes back into it.

ScalarType values only exist inside compiled units (they don't cross the backend's
kernel ABI): nodes connected by scalars are always one unit, with whatever lies on a
path between them. A scalar argument of the function enters as ``TensorFromScalar`` of
it, which the backend passes as the 0-d array the argument arrives in.
"""

from __future__ import annotations

from pytensor.compile import optdb
from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.graph.traversal import toposort_with_orderings
from pytensor.link.llvm.analysis.layout import AttachLayoutFeature
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import (
    Alloc,
    Join,
    Split,
    TensorFromScalar,
    scalar_from_tensor,
    tensor_from_scalar,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Sum
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.type import TensorType


GROUPABLE = (
    DimShuffle,
    Subtensor,
    Alloc,
    Split,
)

SMALL_GROUPABLE = (
    Elemwise,
    CAReduce,
    DeepCopyOp,
    Sum,
    FusedElemwise,
    Join,
    Sum,
)


def _small(v, max_elems):
    t = v.type
    if not isinstance(t, TensorType) or any(s is None for s in t.shape):
        return False
    n = 1
    for s in t.shape:
        n *= s
    return n <= max_elems


def _scalar(v):
    # constant scalars are folded into the code: they need no unit
    return isinstance(v.type, ScalarType) and not isinstance(v, Constant)


def _boxed_argument(node):
    """`TensorFromScalar` of a scalar argument of the function: the 0-d array the
    argument arrives in (see `GroupIntoUnits.box_scalar_arguments`)."""
    return isinstance(node.op, TensorFromScalar) and node.inputs[0].owner is None


def _unit_views(fgraph, group, inputs, outs):
    """The `view_map` of a unit of the nodes `group`: the outputs that may be views of
    an input, from the `view_map`s of its nodes. With it the destroy handler keeps
    ordering the readers of such an output before what overwrites the input.

    A unit never overwrites its inputs (the planner gives the nodes in it copies to
    overwrite) and its other outputs are new memory, so that is all it must declare.
    Returns None when it can't: an output that may view several inputs, one of which
    something overwrites."""
    pos = {v: k for k, v in enumerate(inputs)}
    views = {}  # inner variable -> positions of the inputs it may be a view of
    for n in group:
        for k, srcs in n.op.view_map.items():
            viewed = set().union(
                *(
                    {pos[i]} if i in pos else views.get(i, set())
                    for i in (n.inputs[s] for s in srcs)
                )
            )
            if viewed:
                views[n.outputs[k]] = viewed
    view_map = {}
    for j, o in enumerate(outs):
        viewed = views.get(o, set())
        if len(viewed) == 1:
            view_map[j] = list(viewed)
        elif (
            viewed
            and hasattr(fgraph, "destroy_handler")
            and fgraph.has_destroyers([inputs[i] for i in viewed])
        ):
            return None
    return view_map


def _has_scalars(group):
    return any(_scalar(i) for n in group if not _boxed_argument(n) for i in n.inputs)


class GroupIntoUnits(GraphRewriter):
    def __init__(self, max_elems=128):
        super().__init__()
        self.max_elems = max_elems

    def eligible(self, node):
        if isinstance(node.op, GROUPABLE):
            return True
        return isinstance(node.op, SMALL_GROUPABLE) and all(
            _small(v, self.max_elems) for v in (*node.inputs, *node.outputs)
        )

    @staticmethod
    def scalar_clusters(fgraph, order, before, after):
        """Sets of nodes that must be one unit: those connected by ScalarType values,
        with what lies on a path between them (so that the unit is convex).

        `order`: the nodes in topological order; `before`/`after`: node -> the nodes it
        depends on / that depend on it (data and orderings)."""
        parent = {}

        def find(n):
            while parent.get(n, n) is not n:
                n = parent[n]
            return n

        def union(a, b):
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra is not rb:
                parent[ra] = rb

        for node in order:
            for i in node.inputs:
                if _scalar(i) and i.owner is not None:
                    union(node, i.owner)
        clusters = {}
        for n in parent:
            clusters.setdefault(find(n), set()).add(n)
        clusters = list(clusters.values())

        def reach(start, edges):
            seen, stack = set(start), list(start)
            while stack:
                for m in edges[stack.pop()]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            return seen

        # convex closure; clusters that come to overlap are merged
        done = []
        while clusters:
            c = clusters.pop()
            closed = reach(c, after) & reach(c, before)
            merged = [o for o in clusters + done if o & closed]
            if merged:
                for o in merged:
                    (clusters if o in clusters else done).remove(o)
                clusters.append(closed.union(*merged))
            else:
                done.append(closed)
        return done

    def groups(self, fgraph):
        """Convex groups of nodes, in topological order: the scalar clusters, each
        whole, and runs of eligible nodes joining them or each other.

        Convex with respect to the data edges and to the orderings the graph's features
        impose (e.g. an inplace node must run after the other readers of its input): a
        node that is ordered after a group member counts as depending on it.
        """
        orderings = fgraph.orderings()
        order = list(toposort_with_orderings(fgraph.outputs, orderings=orderings))
        before = {n: set() for n in order}
        for n in order:
            before[n].update(i.owner for i in n.inputs if i.owner is not None)
            before[n].update(orderings.get(n, ()))
        after = {n: set() for n in order}
        for n, bs in before.items():
            for m in bs:
                after[m].add(n)

        # the graph with each cluster as one item, in topological order
        item_of = {n: (n,) for n in order}
        for c in self.scalar_clusters(fgraph, order, before, after):
            item = tuple(n for n in order if n in c)
            for n in item:
                item_of[n] = item
        pos = {n: k for k, n in enumerate(order)}
        items = list(dict.fromkeys(item_of[n] for n in order))
        deps = {it: {item_of[m] for n in it for m in before[n]} - {it} for it in items}
        waiting = {it: len(d) for it, d in deps.items()}
        users = {it: [] for it in items}
        for it, d in deps.items():
            for m in d:
                users[m].append(it)
        ready = sorted(
            (it for it in items if not waiting[it]), key=lambda it: pos[it[0]]
        )
        ordered = []
        while ready:
            it = ready.pop(0)
            ordered.append(it)
            for u in users[it]:
                waiting[u] -= 1
                if not waiting[u]:
                    ready.append(u)
                    ready.sort(key=lambda it: pos[it[0]])

        groups, current = [], []
        in_group = set()  # items in the current group
        tainted = set()  # items that depend on the current group through other items
        for it in ordered:
            cluster = len(it) > 1 or (
                any(_scalar(i) for i in it[0].inputs) and not _boxed_argument(it[0])
            )
            if cluster or self.eligible(it[0]):
                if deps[it] & tainted:
                    # would make the group non-convex: start a new one
                    groups.append(current)
                    current, in_group, tainted = [], set(), set()
                current.extend(it)
                in_group.add(it)
            elif deps[it] & (in_group | tainted):
                tainted.add(it)
        groups.append(current)

        # a group with scalars is a unit even on its own
        return [g for g in groups if len(g) > 1 or _has_scalars(g)]

    def apply(self, fgraph):
        self.box_scalar_arguments(fgraph)
        clients = fgraph.clients
        for group in self.groups(fgraph):
            members = set(group)
            graph_outputs = set(fgraph.outputs)
            produced = {o for n in group for o in n.outputs}
            inputs = []
            for n in group:
                for i in n.inputs:
                    if (
                        i not in produced
                        and not isinstance(i, Constant)
                        and i not in inputs
                    ):
                        inputs.append(i)
            outs = [
                o
                for n in group
                for o in n.outputs
                if o in graph_outputs or any(c not in members for c, _ in clients[o])
            ]
            if not outs:
                continue
            view_map = _unit_views(fgraph, group, inputs, outs)
            if view_map is None:
                if _has_scalars(group):
                    raise NotImplementedError(
                        "LLVM backend: a unit output may view several inputs that "
                        "are overwritten"
                    )
                continue
            op = OpFromGraph(
                inputs,
                outs,
                inline=False,
                execution=Execution.COMPILED,
                name="unit",
                view_map=view_map,
            )
            new = op(*inputs, return_list=True)
            fgraph.replace_all(list(zip(outs, new, strict=True)), reason="llvm_units")

    @staticmethod
    def box_scalar_arguments(fgraph):
        """Scalar arguments of the function are read as `TensorFromScalar` of them, the
        0-d array they arrive in, so their readers can be units."""
        for s in fgraph.inputs:
            if not isinstance(s.type, ScalarType):
                continue
            uses = list(fgraph.clients[s])
            if not uses:
                continue
            box = tensor_from_scalar(s)
            back = scalar_from_tensor(box)
            for c, k in uses:
                if c != "output" and isinstance(c.op, TensorFromScalar):
                    fgraph.replace(c.outputs[0], box, reason="llvm_units")
                else:
                    fgraph.change_node_input(c, k, back, reason="llvm_units")


# opt-in by name (the LLVM linker requires them)
optdb.register("llvm_layouts", AttachLayoutFeature(), position=-0.5)
optdb.register("llvm_units", GroupIntoUnits(), position=200)
