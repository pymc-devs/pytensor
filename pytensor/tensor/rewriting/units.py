"""Group small nodes into compiled units (for the LLVM_V2 backend).

A late rewrite that wraps runs of small, statically shaped nodes in
``OpFromGraph(unit=UnitKind.COMPILED)``: the backend generates each group as one
function, so a chain of tiny Elemwise / CAReduce / DimShuffle nodes doesn't become one
interpreter step (and one compiled function) per node.

Groups are convex: no path leaves a group and comes back into it.
"""

from __future__ import annotations

from pytensor.compile.builders import OpFromGraph, UnitKind
from pytensor.compile import optdb
from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.type import TensorType


GROUPABLE = (Elemwise, CAReduce, DimShuffle)


def _small(v, max_elems):
    t = v.type
    if not isinstance(t, TensorType) or any(s is None for s in t.shape):
        return False
    n = 1
    for s in t.shape:
        n *= s
    return n <= max_elems


class GroupIntoUnits(GraphRewriter):
    def __init__(self, max_elems=64):
        super().__init__()
        self.max_elems = max_elems

    def eligible(self, node):
        return isinstance(node.op, GROUPABLE) and all(
            _small(v, self.max_elems) for v in (*node.inputs, *node.outputs) if not isinstance(v, Constant)
        )

    def groups(self, fgraph):
        """Convex groups of eligible nodes, in topological order."""
        groups, current = [], []
        in_group = set()  # variables produced by the current group
        tainted = set()  # variables that depend on the current group through other nodes
        for node in fgraph.toposort():
            if self.eligible(node):
                if any(i in tainted for i in node.inputs):
                    # would make the group non-convex: start a new one
                    groups.append(current)
                    current, in_group, tainted = [], set(), set()
                current.append(node)
                in_group.update(node.outputs)
            elif any(i in in_group or i in tainted for i in node.inputs):
                tainted.update(node.outputs)
        groups.append(current)
        return [g for g in groups if len(g) > 1]

    def apply(self, fgraph):
        clients = fgraph.clients
        outputs = set(fgraph.outputs)
        for group in self.groups(fgraph):
            members = set(group)
            produced = {o for n in group for o in n.outputs}
            inputs = []
            for n in group:
                for i in n.inputs:
                    if i not in produced and not isinstance(i, Constant) and i not in inputs:
                        inputs.append(i)
            outs = [
                o
                for n in group
                for o in n.outputs
                if o in outputs or any(c not in members for c, _ in clients[o])
            ]
            if not outs:
                continue
            op = OpFromGraph(inputs, outs, inline=False, unit=UnitKind.COMPILED, name="unit")
            new = op(*inputs, return_list=True)
            fgraph.replace_all(list(zip(outs, new, strict=True)), reason="llvm_units")


# opt-in by name (the LLVM_V2 mode includes it)
optdb.register("llvm_units", GroupIntoUnits(), position=99)
