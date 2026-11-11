"""Stage 1: the graph as items. No code is generated here."""

from __future__ import annotations

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.link.llvm.plan.values import NodeItem, SubprogramItem, UnitItem, Values
from pytensor.scalar.basic import ScalarType


def flatten(
    values: Values, nodes, scope, inside_unit=False, of_the_graph=False
) -> list:
    """The items running `nodes` (in order). `scope` maps variables to values and is
    filled with the nodes' outputs. OpFromGraphs are inlined, become units
    (`Execution.COMPILED`) or subprograms (`Execution.INTERPRETED`); inside a unit every
    OpFromGraph is inlined. `of_the_graph`: `nodes` are the graph being planned itself
    (marks the items, see `NodeItem`)."""
    items = []
    for node in nodes:
        op = node.op
        execution = (
            getattr(op, "execution", None) if isinstance(op, OpFromGraph) else None
        )
        if isinstance(op, OpFromGraph) and execution is None and fusable(op, node):
            # a FusedElemwise we can generate as one loop: keep it as one node
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(o.type) for o in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            items.append(NodeItem(node, ins, outs))
        elif isinstance(op, OpFromGraph) and (execution is None or inside_unit):
            # (inside generated code every OpFromGraph is inlined)
            inner = op.fgraph
            sub = {
                iv: values.of(ov, scope)
                for iv, ov in zip(inner.inputs, node.inputs, strict=True)
            }
            items += flatten(values, inner.toposort(), sub, inside_unit)
            _inlined_outputs(values, inner, node, sub, scope)
        elif isinstance(op, OpFromGraph) and execution is Execution.COMPILED:
            inner = op.fgraph
            sub = {
                iv: values.of(ov, scope)
                for iv, ov in zip(inner.inputs, node.inputs, strict=True)
            }
            inner_scope = dict(sub)
            try:
                inner_items = flatten(
                    values, inner.toposort(), inner_scope, inside_unit=True
                )
            except NotImplementedError:
                # can't be one unit (e.g. it contains an interpreted OpFromGraph):
                # run its nodes as tasks
                items += flatten(values, inner.toposort(), sub, inside_unit)
                _inlined_outputs(values, inner, node, sub, scope)
                continue
            sub = inner_scope
            outs = []
            for io, oo in zip(inner.outputs, node.outputs, strict=True):
                k = values.new(oo.type)
                scope[oo] = k
                outs.append((values.of(io, sub), k))
            items.append(
                UnitItem(inner_items, values.unit_inputs(inner_items), outs, str(op))
            )
        elif isinstance(op, OpFromGraph) and execution is Execution.INTERPRETED:
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(o.type) for o in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            items.append(SubprogramItem(node, ins, outs))
        else:
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(o.type) for o in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            items.append(NodeItem(node, ins, outs, of_the_graph))
    return items


def _inlined_outputs(values: Values, inner, node, inner_scope, scope) -> None:
    """The outputs of an inlined OpFromGraph are the values of its inner outputs. Where
    the inner graph passes an input through, or returns one variable twice, a value
    now stands for several variables of the outer graph (`Values.shared`)."""
    seen = set()
    for inner_output, output in zip(inner.outputs, node.outputs, strict=True):
        k = values.of(inner_output, inner_scope)
        if inner_output.owner is None or k in seen:
            values.shared.add(k)
        seen.add(k)
        scope[output] = k


def group_scalars(values: Values, items, graph_outputs) -> list:
    """Group items that touch ScalarType values into compiled units."""
    is_scalar = {k for k, t in values.types.items() if isinstance(t, ScalarType)}

    def touches(it):
        # constant scalars are folded into a node's code: they don't need a unit
        return isinstance(it, NodeItem) and any(
            k in is_scalar and k not in values.consts for k in (*it.inputs, *it.outputs)
        )

    if not any(touches(it) for it in items):
        return items
    from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar

    # Greedy convex groups in program order (like the `llvm_units` rewrite): an item
    # touching scalars joins the current group unless it depends on an item outside
    # the group that depends on the group; then a new group starts. A scalar made in
    # an earlier group crosses into a later one as a 0-d tensor.
    groups = []  # lists of items
    made_in = {}  # scalar vid -> index of the group producing it
    as_tensor = {}  # scalar vid -> 0-d tensor vid carrying it out of its group
    out = []  # items outside groups, and group placeholders (ints)
    produced, tainted = set(), set()
    for it in items:
        if not touches(it):
            if any(k in produced or k in tainted for k in it.inputs):
                tainted.update(o[1] if isinstance(o, tuple) else o for o in it.outputs)
            out.append(it)
            continue
        if not groups or any(k in tainted for k in it.inputs):
            groups.append([])
            out.append(len(groups) - 1)
            produced, tainted = set(), set()
        g = len(groups) - 1
        ins = list(it.inputs)
        for pos, k in enumerate(ins):
            if k not in is_scalar or k in values.consts:
                continue
            if k not in made_in:
                raise NotImplementedError("a ScalarType graph input")
            if made_in[k] == g:
                continue
            if k not in as_tensor:
                conv = tensor_from_scalar(values.types[k]()).owner
                t = values.new(conv.outputs[0].type)
                groups[made_in[k]].append(NodeItem(conv, [k], [t]))
                as_tensor[k] = t
            t = as_tensor[k]
            back = scalar_from_tensor(values.types[t]()).owner
            k2 = values.new(values.types[k])
            groups[g].append(NodeItem(back, [t], [k2]))
            made_in[k2] = g
            produced.add(k2)
            ins[pos] = k2
        it = NodeItem(it.node, ins, it.outputs) if ins != it.inputs else it
        groups[g].append(it)
        for k in it.outputs:
            produced.add(k)
            if k in is_scalar:
                made_in[k] = g
    consumers = {}
    for x in out:
        for s in groups[x] if isinstance(x, int) else [x]:
            for k in s.inputs:
                consumers.setdefault(k, set()).add(id(s))
    result = []
    for x in out:
        if not isinstance(x, int):
            result.append(x)
            continue
        sub = groups[x]
        inside = {id(s) for s in sub}
        outputs = []
        for s in sub:
            for k in s.outputs:
                if k in graph_outputs or any(
                    c not in inside for c in consumers.get(k, ())
                ):
                    if k in is_scalar:
                        raise NotImplementedError(
                            "a ScalarType value used outside its compiled unit"
                        )
                    outputs.append((k, k))
        result.append(UnitItem(sub, values.unit_inputs(sub), outputs, "scalars"))
    return toposort_items(result)


def fusable(op, node) -> bool:
    """A FusedElemwise we can generate as one loop?"""
    from pytensor.link.llvm.codegen.fused import fused_spec
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    if not isinstance(op, FusedElemwise):
        return False
    try:
        fused_spec(op, node)
    except NotImplementedError:
        return False
    return True


def _outputs_of(it):
    if isinstance(it, UnitItem):
        return [o for _, o in it.outputs]
    return it.outputs


def toposort_items(items):
    """Order items by data dependencies, keeping the given order where possible."""
    producer = {k: i for i, it in enumerate(items) for k in _outputs_of(it)}
    deps = [
        {producer[k] for k in it.inputs if k in producer and producer[k] != i}
        for i, it in enumerate(items)
    ]
    done, order = set(), []
    pending = list(range(len(items)))
    while pending:
        progress = False
        rest = []
        for i in pending:
            if deps[i] <= done:
                order.append(items[i])
                done.add(i)
                progress = True
            else:
                rest.append(i)
        if not progress:
            raise NotImplementedError(
                "scalar operations interleaved with other nodes (a cycle)"
            )
        pending = rest
    return order
