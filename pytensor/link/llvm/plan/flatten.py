"""Stage 1: the graph as items. No code is generated here."""

from __future__ import annotations

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.link.llvm.plan.values import NodeItem, SubprogramItem, UnitItem, Values


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
    stands for several variables of the outer graph: the graph's `ViewFeature` knows."""
    for inner_output, output in zip(inner.outputs, node.outputs, strict=True):
        scope[output] = values.of(inner_output, inner_scope)


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
