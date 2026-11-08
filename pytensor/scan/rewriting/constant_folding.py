"""Fold constant Scan computations together with their temporary input buffers."""

from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import HasMultipleInnerGraph
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.graph.traversal import graph_inputs
from pytensor.scan.op import Scan
from pytensor.tensor.basic import Alloc, AllocEmpty
from pytensor.tensor.rewriting.basic import (
    constant_folding,
    local_remove_useless_assert,
    unconditional_constant_folding,
)


@node_rewriter([Alloc, AllocEmpty])
def fold_buffer(fgraph, node):
    return unconditional_constant_folding.transform(fgraph, node)


scan_inputs_constant_folding = in2out(
    local_remove_useless_assert,
    fold_buffer,
    constant_folding,
    ignore_newtrees=True,
)


@node_rewriter([Scan])
def scan_constant_folding(fgraph, node):
    """Fold a closed constant Scan, allowing temporary Alloc/AllocEmpty inputs.

    Buffer construction is evaluated on a copy. Only the completed Scan outputs
    replace the original graph, so intermediate buffers never become persistent
    constants when the Scan cannot be folded.
    """
    if not all(isinstance(inp, Constant) for inp in graph_inputs(node.inputs)):
        return None

    # Allow temporary allocations, but honor folding opt-outs in all inner graphs.
    pending = [node.op.fgraph]
    seen = set()
    while pending:
        graph = pending.pop()
        if id(graph) in seen:
            continue
        seen.add(id(graph))
        for inner_node in graph.apply_nodes:
            if isinstance(inner_node.op, Alloc | AllocEmpty):
                continue
            if not inner_node.op.do_constant_folding(graph, inner_node):
                return None
            if isinstance(inner_node.op, HasMultipleInnerGraph):
                pending.extend(inner_node.op.inner_graphs.values())

    inputs_fg = FunctionGraph(outputs=node.inputs, clone=True)
    scan_inputs_constant_folding.rewrite(inputs_fg)
    if not all(isinstance(inp, Constant) for inp in inputs_fg.outputs):
        return None

    constant_node = node.op.make_node(*inputs_fg.outputs)
    return constant_folding.transform(fgraph, constant_node)
