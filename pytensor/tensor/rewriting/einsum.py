from typing import cast

from pytensor.graph import Apply, FunctionGraph, node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.einsum import Einsum, einsum
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.ofg import inline_ofg_node
from pytensor.tensor.variable import TensorVariable


@register_specialize
@node_rewriter([Einsum])
def optimize_einsum_inner_graph(
    fgraph: FunctionGraph, node: Apply
) -> list[TensorVariable] | None:
    """Try to optimize an einsum that was not optimizable at definition time.

    This can happen when users replace a graph without rebuilding

    Or when during the course of rewrites more specialized static shapes are found
    """
    op: Einsum = node.op

    if op.optimized:
        # Already optimized
        return None

    operands = node.inputs
    if any(None in operand.type.shape for operand in operands):
        return None

    new_out = einsum(op.subscripts, *operands)
    assert new_out.owner.op.optimized

    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_specialize
@node_rewriter([Einsum])
def inline_optimized_einsum(
    fgraph: FunctionGraph, node: Apply
) -> list[TensorVariable] | None:
    """Inline einsums that are already optimized.

    This allows the inner garph to be optimized with the rest of the graph, now that we got ordering right.
    """
    op: Einsum = node.op

    if not op.optimized:
        return None

    return cast(list[TensorVariable], inline_ofg_node(node))
