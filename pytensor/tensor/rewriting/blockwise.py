from pytensor.compile.mode import optdb
from pytensor.graph import node_rewriter
from pytensor.graph.replace import vectorize_node
from pytensor.graph.rewriting.basic import copy_stack_trace, out2in
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import _matrix_matrix_matmul
from pytensor.tensor.rewriting.basic import register_canonicalize


@node_rewriter([Blockwise])
def local_useless_blockwise(fgraph, node):
    """
    If there is a dispatch implementation that does not require Blockwise, use that instead.
    This means a user created a Blockwise manually when there was no need.

    Note: This rewrite is not registered by default anywhere
    """
    op = node.op
    inputs = node.inputs
    dummy_core_node = op._create_dummy_core_node(node.inputs)
    vect_node = vectorize_node(dummy_core_node, *inputs)
    if not isinstance(vect_node.op, Blockwise):
        return copy_stack_trace(node.outputs, vect_node.outputs)


@node_rewriter([Blockwise])
def local_useless_unbatched_blockwise(fgraph, node):
    """Remove Blockwise that don't have any batched dims."""
    op = node.op
    inputs = node.inputs

    if max(inp.type.ndim - len(sig) for inp, sig in zip(inputs, op.inputs_sig)) == 0:
        return copy_stack_trace(node.outputs, op.core_op.make_node(*inputs).outputs)


# We register this rewrite late, so that other rewrites need only target Blockwise Ops
optdb.register(
    "local_useless_unbatched_blockwise",
    out2in(local_useless_unbatched_blockwise, ignore_newtrees=True),
    "fast_run",
    "fast_compile",
    "blockwise",
    position=49,
)


# Avoid redundant cases early on for Ops whose default form is not Blockwised
@register_canonicalize
@node_rewriter(tracks=[_matrix_matrix_matmul])
def local_eager_useless_unbatched_blockwise(fgraph, node):
    return local_useless_unbatched_blockwise.fn(fgraph, node)
