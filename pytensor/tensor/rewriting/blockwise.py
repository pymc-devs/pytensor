from pytensor.compile.mode import optdb
from pytensor.graph import node_rewriter
from pytensor.graph.replace import vectorize_node
from pytensor.graph.rewriting.basic import copy_stack_trace, out2in
from pytensor.tensor.basic import Alloc, ARange, shape_padleft
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedSubtensor, Subtensor


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

    batch_ndims = node.op.batch_ndim(node)
    if all(all(inp.type.broadcastable[:batch_ndims]) for inp in inputs):
        if batch_ndims:
            # Remove dummy batch dims
            axis = tuple(range(batch_ndims))
            inputs = [inp.squeeze(axis) for inp in inputs]
        new_outs = op.core_op.make_node(*inputs).outputs
        if batch_ndims:
            # Reintroduce dummy batch dims
            new_outs = [shape_padleft(out, batch_ndims) for out in new_outs]
        return copy_stack_trace(node.outputs, new_outs)


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
@register_stabilize
@register_specialize
@node_rewriter(tracks=[Blockwise])
def local_eager_useless_unbatched_blockwise(fgraph, node):
    if isinstance(
        node.op.core_op,
        (
            # Many Dot-related rewrites (e.g., all of BlasOpt) happen before specialize
            Dot,
            # These Ops can't always be trivially vectorized at runtime,
            # Since their inputs may imply non-rectangular shapes.
            Alloc,
            ARange,
            Subtensor,
            AdvancedSubtensor,
            AdvancedIncSubtensor,
        ),
    ):
        return local_useless_unbatched_blockwise.fn(fgraph, node)
