from typing import Optional

from pytensor.compile.mode import optdb
from pytensor.graph import Constant, node_rewriter
from pytensor.graph.replace import vectorize_node
from pytensor.graph.rewriting.basic import copy_stack_trace, out2in
from pytensor.tensor.basic import Alloc, ARange, alloc, shape_padleft
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


def _squeeze_left(x, stop_at_dim: Optional[int] = None):
    """Squeeze any leading dims of `x` until a real dim or `stop_at_dim` (if not None) is reached."""
    x_dims = x.type.broadcastable
    squeeze_ndim = len(x_dims) if all(x_dims) else x_dims.index(False)
    if stop_at_dim is not None:
        squeeze_ndim = min(squeeze_ndim, stop_at_dim)
    if squeeze_ndim == 0:
        return x
    return x.squeeze(axis=tuple(range(squeeze_ndim)))


@register_specialize("shape_unsafe")
@node_rewriter([Blockwise])
def local_blockwise_alloc(fgraph, node):
    """Push Allocs from the inputs to the output of Blockwise Ops.

    BOp = Blockwise(Op, signature="(x),(x)->(x)")
    BOp(vector, alloc(vector, 10, 5)) -> alloc(BOp)(vector, vector), 10, 5)
    BOp(vector, alloc(scalar, 10, 5)) -> alloc(BOp)(vector, alloc(scalar, 5), 10, 5)
    BOp(matrix, alloc(vector, 10, 5)) -> BOp(matrix, vector)
    """

    if not any(isinstance(inp.owner.op, Alloc) for inp in node.inputs if inp.owner):
        return None

    op: Blockwise = node.op  # type: ignore

    batch_ndim = op.batch_ndim(node)
    if not batch_ndim:
        return None

    new_inputs = []
    batch_shapes = []
    can_push_any_alloc = False
    for inp, inp_sig in zip(node.inputs, op.inputs_sig):
        if inp.owner and isinstance(inp.owner.op, Alloc):
            # Push batch dims from Alloc
            value, *shape = inp.owner.inputs

            # Check what to do with the value of the Alloc
            squeezed_value = _squeeze_left(value, batch_ndim)
            missing_ndim = len(shape) - value.type.ndim
            if (
                (((1,) * missing_ndim + value.type.broadcastable)[batch_ndim:])
                != inp.type.broadcastable[batch_ndim:]
            ):
                # We still need an Alloc for the core dims
                core_shape = shape[batch_ndim:]
                # And the batch dims of the squeezed value
                squeezed_value_batch_ndim = squeezed_value.type.ndim - len(core_shape)
                batch_shape = [
                    1 if broadcastable else dim
                    for broadcastable, dim in zip(
                        squeezed_value.type.broadcastable[:squeezed_value_batch_ndim],
                        tuple(squeezed_value.shape)[:squeezed_value_batch_ndim],
                    )
                ]
                squeezed_value = alloc(squeezed_value, *batch_shape, *core_shape)
                if squeezed_value.type.broadcastable == inp.type.broadcastable:
                    # We can't change anything about this Alloc input
                    new_inputs.append(inp)
                    continue

            # We can push batch dims of this Alloc input
            batch_shapes.append(
                tuple(
                    1 if broadcastable else dim
                    for broadcastable, dim in zip(
                        inp.type.broadcastable, shape[:batch_ndim]
                    )
                )
            )
            new_inputs.append(squeezed_value)
            can_push_any_alloc = True

        else:
            # Nothing to do with this input other than removing dummy batch dims
            new_inputs.append(_squeeze_left(inp, batch_ndim))

    if not can_push_any_alloc:
        return None

    new_outs = node.op.make_node(*new_inputs).outputs

    new_out_type = new_outs[0].type
    old_out_type = node.outputs[0].type
    if new_out_type.broadcastable != old_out_type.broadcastable:
        # An Alloc is still needed to broadcast the new output to the original shape
        # We pick the most parsimonious batch dim from the pushed Alloc
        missing_ndim = old_out_type.ndim - new_out_type.ndim
        batch_shape = ([1] * missing_ndim + list(new_outs[0].shape))[:batch_ndim]
        for i, batch_dims in enumerate(zip(*batch_shapes)):  # Transpose shape tuples
            for batch_dim in batch_dims:
                if batch_dim == 1:
                    continue
                if isinstance(batch_dim, Constant):
                    # Give preference to Constants
                    batch_shape[i] = batch_dim
                    break
                elif old_out_type.broadcastable[i]:
                    # Only use non Constant shapes if absolutely necessary
                    # Otherwise, we use the shape of the non-alloc output
                    batch_shape[i] = batch_dim

        copy_stack_trace(node.outputs, new_outs)
        new_outs = [
            alloc(
                new_out,
                *batch_shape,
                *new_out.shape[batch_ndim - missing_ndim :],
            )
            for new_out in new_outs
        ]
    assert new_outs[0].type.broadcastable == old_out_type.broadcastable
    copy_stack_trace(node.outputs, new_outs)
    return new_outs
