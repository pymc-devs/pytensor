from pytensor.compile.mode import optdb
from pytensor.graph import Constant, Op, node_rewriter
from pytensor.graph.destroyhandler import inplace_candidates
from pytensor.graph.replace import vectorize_node
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter
from pytensor.graph.rewriting.unify import OpPattern, OpPatternOpTypeType
from pytensor.tensor.basic import Alloc, ARange, alloc, shape_padleft
from pytensor.tensor.blockwise import Blockwise, _squeeze_left
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.elemwise import InplaceGraphOptimizer
from pytensor.tensor.shape import Reshape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    Subtensor,
)


def blockwise_of(core_op: OpPatternOpTypeType | OpPattern) -> OpPattern:
    if not isinstance(core_op, Op | OpPattern):
        core_op = OpPattern(core_op)
    return OpPattern(Blockwise, core_op=core_op)


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
# We do it after position>=60 so that Blockwise inplace rewrites will work also on useless Blockwise Ops
optdb.register(
    "local_useless_unbatched_blockwise",
    dfs_rewriter(local_useless_unbatched_blockwise, ignore_newtrees=True),
    "fast_run",
    "fast_compile",
    "blockwise",
    position=60,
)


# Avoid redundant cases early on for Ops whose default form is not Blockwised
@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter(
    tracks=[
        blockwise_of(
            Dot
            | Alloc
            | ARange
            | Subtensor
            | AdvancedSubtensor
            | AdvancedIncSubtensor
            | Reshape
        )
    ]
)
def local_eager_useless_unbatched_blockwise(fgraph, node):
    # Many Dot-related rewrites (eg, all of BlasOpt) happen before specialize
    # These other Ops can't always be trivially vectorized at runtime,
    # since their inputs may imply non-rectangular shapes.
    return local_useless_unbatched_blockwise.fn(fgraph, node)


@register_specialize("shape_unsafe")
@node_rewriter([Blockwise])
def local_blockwise_alloc(fgraph, node):
    """Push Allocs from the inputs to the output of Blockwise Ops.

    BOp = Blockwise(Op, signature="(x),(x)->(x)")
    BOp(vector, alloc(vector, 10, 5)) -> alloc(BOp)(vector, vector), 10, 5)
    BOp(vector, alloc(scalar, 10, 5)) -> alloc(BOp)(vector, alloc(scalar, 5), 10, 5)
    BOp(matrix, alloc(vector, 10, 5)) -> BOp(matrix, vector)

    This is critical to remove many unnecessary Blockwise, or to reduce the work done by it
    """

    op: Blockwise = node.op

    batch_ndim = op.batch_ndim(node)
    if not batch_ndim:
        return None

    if not any(var.owner and isinstance(var.owner.op, Alloc) for var in node.inputs):
        return None

    new_inputs = []
    batch_shapes = []
    can_push_any_alloc = False
    for inp, inp_sig in zip(node.inputs, op.inputs_sig, strict=True):
        if not all(inp.type.broadcastable[:batch_ndim]):
            if inp.owner and isinstance(inp.owner.op, Alloc):
                # Push batch dims from Alloc
                value, *shape = inp.owner.inputs

                # Check what to do with the value of the Alloc
                missing_ndim = inp.type.ndim - value.type.ndim
                squeezed_value = _squeeze_left(value, (batch_ndim - missing_ndim))
                if (
                    (((1,) * missing_ndim + value.type.broadcastable)[batch_ndim:])
                    != inp.type.broadcastable[batch_ndim:]
                ):
                    # We still need an Alloc for the core dims
                    core_shape = shape[batch_ndim:]
                    # And the batch dims of the squeezed value
                    squeezed_value_batch_ndim = squeezed_value.type.ndim - len(
                        core_shape
                    )
                    batch_shape = [
                        1 if broadcastable else dim
                        for broadcastable, dim in zip(
                            squeezed_value.type.broadcastable[
                                :squeezed_value_batch_ndim
                            ],
                            tuple(squeezed_value.shape)[:squeezed_value_batch_ndim],
                            strict=True,
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
                            inp.type.broadcastable, shape[:batch_ndim], strict=False
                        )
                    )
                )
                new_inputs.append(squeezed_value)
                can_push_any_alloc = True
                continue

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
        for i, batch_dims in enumerate(
            zip(*batch_shapes, strict=True)
        ):  # Transpose shape tuples
            if old_out_type.broadcastable[i]:
                continue
            for batch_dim in batch_dims:
                if batch_dim == 1:
                    continue
                batch_shape[i] = batch_dim
                if isinstance(batch_dim, Constant):
                    # Give preference to Constants
                    break

        copy_stack_trace(node.outputs, new_outs)
        new_outs = [
            alloc(
                new_out,
                *batch_shape,
                *new_out.shape[batch_ndim - missing_ndim :],
            )
            for new_out in new_outs
        ]
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


@register_specialize
@node_rewriter([blockwise_of(Reshape)])
def local_blockwise_reshape(fgraph, node):
    """Rewrite away square Blockwise reshapes.

    Reshape is tricky to vectorize eagerly, because a graph like
    `x.reshape([x.shape[0] * x.shape[1], -1])` has many operations
    that must be vectorized before we arrive at the reshape operation.

    For the square Reshape case, we must wait for all the intermediate
    operations to be lifted as Allocs
    """
    x, output_shape = node.inputs
    batch_ndim = node.op.batch_ndim(node)
    if all(output_shape.type.broadcastable[:batch_ndim]):
        batched_shape = x.shape[:batch_ndim]
        core_reshape = _squeeze_left(output_shape, batch_ndim)
        new_out = x.reshape([*tuple(batched_shape), *tuple(core_reshape)])
        copy_stack_trace(node.outputs[0], new_out)
        return [new_out]


class InplaceBlockwiseOptimizer(InplaceGraphOptimizer):
    op = Blockwise

    def filter_candidate_pairs(self, fgraph, node, protected_inputs):
        blockwise_op = node.op
        batch_ndim = blockwise_op.batch_ndim(node)
        out_batch_bcast = node.outputs[0].type.broadcastable[:batch_ndim]
        inputs = node.inputs

        candidate_inputs = set(
            inplace_candidates(
                fgraph,
                [
                    inp
                    for inp in inputs
                    if inp.type.broadcastable[:batch_ndim] == out_batch_bcast
                ],
                protected_inputs=protected_inputs,
            )
        )

        allowed_inplace_inputs = [
            i for i, inp in enumerate(inputs) if inp in candidate_inputs
        ]
        destroy_map = blockwise_op.core_op.inplace_on_inputs(
            allowed_inplace_inputs=allowed_inplace_inputs
        ).destroy_map

        if not destroy_map:
            return []

        outputs = node.outputs
        return [
            ((out_idx, outputs[out_idx]), (inp_idx, inputs[inp_idx]))
            for out_idx, inp_idxs in destroy_map.items()
            for inp_idx in inp_idxs
        ]

    def create_inplace_node(self, node, inplace_pattern):
        blockwise_op = node.op
        allowed_inplace_inputs = tuple(v[0] for v in inplace_pattern.values())
        inplace_core_op = blockwise_op.core_op.inplace_on_inputs(
            allowed_inplace_inputs=allowed_inplace_inputs
        )

        if not inplace_core_op.destroy_map:
            return node

        # Check Op is not trying to inplace on non-candidate inputs
        for destroyed_inputs in inplace_core_op.destroy_map.values():
            for destroyed_input in destroyed_inputs:
                if destroyed_input not in allowed_inplace_inputs:
                    raise ValueError(
                        f"Op {blockwise_op.core_op} destroy_map does not respect allowed_inplace_inputs {allowed_inplace_inputs}"
                    )

        # Recreate core_op with inplace
        inplace_blockwise_op = type(blockwise_op)(
            core_op=inplace_core_op,
            signature=blockwise_op.signature,
            name=blockwise_op.name,
            gufunc_spec=blockwise_op.gufunc_spec,
            destroy_map=inplace_core_op.destroy_map,
        )

        return inplace_blockwise_op.make_node(*node.inputs)


optdb.register(
    "blockwise_inplace",
    InplaceBlockwiseOptimizer(),
    "fast_run",
    "inplace",
    position=50.1,
)
