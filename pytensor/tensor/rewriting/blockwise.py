import itertools

from pytensor.compile import Supervisor
from pytensor.compile.mode import optdb
from pytensor.graph import Constant, node_rewriter
from pytensor.graph.replace import vectorize_node
from pytensor.graph.rewriting.basic import copy_stack_trace, in2out, out2in
from pytensor.tensor.basic import Alloc, ARange, alloc, shape_padleft
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.shape import Reshape
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
# We do it after position>=60 so that Blockwise inplace rewrites will work also on useless Blockwise Ops
optdb.register(
    "local_useless_unbatched_blockwise",
    out2in(local_useless_unbatched_blockwise, ignore_newtrees=True),
    "fast_run",
    "fast_compile",
    "blockwise",
    position=60,
)


# Avoid redundant cases early on for Ops whose default form is not Blockwised
@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter(tracks=[Blockwise])
def local_eager_useless_unbatched_blockwise(fgraph, node):
    if isinstance(
        node.op.core_op,
        Dot
        | Alloc
        | ARange
        | Subtensor
        | AdvancedSubtensor
        | AdvancedIncSubtensor
        | Reshape,
    ):
        # Many Dot-related rewrites (eg, all of BlasOpt) happen before specialize
        # These other Ops can't always be trivially vectorized at runtime,
        # since their inputs may imply non-rectangular shapes.
        return local_useless_unbatched_blockwise.fn(fgraph, node)


def _squeeze_left(x, stop_at_dim: int | None = None):
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

    op: Blockwise = node.op  # type: ignore

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
@node_rewriter([Blockwise])
def local_blockwise_reshape(fgraph, node):
    """Rewrite away square Blockwise reshapes.

    Reshape is tricky to vectorize eagerly, because a graph like
    `x.reshape([x.shape[0] * x.shape[1], -1])` has many operations
    that must be vectorized before we arrize at the reshape operation.

    For the square Reshape case, we must wait for all the intemediate
    operations to be lifted as Allocs
    """
    if not isinstance(node.op.core_op, Reshape):
        return None

    x, output_shape = node.inputs
    batch_ndim = node.op.batch_ndim(node)
    if all(output_shape.type.broadcastable[:batch_ndim]):
        batched_shape = x.shape[:batch_ndim]
        core_reshape = _squeeze_left(output_shape, batch_ndim)
        new_out = x.reshape([*tuple(batched_shape), *tuple(core_reshape)])
        copy_stack_trace(node.outputs[0], new_out)
        return [new_out]


@node_rewriter(tracks=[Blockwise], inplace=True)
def blockwise_inplace(fgraph, node):
    blockwise_op = node.op

    if blockwise_op.destroy_map:
        # Op already has inplace
        return

    # Find out valid inputs for inplacing
    batch_ndim = blockwise_op.batch_ndim(node)
    out_batch_bcast = node.outputs[0].type.broadcastable[:batch_ndim]

    protected_inputs = [
        f.protected for f in fgraph._features if isinstance(f, Supervisor)
    ]
    protected_inputs = list(itertools.chain.from_iterable(protected_inputs))
    protected_inputs.extend(fgraph.outputs)
    allowed_inplace_inputs = [
        idx
        for idx, inp in enumerate(node.inputs)
        if
        (
            # Constants would need to be recreated every time if inplaced
            not isinstance(inp, Constant)
            # We can only inplace on inputs that are not being broadcasted
            # As those are reused across iterations of Blockwise
            and node.inputs[idx].type.broadcastable[:batch_ndim] == out_batch_bcast
            # Inputs that are marked as protected or destroyed can't be inplaced
            and not fgraph.has_destroyers([inp])
            and inp not in protected_inputs
        )
    ]

    if not allowed_inplace_inputs:
        return None

    inplace_core_op = blockwise_op.core_op.inplace_on_inputs(
        allowed_inplace_inputs=allowed_inplace_inputs
    )

    if not inplace_core_op.destroy_map:
        return None

    # Check Op is not trying to inplace on non-candidate inputs
    for destroyed_inputs in inplace_core_op.destroy_map.values():
        for destroyed_input in destroyed_inputs:
            if destroyed_input not in allowed_inplace_inputs:
                raise ValueError(
                    f"Op {blockwise_op.core_op} destroy_map does not respect allowed_inplace_inputs {allowed_inplace_inputs}"
                )

    # Recreate core_op with inplace
    inplace_blockwise_op = Blockwise(
        core_op=inplace_core_op,
        signature=blockwise_op.signature,
        name=blockwise_op.name,
        gufunc_spec=blockwise_op.gufunc_spec,
        destroy_map=inplace_core_op.destroy_map,
    )

    out = inplace_blockwise_op.make_node(*node.inputs).outputs
    copy_stack_trace(node.outputs, out)
    return out


optdb.register(
    "blockwise_inplace",
    in2out(blockwise_inplace),
    "fast_run",
    "inplace",
    position=50.1,
)
