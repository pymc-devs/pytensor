import pytensor.tensor as pt
from pytensor.graph import node_rewriter
from pytensor.tensor import (
    broadcast_to,
    concat_with_broadcast,
    expand_dims,
    moveaxis,
    specify_shape,
    squeeze,
)
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.rewriting.basic import register_lower_xtensor
from pytensor.xtensor.rewriting.utils import lower_aligned
from pytensor.xtensor.shape import (
    Broadcast,
    Concat,
    ExpandDims,
    Squeeze,
    Stack,
    Transpose,
    UnStack,
)


@register_lower_xtensor
@node_rewriter(tracks=[Stack])
def lower_stack(fgraph, node):
    [x] = node.inputs
    batch_ndim = x.type.ndim - len(node.op.stacked_dims)
    stacked_axes = [
        i for i, dim in enumerate(x.type.dims) if dim in node.op.stacked_dims
    ]
    end = tuple(range(-len(stacked_axes), 0))

    x_tensor = tensor_from_xtensor(x)
    x_tensor_transposed = moveaxis(x_tensor, source=stacked_axes, destination=end)
    if batch_ndim == (x.type.ndim - 1):
        # This happens when we stack a "single" dimension, in this case all we need is the transpose
        # Note: If we have meaningful rewrites before lowering, consider canonicalizing this as a Transpose + Rename
        final_tensor = x_tensor_transposed
    else:
        final_shape = (*tuple(x_tensor_transposed.shape)[:batch_ndim], -1)
        final_tensor = x_tensor_transposed.reshape(final_shape)

    new_out = xtensor_from_tensor(final_tensor, dims=node.outputs[0].type.dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter(tracks=[UnStack])
def lower_unstack(fgraph, node):
    x = node.inputs[0]
    unstacked_lengths = node.inputs[1:]
    axis_to_unstack = x.type.dims.index(node.op.old_dim_name)

    x_tensor = tensor_from_xtensor(x)
    x_tensor_transposed = moveaxis(x_tensor, source=[axis_to_unstack], destination=[-1])
    final_tensor = x_tensor_transposed.reshape(
        (*x_tensor_transposed.shape[:-1], *unstacked_lengths)
    )
    # Reintroduce any static shape information that was lost during the reshape
    final_tensor = specify_shape(final_tensor, node.outputs[0].type.shape)

    new_out = xtensor_from_tensor(final_tensor, dims=node.outputs[0].type.dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter(tracks=[Concat])
def lower_concat(fgraph, node):
    out_dims = node.outputs[0].type.dims
    concat_dim = node.op.dim
    concat_axis = out_dims.index(concat_dim)

    # Convert input XTensors to Tensors and align batch dimensions
    tensor_inputs = [lower_aligned(inp, out_dims) for inp in node.inputs]
    joined_tensor = concat_with_broadcast(tensor_inputs, axis=concat_axis)
    new_out = xtensor_from_tensor(joined_tensor, dims=out_dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter(tracks=[Transpose])
def lower_transpose(fgraph, node):
    [x] = node.inputs
    # Use the final dimensions that were already computed in make_node
    out_dims = node.outputs[0].type.dims
    in_dims = x.type.dims

    # Compute the permutation based on the final dimensions
    perm = tuple(in_dims.index(d) for d in out_dims)
    x_tensor = tensor_from_xtensor(x)
    x_tensor_transposed = x_tensor.transpose(perm)
    new_out = xtensor_from_tensor(x_tensor_transposed, dims=out_dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter([Squeeze])
def lower_squeeze(fgraph, node):
    """Rewrite Squeeze to tensor.squeeze."""
    [x] = node.inputs
    x_tensor = tensor_from_xtensor(x)
    x_dims = x.type.dims
    dims_to_remove = node.op.dims
    axes_to_squeeze = tuple(x_dims.index(d) for d in dims_to_remove)
    x_tensor_squeezed = squeeze(x_tensor, axis=axes_to_squeeze)

    new_out = xtensor_from_tensor(x_tensor_squeezed, dims=node.outputs[0].type.dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter([ExpandDims])
def lower_expand_dims(fgraph, node):
    """Rewrite ExpandDims using tensor operations."""
    x, size = node.inputs
    out = node.outputs[0]

    # Convert inputs to tensors
    x_tensor = tensor_from_xtensor(x)
    size_tensor = tensor_from_xtensor(size)

    # Get the new dimension name and position
    new_axis = 0  # Always insert at front

    # Use tensor operations
    if out.type.shape[0] == 1:
        # Simple case: just expand with size 1
        result_tensor = expand_dims(x_tensor, new_axis)
    else:
        # Otherwise broadcast to the requested size
        result_tensor = broadcast_to(x_tensor, (size_tensor, *x_tensor.shape))

    # Preserve static shape information
    result_tensor = specify_shape(result_tensor, out.type.shape)

    # Convert result back to xtensor
    result = xtensor_from_tensor(result_tensor, dims=out.type.dims)
    return [result]


@register_lower_xtensor
@node_rewriter(tracks=[Broadcast])
def lower_broadcast(fgraph, node):
    """Rewrite XBroadcast using tensor operations."""

    excluded_dims = node.op.exclude

    tensor_inputs = [
        lower_aligned(inp, out.type.dims)
        for inp, out in zip(node.inputs, node.outputs, strict=True)
    ]

    if not excluded_dims:
        # Simple case: All dimensions are broadcasted
        tensor_outputs = pt.broadcast_arrays(*tensor_inputs)

    else:
        # Complex case: Some dimensions are excluded from broadcasting
        # Pick the first dimension_length for each dim
        broadcast_dims = {
            d: None for d in node.outputs[0].type.dims if d not in excluded_dims
        }
        for xtensor_inp in node.inputs:
            for dim, dim_length in xtensor_inp.sizes.items():
                if dim in broadcast_dims and broadcast_dims[dim] is None:
                    # If the dimension is not excluded, set its shape
                    broadcast_dims[dim] = dim_length
        assert not any(value is None for value in broadcast_dims.values()), (
            "All dimensions must have a length"
        )

        # Create zeros with the broadcast dimensions, to then broadcast each input against
        # PyTensor will rewrite into using only the shapes of the zeros tensor
        broadcast_dims = pt.zeros(
            tuple(broadcast_dims.values()),
            dtype=node.outputs[0].type.dtype,
        )
        n_broadcast_dims = broadcast_dims.ndim

        tensor_outputs = []
        for tensor_inp, xtensor_out in zip(tensor_inputs, node.outputs, strict=True):
            n_excluded_dims = tensor_inp.type.ndim - n_broadcast_dims
            # Excluded dimensions are on the right side of the output tensor so we padright the broadcast_dims
            # second is equivalent to `np.broadcast_arrays(x, y)[1]` in PyTensor
            tensor_outputs.append(
                pt.second(
                    pt.shape_padright(broadcast_dims, n_excluded_dims),
                    tensor_inp,
                )
            )

    new_outs = [
        xtensor_from_tensor(out_tensor, dims=out.type.dims)
        for out_tensor, out in zip(tensor_outputs, node.outputs)
    ]
    return new_outs
