from pytensor.graph import node_rewriter
from pytensor.tensor import moveaxis
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.rewriting.basic import register_xcanonicalize
from pytensor.xtensor.shape import Stack, UnStack


@register_xcanonicalize
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


@register_xcanonicalize
@node_rewriter(tracks=[UnStack])
def lower_unstack(fgraph, node):
    [x] = node.inputs
    axis_to_unstack = x.type.dims.index(node.op.old_dim_name)

    x_tensor = tensor_from_xtensor(x)
    x_tensor_transposed = moveaxis(x_tensor, source=[axis_to_unstack], destination=[-1])
    final_tensor = x_tensor_transposed.reshape(
        (*x_tensor_transposed.shape[:-1], *node.op.unstacked_lengths)
    )

    new_out = xtensor_from_tensor(final_tensor, dims=node.outputs[0].type.dims)
    return [new_out]
