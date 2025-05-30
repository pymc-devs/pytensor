from pytensor.graph import node_rewriter
from pytensor.tensor import broadcast_to, join, moveaxis, specify_shape
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.rewriting.basic import register_xcanonicalize
from pytensor.xtensor.shape import Concat, ExpandDims, Squeeze, Stack, Transpose, UnStack


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


@register_xcanonicalize("shape_unsafe")
@node_rewriter(tracks=[Concat])
def lower_concat(fgraph, node):
    out_dims = node.outputs[0].type.dims
    concat_dim = node.op.dim
    concat_axis = out_dims.index(concat_dim)

    # Convert input XTensors to Tensors and align batch dimensions
    tensor_inputs = []
    for inp in node.inputs:
        inp_dims = inp.type.dims
        order = [
            inp_dims.index(out_dim) if out_dim in inp_dims else "x"
            for out_dim in out_dims
        ]
        tensor_inp = tensor_from_xtensor(inp).dimshuffle(order)
        tensor_inputs.append(tensor_inp)

    # Broadcast non-concatenated dimensions of each input
    non_concat_shape = [None] * len(out_dims)
    for tensor_inp in tensor_inputs:
        # TODO: This is assuming the graph is correct and every non-concat dimension matches in shape at runtime
        # I'm running this as "shape_unsafe" to simplify the logic / returned graph
        for i, (bcast, sh) in enumerate(
            zip(tensor_inp.type.broadcastable, tensor_inp.shape)
        ):
            if bcast or i == concat_axis or non_concat_shape[i] is not None:
                continue
            non_concat_shape[i] = sh

    assert non_concat_shape.count(None) == 1

    bcast_tensor_inputs = []
    for tensor_inp in tensor_inputs:
        # We modify the concat_axis in place, as we don't need the list anywhere else
        non_concat_shape[concat_axis] = tensor_inp.shape[concat_axis]
        bcast_tensor_inputs.append(broadcast_to(tensor_inp, non_concat_shape))

    joined_tensor = join(concat_axis, *bcast_tensor_inputs)
    new_out = xtensor_from_tensor(joined_tensor, dims=out_dims)
    return [new_out]


@register_xcanonicalize
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


@register_xcanonicalize
@node_rewriter([ExpandDims])
def local_expand_dims_reshape(fgraph, node):
    """Rewrite rule to convert expand_dims to reshape."""
    if not isinstance(node.op, ExpandDims):
        return False

    x = node.inputs[0]
    dim = node.op.dim

    # Create new dimensions list with the new dimension
    new_dims = list(x.type.dims)
    new_dims.append(dim)

    # Create new shape with the new dimension
    new_shape = list(x.type.shape)
    new_shape.append(1)

    # Create a new reshape operation
    from pytensor.xtensor.shape import reshape

    return [reshape(x, new_shape, new_dims)]


@register_xcanonicalize
@node_rewriter([Squeeze])
def local_squeeze_reshape(fgraph, node):
    """Rewrite rule to convert squeeze to reshape."""
    if not isinstance(node.op, Squeeze):
        return False

    x = node.inputs[0]
    dim = node.op.dim

    # Get the index of the dimension to remove
    if dim is not None:
        if dim not in x.type.dims:
            return False
        dim_idx = x.type.dims.index(dim)
        if x.type.shape[dim_idx] != 1:
            return False
    else:
        # Find all dimensions of size 1
        dim_idx = [i for i, s in enumerate(x.type.shape) if s == 1]
        if not dim_idx:
            return False

    # Create new dimensions and shape lists
    new_dims = list(x.type.dims)
    new_shape = list(x.type.shape)
    if dim is not None:
        new_dims.pop(dim_idx)
        new_shape.pop(dim_idx)
    else:
        # Remove all dimensions of size 1
        new_dims = [d for i, d in enumerate(new_dims) if i not in dim_idx]
        new_shape = [s for i, s in enumerate(new_shape) if i not in dim_idx]

    # Create a new reshape operation
    from pytensor.xtensor.shape import reshape

    return [reshape(x, new_shape, new_dims)]
