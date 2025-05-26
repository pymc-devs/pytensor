from pytensor.graph import node_rewriter
from pytensor.tensor import broadcast_to, join, moveaxis
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.rewriting.basic import register_xcanonicalize
from pytensor.xtensor.shape import Concat, Stack


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
