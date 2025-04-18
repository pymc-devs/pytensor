import mlx.core as mx

from pytensor.graph import FunctionGraph
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    # Create a function graph for the core operation
    core_node = op._create_dummy_core_node(node.inputs)
    core_fgraph = FunctionGraph(inputs=core_node.inputs, outputs=core_node.outputs)

    # Convert the core function graph to an MLX function
    tuple_core_fn = mlx_funcify(core_fgraph, **kwargs)

    # If there's only one output, unwrap it from the tuple
    if len(node.outputs) == 1:

        def core_fn(*inputs):
            return tuple_core_fn(*inputs)[0]
    else:
        core_fn = tuple_core_fn

    # Apply vmap for each batch dimension
    batch_ndims = op.batch_ndim(node)
    vmap_fn = core_fn
    for _ in range(batch_ndims):
        vmap_fn = mx.vmap(vmap_fn)

    def blockwise_fn(*inputs):
        # Check for runtime broadcasting compatibility
        op._check_runtime_broadcast(node, inputs)

        # Handle broadcasting for batched dimensions
        if batch_ndims > 0:
            # Get batch shapes for broadcasting
            batch_shapes = [inp.shape[:batch_ndims] for inp in inputs]

            # Calculate the broadcasted batch shape
            from functools import reduce

            def broadcast_shapes(shape1, shape2):
                return tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2, strict=True))

            if batch_shapes:
                broadcasted_shape = reduce(broadcast_shapes, batch_shapes)

                # Broadcast inputs to the common batch shape
                broadcasted_inputs = []
                for inp in inputs:
                    if inp.shape[:batch_ndims] != broadcasted_shape:
                        # Create the full target shape
                        target_shape = broadcasted_shape + inp.shape[batch_ndims:]
                        # Broadcast the input
                        broadcasted_inputs.append(mx.broadcast_to(inp, target_shape))
                    else:
                        broadcasted_inputs.append(inp)

                # Apply the vectorized function to the broadcasted inputs
                return vmap_fn(*broadcasted_inputs)

        # No broadcasting needed
        return vmap_fn(*inputs)

    return blockwise_fn
