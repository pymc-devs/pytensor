import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    # Get the core python function for this Blockwise operation
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, core_node)

    # Determine how many batch dimensions are present in the output
    n_batch = op.batch_ndim(node)

    # If there are no batch dimensions, just return the core function
    if n_batch == 0:
        return core_f

    # Build in_axes specification for mx.vmap
    # Each input can be vectorized (axis=0) or static (axis=None)
    in_axes: list[int | None] = []
    for inp, sig in zip(node.inputs, op.inputs_sig):
        batch_ndim = inp.type.ndim - len(sig)
        if batch_ndim == 0:
            # Input has no batch dimensions - treat as static
            in_axes.append(None)
            continue

        batch_bcast = inp.type.broadcastable[:batch_ndim]
        # If all batch dims are broadcastable (size 1), treat input as static
        # Otherwise, vectorize over the first dimension (axis=0)
        in_axes.append(0 if not all(batch_bcast) else None)

    # If all inputs are static (no actual vectorization needed), return core function
    # This prevents calling mx.vmap with all-None in_axes, which would raise:
    # "ValueError: At least one of in_axes must be non-None"
    if not any(axis == 0 for axis in in_axes):
        return core_f

    # Apply mx.vmap to vectorize the core function over batch dimensions
    return mx.vmap(core_f, in_axes=tuple(in_axes))
