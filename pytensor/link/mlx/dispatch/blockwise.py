import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    # 2) Otherwise, get the core python function for this Blockwise
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, core_node)

    # 3) Determine how many inputs correspond to batch dimensions
    n_batch = op.batch_ndim(node)

    # 4) Handle case where no vectorization is needed
    if n_batch == 0:
        return core_f

    # 5) Vectorize using mx.vmap over any batched inputs
    in_axes: list[int | None] = []
    for inp, sig in zip(node.inputs, op.inputs_sig):
        batch_ndim = inp.type.ndim - len(sig)
        if batch_ndim == 0:
            in_axes.append(None)
            continue

        batch_bcast = inp.type.broadcastable[:batch_ndim]
        # If all batch dims are broadcastable (size 1), treat input as static
        in_axes.append(0 if not all(batch_bcast) else None)

    if not any(axis == 0 for axis in in_axes):
        return core_f

    return mx.vmap(core_f, in_axes=tuple(in_axes))
