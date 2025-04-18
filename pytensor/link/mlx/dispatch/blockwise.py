import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise

@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    core_f = mlx_funcify(op.core_op)
    batched_f = core_f
    for _ in range(op.batch_ndim(node)):
        batched_f = mx.vmap(batched_f)
        
    def wrapped_blockwise_f(*inputs):
        return batched_f(*inputs)

    return wrapped_blockwise_f
