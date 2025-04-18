import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, core_node)
    blockwise_f = core_f
    for i in range(op.batch_ndim(node)):
        blockwise_f = mx.vmap(blockwise_f)

    def blockwise_fun(*inputs):
        return blockwise_f(*inputs)

    return blockwise_fun
