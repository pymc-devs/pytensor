import torch

from pytensor.graph import FunctionGraph
from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.blockwise import Blockwise


@pytorch_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    batched_dims = op.batch_ndim(node)
    core_node = op._create_dummy_core_node(node.inputs)
    core_fgraph = FunctionGraph(inputs=core_node.inputs, outputs=core_node.outputs)
    core_func = pytorch_funcify(core_fgraph)
    if len(node.outputs) == 1:

        def inner_func(*inputs):
            return core_func(*inputs)[0]
    else:
        inner_func = core_func

    for _ in range(batched_dims):
        inner_func = torch.vmap(inner_func)

    def batcher(*inputs):
        op._check_runtime_broadcast(node, inputs)
        return inner_func(*inputs)

    return batcher
