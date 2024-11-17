import torch

from pytensor.graph import FunctionGraph
from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.blockwise import Blockwise


@pytorch_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    batched_dims = op.batch_ndim(node)
    core_node = op._create_dummy_core_node(node.inputs)
    core_fgraph = FunctionGraph(inputs=core_node.inputs, outputs=core_node.outputs)
    inner_func = pytorch_funcify(
        core_fgraph, squeeze_output=len(node.outputs) == 1, **kwargs
    )

    for _ in range(batched_dims):
        inner_func = torch.vmap(inner_func)

    def batcher(*inputs):
        op._check_runtime_broadcast(node, inputs)
        # broadcast on batched_dims
        all_batched_dims = tuple(t.shape[:batched_dims] for t in inputs)
        batched_shape = torch.broadcast_shapes(*all_batched_dims)
        broadcast_inputs = [
            torch.broadcast_to(i, batched_shape + i.shape[batched_dims:])
            for i in inputs
        ]
        res = inner_func(*broadcast_inputs)
        return res

    return batcher
