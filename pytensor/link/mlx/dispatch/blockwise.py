import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.signal.conv import Conv1d

import numpy as np

def blockwise_conv1d(op, node, **kwargs):
    if op.core_op.mode != "valid":
        raise NotImplementedError("Only 'valid' mode is supported for conv1d")
    # batches_ndim = op.batch_ndim(node)
    # if batches_ndim != 1:
    #     raise NotImplementedError("Only 1D batches are supported for conv1d")
    
    # _, kernel = node.inputs
    # if not all(kernel.type.broadcastable[:batches_ndim]):
    #     raise NotImplementedError("Only 1D batches are supported for conv1d")
    
    def inner_f(x, kernel):
        *bx, t = x.shape
        *bk, h = kernel.shape

        b = np.broadcast_shapes(bx, bk)

        x = x.reshape(b + (t,))
        kernel = kernel.reshape(b + (h,))

        x_reshaped = x.reshape(-1, t).T # shape equals to (N, B) -> N Time as batches all together
        kernel_squeeze = kernel.reshape(-1, h)
        b_prod = kernel_squeeze.shape[0]

        kernel_reshaped = mx.broadcast_to(a=kernel_squeeze[None, :, None], shape=(b_prod, h, b_prod))
        conv_result = mx.conv1d(x_reshaped[None, :, :], kernel_reshaped, stride=1, padding=0, dilation=1)
        _, conv_shape, _ = conv_result.shape
        return mx.moveaxis(a=conv_result, source=-1, destination=0).reshape(b + (conv_shape,))
    return inner_f

@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    if isinstance(op.core_op, Conv1d):
        return blockwise_conv1d(op, node, **kwargs)
    
    core_f = mlx_funcify(op.core_op)

    def blockwise_f(*inputs):
        return blockwise_f(*inputs)
    core_node = op._create_dummy_core_node(node.inputs)
    
    core_f = mlx_funcify(op.core_op, core_node)
    blockwise_f = core_f
    for i in range(op.batch_ndim(node)):
        blockwise_f = mx.vmap(blockwise_f)

    def blockwise_fun(*inputs):
        return blockwise_f(*inputs)

    return blockwise_fun
