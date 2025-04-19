import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.signal.conv import Conv1d

def blockwise_conv1d(op, node):
    if op.core_op.mode != "valid":
        raise NotImplementedError("Only 'valid' mode is supported for conv1d")
    batches_ndim = op.batch_ndim(node)
    if batches_ndim != 1:
        raise NotImplementedError("Only 1D batches are supported for conv1d")
    
    _, kernel = node.inputs
    if not all(kernel.type.broadcastable[:batches_ndim]):
        raise NotImplementedError("Only 1D batches are supported for conv1d")
    
    def inner_f(x, kernel):
        x_reshaped = x.reshape(-1, x.shape[-1]).T # shape equals to (N, B) -> N Time as batches all together
        b = x_reshaped.shape[1] #
        kernel_squeeze = kernel.reshape(-1)
        f = kernel_squeeze.shape[0] # Number of filters
        kernel_reshaped = mx.broadcast_to(a=kernel_squeeze[None, :, None], shape=(b, f, b))
        conv_result = mx.conv1d(x_reshaped[None, :, :], kernel_reshaped, stride=1, padding=0, dilation=1)
        _, conv_shape, _ = conv_result.shape
        return mx.moveaxis(a=conv_result, source=-1, destination=0).reshape(x.shape[:-1] + (conv_shape,))
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
