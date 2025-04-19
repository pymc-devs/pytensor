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
        # 1) Validate shapes
        B, T = x.shape
        Bk, K = kernel.shape
        if B != Bk:
            raise ValueError(f"Batch mismatch: x has {B}, kernels has {Bk}")

        # 2) Reshape x so that 'channels' = B, batch size = 1
        #    → input shape (N=1, H=T, C_in=B)
        x_in = x.T[None, :, :]  # shape (1, T, B)

        # 3) Build weight array of shape (C_out=B, H_f=K, C_in=1)
        #    groups = B will slice C_in into B single-channel groups
        w = kernel[:, :, None]  # shape (B, K, 1)

        # 4) Convolve with one group per sequence
        y = mx.conv1d(x_in, w,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=B)

        # 5) y has shape (1, T - K + 1, B); drop the batch axis and transpose
        return y[0].T  # final shape (B, T - K + 1)
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
