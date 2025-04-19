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
        B, T = x.shape
        Bk, K = kernel.shape
        if B != Bk:
            raise ValueError(f"Batch mismatch: x has {B}, kernels has {Bk}")

        # 1) Flip each kernel for true convolution
        kernels_flipped = kernel[:, ::-1]  # shape (B, K)

        # 2) Reshape input into (N=1, H=T, C_in=B)
        x_in = x.T[None, :, :]              

        # 3) Build weight tensor of shape (C_out=B, H_f=K, C_in=1)
        w = kernels_flipped[:, :, None]     

        # 4) Convolve with one group per channel → valid mode
        y = mx.conv1d(
            x_in, w,
            stride=1,
            padding=0,
            dilation=1,
            groups=B
        )
        # y: (1, T-K+1, B) → drop batch and transpose to (B, T-K+1)
        return y[0].T
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
