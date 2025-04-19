import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.signal.conv import Conv1d

import numpy as np

def blockwise_conv1d(op, node, **kwargs):
    # if op.core_op.mode != "valid":
    #     raise NotImplementedError("Only 'valid' mode is supported for conv1d")
    
    # def inner_f(x, kernel):
    #     B, T = x.shape
    #     Bk, K = kernel.shape
    #     if B != Bk:
    #         raise ValueError(f"Batch mismatch: x has {B}, kernels has {Bk}")

    #     # 1) Flip each kernel for true convolution
    #     kernels_flipped = kernel[:, ::-1]  # shape (B, K)

    #     # 2) Reshape input into (N=1, H=T, C_in=B)
    #     x_in = x.T[None, :, :]              

    #     # 3) Build weight tensor of shape (C_out=B, H_f=K, C_in=1)
    #     w = kernels_flipped[:, :, None]     

    #     # 4) Convolve with one group per channel → valid mode
    #     y = mx.conv1d(
    #         x_in, w,
    #         stride=1,
    #         padding=0,
    #         dilation=1,
    #         groups=B
    #     )
    #     # y: (1, T-K+1, B) → drop batch and transpose to (B, T-K+1)
    #     return y[0].T
    
    def batched_conv1d(
            x: mx.array,
            kernels: mx.array,
            mode: str = op.core_op.mode,
            stride: int = 1,
            dilation: int = 1) -> mx.array:
        """
        Apply B separate 1D convolutions (full or valid) to B sequences in parallel.

        Parameters
        ----------
        x        : array of shape (B, T)
                B sequences of length T.
        kernels  : array of shape (B, K)
                B kernels of length K.
        mode     : {"valid", "full"}
                "valid" → no padding, output length = T - K + 1
                "full"  → zero‑pad so output length = T + K - 1
        stride   : int, convolution stride (default=1)
        dilation : int, convolution dilation (default=1)

        Returns
        -------
        out      : array of shape (B, L)
                where L = 
                    - T - K + 1   if mode="valid"
                    - T + K - 1   if mode="full"
        """
        # --- 1) shape checks ---
        B, T = x.shape
        Bk, K = kernels.shape
        if B != Bk:
            raise ValueError(f"Batch mismatch: x has {B}, kernels has {Bk}")

        # --- 2) flip kernels for convolution ---
        kernels_flipped = kernels[:, ::-1]  # shape (B, K)

        # --- 3) decide padding ---
        if mode == "valid":
            pad = 0
        elif mode == "full":
            pad = (K - 1) * dilation
        else:
            raise ValueError(f"Unsupported mode {mode!r}: choose 'valid' or 'full'")

        # --- 4) reshape into MLX conv1d form ---
        #   input: (N=1, H=T, C_in=B)
        x_in = x.T[None, :, :]

        #   weight: (C_out=B, H_f=K, C_in=1)
        w = kernels_flipped[:, :, None]

        # --- 5) run grouped conv1d ---
        y = mx.conv1d(
            x_in, w,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=B
        )
        # y shape: (1, H_out, B)

        # --- 6) return shape (B, H_out) ---
        return y[0].T

    return batched_conv1d

@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    # 1) If it's a Conv1d Blockwise, use the custom implementation
    if isinstance(op.core_op, Conv1d):
        return blockwise_conv1d(op, node, **kwargs)

    # 2) Otherwise, get the core python function for this Blockwise
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, core_node)

    # 3) Determine how many inputs correspond to batch dimensions
    n_batch = op.batch_ndim(node)

    # 4) Build in_axes: map only the first n_batch args, keep the rest static
    in_axes = tuple(0 if i < n_batch else None for i in range(len(node.inputs)))

    # 5) Vectorize (vmap) with in_axes
    blockwise_f = mx.vmap(core_f, in_axes=in_axes)

    # 6) Return the mapped function
    def blockwise_fun(*inputs):
        return blockwise_f(*inputs)

    return blockwise_fun
