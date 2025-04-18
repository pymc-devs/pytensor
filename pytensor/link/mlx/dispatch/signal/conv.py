from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.signal.conv import Conv1d

import mlx.core as mx


@mlx_funcify.register(Conv1d)
def mlx_funcify_Conv1d(op, node, **kwargs):
    mode = op.mode

    def conv1d(data, kernel):
        return mx.convolve(data, kernel, mode=mode)

    return conv1d
