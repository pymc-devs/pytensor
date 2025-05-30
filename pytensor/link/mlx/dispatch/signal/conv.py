import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.signal.conv import Conv1d


@mlx_funcify.register(Conv1d)
def mlx_funcify_Conv1d(op, node=None, **kwargs):
    mode = op.mode

    def conv1d(data, kernel):
        return mx.convolve(data, kernel, mode=mode)

    return conv1d
