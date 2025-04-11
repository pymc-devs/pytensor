import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.math import Dot


@mlx_funcify.register(Dot)
def mlx_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return mx.matmul(x, y)

    return dot
