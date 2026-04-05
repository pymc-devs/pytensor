import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.einsum import Einsum


@mlx_funcify.register(Einsum)
def mlx_funcify_Einsum(op, **kwargs):
    subscripts = op.subscripts

    def einsum(*operands):
        return mx.einsum(subscripts, *operands)

    return einsum
