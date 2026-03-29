import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blas import BatchedDot


@mlx_funcify.register(BatchedDot)
def mlx_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match along the first dimension of BatchedDot")
        return mx.matmul(a, b)

    return batched_dot
