import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.blas import BatchedDot


@jax_funcify.register(BatchedDot)
def jax_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match along the first dimension of BatchedDot")
        return jnp.matmul(a, b)

    return batched_dot
