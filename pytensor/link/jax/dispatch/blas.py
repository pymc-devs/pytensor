import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.blas import BatchedDot, Gemm, Gemv, Ger


@jax_funcify.register(BatchedDot)
def jax_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match along the first dimension of BatchedDot")
        return jnp.matmul(a, b)

    return batched_dot


@jax_funcify.register(Gemm)
def jax_funcify_Gemm(op, **kwargs):
    def gemm(z, alpha, x, y, beta):
        # Written out rather than fused by hand: XLA contracts this to a single dot with
        # alpha and beta folded into it.
        return beta * z + alpha * jnp.matmul(x, y)

    return gemm


@jax_funcify.register(Gemv)
def jax_funcify_Gemv(op, **kwargs):
    def gemv(y, alpha, A, x, beta):
        # As with Gemm above, XLA folds the scalars into the contraction itself.
        return beta * y + alpha * jnp.matmul(A, x)

    return gemv


@jax_funcify.register(Ger)
def jax_funcify_Ger(op, **kwargs):
    def ger(A, alpha, x, y):
        # As with Gemm above, there is no fused primitive to call and none is needed.
        return A + alpha * jnp.outer(x, y)

    return ger
