import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv


@jax_funcify.register(MatrixInverse)
def jax_funcify_MatrixInverse(op, **kwargs):
    def matrix_inverse(x):
        return jnp.linalg.inv(x)

    return matrix_inverse


@jax_funcify.register(MatrixPinv)
def jax_funcify_Pinv(op, **kwargs):
    def pinv(x):
        return jnp.linalg.pinv(x, hermitian=op.hermitian)

    return pinv
