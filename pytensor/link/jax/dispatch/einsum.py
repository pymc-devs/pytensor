import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.einsum import Einsum


@jax_funcify.register(Einsum)
def jax_funcify_Einsum(op, **kwargs):
    subscripts = op.subscripts
    optimize = op.optimize

    def einsum(*operands):
        return jnp.einsum(subscripts, *operands, optimize=optimize)

    return einsum
