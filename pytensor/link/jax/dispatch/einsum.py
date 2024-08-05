import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.einsum import Einsum


@jax_funcify.register(Einsum)
def jax_funcify_Einsum(op, **kwargs):
    """Dispatch einsum to JAX.

    This dispatch is triggered only when we couldn't optimize einsum at the PyTensor level.
    This happens when some of the dimension lengths are unknown. This is never a problem in JAX,
    as it always compiles a function per runtime input shape.
    """
    subscripts = op.subscripts

    def einsum(*operands):
        return jnp.einsum(subscripts, *operands, optimize="optimal")

    return einsum
