import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor._linalg.products import Expm, KroneckerProduct


@jax_funcify.register(KroneckerProduct)
def jax_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return jnp.kron(x, y)

    return _kron


@jax_funcify.register(Expm)
def jax_funcify_Expm(op, **kwargs):
    def expm(x):
        return jax.scipy.linalg.expm(x)

    return expm
