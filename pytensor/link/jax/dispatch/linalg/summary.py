import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.linalg.summary import Det, SLogDet


@jax_funcify.register(Det)
def jax_funcify_Det(op, **kwargs):
    def det(x):
        return jnp.linalg.det(x)

    return det


@jax_funcify.register(SLogDet)
def jax_funcify_SLogDet(op, **kwargs):
    def slogdet(x):
        return jnp.linalg.slogdet(x)

    return slogdet
