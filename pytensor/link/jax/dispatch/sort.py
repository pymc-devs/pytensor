from jax import numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.sort import SortOp


@jax_funcify.register(SortOp)
def jax_funcify_Sort(op, **kwargs):
    def sort(arr, axis):
        return jnp.sort(arr, axis=axis)

    return sort
