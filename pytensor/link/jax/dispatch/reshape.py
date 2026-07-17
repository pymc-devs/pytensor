import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.reshape import (
    JoinDims,
    SplitDims,
)


@jax_funcify.register(JoinDims)
def jax_funcify_JoinDims(op, node, **kwargs):
    start_axis = op.start_axis
    n_axes = op.n_axes

    def join_dims(x):
        output_shape = (*x.shape[:start_axis], -1, *x.shape[start_axis + n_axes :])
        return jnp.reshape(x, output_shape)

    return join_dims


@jax_funcify.register(SplitDims)
def jax_funcify_SplitDims(op, node, **kwargs):
    axis = op.axis

    def split_dims(x, shape):
        output_shape = (*x.shape[:axis], *shape, *x.shape[axis + 1 :])
        return jnp.reshape(x, output_shape)

    return split_dims
