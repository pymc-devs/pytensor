import jax.numpy as jnp
import numpy as np

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.math import Argmax, Dot, Max


@jax_funcify.register(Dot)
def jax_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return jnp.dot(x, y)

    return dot


@jax_funcify.register(Max)
def jax_funcify_Max(op, **kwargs):
    axis = op.axis

    def max(x):
        max_res = jnp.max(x, axis)

        return max_res

    return max


@jax_funcify.register(Argmax)
def jax_funcify_Argmax(op, **kwargs):
    axis = op.axis

    def argmax(x):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes], dtype="int64")
        # Not-reduced axes in front
        transposed_x = jnp.transpose(
            x, tuple(np.concatenate((keep_axes, np.array(axes, dtype="int64"))))
        )
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = (
            *kept_shape,
            np.prod(np.array(reduced_shape, dtype="int64"), dtype="int64"),
        )
        reshaped_x = transposed_x.reshape(tuple(new_shape))

        max_idx_res = jnp.argmax(reshaped_x, axis=-1).astype("int64")

        return max_idx_res

    return argmax
