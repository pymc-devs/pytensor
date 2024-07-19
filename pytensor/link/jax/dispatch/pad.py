import jax.numpy as jnp
import numpy as np

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.pad import Pad


@jax_funcify.register(Pad)
def jax_funcify_pad(op, **kwargs):
    pad_mode = op.pad_mode
    reflect_type = op.reflect_type
    has_stat_length = op.has_stat_length

    if pad_mode == "constant":

        def constant_pad(x, pad_width, constant_values):
            return jnp.pad(x, pad_width, mode=pad_mode, constant_values=constant_values)

        return constant_pad

    elif pad_mode == "linear_ramp":

        def lr_pad(x, pad_width, end_values):
            # JAX does not allow a dynamic input if end_values is non-scalar
            if not isinstance(end_values, int | float):
                end_values = tuple(np.array(end_values))
            return jnp.pad(x, pad_width, mode=pad_mode, end_values=end_values)

        return lr_pad

    elif pad_mode in ["maximum", "minimum", "mean"] and has_stat_length:

        def stat_pad(x, pad_width, stat_length):
            # JAX does not allow a dynamic input here, need to cast to tuple
            return jnp.pad(
                x, pad_width, mode=pad_mode, stat_length=tuple(np.array(stat_length))
            )

        return stat_pad

    elif pad_mode in ["reflect", "symmetric"]:

        def loop_pad(x, pad_width):
            return jnp.pad(x, pad_width, mode=pad_mode, reflect_type=reflect_type)

        return loop_pad

    else:

        def pad(x, pad_width):
            return jnp.pad(x, pad_width, mode=pad_mode)

        return pad
