import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.pad import Pad, allowed_kwargs


@jax_funcify([Pad])
def jax_funcify_pad(op, **kwargs):
    pad_mode = op.pad_mode
    expected_kwargs = allowed_kwargs[pad_mode]
    mode_kwargs = {kwarg: getattr(op, kwarg) for kwarg in expected_kwargs}

    def pad(x, pad_width, pad_mode=pad_mode):
        return jnp.pad(x, pad_width=pad_width, pad_mode=pad_mode, **mode_kwargs)

    return pad
