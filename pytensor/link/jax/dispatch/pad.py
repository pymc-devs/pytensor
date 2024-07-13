import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.pad import Pad


fixed_kwargs = {"reflect": ["reflect_type"], "symmetric": ["reflect_type"]}


@jax_funcify.register(Pad)
def jax_funcify_pad(op, **kwargs):
    pad_mode = op.pad_mode
    expected_kwargs = fixed_kwargs.get(pad_mode, {})
    mode_kwargs = {kwarg: getattr(op, kwarg) for kwarg in expected_kwargs}

    def pad(x, pad_width, *args):
        print(args)
        return jnp.pad(x, pad_width, mode=pad_mode, **mode_kwargs)

    return pad
