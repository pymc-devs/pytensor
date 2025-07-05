import jax

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.basic import get_underlying_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.signal.conv import Convolve1d


@jax_funcify.register(Convolve1d)
def jax_funcify_Convolve1d(op, node, **kwargs):
    _, _, is_full_mode = node.inputs
    try:
        is_full_mode = get_underlying_scalar_constant_value(is_full_mode)
    except NotScalarConstantError:
        raise NotImplementedError(
            "Cannot compile Convolve1D to jax without static mode"
        )
    static_mode = "full" if is_full_mode else "valid"

    def conv1d(data, kernel, _):
        return jax.numpy.convolve(data, kernel, mode=static_mode)

    return conv1d
