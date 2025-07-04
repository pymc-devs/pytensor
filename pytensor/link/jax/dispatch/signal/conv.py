import jax

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.basic import get_underlying_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.signal.conv import Convolve1d


@jax_funcify.register(Convolve1d)
def jax_funcify_Convolve1d(op, node, **kwargs):
    _, _, full_mode = node.inputs
    try:
        full_mode = get_underlying_scalar_constant_value(full_mode)
    except NotScalarConstantError:
        raise NotImplementedError(
            "Cannot compile Convolve1D to jax without static mode"
        )
    static_mode = "full" if full_mode else "valid"

    def conv1d(data, kernel, _runtime_full_mode):
        # _runtime_full_mode is not used, as we only support static mode
        return jax.numpy.convolve(data, kernel, mode=static_mode)

    return conv1d
