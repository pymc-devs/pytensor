import jax

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.signal.conv import Convolve1d


@jax_funcify.register(Convolve1d)
def jax_funcify_Convolve1d(op, node, **kwargs):
    mode = op.mode

    def conv1d(data, kernel):
        return jax.numpy.convolve(data, kernel, mode=mode)

    return conv1d
