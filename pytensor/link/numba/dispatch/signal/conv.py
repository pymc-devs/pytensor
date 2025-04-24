import numpy as np

from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.tensor.signal.conv import Convolve1d


@numba_funcify.register(Convolve1d)
def numba_funcify_Convolve1d(op, node, **kwargs):
    mode = op.mode

    @numba_njit
    def conv1d(data, kernel):
        return np.convolve(data, kernel, mode=mode)

    return conv1d
