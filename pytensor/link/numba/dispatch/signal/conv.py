import numpy as np
from numba.np.arraymath import _get_inner_prod

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_default_op_cache_key,
)
from pytensor.tensor.signal.conv import Convolve1d


@register_funcify_default_op_cache_key(Convolve1d)
def numba_funcify_Convolve1d(op, node, **kwargs):
    # This specialized version is faster than the overloaded numba np.convolve
    a_dtype, b_dtype = node.inputs[0].type.dtype, node.inputs[1].type.dtype
    out_dtype = node.outputs[0].type.dtype
    innerprod = _get_inner_prod(a_dtype, b_dtype)

    @numba_basic.numba_njit
    def valid_convolve1d(x, y):
        nx = len(x)
        ny = len(y)
        if nx < ny:
            x, y = y, x
            nx, ny = ny, nx
        y_flipped = y[::-1]

        length = nx - ny + 1
        ret = np.empty(length, out_dtype)

        for i in range(length):
            ret[i] = innerprod(x[i : i + ny], y_flipped)

        return ret

    @numba_basic.numba_njit
    def full_convolve1d(x, y):
        nx = len(x)
        ny = len(y)
        if nx < ny:
            x, y = y, x
            nx, ny = ny, nx
        y_flipped = y[::-1]

        length = nx + ny - 1
        ret = np.empty(length, out_dtype)
        idx = 0

        for i in range(ny - 1):
            k = i + 1
            ret[idx] = innerprod(x[:k], y_flipped[-k:])
            idx = idx + 1

        for i in range(nx - ny + 1):
            ret[idx] = innerprod(x[i : i + ny], y_flipped)
            idx = idx + 1

        for i in range(ny - 1):
            k = ny - i - 1
            ret[idx] = innerprod(x[-k:], y_flipped[:k])
            idx = idx + 1

        return ret

    @numba_basic.numba_njit
    def convolve_1d(x, y, mode):
        if mode:
            return full_convolve1d(x, y)
        else:
            return valid_convolve1d(x, y)

    return convolve_1d
