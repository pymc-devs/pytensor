import numpy as np
from numba.np.arraymath import _get_inner_prod

from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.tensor.signal.conv import Convolve1d


@numba_funcify.register(Convolve1d)
def numba_funcify_Convolve1d(op, node, **kwargs):
    # This specialized version is faster than the overloaded numba np.convolve
    mode = op.mode
    a_dtype, b_dtype = node.inputs[0].type.dtype, node.inputs[1].type.dtype
    out_dtype = node.outputs[0].type.dtype
    innerprod = _get_inner_prod(a_dtype, b_dtype)

    if mode == "valid":

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

        return numba_njit(valid_convolve1d)

    elif mode == "full":

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

        return numba_njit(full_convolve1d)

    else:
        raise ValueError(f"Unsupported mode: {mode}")
