import numpy as np
from numba.np.arraymath import _get_inner_prod

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    default_hash_key_from_props,
    register_funcify_and_cache_key,
)
from pytensor.tensor.signal.conv import Convolve1d


@register_funcify_and_cache_key(Convolve1d)
def numba_funcify_Convolve1d(op, node, **kwargs):
    a_dtype, b_dtype = node.inputs[0].type.dtype, node.inputs[1].type.dtype
    out_dtype = node.outputs[0].type.dtype
    innerprod = _get_inner_prod(a_dtype, b_dtype)

    a_static_len = node.inputs[0].type.shape[-1]
    b_static_len = node.inputs[1].type.shape[-1]
    a_known = isinstance(a_static_len, int)
    b_known = isinstance(b_static_len, int)

    kernel_len = None
    kernel_is_b = True
    if a_known and b_known:
        if b_static_len <= a_static_len:
            kernel_len, kernel_is_b = b_static_len, True
        else:
            kernel_len, kernel_is_b = a_static_len, False
    elif b_known:
        kernel_len, kernel_is_b = b_static_len, True
    elif a_known:
        kernel_len, kernel_is_b = a_static_len, False

    use_static = kernel_len is not None

    if use_static:

        @numba_basic.numba_njit(boundscheck=False)
        def valid_convolve1d(x, y):
            if kernel_is_b:
                signal, kernel = x, y
            else:
                signal, kernel = y, x
            ns = signal.shape[0]
            if ns >= kernel_len:
                length = ns - kernel_len + 1
                ret = np.empty(length, out_dtype)
                for i in range(length):
                    acc = 0.0
                    for k in range(kernel_len):
                        acc += signal[i + k] * kernel[kernel_len - 1 - k]
                    ret[i] = acc
            else:
                signal, kernel = kernel, signal
                nk = ns
                ns = signal.shape[0]
                length = ns - nk + 1
                ret = np.empty(length, out_dtype)
                for i in range(length):
                    acc = 0.0
                    for k in range(nk):
                        acc += signal[i + k] * kernel[nk - 1 - k]
                    ret[i] = acc
            return ret

    else:

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

    if use_static:
        extra_kwargs = dict(kernel_len=kernel_len, kernel_is_b=kernel_is_b)
    else:
        extra_kwargs = {}
    cache_key = default_hash_key_from_props(op, **extra_kwargs, cache_version=1)
    return convolve_1d, cache_key
