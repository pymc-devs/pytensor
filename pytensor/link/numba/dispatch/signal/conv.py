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
    match a_static_len, b_static_len:
        case int(), int():
            if b_static_len <= a_static_len:
                kernel_len, kernel_is_b = b_static_len, True
            else:
                kernel_len, kernel_is_b = a_static_len, False
        case None, int():
            kernel_len, kernel_is_b = b_static_len, True
        case int(), None:
            kernel_len, kernel_is_b = a_static_len, False
        case None, None:
            kernel_len, kernel_is_b = None, True

    use_static = kernel_len is not None

    if use_static:

        @numba_basic.numba_njit(boundscheck=False)
        def valid_convolve1d(x, y, out=None):
            if kernel_is_b:
                signal, kernel = x, y
            else:
                signal, kernel = y, x
            ns = signal.shape[0]
            if ns >= kernel_len:
                length = ns - kernel_len + 1
                if out is None:
                    out = np.empty(length, out_dtype)
                for i in range(length):
                    acc = 0.0
                    for k in range(kernel_len):
                        acc += signal[i + k] * kernel[kernel_len - 1 - k]
                    out[i] = acc
            else:
                signal, kernel = kernel, signal
                nk = ns
                ns = signal.shape[0]
                length = ns - nk + 1
                if out is None:
                    out = np.empty(length, out_dtype)
                for i in range(length):
                    acc = 0.0
                    for k in range(nk):
                        acc += signal[i + k] * kernel[nk - 1 - k]
                    out[i] = acc
            return out

    else:

        @numba_basic.numba_njit
        def valid_convolve1d(x, y, out=None):
            nx = len(x)
            ny = len(y)
            if nx < ny:
                x, y = y, x
                nx, ny = ny, nx
            length = nx - ny + 1
            if out is None:
                out = np.empty(length, out_dtype)
            y_flipped = y[::-1]
            for i in range(length):
                out[i] = innerprod(x[i : i + ny], y_flipped)
            return out

    @numba_basic.numba_njit
    def full_convolve1d(x, y, out=None):
        nx = len(x)
        ny = len(y)
        if nx < ny:
            x, y = y, x
            nx, ny = ny, nx

        if out is None:
            length = nx + ny - 1
            out = np.empty(length, out_dtype)

        y_flipped = y[::-1]
        idx = 0
        for i in range(ny - 1):
            k = i + 1
            out[idx] = innerprod(x[:k], y_flipped[-k:])
            idx = idx + 1
        for i in range(nx - ny + 1):
            out[idx] = innerprod(x[i : i + ny], y_flipped)
            idx = idx + 1
        for i in range(ny - 1):
            k = ny - i - 1
            out[idx] = innerprod(x[-k:], y_flipped[:k])
            idx = idx + 1
        return out

    @numba_basic.numba_njit
    def convolve_1d(x, y, mode, out=None):
        if mode:
            return full_convolve1d(x, y, out=out)
        else:
            return valid_convolve1d(x, y, out=out)

    convolve_1d.handles_out = True

    if use_static:
        extra_kwargs = dict(kernel_len=kernel_len, kernel_is_b=kernel_is_b)
    else:
        extra_kwargs = {}
    cache_key = default_hash_key_from_props(op, **extra_kwargs, cache_version=2)
    return convolve_1d, cache_key
