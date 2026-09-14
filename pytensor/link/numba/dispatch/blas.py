import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg.products import _gemm, _ger
from pytensor.tensor.blas import Gemm, Gemv, Ger


@register_funcify_default_op_cache_key(Gemm)
def numba_funcify_Gemm(op, node, **kwargs):
    """Dispatch ``Gemm`` to one BLAS call, with its scalars carried as gemm's own alpha and beta."""
    dtype = node.outputs[0].type.numpy_dtype

    if op.inplace:

        @numba_basic.numba_njit
        def gemm(Z, alpha, X, Y, beta):
            return _gemm(X, Y, Z, False, False, alpha.item(), beta.item())

    else:

        @numba_basic.numba_njit
        def gemm(Z, alpha, X, Y, beta):
            # One pass adds beta * Z, which may broadcast, without a temporary.
            out = np.empty((X.shape[0], Y.shape[1]), dtype=dtype)
            _gemm(X, Y, out, False, False, alpha.item(), 0.0)
            beta_value = beta.item()
            if beta_value != 0.0:
                Z_full = np.broadcast_to(Z, out.shape)
                for i in range(out.shape[0]):
                    for j in range(out.shape[1]):
                        out[i, j] += beta_value * Z_full[i, j]
            return out

    cache_version = 5
    return gemm, cache_version


@numba_basic.numba_njit(inline="always")
def _gemv_into(y, alpha, A, x, beta):
    # A reversed axis of A is read forwards by reversing the vector that runs along
    # it, which turns a layout gemm would copy into one it addresses in place.
    if A.strides[0] < 0:
        A = A[::-1]
        y = y[::-1]
    if A.strides[1] < 0:
        A = A[:, ::-1]
        x = x[::-1]
    _gemm(A, np.expand_dims(x, 1), np.expand_dims(y, 1), False, False, alpha, beta)


@register_funcify_default_op_cache_key(Gemv)
def numba_funcify_Gemv(op, node, **kwargs):
    """Dispatch ``Gemv`` through gemm on one-column matrices, which reads the same
    buffers and already resolves each operand's layout."""
    if op.inplace:

        @numba_basic.numba_njit
        def gemv(y, alpha, A, x, beta):
            _gemv_into(y, alpha.item(), A, x, beta.item())
            return y

    else:

        @numba_basic.numba_njit
        def gemv(y, alpha, A, x, beta):
            out = y.copy()
            _gemv_into(out, alpha.item(), A, x, beta.item())
            return out

    cache_version = 2
    return gemv, cache_version


@register_funcify_default_op_cache_key(Ger)
def numba_funcify_Ger(op, node, **kwargs):
    """Dispatch ``Ger`` to one BLAS rank-1 update."""
    dtype = node.outputs[0].type.numpy_dtype

    if op.inplace:

        @numba_basic.numba_njit
        def ger(A, alpha, x, y):
            return _ger(alpha.item(), x, y, A)

    else:

        @numba_basic.numba_njit
        def ger(A, alpha, x, y):
            # One pass writes A and the update together instead of copying A in and
            # letting BLAS accumulate on top.
            rows = x.shape[0]
            cols = y.shape[0]
            out = np.empty((rows, cols), dtype=dtype)
            a = alpha.item()
            for i in range(rows):
                scaled = a * x[i]
                for j in range(cols):
                    out[i, j] = A[i, j] + scaled * y[j]
            return out

    cache_version = 4
    return ger, cache_version
