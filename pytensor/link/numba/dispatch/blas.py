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
            # `Z` is only broadcast against the product, so the accumulator gemm writes into takes
            # the product's shape rather than `Z`'s. Copying also leaves `Z` intact, which is the
            # whole difference between this op and its inplace form.
            out = np.empty((X.shape[0], Y.shape[1]), dtype=dtype)
            out[:] = Z
            return _gemm(X, Y, out, False, False, alpha.item(), beta.item())

    cache_version = 2
    return gemm, cache_version


@register_funcify_default_op_cache_key(Gemv)
def numba_funcify_Gemv(op, node, **kwargs):
    """Dispatch ``Gemv`` to a single BLAS call, with its scalars carried as gemm's own
    alpha and beta."""
    # The vectors reach `_gemm` as one-column matrices rather than through a separate
    # gemv binding: BLAS reads the same buffers either way, and gemm already resolves
    # each operand's memory order without copying.
    if op.inplace:

        @numba_basic.numba_njit
        def gemv(y, alpha, A, x, beta):
            _gemm(
                A,
                np.expand_dims(x, 1),
                np.expand_dims(y, 1),
                False,
                False,
                alpha.item(),
                beta.item(),
            )
            return y

    else:

        @numba_basic.numba_njit
        def gemv(y, alpha, A, x, beta):
            # Accumulating into a copy leaves `y` intact, which is the whole difference
            # between this op and its inplace form.
            out = y.copy()
            _gemm(
                A,
                np.expand_dims(x, 1),
                np.expand_dims(out, 1),
                False,
                False,
                alpha.item(),
                beta.item(),
            )
            return out

    cache_version = 1
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
            # `A` is only broadcast against the outer product, so the buffer the update
            # writes into takes the product's shape rather than `A`'s. Copying also leaves
            # `A` intact, which is the whole difference between this op and its inplace form.
            out = np.empty((x.shape[0], y.shape[0]), dtype=dtype)
            out[:] = A
            return _ger(alpha.item(), x, y, out)

    # Bump whenever `_ger` changes: it is inlined here, so its source is not part of
    # this key.
    cache_version = 1
    return ger, cache_version
