import ctypes

from numba.core.extending import get_cython_function_address
from numba.np.linalg import ensure_blas, ensure_lapack, get_blas_kind

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _get_float_pointer_for_dtype,
    _ptr_int,
)


def _get_blas_ptr_and_ptr_type(dtype, name):
    d = get_blas_kind(dtype)
    func_name = f"{d}{name}"
    float_pointer = _get_float_pointer_for_dtype(d)
    lapack_ptr = get_cython_function_address("scipy.linalg.cython_blas", func_name)

    return lapack_ptr, float_pointer


class _BLAS:
    """
    Functions to return type signatures for wrapped BLAS functions.

    Here we are specifically concered with BLAS functions exposed by scipy, and not used by numpy.

    Patterned after https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L74
    """

    def __init__(self):
        ensure_lapack()
        ensure_blas()

    @classmethod
    def numba_xgemv(cls, dtype):
        """
        xGEMV performs one of the following matrix operations:

            y = alpha * A @ x + beta * y,   or   y = alpha * A.T @ x + beta * y

        Where alpha and beta are scalars, x and y are vectors, and A is a general matrix.
        """

        blas_ptr, float_pointer = _get_blas_ptr_and_ptr_type(dtype, "gemv")

        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # TRANS
            _ptr_int,  # M
            _ptr_int,  # N
            float_pointer,  # ALPHA
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # X
            _ptr_int,  # INCX
            float_pointer,  # BETA
            float_pointer,  # Y
            _ptr_int,  # INCY
        )

        return functype(blas_ptr)

    @classmethod
    def numba_xgbmv(cls, dtype):
        """
        xGBMV performs one of the following matrix operations:

            y = alpha * A @ x + beta * y,   or   y = alpha * A.T @ x + beta * y

        Where alpha and beta are scalars, x and y are vectors, and A is a band matrix with kl sub-diagonals and ku
        super-diagonals.
        """

        blas_ptr, float_pointer = _get_blas_ptr_and_ptr_type(dtype, "gbmv")

        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # TRANS
            _ptr_int,  # M
            _ptr_int,  # N
            _ptr_int,  # KL
            _ptr_int,  # KU
            float_pointer,  # ALPHA
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # X
            _ptr_int,  # INCX
            float_pointer,  # BETA
            float_pointer,  # Y
            _ptr_int,  # INCY
        )

        return functype(blas_ptr)
