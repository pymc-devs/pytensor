import numpy as np
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from numba.types import Float
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


def _cholesky(a, lower=False, overwrite_a=False):
    return linalg.cholesky(a, lower=lower, overwrite_a=overwrite_a, check_finite=False)


@overload(_cholesky)
def cholesky_impl(A, lower=0, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=Float, func_name="cholesky")
    dtype = A.dtype

    numba_potrf = _LAPACK().numba_xpotrf(dtype)

    def impl(A, lower=False, overwrite_a=False):
        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")

        transposed = False
        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        elif overwrite_a and A.flags.c_contiguous:
            # We can work on the transpose of A directly
            A_copy = A.T
            transposed = True
            lower = not lower
        else:
            A_copy = _copy_to_fortran_order(A)

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_potrf(
            UPLO,
            N,
            A_copy.ctypes,
            LDA,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy = np.full_like(A_copy, np.nan)
            return A_copy

        if lower:
            for j in range(1, _N):
                for i in range(j):
                    A_copy[i, j] = 0.0
        else:
            for j in range(_N):
                for i in range(j + 1, _N):
                    A_copy[i, j] = 0.0

        if transposed:
            return A_copy.T
        else:
            return A_copy

    return impl
