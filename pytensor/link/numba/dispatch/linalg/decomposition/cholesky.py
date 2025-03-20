import numpy as np
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_scipy_linalg_matrix


def _cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    return (
        linalg.cholesky(
            a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite
        ),
        0,
    )


@overload(_cholesky)
def cholesky_impl(A, lower=0, overwrite_a=False, check_finite=True):
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "cholesky")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_potrf = _LAPACK().numba_xpotrf(dtype)

    def impl(A, lower=0, overwrite_a=False, check_finite=True):
        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        numba_potrf(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            INFO,
        )

        if lower:
            for j in range(1, _N):
                for i in range(j):
                    A_copy[i, j] = 0.0
        else:
            for j in range(_N):
                for i in range(j + 1, _N):
                    A_copy[i, j] = 0.0

        return A_copy, int_ptr_to_val(INFO)

    return impl
