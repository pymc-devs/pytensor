import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import (
    _c_work_array,
    _check_linalg_matrix,
    _fortran_work_array,
    _full_fortran_order,
)


def _cholesky(a, lower=False, overwrite_a=False):
    return linalg.cholesky(a, lower=lower, overwrite_a=overwrite_a, check_finite=False)


@overload(_cholesky)
def cholesky_impl(A, lower=0, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="cholesky")
    dtype = A.dtype

    numba_potrf = _LAPACK().numba_xpotrf(dtype)
    # A c_contiguous A reinterpreted as f_contiguous is A^T. potrf(A^T, UPLO='U')
    # produces U where U.T == L (the correct lower factor), even for complex Hermitian
    # matrices, and the .T return corrects the result. Choosing this from the type
    # keeps every return path one layout.
    transposed = A.layout == "C"

    def impl(A, lower=False, overwrite_a=False):
        if A.size == 0:
            if transposed:
                return np.zeros(A.shape, dtype=A.dtype)
            return _full_fortran_order(A.shape, 0, A.dtype)

        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")

        if transposed:
            A_copy = _c_work_array(A, overwrite_a).T
            lower = not lower
        else:
            A_copy = _fortran_work_array(A, overwrite_a)

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
        elif lower:
            for j in range(1, _N):
                for i in range(j):
                    A_copy[i, j] = 0.0
        else:
            for j in range(_N):
                for i in range(j + 1, _N):
                    A_copy[i, j] = 0.0

        if transposed:
            return A_copy.T
        return A_copy

    return impl
