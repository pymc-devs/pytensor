from collections.abc import Callable

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
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_scipy_linalg_matrix,
)


def _getrf(A, overwrite_a=False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Underlying LAPACK function used for LU factorization. Compared to scipy.linalg.lu_factorize, this function also
    returns an info code with diagnostic information.
    """
    (getrf,) = linalg.get_lapack_funcs("getrf", (A,))
    A_copy, ipiv, info = getrf(A, overwrite_a=overwrite_a)

    return A_copy, ipiv, info


@overload(_getrf)
def getrf_impl(
    A: np.ndarray, overwrite_a: bool = False
) -> Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "getrf")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_getrf = _LAPACK().numba_xgetrf(dtype)

    def impl(
        A: np.ndarray, overwrite_a: bool = False
    ) -> tuple[np.ndarray, np.ndarray, int]:
        _M, _N = np.int32(A.shape[-2:])  # type: ignore

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        M = val_to_int_ptr(_M)  # type: ignore
        N = val_to_int_ptr(_N)  # type: ignore
        LDA = val_to_int_ptr(_M)  # type: ignore
        IPIV = np.empty(_N, dtype=np.int32)  # type: ignore
        INFO = val_to_int_ptr(0)

        numba_getrf(M, N, A_copy.view(w_type).ctypes, LDA, IPIV.ctypes, INFO)

        return A_copy, IPIV, int_ptr_to_val(INFO)

    return impl


def _lu_factor(A: np.ndarray, overwrite_a: bool = False):
    """
    Thin wrapper around scipy.linalg.lu_factor. Used as an overload target to avoid side-effects on users who import
    Pytensor.
    """
    return linalg.lu_factor(A, overwrite_a=overwrite_a)


@overload(_lu_factor)
def lu_factor_impl(
    A: np.ndarray, overwrite_a: bool = False
) -> Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "lu_factor")

    def impl(A: np.ndarray, overwrite_a: bool = False) -> tuple[np.ndarray, np.ndarray]:
        A_copy, IPIV, INFO = _getrf(A, overwrite_a=overwrite_a)
        IPIV -= 1  # LAPACK uses 1-based indexing, convert to 0-based

        if INFO != 0:
            raise np.linalg.LinAlgError("LU decomposition failed")
        return A_copy, IPIV

    return impl
