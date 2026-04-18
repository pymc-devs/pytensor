from typing import Any

import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy.linalg import schur

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


def schur_real(
    A: np.ndarray,
    lwork: Any | None = None,
    overwrite_a: Any = False,
):
    return schur(
        a=A,
        output="real",
        lwork=lwork,
        overwrite_a=overwrite_a,
        sort=None,
        check_finite=False,
    )


def schur_complex(
    A: np.ndarray,
    lwork: Any | None = None,
    overwrite_a: Any = False,
):
    return schur(
        a=A,
        output="complex",
        lwork=lwork,
        overwrite_a=overwrite_a,
        sort=None,
        check_finite=False,
    )


@overload(schur_real)
def schur_real_impl(A, lwork, overwrite_a):
    """Overload for real Schur decomposition."""
    ensure_lapack()

    _check_linalg_matrix(A, ndim=2, dtype=(Float,), func_name="schur")

    dtype = A.dtype
    numba_xgees = _LAPACK().numba_xgees(dtype)

    def real_schur_impl(A, lwork, overwrite_a):
        _N = np.int32(A.shape[-1])
        if lwork is None:
            lwork = -1

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)

        JOBVS = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        WR = np.empty(_N, dtype=dtype)
        WI = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        if lwork == -1:
            numba_xgees(
                JOBVS,
                SORT,
                SELECT,
                N,
                A_copy.ctypes,
                LDA,
                SDIM,
                WR.ctypes,
                WI.ctypes,
                VS.ctypes,
                LDVS,
                WORK.ctypes,
                LWORK,
                BWORK,
                INFO,
            )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        numba_xgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            SDIM,
            WR.ctypes,
            WI.ctypes,
            VS.ctypes,
            LDVS,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan

        return A_copy, VS.T

    return real_schur_impl


@overload(schur_complex)
def schur_complex_impl(A, lwork, overwrite_a):
    """Overload for complex Schur decomposition."""
    ensure_lapack()

    _check_linalg_matrix(A, ndim=2, dtype=(Complex,), func_name="schur")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_xgees = _LAPACK().numba_xgees(dtype)

    def complex_schur_impl(A, lwork, overwrite_a):
        _N = np.int32(A.shape[-1])
        if lwork is None:
            lwork = -1

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)

        JOBVS = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        W = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        RWORK = np.empty(_N, dtype=w_type)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        if lwork == -1:
            numba_xgees(
                JOBVS,
                SORT,
                SELECT,
                N,
                A_copy.ctypes,
                LDA,
                SDIM,
                W.ctypes,
                VS.ctypes,
                LDVS,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                BWORK,
                INFO,
            )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        numba_xgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            SDIM,
            W.ctypes,
            VS.ctypes,
            LDVS,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan

        return A_copy, VS.T

    return complex_schur_impl
