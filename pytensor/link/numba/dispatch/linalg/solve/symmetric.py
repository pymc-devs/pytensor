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
from pytensor.link.numba.dispatch.linalg.solve.norm import _xlange
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_scipy_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _solve_check,
)


def _sysv(
    A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Placeholder for solving a linear system with a symmetric matrix; used by linalg.solve.
    """
    return  # type: ignore


@overload(_sysv)
def sysv_impl(
    A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
) -> Callable[
    [np.ndarray, np.ndarray, bool, bool, bool],
    tuple[np.ndarray, np.ndarray, np.ndarray, int],
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "sysv")
    _check_scipy_linalg_matrix(B, "sysv")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_sysv = _LAPACK().numba_xsysv(dtype)

    def impl(
        A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
    ):
        _LDA, _N = np.int32(A.shape[-2:])  # type: ignore
        _solve_check_input_shapes(A, B)

        if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
            A_copy = A
            if A.flags.c_contiguous:
                # An upper/lower triangular c_contiguous is the same as a lower/upper triangular f_contiguous
                lower = not lower
        else:
            A_copy = _copy_to_fortran_order(A)

        B_is_1d = B.ndim == 1

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order_even_if_1d(B)

        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        NRHS = 1 if B_is_1d else int(B.shape[-1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)  # type: ignore
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_LDA)  # type: ignore
        IPIV = np.empty(_N, dtype=np.int32)  # type: ignore
        LDB = val_to_int_ptr(_N)  # type: ignore
        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_sysv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual solve
        numba_sysv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )

        if B_is_1d:
            B_copy = B_copy[..., 0]
        return A_copy, B_copy, IPIV, int_ptr_to_val(INFO)

    return impl


def _sycon(A: np.ndarray, ipiv: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a symmetric matrix; used by linalg.solve. Never called in
    python mode.
    """
    return  # type: ignore


@overload(_sycon)
def sycon_impl(
    A: np.ndarray, ipiv: np.ndarray, anorm: float
) -> Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "sycon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_sycon = _LAPACK().numba_xsycon(dtype)

    def impl(A: np.ndarray, ipiv: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        UPLO = val_to_int_ptr(ord("U"))
        ANORM = np.array(anorm, dtype=dtype)
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(2 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_sycon(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            ipiv.ctypes,
            ANORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _solve_symmetric(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve for symmetric matrices. Used as an overload target for numba to avoid
    unexpected side-effects when users import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a="sym",
        transposed=transposed,
    )


@overload(_solve_symmetric)
def solve_symmetric_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> np.ndarray:
        _solve_check_input_shapes(A, B)

        lu, x, ipiv, info = _sysv(A, B, lower, overwrite_a, overwrite_b)
        _solve_check(A.shape[-1], info)

        rcond, info = _sycon(lu, ipiv, _xlange(A, order="I"))
        _solve_check(A.shape[-1], info, True, rcond)

        return x

    return impl
