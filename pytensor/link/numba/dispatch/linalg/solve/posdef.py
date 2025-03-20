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


def _posv(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Placeholder for solving a linear system with a positive-definite matrix; used by linalg.solve.
    """
    return  # type: ignore


@overload(_posv)
def posv_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[
    [np.ndarray, np.ndarray, bool, bool, bool, bool, bool],
    tuple[np.ndarray, np.ndarray, int],
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_posv = _LAPACK().numba_xposv(dtype)

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        _solve_check_input_shapes(A, B)

        _N = np.int32(A.shape[-1])

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

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        NRHS = 1 if B_is_1d else int(B.shape[-1])

        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_posv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            B_copy = B_copy[..., 0]

        return A_copy, B_copy, int_ptr_to_val(INFO)

    return impl


def _pocon(A: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a cholesky-factorized positive-definite matrix. Used by
    linalg.solve when assume_a = "pos".
    """
    return  # type: ignore


@overload(_pocon)
def pocon_impl(
    A: np.ndarray, anorm: float
) -> Callable[[np.ndarray, float], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "pocon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_pocon = _LAPACK().numba_xpocon(dtype)

    def impl(A: np.ndarray, anorm: float):
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        UPLO = val_to_int_ptr(ord("L"))
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        ANORM = np.array(anorm, dtype=dtype)
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(3 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_pocon(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            ANORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _solve_psd(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve for positive-definite matrices. Used as an overload target for numba to
    avoid unexpected side-effects when users import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        transposed=transposed,
        assume_a="pos",
    )


@overload(_solve_psd)
def solve_psd_impl(
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

        C, x, info = _posv(
            A, B, lower, overwrite_a, overwrite_b, check_finite, transposed
        )
        _solve_check(A.shape[-1], info)

        rcond, info = _pocon(C, _xlange(A))
        _solve_check(A.shape[-1], info=info, lamch=True, rcond=rcond)

        return x

    return impl
