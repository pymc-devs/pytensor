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
    _trans_char_to_int,
)


def _xgecon(A: np.ndarray, A_norm: float, norm: str) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a matrix; used by linalg.solve. Not used by pytensor to numbify
    graphs.
    """
    return  # type: ignore


@overload(_xgecon)
def xgecon_impl(
    A: np.ndarray, A_norm: float, norm: str
) -> Callable[[np.ndarray, float, str], tuple[np.ndarray, int]]:
    """
    Compute the condition number of a matrix A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "gecon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_gecon = _LAPACK().numba_xgecon(dtype)

    def impl(A: np.ndarray, A_norm: float, norm: str) -> tuple[np.ndarray, int]:
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        A_NORM = np.array(A_norm, dtype=dtype)
        NORM = val_to_int_ptr(ord(norm))
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(4 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(1)

        numba_gecon(
            NORM,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            A_NORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _getrf(A, overwrite_a=False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Placeholder for LU factorization; used by linalg.solve.

    # TODO: Implement an LU_factor Op, then dispatch to this function in numba mode.
    """
    return  # type: ignore


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


def _getrs(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
) -> tuple[np.ndarray, int]:
    """
    Placeholder for solving a linear system with a matrix that has been LU-factored; used by linalg.solve.

    # TODO: Implement an LU_solve Op, then dispatch to this function in numba mode.
    """
    return  # type: ignore


@overload(_getrs)
def getrs_impl(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int, bool], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(LU, "getrs")
    _check_scipy_linalg_matrix(B, "getrs")
    dtype = LU.dtype
    w_type = _get_underlying_float(dtype)
    numba_getrs = _LAPACK().numba_xgetrs(dtype)

    def impl(
        LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
    ) -> tuple[np.ndarray, int]:
        _N = np.int32(LU.shape[-1])
        _solve_check_input_shapes(LU, B)

        B_is_1d = B.ndim == 1

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order_even_if_1d(B)

        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        NRHS = 1 if B_is_1d else int(B_copy.shape[-1])

        TRANS = val_to_int_ptr(_trans_char_to_int(trans))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        IPIV = _copy_to_fortran_order(IPIV)
        INFO = val_to_int_ptr(0)

        numba_getrs(
            TRANS,
            N,
            NRHS,
            LU.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            B_copy = B_copy[..., 0]

        return B_copy, int_ptr_to_val(INFO)

    return impl


def _solve_gen(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve. Used as an overload target for numba to avoid unexpected side-effects
    for users who import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a="gen",
        transposed=transposed,
    )


@overload(_solve_gen)
def solve_gen_impl(
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
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        if overwrite_a and A.flags.c_contiguous:
            # Work with the transposed system to avoid copying A
            A = A.T
            transposed = not transposed

        order = "I" if transposed else "1"
        norm = _xlange(A, order=order)

        N = A.shape[1]
        LU, IPIV, INFO = _getrf(A, overwrite_a=overwrite_a)
        _solve_check(N, INFO)

        X, INFO = _getrs(
            LU=LU, B=B, IPIV=IPIV, trans=transposed, overwrite_b=overwrite_b
        )
        _solve_check(N, INFO)

        RCOND, INFO = _xgecon(LU, norm, "1")
        _solve_check(N, INFO, True, RCOND)

        return X

    return impl
