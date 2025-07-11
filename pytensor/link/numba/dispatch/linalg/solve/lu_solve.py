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
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_scipy_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _solve_check,
    _trans_char_to_int,
)


def _getrs(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
) -> tuple[np.ndarray, int]:
    """
    Placeholder for solving a linear system with a matrix that has been LU-factored. Used by linalg.lu_solve.
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


def _lu_solve(
    lu_and_piv: tuple[np.ndarray, np.ndarray],
    b: np.ndarray,
    trans: int,
    overwrite_b: bool,
    check_finite: bool,
):
    """
    Thin wrapper around scipy.lu_solve, used to avoid side effects from numba overloads on users who import Pytensor.
    """
    return linalg.lu_solve(
        lu_and_piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite
    )


@overload(_lu_solve)
def lu_solve_impl(
    lu_and_piv: tuple[np.ndarray, np.ndarray],
    b: np.ndarray,
    trans: int,
    overwrite_b: bool,
    check_finite: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(lu_and_piv[0], "lu_solve")
    _check_scipy_linalg_matrix(b, "lu_solve")

    def impl(
        lu: np.ndarray,
        piv: np.ndarray,
        b: np.ndarray,
        trans: int,
        overwrite_b: bool,
        check_finite: bool,
    ) -> np.ndarray:
        n = np.int32(lu.shape[0])

        X, INFO = _getrs(LU=lu, B=b, IPIV=piv, trans=trans, overwrite_b=overwrite_b)

        _solve_check(n, INFO)

        return X

    return impl
