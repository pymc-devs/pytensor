from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
from numba.core.extending import overload
from numba.core.types import Float, int32
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_dtypes_match,
    _check_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _trans_char_to_int,
)


_Trans: TypeAlias = Literal[0, 1, 2]


def _getrs(
    LU: np.ndarray,
    B: np.ndarray,
    IPIV: np.ndarray,
    trans: _Trans | bool,  # mypy does not realize that `bool <: Literal[0, 1]`
    overwrite_b: bool,
) -> tuple[np.ndarray, int]:
    """
    Placeholder for solving a linear system with a matrix that has been LU-factored. Used by linalg.lu_solve.
    """
    return  # type: ignore


@overload(_getrs)
def getrs_impl(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: _Trans, overwrite_b: bool
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, _Trans, bool], tuple[np.ndarray, int]
]:
    ensure_lapack()
    _check_linalg_matrix(LU, ndim=2, dtype=Float, func_name="getrs")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="getrs")
    _check_dtypes_match((LU, B), func_name="getrs")
    _check_linalg_matrix(IPIV, ndim=1, dtype=int32, func_name="getrs")
    dtype = LU.dtype
    numba_getrs = _LAPACK().numba_xgetrs(dtype)

    def impl(
        LU: np.ndarray,
        B: np.ndarray,
        IPIV: np.ndarray,
        trans: _Trans,
        overwrite_b: bool,
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
            LU.ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.ctypes,
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
    trans: _Trans,
    overwrite_b: bool,
):
    """
    Thin wrapper around scipy.lu_solve, used to avoid side effects from numba overloads on users who import Pytensor.
    """
    return linalg.lu_solve(lu_and_piv, b, trans=trans, overwrite_b=overwrite_b)


@overload(_lu_solve)
def lu_solve_impl(
    lu_and_piv: tuple[np.ndarray, np.ndarray],
    b: np.ndarray,
    trans: _Trans,
    overwrite_b: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, _Trans, bool], np.ndarray]:
    ensure_lapack()
    lu, _piv = lu_and_piv
    _check_linalg_matrix(lu, ndim=2, dtype=Float, func_name="lu_solve")
    _check_linalg_matrix(b, ndim=(1, 2), dtype=Float, func_name="lu_solve")
    _check_dtypes_match((lu, b), func_name="lu_solve")

    def impl(
        lu: np.ndarray,
        piv: np.ndarray,
        b: np.ndarray,
        trans: _Trans,
        overwrite_b: bool,
    ) -> np.ndarray:
        X, info = _getrs(LU=lu, B=b, IPIV=piv, trans=trans, overwrite_b=overwrite_b)

        if info != 0:
            X = np.full_like(X, np.nan)

        return X

    return impl
