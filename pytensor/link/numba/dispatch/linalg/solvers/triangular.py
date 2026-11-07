import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import ensure_blas
from scipy import linalg

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.linalg._BLAS import _BLAS
from pytensor.link.numba.dispatch.linalg._LAPACK import val_to_int_ptr
from pytensor.link.numba.dispatch.linalg.solvers.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _c_work_array,
    _check_dtypes_match,
    _check_linalg_matrix,
    _fortran_work_array,
    _full_fortran_order,
)


def _solve_triangular(A, B, lower=False, unit_diagonal=False, overwrite_b=False):
    """
    Thin wrapper around scipy.linalg.solve_triangular.

    This function is overloaded instead of the original scipy function to avoid unexpected side-effects to users who
    import pytensor.
    """
    return linalg.solve_triangular(
        A,
        B,
        lower=lower,
        unit_diagonal=unit_diagonal,
        overwrite_b=overwrite_b,
        check_finite=False,
    )


@overload(_solve_triangular)
def solve_triangular_impl(A, B, lower, unit_diagonal, overwrite_b):
    ensure_blas()

    _check_linalg_matrix(
        A, ndim=2, dtype=(Float, Complex), func_name="solve_triangular"
    )
    _check_linalg_matrix(
        B, ndim=(1, 2), dtype=(Float, Complex), func_name="solve_triangular"
    )
    _check_dtypes_match((A, B), func_name="solve_triangular")
    dtype = A.dtype
    numba_trsm = _BLAS().numba_xtrsm(dtype)
    ALPHA = np.ones(1, dtype=np.dtype(dtype.name))
    B_is_1d = B.ndim == 1
    # A C-typed B is solved in its own buffer, so every return path keeps its type.
    # Other types go through a Fortran copy, so every return path is Fortran.
    solve_in_row_major_frame = B.ndim == 2 and B.layout == "C"

    def impl(A, B, lower, unit_diagonal, overwrite_b):
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        if B.size == 0:
            if solve_in_row_major_frame:
                return np.zeros_like(B)
            return _full_fortran_order(B.shape, 0, B.dtype)

        # trsm reports no INFO, so a zero pivot has to be caught before the call. A is
        # scanned as given, ahead of any copy: the diagonal does not depend on layout,
        # and a singular system should not pay to be reordered first.
        if not unit_diagonal and _has_zero_on_diagonal(A):
            if solve_in_row_major_frame:
                return np.full_like(B, np.nan)
            return _full_fortran_order(B.shape, np.nan, B.dtype)

        # A c_contiguous matrix reinterpreted as f_contiguous is A^T (plain transpose, no
        # conjugation). An upper/lower triangular A^T is lower/upper triangular.
        A_f_is_transposed = A.flags.c_contiguous
        if A_f_is_transposed:
            lower = not lower

        if A.flags.f_contiguous or A_f_is_transposed:
            A_f = A
        else:
            A_f = np.asfortranarray(A)

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        DIAG = val_to_int_ptr(ord("U") if unit_diagonal else ord("N"))
        LDA = val_to_int_ptr(_N)

        if solve_in_row_major_frame:
            # A c_contiguous B reinterpreted as f_contiguous is B^T, so A X = B can be
            # solved as X^T A^T = B^T, which is what trsm computes with side="R",
            # straight out of B's own buffer.
            B_work = _c_work_array(B, overwrite_b)
            _NRHS = np.int32(B_work.shape[-1])
            SIDE = val_to_int_ptr(ord("R"))
            TRANSA = val_to_int_ptr(ord("N") if A_f_is_transposed else ord("T"))
            M = val_to_int_ptr(_NRHS)
            N = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_NRHS)
        else:
            B_work = _fortran_work_array(B, overwrite_b)
            _NRHS = np.int32(1 if B_is_1d else B_work.shape[-1])
            SIDE = val_to_int_ptr(ord("L"))
            TRANSA = val_to_int_ptr(ord("T") if A_f_is_transposed else ord("N"))
            M = val_to_int_ptr(_N)
            N = val_to_int_ptr(_NRHS)
            LDB = val_to_int_ptr(_N)

        numba_trsm(
            SIDE,
            UPLO,
            TRANSA,
            DIAG,
            M,
            N,
            ALPHA.ctypes,
            A_f.ctypes,
            LDA,
            B_work.ctypes,
            LDB,
        )

        return B_work

    return impl


@numba_basic.numba_njit(inline="always")
def _has_zero_on_diagonal(A):
    for i in range(A.shape[0]):
        if A[i, i] == 0:
            return True
    return False
