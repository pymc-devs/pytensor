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
    _check_dtypes_match,
    _check_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _trans_char_to_int,
)


def _solve_triangular(
    A, B, trans=0, lower=False, unit_diagonal=False, overwrite_b=False
):
    """
    Thin wrapper around scipy.linalg.solve_triangular.

    This function is overloaded instead of the original scipy function to avoid unexpected side-effects to users who
    import pytensor.
    """
    return linalg.solve_triangular(
        A,
        B,
        trans=trans,
        lower=lower,
        unit_diagonal=unit_diagonal,
        overwrite_b=overwrite_b,
        check_finite=False,
    )


# PyTensor always calls this with trans=0, transposing on the graph instead, so nothing
# in the test suite reaches the trans=1 or trans=2 paths below.
@overload(_solve_triangular)
def solve_triangular_impl(A, B, trans, lower, unit_diagonal, overwrite_b):
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

    def impl(A, B, trans, lower, unit_diagonal, overwrite_b):
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        # trsm reports no INFO, so a zero pivot has to be caught before the call. A is
        # scanned as given, ahead of any copy: the diagonal does not depend on layout,
        # and a singular system should not pay to be reordered first.
        if not unit_diagonal and _has_zero_on_diagonal(A):
            return np.full_like(B, np.nan)

        if A.flags.f_contiguous or (A.flags.c_contiguous and trans in (0, 1)):
            A_f = A
            if A.flags.c_contiguous:
                # A c_contiguous matrix reinterpreted as f_contiguous is A^T (plain transpose, no conjugation).
                # An upper/lower triangular A^T is lower/upper triangular, so we flip lower.
                lower = not lower
                trans = 1 - trans
        else:
            A_f = np.asfortranarray(A)

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        DIAG = val_to_int_ptr(ord("U") if unit_diagonal else ord("N"))
        LDA = val_to_int_ptr(_N)

        # A c_contiguous B reinterpreted as f_contiguous is B^T, so op(A) X = B can be solved as
        # X^T op(A)^T = B^T, which is what trsm computes with side="R", straight out of B's own buffer.
        # op(A)^T is not expressible for trans=2, since trsm has no conjugate-only mode.
        if (
            not B_is_1d
            and trans != 2
            and B.flags.c_contiguous
            and not B.flags.f_contiguous
        ):
            B_work = B if overwrite_b else B.copy()
            _NRHS = np.int32(B_work.shape[-1])
            SIDE = val_to_int_ptr(ord("R"))
            # op(A)^T is A^T for trans=0 and A for trans=1
            TRANSA = val_to_int_ptr(ord("T") if trans == 0 else ord("N"))
            M = val_to_int_ptr(_NRHS)
            N = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_NRHS)
        else:
            if overwrite_b and B.flags.f_contiguous:
                B_work = B
            else:
                B_work = _copy_to_fortran_order_even_if_1d(B)
            _NRHS = np.int32(1 if B_is_1d else B_work.shape[-1])
            SIDE = val_to_int_ptr(ord("L"))
            TRANSA = val_to_int_ptr(_trans_char_to_int(trans))
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
