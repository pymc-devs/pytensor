from collections.abc import Callable

import numpy as np
from numba.core.extending import overload
from numba.core.types import Float
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
)


def _solve_psd(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
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
        check_finite=False,
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
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=Float, func_name="solve")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="solve")
    _check_dtypes_match((A, B), func_name="solve")
    numba_posv = _LAPACK().numba_xposv(A.dtype)

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        transposed: bool,
    ) -> np.ndarray:
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
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            B_copy = B_copy[..., 0]

        if int_ptr_to_val(INFO) != 0:
            B_copy = np.full_like(B_copy, np.nan)

        return B_copy

    return impl
