import numpy as np
from numba.core.extending import overload
from numba.core.types import Float
from numba.np.linalg import ensure_lapack
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


def _cho_solve(C: np.ndarray, B: np.ndarray, lower: bool, overwrite_b: bool):
    """
    Solve a positive-definite linear system using the Cholesky decomposition.
    """
    return linalg.cho_solve(
        (C, lower),
        b=B,
        overwrite_b=overwrite_b,
        check_finite=False,
    )


@overload(_cho_solve)
def cho_solve_impl(C, B, lower=False, overwrite_b=False):
    ensure_lapack()
    _check_linalg_matrix(C, ndim=2, dtype=Float, func_name="cho_solve")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="cho_solve")
    _check_dtypes_match((C, B), func_name="cho_solve")
    dtype = C.dtype
    numba_potrs = _LAPACK().numba_xpotrs(dtype)

    def impl(C, B, lower=False, overwrite_b=False):
        _solve_check_input_shapes(C, B)

        _N = np.int32(C.shape[-1])
        if C.flags.f_contiguous or C.flags.c_contiguous:
            C_f = C
            if C.flags.c_contiguous:
                # An upper/lower triangular c_contiguous is the same as a lower/upper triangular f_contiguous
                lower = not lower
        else:
            C_f = np.asfortranarray(C)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order_even_if_1d(B)

        B_is_1d = B.ndim == 1
        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        NRHS = 1 if B_is_1d else int(B.shape[-1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_potrs(
            UPLO,
            N,
            NRHS,
            C_f.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            B_copy = np.full_like(B_copy, np.nan)

        if B_is_1d:
            return B_copy[..., 0]
        return B_copy

    return impl
