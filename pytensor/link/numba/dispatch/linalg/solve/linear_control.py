from collections.abc import Callable
from typing import cast

import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy.linalg import get_lapack_funcs

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_dtypes_match,
    _check_linalg_matrix,
)


def _trsyl(a: np.ndarray, b: np.ndarray, c: np.ndarray, overwrite_c):
    """Placeholder for real TRSYL (Sylvester equation solver)."""
    fn = cast(Callable, get_lapack_funcs("trsyl", (a, b, c)))
    x, scale, info = fn(a, b, c, overwrite_c=overwrite_c)
    if info < 0:
        return np.full_like(c, np.nan)
    x *= scale
    return x


@overload(_trsyl)
def trsyl_impl(A, B, C, overwrite_c):
    """
    Overload for real TRSYL to solve Sylvester equation for inputs A and B in standard
    Schur form.
    """
    ensure_lapack()

    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="trsyl")
    _check_linalg_matrix(B, ndim=2, dtype=(Float, Complex), func_name="trsyl")
    _check_linalg_matrix(C, ndim=2, dtype=(Float, Complex), func_name="trsyl")
    _check_dtypes_match((A, B, C), func_name="trsyl")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_xtrsyl = _LAPACK().numba_xtrsyl(dtype)

    def impl(A, B, C, overwrite_c):
        _M = np.int32(A.shape[-1])
        _N = np.int32(B.shape[-1])

        A_copy = _copy_to_fortran_order(A)
        B_copy = _copy_to_fortran_order(B)

        if overwrite_c and C.flags.f_contiguous:
            C_copy = C
        else:
            C_copy = _copy_to_fortran_order(C)

        TRANA = val_to_int_ptr(ord("N"))
        TRANB = val_to_int_ptr(ord("N"))
        ISGN = val_to_int_ptr(1)

        M = val_to_int_ptr(_M)
        N = val_to_int_ptr(_N)

        LDA = val_to_int_ptr(_M)
        LDB = val_to_int_ptr(_N)
        LDC = val_to_int_ptr(_M)

        SCALE = np.array(1.0, dtype=w_type)
        INFO = val_to_int_ptr(0)

        # Call LAPACK trsyl
        numba_xtrsyl(
            TRANA,
            TRANB,
            ISGN,
            M,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            C_copy.ctypes,
            LDC,
            SCALE.ctypes,
            INFO,
        )

        if int_ptr_to_val(INFO) < 0:
            return np.full_like(C_copy, np.nan)

        # CC now contains the solution, scale it
        C_copy *= SCALE
        return C_copy

    return impl
