from collections.abc import Callable

import numpy as np
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_scipy_linalg_matrix


def _xlange(A: np.ndarray, order: str | None = None) -> float:
    """
    Placeholder for computing the norm of a matrix; used by linalg.solve. Will never be called in python mode.
    """
    return  # type: ignore


@overload(_xlange)
def xlange_impl(
    A: np.ndarray, order: str | None = None
) -> Callable[[np.ndarray, str], float]:
    """
    xLANGE returns the value of the one norm, or the Frobenius norm, or the infinity norm, or the  element of
    largest absolute value of a matrix A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "norm")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_lange = _LAPACK().numba_xlange(dtype)

    def impl(A: np.ndarray, order: str | None = None):
        _M, _N = np.int32(A.shape[-2:])  # type: ignore

        A_copy = _copy_to_fortran_order(A)

        M = val_to_int_ptr(_M)  # type: ignore
        N = val_to_int_ptr(_N)  # type: ignore
        LDA = val_to_int_ptr(_M)  # type: ignore

        NORM = (
            val_to_int_ptr(ord(order))
            if order is not None
            else val_to_int_ptr(ord("1"))
        )
        WORK = np.empty(_M, dtype=dtype)  # type: ignore

        result = numba_lange(
            NORM, M, N, A_copy.view(w_type).ctypes, LDA, WORK.view(w_type).ctypes
        )

        return result

    return impl
