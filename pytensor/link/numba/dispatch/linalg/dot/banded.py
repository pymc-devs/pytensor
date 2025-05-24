from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit as numba_njit
from numba.core.extending import overload
from numba.np.linalg import ensure_blas, ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._BLAS import _BLAS
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _get_underlying_float,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_scipy_linalg_matrix


@numba_njit(inline="always")
def A_to_banded(A: np.ndarray, kl: int, ku: int, order="C") -> np.ndarray:
    m, n = A.shape
    if order == "C":
        A_banded = np.zeros((kl + ku + 1, n), dtype=A.dtype)
    else:
        A_banded = np.zeros((n, kl + ku + 1), dtype=A.dtype).T

    for i, k in enumerate(range(ku, -kl - 1, -1)):
        if k >= 0:
            A_banded[i, k:] = np.diag(A, k=k)
        else:
            A_banded[i, : n + k] = np.diag(A, k=k)

    return A_banded


def _dot_banded(A: np.ndarray, x: np.ndarray, kl: int, ku: int) -> Any:
    """
    Thin wrapper around gmbv. This code will only be called if njit is disabled globally
    (e.g. during testing)
    """
    fn = linalg.get_blas_funcs("gbmv", (A, x))
    m, n = A.shape
    A_banded = A_to_banded(A, kl=kl, ku=ku, order="F")

    return fn(m=m, n=n, kl=kl, ku=ku, alpha=1, a=A_banded, x=x)


@overload(_dot_banded)
def dot_banded_impl(
    A: np.ndarray, x: np.ndarray, kl: int, ku: int
) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
    ensure_lapack()
    ensure_blas()
    _check_scipy_linalg_matrix(A, "dot_banded")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_gbmv = _BLAS().numba_xgbmv(dtype)

    def impl(A: np.ndarray, x: np.ndarray, kl: int, ku: int) -> np.ndarray:
        m, n = A.shape

        A_banded = A_to_banded(A, kl=kl, ku=ku, order="F")

        TRANS = val_to_int_ptr(ord("N"))
        M = val_to_int_ptr(m)
        N = val_to_int_ptr(n)
        LDA = val_to_int_ptr(A_banded.shape[0])

        KL = val_to_int_ptr(kl)
        KU = val_to_int_ptr(ku)

        ALPHA = np.array(1.0, dtype=dtype)
        INCX = val_to_int_ptr(1)
        BETA = np.array(0.0, dtype=dtype)
        Y = np.empty(m, dtype=dtype)
        INCY = val_to_int_ptr(1)

        numba_gbmv(
            TRANS,
            M,
            N,
            KL,
            KU,
            ALPHA.view(w_type).ctypes,
            A_banded.view(w_type).ctypes,
            LDA,
            x.view(w_type).ctypes,
            INCX,
            BETA.view(w_type).ctypes,
            Y.view(w_type).ctypes,
            INCY,
        )

        return Y

    return impl
