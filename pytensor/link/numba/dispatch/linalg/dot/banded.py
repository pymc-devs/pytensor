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
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_scipy_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _trans_char_to_int,
)


@numba_njit(inline="always")
def A_to_banded(A: np.ndarray, kl: int, ku: int) -> np.ndarray:
    m, n = A.shape

    # This matrix is build backwards then transposed to get it into Fortran order
    # (order="F" is not allowed in Numba land)
    A_banded = np.zeros((n, kl + ku + 1), dtype=A.dtype).T

    for i, k in enumerate(range(ku, -kl - 1, -1)):
        if k >= 0:
            A_banded[i, k:] = np.diag(A, k=k)
        else:
            A_banded[i, : n + k] = np.diag(A, k=k)

    return A_banded


def _gbmv(
    alpha: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    kl: int,
    ku: int,
    beta: np.ndarray | None = None,
    y: np.ndarray | None = None,
    overwrite_y: bool = False,
    trans: int = 1,
) -> Any:
    """
    Thin wrapper around gmbv. This code will only be called if njit is disabled globally
    (e.g. during testing)
    """
    (fn,) = linalg.get_blas_funcs(("gbmv",), (A, x))
    m, n = A.shape
    A_banded = A_to_banded(A, kl=kl, ku=ku)

    incx = x.strides[0] // x.itemsize
    offx = 0 if incx >= 0 else -x.size + 1

    if y is not None:
        incy = y.strides[0] // y.itemsize
        offy = 0 if incy >= 0 else -y.size + 1
    else:
        incy = 1
        offy = 0

    return fn(
        m=m,
        n=n,
        kl=kl,
        ku=ku,
        a=A_banded,
        alpha=alpha,
        x=x,
        incx=incx,
        offx=offx,
        beta=beta,
        y=y,
        overwrite_y=overwrite_y,
        incy=incy,
        offy=offy,
        trans=trans,
    )


@overload(_gbmv)
def gbmv_impl(
    alpha: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    kl: int,
    ku: int,
    beta: np.ndarray | None = None,
    y: np.ndarray | None = None,
    overwrite_y: bool = False,
    trans: int = 1,
) -> Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        np.ndarray | None,
        np.ndarray | None,
        bool,
        int,
    ],
    np.ndarray,
]:
    ensure_lapack()
    ensure_blas()
    _check_scipy_linalg_matrix(A, "dot_banded")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_gbmv = _BLAS().numba_xgbmv(dtype)

    def impl(
        alpha: np.ndarray,
        A: np.ndarray,
        x: np.ndarray,
        kl: int,
        ku: int,
        beta: np.ndarray | None = None,
        y: np.ndarray | None = None,
        overwrite_y: bool = False,
        trans: int = 1,
    ) -> np.ndarray:
        m, n = A.shape

        A_banded = A_to_banded(A, kl=kl, ku=ku)
        x_stride = x.strides[0] // x.itemsize

        if beta is None:
            beta = np.zeros((), dtype=dtype)

        if y is None:
            y_copy = np.empty(shape=(m,), dtype=dtype)
        elif overwrite_y and y.flags.f_contiguous:
            y_copy = y
        else:
            y_copy = _copy_to_fortran_order_even_if_1d(y)

        y_stride = y_copy.strides[0] // y_copy.itemsize

        TRANS = val_to_int_ptr(_trans_char_to_int(trans))
        M = val_to_int_ptr(m)
        N = val_to_int_ptr(n)
        LDA = val_to_int_ptr(A_banded.shape[0])

        KL = val_to_int_ptr(kl)
        KU = val_to_int_ptr(ku)

        INCX = val_to_int_ptr(x_stride)
        INCY = val_to_int_ptr(y_stride)

        numba_gbmv(
            TRANS,
            M,
            N,
            KL,
            KU,
            alpha.view(w_type).ctypes,
            A_banded.view(w_type).ctypes,
            LDA,
            # x.view().ctypes is creating a pointer to the beginning of the memory where the array is. When we have
            # a negative stride, we need to trick BLAS by pointing to the last element of the array.
            # The [-1:] slice is a workaround to make sure x remains an array (otherwise it has no .ctypes)
            (x if x_stride >= 0 else x[-1:]).view(w_type).ctypes,
            INCX,
            beta.view(w_type).ctypes,
            y_copy.view(w_type).ctypes,
            INCY,
        )

        return y_copy

    return impl
