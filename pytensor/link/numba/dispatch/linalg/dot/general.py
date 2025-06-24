from collections.abc import Callable

import numpy as np
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


def _matrix_vector_product(
    alpha: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray | None = None,
    y: np.ndarray | None = None,
    overwrite_y: bool = False,
    trans: int = 1,
) -> np.ndarray:
    """
    Thin wrapper around gmv. This code will only be called if njit is disabled globally
    (e.g. during testing)
    """
    (fn,) = linalg.get_blas_funcs(("gemv",), (A, x))

    incx = x.strides[0] // x.itemsize
    incy = y.strides[0] // y.itemsize if y is not None else 1

    offx = 0 if incx >= 0 else -x.size + 1
    offy = 0 if incy >= 0 else -y.size + 1

    return fn(
        alpha=alpha,
        a=A,
        x=x,
        beta=beta,
        y=y,
        overwrite_y=overwrite_y,
        offx=offx,
        incx=incx,
        offy=offy,
        incy=incy,
        trans=trans,
    )


@overload(_matrix_vector_product)
def matrix_vector_product_impl(
    alpha: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray | None = None,
    y: np.ndarray | None = None,
    overwrite_y: bool = False,
    trans: int = 1,
) -> Callable[
    [float, np.ndarray, np.ndarray, float, np.ndarray, int, int, int, int, int],
    np.ndarray,
]:
    ensure_lapack()
    ensure_blas()
    _check_scipy_linalg_matrix(A, "matrix_vector_product")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_gemv = _BLAS().numba_xgemv(dtype)

    def impl(
        alpha: np.ndarray,
        A: np.ndarray,
        x: np.ndarray,
        beta: np.ndarray | None = None,
        y: np.ndarray | None = None,
        overwrite_y: bool = False,
        trans: int = 1,
    ) -> np.ndarray:
        m, n = A.shape
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
        LDA = val_to_int_ptr(A.shape[0])

        # ALPHA = np.array(alpha, dtype=dtype)

        INCX = val_to_int_ptr(x_stride)
        # BETA = np.array(beta, dtype=dtype)
        INCY = val_to_int_ptr(y_stride)

        numba_gemv(
            TRANS,
            M,
            N,
            alpha.view(w_type).ctypes,
            A.view(w_type).ctypes,
            LDA,
            # x.view().ctypes is creating a pointer to the beginning of the memory where the array is. When we have
            # a negative stride, we need to trick BLAS by pointing to the last element of the array.
            # The [-1:] slice is a workaround to make sure x remains an array (otherwise it has no .ctypes)
            (x if x_stride >= 0 else x[-1:]).view(w_type).ctypes,
            INCX,
            beta.view(w_type).ctypes,
            y_copy.view(w_type).ctypes,
            # (y_copy if y_stride >= 0 else y_copy[:-1:]).view(w_type).ctypes,
            INCY,
        )

        return y_copy

    return impl
