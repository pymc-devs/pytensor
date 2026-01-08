from typing import Literal

import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy.linalg import qr

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


def _qr_full_pivot(
    x: np.ndarray,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode not "r" or "raw", and pivoting is True, resulting in a return of arrays Q, R, and
    P.
    """
    return qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )


def _qr_full_no_pivot(
    x: np.ndarray,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode not "r" or "raw", and pivoting is False, resulting in a return of arrays Q and R.
    """
    return qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )


def _qr_r_pivot(
    x: np.ndarray,
    mode: Literal["r", "raw"] = "r",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode is "r" or "raw", and pivoting is True, resulting in a return of arrays R and P.
    """
    return qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )


def _qr_r_no_pivot(
    x: np.ndarray,
    mode: Literal["r", "raw"] = "r",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode is "r" or "raw", and pivoting is False, resulting in a return of array R.
    """
    return qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )


def _qr_raw_no_pivot(
    x: np.ndarray,
    mode: Literal["raw"] = "raw",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode is "raw", and pivoting is False, resulting in a return of arrays H, tau, and R.
    """
    (H, tau), R = qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )

    return H, tau, R


def _qr_raw_pivot(
    x: np.ndarray,
    mode: Literal["raw"] = "raw",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, used to avoid side effects when users import pytensor and scipy in the same
    script.

    Corresponds to the case where mode is "raw", and pivoting is True, resulting in a return of arrays H, tau, R, and P.
    """
    (H, tau), R, P = qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=False,
        lwork=lwork,
    )

    return H, tau, R, P


@overload(_qr_full_pivot)
def qr_full_pivot_impl(x, mode="full", pivoting=True, overwrite_a=False, lwork=None):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")
    dtype = x.dtype

    is_complex = isinstance(dtype, Complex)

    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)
    orgqr = (
        _LAPACK().numba_xorgqr(dtype)
        if isinstance(dtype, Float)
        else _LAPACK().numba_xungqr(dtype)
    )

    def impl(
        x,
        mode="full",
        pivoting=True,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])
        K = min(M, N)

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        TAU = np.empty(K, dtype=dtype)
        JPVT = np.zeros(N, dtype=np.int32)
        if is_complex:
            RWORK = np.empty(2 * N, dtype=w_type)

        if lwork is None:
            lwork = -1

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            if is_complex:
                geqp3(
                    val_to_int_ptr(M),
                    val_to_int_ptr(N),
                    x_copy.ctypes,
                    LDA,
                    JPVT.ctypes,
                    TAU.ctypes,
                    WORK.ctypes,
                    val_to_int_ptr(-1),  # LWORK
                    RWORK.ctypes,
                    val_to_int_ptr(1),  # INFO
                )
            else:
                geqp3(
                    val_to_int_ptr(M),
                    val_to_int_ptr(N),
                    x_copy.ctypes,
                    LDA,
                    JPVT.ctypes,
                    TAU.ctypes,
                    WORK.ctypes,
                    val_to_int_ptr(-1),
                    val_to_int_ptr(1),
                )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        if is_complex:
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                JPVT.ctypes,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(lwork_val),
                RWORK.ctypes,
                INFO,
            )
        else:
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                JPVT.ctypes,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(lwork_val),
                INFO,
            )
        JPVT = (JPVT - 1).astype(np.int32)

        if mode == "full" or M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        if M < N:
            Q_in = x_copy[:, :M]
        elif M == N or mode == "economic":
            Q_in = x_copy
        else:
            # Transpose to put the matrix into Fortran order
            Q_in = np.empty((M, M), dtype=dtype).T
            Q_in[:, :N] = x_copy

        if lwork == -1:
            WORKQ = np.empty(1, dtype=dtype)
            orgqr(
                val_to_int_ptr(M),
                val_to_int_ptr(Q_in.shape[1]),
                val_to_int_ptr(K),
                Q_in.ctypes,
                val_to_int_ptr(M),
                TAU.ctypes,
                WORKQ.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_q = int(WORKQ.item().real)

        else:
            lwork_q = lwork

        WORKQ = np.empty(lwork_q, dtype=dtype)
        INFOQ = val_to_int_ptr(1)
        orgqr(
            val_to_int_ptr(M),
            val_to_int_ptr(Q_in.shape[1]),
            val_to_int_ptr(K),
            Q_in.ctypes,
            val_to_int_ptr(M),
            TAU.ctypes,
            WORKQ.ctypes,
            val_to_int_ptr(lwork_q),
            INFOQ,
        )
        return Q_in, R, JPVT

    return impl


@overload(_qr_full_no_pivot)
def qr_full_no_pivot_impl(
    x, mode="full", pivoting=False, overwrite_a=False, lwork=None
):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")
    dtype = x.dtype
    geqrf = _LAPACK().numba_xgeqrf(dtype)
    orgqr = (
        _LAPACK().numba_xorgqr(dtype)
        if isinstance(dtype, Float)
        else _LAPACK().numba_xungqr(dtype)
    )

    def impl(
        x,
        mode="full",
        pivoting=False,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])
        K = min(M, N)

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        TAU = np.empty(K, dtype=dtype)

        if lwork is None:
            lwork = -1

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            geqrf(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.ctypes,
            LDA,
            TAU.ctypes,
            WORK.ctypes,
            val_to_int_ptr(lwork_val),
            INFO,
        )

        if M < N or mode == "full":
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        if M < N:
            Q_in = x_copy[:, :M]
        elif M == N or mode == "economic":
            Q_in = x_copy
        else:
            # Transpose to put the matrix into Fortran order
            Q_in = np.empty((M, M), dtype=dtype).T
            Q_in[:, :N] = x_copy

        if lwork == -1:
            WORKQ = np.empty(1, dtype=dtype)
            orgqr(
                val_to_int_ptr(M),
                val_to_int_ptr(Q_in.shape[1]),
                val_to_int_ptr(K),
                Q_in.ctypes,
                val_to_int_ptr(M),
                TAU.ctypes,
                WORKQ.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_q = int(WORKQ.real.item())
        else:
            lwork_q = lwork

        WORKQ = np.empty(lwork_q, dtype=dtype)
        INFOQ = val_to_int_ptr(1)

        orgqr(
            val_to_int_ptr(M),  # M
            val_to_int_ptr(Q_in.shape[1]),  # N
            val_to_int_ptr(K),  # K
            Q_in.ctypes,  # A
            val_to_int_ptr(M),  # LDA
            TAU.ctypes,  # TAU
            WORKQ.ctypes,  # WORK
            val_to_int_ptr(lwork_q),  # LWORK
            INFOQ,  # INFO
        )
        return Q_in, R

    return impl


@overload(_qr_r_pivot)
def qr_r_pivot_impl(x, mode="r", pivoting=True, overwrite_a=False, lwork=None):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")
    dtype = x.dtype
    geqp3 = _LAPACK().numba_xgeqp3(dtype)

    def impl(
        x,
        mode="r",
        pivoting=True,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        K = min(M, N)
        TAU = np.empty(K, dtype=dtype)
        JPVT = np.zeros(N, dtype=np.int32)

        if lwork is None:
            lwork = -1
        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                JPVT.ctypes,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqp3(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.ctypes,
            LDA,
            JPVT.ctypes,
            TAU.ctypes,
            WORK.ctypes,
            val_to_int_ptr(lwork_val),
            INFO,
        )
        JPVT = (JPVT - 1).astype(np.int32)

        if M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        return R, JPVT

    return impl


@overload(_qr_r_no_pivot)
def qr_r_no_pivot_impl(x, mode="r", pivoting=False, overwrite_a=False, lwork=None):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")
    dtype = x.dtype
    geqrf = _LAPACK().numba_xgeqrf(dtype)

    def impl(
        x,
        mode="r",
        pivoting=False,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        K = min(M, N)
        TAU = np.empty(K, dtype=dtype)

        if lwork is None:
            lwork = -1
        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            geqrf(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.ctypes,
            LDA,
            TAU.ctypes,
            WORK.ctypes,
            val_to_int_ptr(lwork_val),
            INFO,
        )

        if M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        # Return a tuple with R only to match the scipy qr interface
        return (R,)

    return impl


@overload(_qr_raw_no_pivot)
def qr_raw_no_pivot_impl(x, mode="raw", pivoting=False, overwrite_a=False, lwork=None):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")
    dtype = x.dtype
    geqrf = _LAPACK().numba_xgeqrf(dtype)

    def impl(
        x,
        mode="raw",
        pivoting=False,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        K = min(M, N)
        TAU = np.empty(K, dtype=dtype)

        if lwork is None:
            lwork = -1
        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            geqrf(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.ctypes,
            LDA,
            TAU.ctypes,
            WORK.ctypes,
            val_to_int_ptr(lwork_val),
            INFO,
        )

        if M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        return x_copy, TAU, R

    return impl


@overload(_qr_raw_pivot)
def qr_raw_pivot_impl(x, mode="raw", pivoting=True, overwrite_a=False, lwork=None):
    ensure_lapack()
    _check_linalg_matrix(x, ndim=2, dtype=(Float, Complex), func_name="qr")

    dtype = x.dtype
    is_complex = isinstance(dtype, Complex)

    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)

    def impl(
        x,
        mode="raw",
        pivoting=True,
        overwrite_a=False,
        lwork=None,
    ):
        M = np.int32(x.shape[0])
        N = np.int32(x.shape[1])

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        K = min(M, N)
        TAU = np.empty(K, dtype=dtype)
        JPVT = np.zeros(N, dtype=np.int32)
        if is_complex:
            RWORK = np.empty(2 * N, dtype=w_type)

        if lwork is None:
            lwork = -1
        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            if is_complex:
                geqp3(
                    val_to_int_ptr(M),
                    val_to_int_ptr(N),
                    x_copy.ctypes,
                    LDA,
                    JPVT.ctypes,
                    TAU.ctypes,
                    WORK.ctypes,
                    val_to_int_ptr(-1),  # LWORK
                    RWORK.ctypes,
                    val_to_int_ptr(1),  # INFO
                )
            else:
                geqp3(
                    val_to_int_ptr(M),
                    val_to_int_ptr(N),
                    x_copy.ctypes,
                    LDA,
                    JPVT.ctypes,
                    TAU.ctypes,
                    WORK.ctypes,
                    val_to_int_ptr(-1),
                    val_to_int_ptr(1),
                )
            lwork_val = int(WORK.item().real)
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        if is_complex:
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                JPVT.ctypes,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(lwork_val),
                RWORK.ctypes,
                INFO,
            )
        else:
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.ctypes,
                LDA,
                JPVT.ctypes,
                TAU.ctypes,
                WORK.ctypes,
                val_to_int_ptr(lwork_val),
                INFO,
            )

        JPVT = (JPVT - 1).astype(np.int32)

        if M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        return x_copy, TAU, R, JPVT

    return impl
