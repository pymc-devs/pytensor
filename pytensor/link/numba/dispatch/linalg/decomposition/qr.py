from typing import Literal

import numpy as np
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy.linalg import get_lapack_funcs, qr

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)


def _xgeqrf(A: np.ndarray, overwrite_a: bool, lwork: int):
    """LAPACK geqrf: Computes a QR factorization of a general M-by-N matrix A."""
    # (geqrf,) = typing_cast(
    #     list[Callable[..., np.ndarray]], get_lapack_funcs(("geqrf",), (A,))
    # )
    funcs = get_lapack_funcs(("geqrf",), (A,))
    assert isinstance(funcs, list)  # narrows `funcs: list[F] | F` to `funcs: list[F]`
    geqrf = funcs[0]

    return geqrf(A, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xgeqrf)
def xgeqrf_impl(A, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    geqrf = _LAPACK().numba_xgeqrf(dtype)

    def impl(A, overwrite_a, lwork):
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        LDA = val_to_int_ptr(M)
        TAU = np.empty(min(M, N), dtype=dtype)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            A_copy.view(w_type).ctypes,
            LDA,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )
        return A_copy, TAU, WORK, int_ptr_to_val(INFO)

    return impl


def _xgeqp3(A: np.ndarray, overwrite_a: bool, lwork: int):
    """LAPACK geqp3: Computes a QR factorization with column pivoting of a general M-by-N matrix A."""
    funcs = get_lapack_funcs(("geqp3",), (A,))
    assert isinstance(funcs, list)  # narrows `funcs: list[F] | F` to `funcs: list[F]`
    geqp3 = funcs[0]

    return geqp3(A, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xgeqp3)
def xgeqp3_impl(A, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)

    def impl(A, overwrite_a, lwork):
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        LDA = val_to_int_ptr(M)
        JPVT = np.zeros(N, dtype=np.int32)
        TAU = np.empty(min(M, N), dtype=dtype)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)
        INFO = val_to_int_ptr(1)

        geqp3(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            A_copy.view(w_type).ctypes,
            LDA,
            JPVT.ctypes,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )
        return A_copy, JPVT, TAU, WORK, int_ptr_to_val(INFO)

    return impl


def _xorgqr(A: np.ndarray, tau: np.ndarray, overwrite_a: bool, lwork: int):
    """LAPACK orgqr: Generates the M-by-N matrix Q with orthonormal columns from a QR factorization (real types)."""
    funcs = get_lapack_funcs(("orgqr",), (A,))
    assert isinstance(funcs, list)  # narrows `funcs: list[F] | F` to `funcs: list[F]`
    orgqr = funcs[0]

    return orgqr(A, tau, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xorgqr)
def xorgqr_impl(A, tau, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    orgqr = _LAPACK().numba_xorgqr(dtype)

    def impl(A, tau, overwrite_a, lwork):
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[1])
        K = np.int32(tau.shape[0])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)

        LDA = val_to_int_ptr(M)
        INFO = val_to_int_ptr(1)

        orgqr(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            val_to_int_ptr(K),
            A_copy.view(w_type).ctypes,
            LDA,
            tau.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )
        return A_copy, WORK, int_ptr_to_val(INFO)

    return impl


def _xungqr(A: np.ndarray, tau: np.ndarray, overwrite_a: bool, lwork: int):
    """LAPACK ungqr: Generates the M-by-N matrix Q with orthonormal columns from a QR factorization (complex types)."""
    funcs = get_lapack_funcs(("ungqr",), (A,))
    assert isinstance(funcs, list)  # narrows `funcs: list[F] | F` to `funcs: list[F]`
    ungqr = funcs[0]

    return ungqr(A, tau, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xungqr)
def xungqr_impl(A, tau, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    ungqr = _LAPACK().numba_xungqr(dtype)

    def impl(A, tau, overwrite_a, lwork):
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[1])
        K = np.int32(tau.shape[0])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)
        LDA = val_to_int_ptr(M)

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            LWORK = val_to_int_ptr(-1)
        else:
            WORK = np.empty(lwork if lwork > 0 else 1, dtype=dtype)
            LWORK = val_to_int_ptr(WORK.size)
        INFO = val_to_int_ptr(1)

        ungqr(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            val_to_int_ptr(K),
            A_copy.view(w_type).ctypes,
            LDA,
            tau.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )

        return A_copy, WORK, int_ptr_to_val(INFO)

    return impl


def _qr_full_pivot(
    x: np.ndarray,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )


def _qr_full_no_pivot(
    x: np.ndarray,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )


def _qr_r_pivot(
    x: np.ndarray,
    mode: Literal["r", "raw"] = "r",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )


def _qr_r_no_pivot(
    x: np.ndarray,
    mode: Literal["r", "raw"] = "r",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )


def _qr_raw_no_pivot(
    x: np.ndarray,
    mode: Literal["raw"] = "raw",
    pivoting: Literal[False] = False,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )

    return H, tau, R


def _qr_raw_pivot(
    x: np.ndarray,
    mode: Literal["raw"] = "raw",
    pivoting: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = False,
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
        check_finite=check_finite,
        lwork=lwork,
    )

    return H, tau, R, P


@overload(_qr_full_pivot)
def qr_full_pivot_impl(
    x, mode="full", pivoting=True, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)
    orgqr = _LAPACK().numba_xorgqr(dtype)

    def impl(
        x,
        mode="full",
        pivoting=True,
        overwrite_a=False,
        check_finite=False,
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

        if lwork is None:
            lwork = -1

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            geqp3(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.view(w_type).ctypes,
                LDA,
                JPVT.ctypes,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())

        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)
        geqp3(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            JPVT.ctypes,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
                Q_in.view(w_type).ctypes,
                val_to_int_ptr(M),
                TAU.view(w_type).ctypes,
                WORKQ.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_q = int(WORKQ.item())

        else:
            lwork_q = lwork

        WORKQ = np.empty(lwork_q, dtype=dtype)
        INFOQ = val_to_int_ptr(1)
        orgqr(
            val_to_int_ptr(M),
            val_to_int_ptr(Q_in.shape[1]),
            val_to_int_ptr(K),
            Q_in.view(w_type).ctypes,
            val_to_int_ptr(M),
            TAU.view(w_type).ctypes,
            WORKQ.view(w_type).ctypes,
            val_to_int_ptr(lwork_q),
            INFOQ,
        )
        return Q_in, R, JPVT

    return impl


@overload(_qr_full_no_pivot)
def qr_full_no_pivot_impl(
    x, mode="full", pivoting=False, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqrf = _LAPACK().numba_xgeqrf(dtype)
    orgqr = _LAPACK().numba_xorgqr(dtype)

    def impl(
        x,
        mode="full",
        pivoting=False,
        overwrite_a=False,
        check_finite=False,
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
                x_copy.view(w_type).ctypes,
                LDA,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
                Q_in.view(w_type).ctypes,
                val_to_int_ptr(M),
                TAU.view(w_type).ctypes,
                WORKQ.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_q = int(WORKQ.item())
        else:
            lwork_q = lwork

        WORKQ = np.empty(lwork_q, dtype=dtype)
        INFOQ = val_to_int_ptr(1)

        orgqr(
            val_to_int_ptr(M),  # M
            val_to_int_ptr(Q_in.shape[1]),  # N
            val_to_int_ptr(K),  # K
            Q_in.view(w_type).ctypes,  # A
            val_to_int_ptr(M),  # LDA
            TAU.view(w_type).ctypes,  # TAU
            WORKQ.view(w_type).ctypes,  # WORK
            val_to_int_ptr(lwork_q),  # LWORK
            INFOQ,  # INFO
        )
        return Q_in, R

    return impl


@overload(_qr_r_pivot)
def qr_r_pivot_impl(
    x, mode="r", pivoting=True, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)

    def impl(
        x,
        mode="r",
        pivoting=True,
        overwrite_a=False,
        check_finite=False,
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
                x_copy.view(w_type).ctypes,
                LDA,
                JPVT.ctypes,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqp3(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            JPVT.ctypes,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
def qr_r_no_pivot_impl(
    x, mode="r", pivoting=False, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqrf = _LAPACK().numba_xgeqrf(dtype)

    def impl(
        x,
        mode="r",
        pivoting=False,
        overwrite_a=False,
        check_finite=False,
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
                x_copy.view(w_type).ctypes,
                LDA,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
def qr_raw_no_pivot_impl(
    x, mode="raw", pivoting=False, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqrf = _LAPACK().numba_xgeqrf(dtype)

    def impl(
        x,
        mode="raw",
        pivoting=False,
        overwrite_a=False,
        check_finite=False,
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
                x_copy.view(w_type).ctypes,
                LDA,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqrf(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
def qr_raw_pivot_impl(
    x, mode="raw", pivoting=True, overwrite_a=False, check_finite=False, lwork=None
):
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    geqp3 = _LAPACK().numba_xgeqp3(dtype)

    def impl(
        x,
        mode="raw",
        pivoting=True,
        overwrite_a=False,
        check_finite=False,
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
                x_copy.view(w_type).ctypes,
                LDA,
                JPVT.ctypes,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_val = int(WORK.item())
        else:
            lwork_val = lwork

        WORK = np.empty(lwork_val, dtype=dtype)
        INFO = val_to_int_ptr(1)

        geqp3(
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            x_copy.view(w_type).ctypes,
            LDA,
            JPVT.ctypes,
            TAU.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
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
