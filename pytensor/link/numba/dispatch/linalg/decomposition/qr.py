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
    (geqrf,) = get_lapack_funcs(("geqrf",), (A,))
    return geqrf(A, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xgeqrf)
def xgeqrf_impl(A, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_geqrf = _LAPACK().numba_xgeqrf(dtype)

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

        numba_geqrf(
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
    (geqp3,) = get_lapack_funcs(("geqp3",), (A,))
    return geqp3(A, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xgeqp3)
def xgeqp3_impl(A, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_geqp3 = _LAPACK().numba_xgeqp3(dtype)

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

        numba_geqp3(
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
    (orgqr,) = get_lapack_funcs(("orgqr",), (A,))
    return orgqr(A, tau, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xorgqr)
def xorgqr_impl(A, tau, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_orgqr = _LAPACK().numba_xorgqr(dtype)

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

        numba_orgqr(
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
    (ungqr,) = get_lapack_funcs(("ungqr",), (A,))
    return ungqr(A, tau, overwrite_a=overwrite_a, lwork=lwork)


@overload(_xungqr)
def xungqr_impl(A, tau, overwrite_a, lwork):
    ensure_lapack()
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_ungqr = _LAPACK().numba_xungqr(dtype)

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

        numba_ungqr(
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


def _qr(
    x: np.ndarray,
    mode: str = "full",
    pivoting: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = False,
    lwork: int | None = None,
):
    """
    Thin wrapper around scipy.linalg.qr, following the logic in slinalg.py::QR::perform.
    """
    return qr(
        x,
        mode=mode,
        pivoting=pivoting,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
        lwork=lwork,
    )


@overload(_qr)
def qr_impl(
    x, mode="full", pivoting=False, overwrite_a=False, check_finite=False, lwork=None
):
    """
    Numba overload for _qr, dispatching to the appropriate LAPACK routines.
    """
    ensure_lapack()
    dtype = x.dtype
    w_type = _get_underlying_float(dtype)
    numba_geqrf = _LAPACK().numba_xgeqrf(dtype)
    numba_geqp3 = _LAPACK().numba_xgeqp3(dtype)
    numba_orgqr = _LAPACK().numba_xorgqr(dtype)

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

        if overwrite_a and x.flags.f_contiguous:
            x_copy = x
        else:
            x_copy = _copy_to_fortran_order(x)

        LDA = val_to_int_ptr(M)
        K = min(M, N)
        TAU = np.empty(K, dtype=dtype)

        if lwork is None:
            lwork = -1

        if pivoting:
            JPVT = np.zeros(N, dtype=np.int32)

            if lwork == -1:
                WORK = np.empty(1, dtype=dtype)
                numba_geqp3(
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

            numba_geqp3(
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

            # LAPACK uses 1-based indexing for JPVT, convert to 0-based
            JPVT = JPVT - 1

        else:
            if lwork == -1:
                WORK = np.empty(1, dtype=np.int32)
                numba_geqrf(
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
            numba_geqrf(
                val_to_int_ptr(M),
                val_to_int_ptr(N),
                x_copy.view(w_type).ctypes,
                LDA,
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(lwork_val),
                INFO,
            )

        if mode not in ["economic", "raw"] or M < N:
            R = np.triu(x_copy)
        else:
            R = np.triu(x_copy[:N, :])

        if mode == "r" and pivoting:
            return R, JPVT
        elif mode == "r":
            return (R,)
        elif mode == "raw" and pivoting:
            return x_copy, TAU, R, JPVT
        elif mode == "raw":
            return x_copy, TAU, R

        # Compute Q
        if M < N:
            Q_in = x_copy[:, :M]
        elif mode == "economic":
            Q_in = x_copy
        else:
            t = x_copy.dtype.char
            Q_in = np.empty((M, M), dtype=t)
            Q_in[:, :N] = x_copy

        if lwork == -1:
            WORK = np.empty(1, dtype=dtype)
            numba_orgqr(
                val_to_int_ptr(M),
                val_to_int_ptr(Q_in.shape[1]),
                val_to_int_ptr(K),
                Q_in.view(w_type).ctypes,
                val_to_int_ptr(M),
                TAU.view(w_type).ctypes,
                WORK.view(w_type).ctypes,
                val_to_int_ptr(-1),
                val_to_int_ptr(1),
            )
            lwork_q = int(WORK.item())
        else:
            lwork_q = lwork

        WORK_Q = np.empty(lwork_q, dtype=dtype)
        INFO_Q = val_to_int_ptr(1)
        numba_orgqr(
            val_to_int_ptr(M),
            val_to_int_ptr(Q_in.shape[1]),
            val_to_int_ptr(K),
            Q_in.view(w_type).ctypes,
            val_to_int_ptr(M),
            TAU.view(w_type).ctypes,
            WORK_Q.view(w_type).ctypes,
            val_to_int_ptr(lwork_q),
            INFO_Q,
        )

        if mode == "full":
            if pivoting:
                return Q_in, R, JPVT
            else:
                return Q_in, R
        elif mode == "economic":
            if pivoting:
                return Q_in, R, JPVT
            else:
                return Q_in, R

    return impl
