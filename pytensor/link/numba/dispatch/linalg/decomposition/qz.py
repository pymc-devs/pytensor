import numpy as np
import scipy.linalg as scipy_linalg
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


@numba_basic.numba_njit
def _lhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) < 0.0
    return out


@numba_basic.numba_njit
def _rhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) > 0.0
    return out


@numba_basic.numba_njit
def _iuc(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    out[~nonzero] = False
    out[nonzero] = np.abs(alpha[nonzero] / beta[nonzero]) < 1.0
    return out


@numba_basic.numba_njit
def _ouc(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    alpha_zero = alpha == 0
    beta_zero = beta == 0
    out[alpha_zero & beta_zero] = False
    out[~alpha_zero & beta_zero] = True
    out[~beta_zero] = np.abs(alpha[~beta_zero] / beta[~beta_zero]) > 1.0
    return out


def _qz_real_nosort_noeig(A, B, overwrite_a=False, overwrite_b=False):
    S, T, Q, Z = scipy_linalg.qz(
        A,
        B,
        output="real",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, Q, Z


def _qz_real_nosort_eig(A, B, overwrite_a=False, overwrite_b=False):
    S, T, Q, Z = scipy_linalg.qz(
        A,
        B,
        output="real",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    # There is no option to return eigenvalues directly from scipy.linalg.qz, so we have to compute them manually.
    # Unlike the complex Schur form, the real Schur form can have 2x2 blocks on the main diagonal for complex conjugate
    # pairs, so we can't just read off the eigenvalues and the diagonal elements of S and T.
    n = S.shape[0]
    alpha = np.empty(
        n,
        dtype=np.complex128
        if _get_underlying_float(S.dtype) == np.float64
        else np.complex64,
    )
    beta = np.empty(n, dtype=S.dtype)
    i = 0
    while i < n:
        if i == n - 1 or S[i + 1, i] == 0:
            # 1x1 block - real eigenvalue
            alpha[i] = S[i, i]
            beta[i] = T[i, i]
            i += 1
        else:
            # 2x2 block - complex conjugate pair
            a11, a12, a21, a22 = S[i, i], S[i, i + 1], S[i + 1, i], S[i + 1, i + 1]
            b11, b22 = T[i, i], T[i + 1, i + 1]
            # For standardized 2x2 blocks, eigenvalues are roots of det(A - lambda*B)

            tr = (a11 * b22 + a22 * b11) / (b11 * b22)
            det = (a11 * a22 - a12 * a21) / (b11 * b22)
            disc = tr * tr / 4 - det
            if disc < 0:
                sqrt_disc = np.sqrt(-disc)
                alpha[i] = tr / 2 + 1j * sqrt_disc
                alpha[i + 1] = tr / 2 - 1j * sqrt_disc
            else:
                sqrt_disc = np.sqrt(disc)
                alpha[i] = tr / 2 + sqrt_disc
                alpha[i + 1] = tr / 2 - sqrt_disc
            beta[i] = 1.0
            beta[i + 1] = 1.0
            i += 2
    return S, T, alpha, beta, Q, Z


def _qz_real_sort_noeig(A, B, sort, overwrite_a=False, overwrite_b=False):
    S, T, _, _, Q, Z = scipy_linalg.ordqz(
        A,
        B,
        sort=sort,
        output="real",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, Q, Z


def _qz_real_sort_eig(A, B, sort, overwrite_a=False, overwrite_b=False):
    S, T, alpha, beta, Q, Z = scipy_linalg.ordqz(
        A,
        B,
        sort=sort,
        output="real",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, alpha, beta, Q, Z


def _qz_complex_nosort_noeig(A, B, overwrite_a=False, overwrite_b=False):
    S, T, Q, Z = scipy_linalg.qz(
        A,
        B,
        output="complex",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, Q, Z


def _qz_complex_nosort_eig(A, B, overwrite_a=False, overwrite_b=False):
    S, T, Q, Z = scipy_linalg.qz(
        A,
        B,
        output="complex",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )

    # For complex Schur form, eigenvalues are simply the diagonal elements
    alpha = np.diag(S)
    beta = np.diag(T)
    return S, T, alpha, beta, Q, Z


def _qz_complex_sort_noeig(A, B, sort, overwrite_a=False, overwrite_b=False):
    S, T, _, _, Q, Z = scipy_linalg.ordqz(
        A,
        B,
        sort=sort,
        output="complex",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, Q, Z


def _qz_complex_sort_eig(A, B, sort, overwrite_a=False, overwrite_b=False):
    S, T, alpha, beta, Q, Z = scipy_linalg.ordqz(
        A,
        B,
        sort=sort,
        output="complex",
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
    )
    return S, T, alpha, beta, Q, Z


@overload(_qz_real_nosort_noeig)
def qz_real_nosort_noeig_impl(A, B, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Float,), func_name="qz")

    dtype = A.dtype
    numba_gges = _LAPACK().numba_xgges(dtype)

    def impl(A, B, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHAR = np.empty(_N, dtype=dtype)
        ALPHAI = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, VSL.T, VSR.T

    return impl


@overload(_qz_real_nosort_eig)
def qz_real_nosort_eig_impl(A, B, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Float,), func_name="qz")

    dtype = A.dtype
    numba_gges = _LAPACK().numba_xgges(dtype)

    def impl(A, B, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHAR = np.empty(_N, dtype=dtype)
        ALPHAI = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        alpha = ALPHAR + 1j * ALPHAI
        return A_copy, B_copy, alpha, BETA, VSL.T, VSR.T

    return impl


@overload(_qz_real_sort_noeig)
def qz_real_sort_noeig_impl(A, B, sort, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Float,), func_name="qz")

    dtype = A.dtype
    numba_gges = _LAPACK().numba_xgges(dtype)
    numba_tgsen = _LAPACK().numba_tgsen(dtype)

    def impl(A, B, sort, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHAR = np.empty(_N, dtype=dtype)
        ALPHAI = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query for gges
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual gges call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan
            return A_copy, B_copy, VSL.T, VSR.T

        # Apply sorting via tgsen
        alpha = ALPHAR + 1j * ALPHAI

        if sort == "lhp":
            select = _lhp(alpha, BETA)
        elif sort == "rhp":
            select = _rhp(alpha, BETA)
        elif sort == "iuc":
            select = _iuc(alpha, BETA)
        else:  # ouc
            select = _ouc(alpha, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        LDQ = val_to_int_ptr(_N)
        LDZ = val_to_int_ptr(_N)
        M = val_to_int_ptr(0)
        PL = np.empty(1, dtype=dtype)
        PR = np.empty(1, dtype=dtype)
        DIF = np.empty(2, dtype=dtype)
        TGSEN_LWORK = val_to_int_ptr(4 * _N + 16)
        TGSEN_WORK = np.empty(4 * _N + 16, dtype=dtype)
        LIWORK = val_to_int_ptr(1)
        IWORK = np.empty(1, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_tgsen(
            IJOB,
            WANTQ,
            WANTZ,
            select.ctypes,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDQ,
            VSR.ctypes,
            LDZ,
            M,
            PL.ctypes,
            PR.ctypes,
            DIF.ctypes,
            TGSEN_WORK.ctypes,
            TGSEN_LWORK,
            IWORK.ctypes,
            LIWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, VSL.T, VSR.T

    return impl


@overload(_qz_real_sort_eig)
def qz_real_sort_eig_impl(A, B, sort, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Float,), func_name="qz")

    dtype = A.dtype
    numba_gges = _LAPACK().numba_xgges(dtype)
    numba_tgsen = _LAPACK().numba_tgsen(dtype)

    def impl(A, B, sort, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHAR = np.empty(_N, dtype=dtype)
        ALPHAI = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query for gges
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual gges call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan
            alpha = ALPHAR + 1j * ALPHAI
            return A_copy, B_copy, alpha, BETA, VSL.T, VSR.T

        # Apply sorting via tgsen
        alpha = ALPHAR + 1j * ALPHAI

        if sort == "lhp":
            select = _lhp(alpha, BETA)
        elif sort == "rhp":
            select = _rhp(alpha, BETA)
        elif sort == "iuc":
            select = _iuc(alpha, BETA)
        else:  # ouc
            select = _ouc(alpha, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        LDQ = val_to_int_ptr(_N)
        LDZ = val_to_int_ptr(_N)
        M = val_to_int_ptr(0)
        PL = np.empty(1, dtype=dtype)
        PR = np.empty(1, dtype=dtype)
        DIF = np.empty(2, dtype=dtype)
        TGSEN_LWORK = val_to_int_ptr(4 * _N + 16)
        TGSEN_WORK = np.empty(4 * _N + 16, dtype=dtype)
        LIWORK = val_to_int_ptr(1)
        IWORK = np.empty(1, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_tgsen(
            IJOB,
            WANTQ,
            WANTZ,
            select.ctypes,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            ALPHAR.ctypes,
            ALPHAI.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDQ,
            VSR.ctypes,
            LDZ,
            M,
            PL.ctypes,
            PR.ctypes,
            DIF.ctypes,
            TGSEN_WORK.ctypes,
            TGSEN_LWORK,
            IWORK.ctypes,
            LIWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        # Recompute alpha after tgsen
        alpha = ALPHAR + 1j * ALPHAI
        return A_copy, B_copy, alpha, BETA, VSL.T, VSR.T

    return impl


@overload(_qz_complex_nosort_noeig)
def qz_complex_nosort_noeig_impl(A, B, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Complex,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Complex,), func_name="qz")

    dtype = A.dtype
    real_dtype = _get_underlying_float(dtype)
    numba_gges = _LAPACK().numba_xgges(dtype)

    def impl(A, B, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHA = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        RWORK = np.empty(8 * _N, dtype=real_dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, VSL.T, VSR.T

    return impl


@overload(_qz_complex_nosort_eig)
def qz_complex_nosort_eig_impl(A, B, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Complex,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Complex,), func_name="qz")

    dtype = A.dtype
    real_dtype = _get_underlying_float(dtype)
    numba_gges = _LAPACK().numba_xgges(dtype)

    def impl(A, B, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHA = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        RWORK = np.empty(8 * _N, dtype=real_dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, ALPHA, BETA, VSL.T, VSR.T

    return impl


@overload(_qz_complex_sort_noeig)
def qz_complex_sort_noeig_impl(A, B, sort, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Complex,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Complex,), func_name="qz")

    dtype = A.dtype
    real_dtype = _get_underlying_float(dtype)
    numba_gges = _LAPACK().numba_xgges(dtype)
    numba_tgsen = _LAPACK().numba_tgsen(dtype)

    def impl(A, B, sort, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHA = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        RWORK = np.empty(8 * _N, dtype=real_dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query for gges
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual gges call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan
            return A_copy, B_copy, VSL.T, VSR.T

        # Apply sorting via tgsen
        if sort == "lhp":
            select = _lhp(ALPHA, BETA)
        elif sort == "rhp":
            select = _rhp(ALPHA, BETA)
        elif sort == "iuc":
            select = _iuc(ALPHA, BETA)
        else:  # ouc
            select = _ouc(ALPHA, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        LDQ = val_to_int_ptr(_N)
        LDZ = val_to_int_ptr(_N)
        M = val_to_int_ptr(0)
        PL = np.empty(1, dtype=real_dtype)
        PR = np.empty(1, dtype=real_dtype)
        DIF = np.empty(2, dtype=real_dtype)
        TGSEN_LWORK = val_to_int_ptr(1)
        TGSEN_WORK = np.empty(1, dtype=dtype)
        LIWORK = val_to_int_ptr(1)
        IWORK = np.empty(1, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_tgsen(
            IJOB,
            WANTQ,
            WANTZ,
            select.ctypes,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDQ,
            VSR.ctypes,
            LDZ,
            M,
            PL.ctypes,
            PR.ctypes,
            DIF.ctypes,
            TGSEN_WORK.ctypes,
            TGSEN_LWORK,
            IWORK.ctypes,
            LIWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, VSL.T, VSR.T

    return impl


@overload(_qz_complex_sort_eig)
def qz_complex_sort_eig_impl(A, B, sort, overwrite_a, overwrite_b):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Complex,), func_name="qz")
    _check_linalg_matrix(B, ndim=2, dtype=(Complex,), func_name="qz")

    dtype = A.dtype
    real_dtype = _get_underlying_float(dtype)
    numba_gges = _LAPACK().numba_xgges(dtype)
    numba_tgsen = _LAPACK().numba_tgsen(dtype)

    def impl(A, B, sort, overwrite_a, overwrite_b):
        _N = np.int32(A.shape[-1])

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order(B)

        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)

        JOBVSL = val_to_int_ptr(ord("V"))
        JOBVSR = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)
        ALPHA = np.empty(_N, dtype=dtype)
        BETA = np.empty(_N, dtype=dtype)
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)
        RWORK = np.empty(8 * _N, dtype=real_dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(0)

        # Workspace query for gges
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual call
        numba_gges(
            JOBVSL,
            JOBVSR,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            SDIM,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDVSL,
            VSR.ctypes,
            LDVSR,
            WORK.ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan
            return A_copy, B_copy, ALPHA, BETA, VSL.T, VSR.T

        # Apply sorting via tgsen
        if sort == "lhp":
            select = _lhp(ALPHA, BETA)
        elif sort == "rhp":
            select = _rhp(ALPHA, BETA)
        elif sort == "iuc":
            select = _iuc(ALPHA, BETA)
        else:  # ouc
            select = _ouc(ALPHA, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        LDQ = val_to_int_ptr(_N)
        LDZ = val_to_int_ptr(_N)
        M = val_to_int_ptr(0)
        PL = np.empty(1, dtype=real_dtype)
        PR = np.empty(1, dtype=real_dtype)
        DIF = np.empty(2, dtype=real_dtype)
        TGSEN_LWORK = val_to_int_ptr(1)
        TGSEN_WORK = np.empty(1, dtype=dtype)
        LIWORK = val_to_int_ptr(1)
        IWORK = np.empty(1, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_tgsen(
            IJOB,
            WANTQ,
            WANTZ,
            select.ctypes,
            N,
            A_copy.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            ALPHA.ctypes,
            BETA.ctypes,
            VSL.ctypes,
            LDQ,
            VSR.ctypes,
            LDZ,
            M,
            PL.ctypes,
            PR.ctypes,
            DIF.ctypes,
            TGSEN_WORK.ctypes,
            TGSEN_LWORK,
            IWORK.ctypes,
            LIWORK,
            INFO,
        )

        if int_ptr_to_val(INFO) != 0:
            A_copy[:] = np.nan
            B_copy[:] = np.nan
            VSL[:] = np.nan
            VSR[:] = np.nan

        return A_copy, B_copy, ALPHA, BETA, VSL.T, VSR.T

    return impl
