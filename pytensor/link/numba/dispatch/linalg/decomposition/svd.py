import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


def _svd_gesvd_full(a, full_matrices=True):
    """Placeholder; overloaded with a direct xGESVD dispatch."""
    return np.linalg.svd(a, full_matrices=full_matrices)


def _svd_gesvd_no_uv(a):
    """Placeholder; overloaded with xGESVD computing singular values only."""
    return np.linalg.svd(a, compute_uv=False)


def _svd_gesdd_full(a, full_matrices=True, overwrite_a=False):
    """Placeholder; overloaded with a direct xGESDD dispatch."""
    return np.linalg.svd(a, full_matrices=full_matrices)


def _svd_gesdd_no_uv(a, overwrite_a=False):
    """Placeholder; overloaded with xGESDD computing singular values only."""
    return np.linalg.svd(a, compute_uv=False)


@overload(_svd_gesvd_full)
def svd_gesvd_full_impl(A, full_matrices=True):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="svd")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)
    real_dtype = _get_underlying_float(dtype) if is_complex else None
    numba_gesvd = _LAPACK().numba_xgesvd(dtype)

    if is_complex:

        def impl(A, full_matrices=True):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            A_copy = _copy_to_fortran_order(A)

            if full_matrices:
                JOBU = val_to_int_ptr(ord("A"))
                JOBVT = val_to_int_ptr(ord("A"))
                U = np.empty((_M, _M), dtype=A.dtype).T
                VT = np.empty((_N, _N), dtype=A.dtype).T
            else:
                JOBU = val_to_int_ptr(ord("S"))
                JOBVT = val_to_int_ptr(ord("S"))
                U = np.empty((_K, _M), dtype=A.dtype).T
                VT = np.empty((_N, _K), dtype=A.dtype).T

            S = np.empty(_K, dtype=real_dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(max(np.int32(1), _M))
            LDVT = val_to_int_ptr(max(np.int32(1), np.int32(VT.shape[0])))

            RWORK = np.empty(max(np.int32(1), 5 * _K), dtype=real_dtype)
            INFO = val_to_int_ptr(0)

            # Workspace query
            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan
                U[:] = np.nan
                VT[:] = np.nan

            return U, S, VT

    else:

        def impl(A, full_matrices=True):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            A_copy = _copy_to_fortran_order(A)

            if full_matrices:
                JOBU = val_to_int_ptr(ord("A"))
                JOBVT = val_to_int_ptr(ord("A"))
                U = np.empty((_M, _M), dtype=A.dtype).T
                VT = np.empty((_N, _N), dtype=A.dtype).T
            else:
                JOBU = val_to_int_ptr(ord("S"))
                JOBVT = val_to_int_ptr(ord("S"))
                U = np.empty((_K, _M), dtype=A.dtype).T
                VT = np.empty((_N, _K), dtype=A.dtype).T

            S = np.empty(_K, dtype=A.dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(max(np.int32(1), _M))
            LDVT = val_to_int_ptr(max(np.int32(1), np.int32(VT.shape[0])))
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                INFO,
            )

            lwork = np.int32(WORK[0])
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan
                U[:] = np.nan
                VT[:] = np.nan

            return U, S, VT

    return impl


@overload(_svd_gesvd_no_uv)
def svd_gesvd_no_uv_impl(A):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="svd")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)
    real_dtype = _get_underlying_float(dtype) if is_complex else None
    numba_gesvd = _LAPACK().numba_xgesvd(dtype)

    if is_complex:

        def impl(A):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            A_copy = _copy_to_fortran_order(A)

            JOBU = val_to_int_ptr(ord("N"))
            JOBVT = val_to_int_ptr(ord("N"))
            # JOBU='N' / JOBVT='N': U and VT are not referenced but still need
            # valid pointers and LDU/LDVT >= 1.
            U = np.empty(1, dtype=A.dtype)
            VT = np.empty(1, dtype=A.dtype)
            S = np.empty(_K, dtype=real_dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(np.int32(1))
            LDVT = val_to_int_ptr(np.int32(1))
            RWORK = np.empty(max(np.int32(1), 5 * _K), dtype=real_dtype)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan

            return S

    else:

        def impl(A):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            A_copy = _copy_to_fortran_order(A)

            JOBU = val_to_int_ptr(ord("N"))
            JOBVT = val_to_int_ptr(ord("N"))
            U = np.empty(1, dtype=A.dtype)
            VT = np.empty(1, dtype=A.dtype)
            S = np.empty(_K, dtype=A.dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(np.int32(1))
            LDVT = val_to_int_ptr(np.int32(1))
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                INFO,
            )

            lwork = np.int32(WORK[0])
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesvd(
                JOBU,
                JOBVT,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan

            return S

    return impl


@overload(_svd_gesdd_full)
def svd_gesdd_full_impl(A, full_matrices=True, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="svd")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)
    real_dtype = _get_underlying_float(dtype) if is_complex else None
    numba_gesdd = _LAPACK().numba_xgesdd(dtype)

    if is_complex:

        def impl(A, full_matrices=True, overwrite_a=False):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)
            _MAX = max(_M, _N)

            # gesdd uses A as scratch and clobbers it regardless of JOBZ, so
            # when the caller donates an f-contig A we reuse the buffer; the
            # post-call contents are meaningless but the alloc is saved.
            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            else:
                A_copy = _copy_to_fortran_order(A)

            if full_matrices:
                JOBZ = val_to_int_ptr(ord("A"))
                U = np.empty((_M, _M), dtype=A.dtype).T
                VT = np.empty((_N, _N), dtype=A.dtype).T
            else:
                JOBZ = val_to_int_ptr(ord("S"))
                U = np.empty((_K, _M), dtype=A.dtype).T
                VT = np.empty((_N, _K), dtype=A.dtype).T

            S = np.empty(_K, dtype=real_dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(max(np.int32(1), _M))
            LDVT = val_to_int_ptr(max(np.int32(1), np.int32(VT.shape[0])))

            # gesdd RWORK sizing (complex, JOBZ != 'N'): LAPACK doc minimum is
            #   max(1, mn * max(5*mn + 7, 2*mx + 2*mn + 1))
            # gesdd has no LRWORK argument and the WORK query does not return
            # an RWORK size, so the formula is the only way to size RWORK.
            lrwork = np.int32(
                max(
                    np.int32(1),
                    max(
                        5 * _K * _K + 7 * _K,
                        2 * _MAX * _K + 2 * _K * _K + _K,
                    ),
                )
            )
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(max(np.int32(1), 8 * _K), dtype=np.int32)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                IWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                IWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan
                U[:] = np.nan
                VT[:] = np.nan

            return U, S, VT

    else:

        def impl(A, full_matrices=True, overwrite_a=False):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            # Real A: a c-contiguous buffer reinterpreted as fortran-order is
            # A.T (M and N swapped in LAPACK's view). For SVD this lets us
            # solve the swapped (N, M) problem and recover A's SVD by swapping
            # roles: if A = U S Vt, then A.T = V S U.T, so LAPACK's U' maps to
            # Vt.T and LAPACK's Vt' maps to U.T.
            swap = False
            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            elif overwrite_a and A.flags.c_contiguous:
                A_copy = A.T
                swap = True
                _M, _N = _N, _M
            else:
                A_copy = _copy_to_fortran_order(A)

            if full_matrices:
                JOBZ = val_to_int_ptr(ord("A"))
                U = np.empty((_M, _M), dtype=A.dtype).T
                VT = np.empty((_N, _N), dtype=A.dtype).T
            else:
                JOBZ = val_to_int_ptr(ord("S"))
                U = np.empty((_K, _M), dtype=A.dtype).T
                VT = np.empty((_N, _K), dtype=A.dtype).T

            S = np.empty(_K, dtype=A.dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(max(np.int32(1), _M))
            LDVT = val_to_int_ptr(max(np.int32(1), np.int32(VT.shape[0])))
            IWORK = np.empty(max(np.int32(1), 8 * _K), dtype=np.int32)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0])
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan
                U[:] = np.nan
                VT[:] = np.nan

            if swap:
                # Solved SVD of A.T; map back: A's U = (LAPACK's VT).T,
                # A's VT = (LAPACK's U).T. The .T's are zero-cost stride swaps.
                return VT.T, S, U.T
            return U, S, VT

    return impl


@overload(_svd_gesdd_no_uv)
def svd_gesdd_no_uv_impl(A, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="svd")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)
    real_dtype = _get_underlying_float(dtype) if is_complex else None
    numba_gesdd = _LAPACK().numba_xgesdd(dtype)

    if is_complex:

        def impl(A, overwrite_a=False):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("N"))
            U = np.empty(1, dtype=A.dtype)
            VT = np.empty(1, dtype=A.dtype)
            S = np.empty(_K, dtype=real_dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(np.int32(1))
            LDVT = val_to_int_ptr(np.int32(1))

            # gesdd RWORK sizing for JOBZ='N'.
            lrwork = max(np.int32(1), 7 * _K)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(max(np.int32(1), 8 * _K), dtype=np.int32)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                IWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                IWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan

            return S

    else:

        def impl(A, overwrite_a=False):
            _M = np.int32(A.shape[0])
            _N = np.int32(A.shape[1])
            _K = min(_M, _N)

            # Real, JOBZ='N': singular values of A.T equal those of A, so the
            # c-contig-as-f-contig reinterpretation needs no fix-up beyond
            # swapping M and N in the LAPACK call.
            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            elif overwrite_a and A.flags.c_contiguous:
                A_copy = A.T
                _M, _N = _N, _M
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("N"))
            U = np.empty(1, dtype=A.dtype)
            VT = np.empty(1, dtype=A.dtype)
            S = np.empty(_K, dtype=A.dtype)
            M = val_to_int_ptr(_M)
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(max(np.int32(1), _M))
            LDU = val_to_int_ptr(np.int32(1))
            LDVT = val_to_int_ptr(np.int32(1))
            IWORK = np.empty(max(np.int32(1), 8 * _K), dtype=np.int32)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A.dtype)
            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                INFO,
            )

            lwork = np.int32(WORK[0])
            WORK = np.empty(lwork, dtype=A.dtype)
            LWORK = val_to_int_ptr(lwork)
            INFO = val_to_int_ptr(0)

            numba_gesdd(
                JOBZ,
                M,
                N,
                A_copy.ctypes,
                LDA,
                S.ctypes,
                U.ctypes,
                LDU,
                VT.ctypes,
                LDVT,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                S[:] = np.nan

            return S

    return impl
