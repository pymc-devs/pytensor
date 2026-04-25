import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix


def _eigh(a, UPLO, overwrite_a=False):
    """Placeholder for the overloaded eigh implementation."""
    is_complex = np.issubdtype(a.dtype, np.complexfloating)
    driver = "heevr" if is_complex else "syevr"
    (syevr,) = linalg.get_lapack_funcs((driver,), (a,))
    w, v, _m, _isuppz, info = syevr(
        a, compute_v=1, lower=int(UPLO == 0), overwrite_a=int(overwrite_a)
    )
    if info != 0:
        return np.full(a.shape[0], np.nan, dtype=w.dtype), np.full(
            a.shape, np.nan, dtype=v.dtype
        )
    return w, v


@overload(_eigh)
def eigh_impl(A, UPLO, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="eigh")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)

    if is_complex:
        numba_heevr = _LAPACK().numba_xheevr(dtype)
        real_dtype = _get_underlying_float(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            # syevr/heevr scratch the input triangle of A (Householder reflectors) and read them back during the
            # back-transform, so A and Z must remain distinct buffers. When A is donated and f-contig we reuse it as the
            # LAPACK input/scratch buffer and still allocate Z separately.
            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("V"))
            RANGE = val_to_int_ptr(ord("A"))
            UPLO_ptr = val_to_int_ptr(ord("L") if UPLO == 0 else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            VL = np.empty(1, dtype=real_dtype)
            VU = np.empty(1, dtype=real_dtype)
            IL = val_to_int_ptr(np.int32(0))
            IU = val_to_int_ptr(np.int32(0))
            ABSTOL = np.empty(1, dtype=real_dtype)
            ABSTOL[0] = 0.0
            M = val_to_int_ptr(np.int32(0))
            W = np.empty(_N, dtype=real_dtype)

            Z = np.asfortranarray(np.empty((_N, _N), dtype=A_copy.dtype))

            LDZ = val_to_int_ptr(max(np.int32(1), _N))
            ISUPPZ = np.empty(2 * _N, dtype=np.int32)
            INFO = val_to_int_ptr(0)

            # Workspace query
            LWORK = val_to_int_ptr(np.int32(-1))
            LRWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            RWORK = np.empty(1, dtype=real_dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_heevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            lrwork = np.int32(RWORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LRWORK = val_to_int_ptr(lrwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_heevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                Z[:] = np.nan

            return W, Z

    else:
        numba_syevr = _LAPACK().numba_xsyevr(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            # See the complex branch above for why A and Z must stay distinct.
            # For real symmetric A, a c-contig buffer reinterpreted as f-contig equals A^T == A, so we can reuse a
            # c-contig donated A by flipping UPLO. Z is allocated separately, so V lands in it correctly.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                if A.flags.c_contiguous:
                    lower = not lower
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("V"))
            RANGE = val_to_int_ptr(ord("A"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            VL = np.empty(1, dtype=A_copy.dtype)
            VU = np.empty(1, dtype=A_copy.dtype)
            IL = val_to_int_ptr(np.int32(0))
            IU = val_to_int_ptr(np.int32(0))
            ABSTOL = np.empty(1, dtype=A_copy.dtype)
            ABSTOL[0] = 0.0
            M = val_to_int_ptr(np.int32(0))
            W = np.empty(_N, dtype=A_copy.dtype)

            Z = np.asfortranarray(np.empty((_N, _N), dtype=A_copy.dtype))

            LDZ = val_to_int_ptr(max(np.int32(1), _N))
            ISUPPZ = np.empty(2 * _N, dtype=np.int32)
            INFO = val_to_int_ptr(0)

            # Workspace query
            LWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_syevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_syevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                Z[:] = np.nan

            return W, Z

    return impl


def _eigh_evd(a, UPLO, overwrite_a=False):
    """Placeholder for the overloaded eigh (divide-and-conquer) implementation."""
    try:
        return linalg.eigh(a, lower=(UPLO == 0), check_finite=False, driver="evd")
    except np.linalg.LinAlgError:
        if np.issubdtype(a.dtype, np.complexfloating):
            real_dtype = np.finfo(a.dtype).dtype
        else:
            real_dtype = a.dtype
        return (
            np.full(a.shape[0], np.nan, dtype=real_dtype),
            np.full(a.shape, np.nan, dtype=a.dtype),
        )


@overload(_eigh_evd)
def eigh_evd_impl(A, UPLO, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="eigh")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)

    if is_complex:
        numba_heevd = _LAPACK().numba_xheevd(dtype)
        real_dtype = _get_underlying_float(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("V"))
            UPLO_ptr = val_to_int_ptr(ord("L") if UPLO == 0 else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=real_dtype)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            LRWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            RWORK = np.empty(1, dtype=real_dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_heevd(
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            lrwork = np.int32(RWORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LRWORK = val_to_int_ptr(lrwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_heevd(
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                A_copy[:] = np.nan

            return W, A_copy

    else:
        numba_syevd = _LAPACK().numba_xsyevd(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            # syevd writes V back into A's buffer. For real symmetric A a c-contig buffer reinterpreted
            # as f-contig equals A, so we can reuse a c-contig donated A by flipping UPLO. The eigenvectors
            # LAPACK writes back are in column-major stride within the buffer; the numpy view is c-contig,
            # so we transpose-on-return (zero-cost stride swap) to expose them as the f-order V the rest of the
            # dispatch expects.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                flip = A.flags.c_contiguous
            else:
                A_copy = _copy_to_fortran_order(A)
                flip = False

            if flip:
                lower = not lower

            JOBZ = val_to_int_ptr(ord("V"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=A_copy.dtype)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_syevd(
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_syevd(
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                A_copy[:] = np.nan

            if flip:
                return W, A_copy.T
            return W, A_copy

    return impl


def _eigh_generalized(a, b, UPLO, overwrite_a=False, overwrite_b=False):
    """Placeholder for the overloaded generalized eigh implementation."""
    try:
        return linalg.eigh(
            a,
            b=b,
            lower=(UPLO == 0),
            check_finite=False,
            overwrite_a=bool(overwrite_a),
            overwrite_b=bool(overwrite_b),
        )
    except np.linalg.LinAlgError:
        w_dtype = np.result_type(a.dtype, b.dtype)
        if np.issubdtype(w_dtype, np.complexfloating):
            real_dtype = np.finfo(w_dtype).dtype
        else:
            real_dtype = w_dtype
        return (
            np.full(a.shape[0], np.nan, dtype=real_dtype),
            np.full(a.shape, np.nan, dtype=w_dtype),
        )


@overload(_eigh_generalized)
def eigh_generalized_impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
    ensure_lapack()
    _check_linalg_matrix(
        A, ndim=2, dtype=(Float, Complex), func_name="eigh_generalized"
    )
    _check_linalg_matrix(
        B, ndim=2, dtype=(Float, Complex), func_name="eigh_generalized"
    )
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)

    if is_complex:
        numba_hegvd = _LAPACK().numba_xhegvd(dtype)
        real_dtype = _get_underlying_float(dtype)

        def impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")
            if B.shape[-1] != _N or B.shape[-2] != _N:
                raise np.linalg.LinAlgError("A and B must have the same shape")

            # hegvd writes eigenvectors back into A, so when overwrite_a is set and A is f-contig we route the
            # LAPACK output into A's buffer directly. The c-contig flip from _eigvalsh_generalized doesn't
            # apply here: returning eigenvectors of A^T (== conj(V) for Hermitian) would require a conjugate-on-return
            # that defeats the alloc savings. B has no such issue since its post-call content is discarded scratch
            # (Cholesky factor).
            if overwrite_a and A.flags.f_contiguous:
                A_copy = A
            else:
                A_copy = _copy_to_fortran_order(A)

            if overwrite_b and B.flags.f_contiguous:
                B_copy = B
            else:
                B_copy = _copy_to_fortran_order(B)

            ITYPE = val_to_int_ptr(np.int32(1))
            JOBZ = val_to_int_ptr(ord("V"))
            UPLO_ptr = val_to_int_ptr(ord("L") if UPLO == 0 else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=real_dtype)
            INFO = val_to_int_ptr(0)

            # scipy's f2py wrapper skips the workspace query for the *gvd family and uses the LAPACK-documented
            # minima; MKL's hegvd workspace query can leave WORK[0] unset, so we do the same here.
            lwork = np.int32(max(np.int32(1), 2 * _N + _N * _N))
            lrwork = np.int32(max(np.int32(1), 1 + 5 * _N + 2 * _N * _N))
            liwork = np.int32(max(np.int32(1), 3 + 5 * _N))
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LRWORK = val_to_int_ptr(lrwork)
            LIWORK = val_to_int_ptr(liwork)

            numba_hegvd(
                ITYPE,
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                B_copy.ctypes,
                LDB,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                A_copy[:] = np.nan

            return W, A_copy

    else:
        numba_sygvd = _LAPACK().numba_xsygvd(dtype)

        def impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")
            if B.shape[-1] != _N or B.shape[-2] != _N:
                raise np.linalg.LinAlgError("A and B must have the same shape")

            # Real symmetric: c-contig A reinterpreted as f-contig equals A, so a c-contig donated A can be reused via
            # a UPLO flip. sygvd writes V back into A's buffer in column-major stride; the numpy view is still c-contig
            # labeled, so we transpose-on-return (zero-cost stride swap) to expose V in f-order. B's matched
            # layout requirement (c-contig when A flipped, f-contig otherwise) lets B also stay in-place; B's post-call
            # content is discarded scratch (Cholesky factor) so its layout doesn't propagate.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                flip = A.flags.c_contiguous
            else:
                A_copy = _copy_to_fortran_order(A)
                flip = False

            if overwrite_b and (
                (flip and B.flags.c_contiguous) or (not flip and B.flags.f_contiguous)
            ):
                B_copy = B
            else:
                B_copy = _copy_to_fortran_order(B)

            if flip:
                lower = not lower

            ITYPE = val_to_int_ptr(np.int32(1))
            JOBZ = val_to_int_ptr(ord("V"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=A_copy.dtype)
            INFO = val_to_int_ptr(0)

            # scipy's f2py wrapper skips the workspace query for the *gvd family
            # and uses the LAPACK-documented minima; MKL's sygvd workspace query
            # can leave WORK[0] unset, so we do the same here.
            lwork = np.int32(max(np.int32(1), 1 + 6 * _N + 2 * _N * _N))
            liwork = np.int32(max(np.int32(1), 3 + 5 * _N))
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LIWORK = val_to_int_ptr(liwork)

            numba_sygvd(
                ITYPE,
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                B_copy.ctypes,
                LDB,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan
                A_copy[:] = np.nan

            if flip:
                return W, A_copy.T
            return W, A_copy

    return impl


def _eigvalsh(a, UPLO, overwrite_a=False):
    """Placeholder for the overloaded eigvalsh implementation (JOBZ='N')."""
    is_complex = np.issubdtype(a.dtype, np.complexfloating)
    driver = "heevr" if is_complex else "syevr"
    (syevr,) = linalg.get_lapack_funcs((driver,), (a,))
    w, _v, _m, _isuppz, info = syevr(
        a, compute_v=0, lower=int(UPLO == 0), overwrite_a=int(overwrite_a)
    )
    if info != 0:
        return np.full(a.shape[0], np.nan, dtype=w.dtype)
    return w


@overload(_eigvalsh)
def eigvalsh_impl(A, UPLO, overwrite_a=False):
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="eigvalsh")
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)

    if is_complex:
        numba_heevr = _LAPACK().numba_xheevr(dtype)
        real_dtype = _get_underlying_float(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            # For c-contig Hermitian A, LAPACK sees conj(A) if we reuse the
            # memory. Eigenvalues of conj(A) equal those of A (both real), so
            # only W is returned — no output fixup, just flip UPLO to read
            # the user's valid triangle.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                if A.flags.c_contiguous:
                    lower = not lower
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("N"))
            RANGE = val_to_int_ptr(ord("A"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            VL = np.empty(1, dtype=real_dtype)
            VU = np.empty(1, dtype=real_dtype)
            IL = val_to_int_ptr(np.int32(0))
            IU = val_to_int_ptr(np.int32(0))
            ABSTOL = np.empty(1, dtype=real_dtype)
            ABSTOL[0] = 0.0
            M = val_to_int_ptr(np.int32(0))
            W = np.empty(_N, dtype=real_dtype)

            # JOBZ='N': Z and ISUPPZ are not referenced, but still need valid pointers.
            Z = np.empty(1, dtype=A_copy.dtype)
            LDZ = val_to_int_ptr(np.int32(1))
            ISUPPZ = np.empty(1, dtype=np.int32)
            INFO = val_to_int_ptr(0)

            # Workspace query
            LWORK = val_to_int_ptr(np.int32(-1))
            LRWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            RWORK = np.empty(1, dtype=real_dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_heevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0].real)
            lrwork = np.int32(RWORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LRWORK = val_to_int_ptr(lrwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_heevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan

            return W

    else:
        numba_syevr = _LAPACK().numba_xsyevr(dtype)

        def impl(A, UPLO, overwrite_a=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")

            # A c-contiguous symmetric matrix is identical to its f-contiguous
            # transpose, which equals itself with UPLO swapped.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                if A.flags.c_contiguous:
                    lower = not lower
            else:
                A_copy = _copy_to_fortran_order(A)

            JOBZ = val_to_int_ptr(ord("N"))
            RANGE = val_to_int_ptr(ord("A"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            VL = np.empty(1, dtype=A_copy.dtype)
            VU = np.empty(1, dtype=A_copy.dtype)
            IL = val_to_int_ptr(np.int32(0))
            IU = val_to_int_ptr(np.int32(0))
            ABSTOL = np.empty(1, dtype=A_copy.dtype)
            ABSTOL[0] = 0.0
            M = val_to_int_ptr(np.int32(0))
            W = np.empty(_N, dtype=A_copy.dtype)

            # JOBZ='N': Z and ISUPPZ are not referenced, but still need valid pointers.
            Z = np.empty(1, dtype=A_copy.dtype)
            LDZ = val_to_int_ptr(np.int32(1))
            ISUPPZ = np.empty(1, dtype=np.int32)
            INFO = val_to_int_ptr(0)

            LWORK = val_to_int_ptr(np.int32(-1))
            LIWORK = val_to_int_ptr(np.int32(-1))
            WORK = np.empty(1, dtype=A_copy.dtype)
            IWORK = np.empty(1, dtype=np.int32)

            numba_syevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            lwork = np.int32(WORK[0])
            liwork = IWORK[0]
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LIWORK = val_to_int_ptr(liwork)
            INFO = val_to_int_ptr(0)

            numba_syevr(
                JOBZ,
                RANGE,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                VL.ctypes,
                VU.ctypes,
                IL,
                IU,
                ABSTOL.ctypes,
                M,
                W.ctypes,
                Z.ctypes,
                LDZ,
                ISUPPZ.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan

            return W

    return impl


def _eigvalsh_generalized(a, b, UPLO, overwrite_a=False, overwrite_b=False):
    """Placeholder for the overloaded generalized eigvalsh implementation (JOBZ='N')."""
    try:
        return linalg.eigvalsh(
            a,
            b=b,
            lower=(UPLO == 0),
            check_finite=False,
            overwrite_a=bool(overwrite_a),
            overwrite_b=bool(overwrite_b),
        )
    except np.linalg.LinAlgError:
        w_dtype = np.result_type(a.dtype, b.dtype)
        if np.issubdtype(w_dtype, np.complexfloating):
            real_dtype = np.finfo(w_dtype).dtype
        else:
            real_dtype = w_dtype
        return np.full(a.shape[0], np.nan, dtype=real_dtype)


@overload(_eigvalsh_generalized)
def eigvalsh_generalized_impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
    ensure_lapack()
    _check_linalg_matrix(
        A, ndim=2, dtype=(Float, Complex), func_name="eigvalsh_generalized"
    )
    _check_linalg_matrix(
        B, ndim=2, dtype=(Float, Complex), func_name="eigvalsh_generalized"
    )
    dtype = A.dtype
    is_complex = isinstance(dtype, Complex)

    if is_complex:
        numba_hegvd = _LAPACK().numba_xhegvd(dtype)
        real_dtype = _get_underlying_float(dtype)

        def impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")
            if B.shape[-1] != _N or B.shape[-2] != _N:
                raise np.linalg.LinAlgError("A and B must have the same shape")

            # A (Hermitian/symmetric) c-contig stored is equivalent to f-contig stored
            # with UPLO flipped. The same flip must apply to B so we only reuse B if
            # its layout matches whichever direction A picks.
            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                flip = A.flags.c_contiguous
            else:
                A_copy = _copy_to_fortran_order(A)
                flip = False

            if overwrite_b and (
                (flip and B.flags.c_contiguous) or (not flip and B.flags.f_contiguous)
            ):
                B_copy = B
            else:
                B_copy = _copy_to_fortran_order(B)

            if flip:
                lower = not lower

            ITYPE = val_to_int_ptr(np.int32(1))
            JOBZ = val_to_int_ptr(ord("N"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=real_dtype)
            INFO = val_to_int_ptr(0)

            # hegvd JOBZ='N' has documented fixed minimum workspace; scipy uses this instead
            # of a workspace call (bug in some LAPACKs)
            lwork = np.int32(max(np.int32(1), _N + 1))
            lrwork = np.int32(max(np.int32(1), _N))
            liwork = np.int32(1)
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            RWORK = np.empty(lrwork, dtype=real_dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LRWORK = val_to_int_ptr(lrwork)
            LIWORK = val_to_int_ptr(liwork)

            numba_hegvd(
                ITYPE,
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                B_copy.ctypes,
                LDB,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                RWORK.ctypes,
                LRWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan

            return W

    else:
        numba_sygvd = _LAPACK().numba_xsygvd(dtype)

        def impl(A, B, UPLO, overwrite_a=False, overwrite_b=False):
            _N = np.int32(A.shape[-1])
            if A.shape[-2] != _N:
                raise np.linalg.LinAlgError("Last 2 dimensions of A must be square")
            if B.shape[-1] != _N or B.shape[-2] != _N:
                raise np.linalg.LinAlgError("A and B must have the same shape")

            lower = UPLO == 0
            if overwrite_a and (A.flags.f_contiguous or A.flags.c_contiguous):
                A_copy = A
                flip = A.flags.c_contiguous
            else:
                A_copy = _copy_to_fortran_order(A)
                flip = False

            if overwrite_b and (
                (flip and B.flags.c_contiguous) or (not flip and B.flags.f_contiguous)
            ):
                B_copy = B
            else:
                B_copy = _copy_to_fortran_order(B)

            if flip:
                lower = not lower

            ITYPE = val_to_int_ptr(np.int32(1))
            JOBZ = val_to_int_ptr(ord("N"))
            UPLO_ptr = val_to_int_ptr(ord("L") if lower else ord("U"))
            N = val_to_int_ptr(_N)
            LDA = val_to_int_ptr(_N)
            LDB = val_to_int_ptr(_N)
            W = np.empty(_N, dtype=A_copy.dtype)
            INFO = val_to_int_ptr(0)

            # sygvd JOBZ='N' doc minima; see _eigh_generalized for why we skip
            # the workspace query for the *gvd family.
            lwork = np.int32(max(np.int32(1), 2 * _N + 1))
            liwork = np.int32(1)
            WORK = np.empty(lwork, dtype=A_copy.dtype)
            IWORK = np.empty(liwork, dtype=np.int32)
            LWORK = val_to_int_ptr(lwork)
            LIWORK = val_to_int_ptr(liwork)

            numba_sygvd(
                ITYPE,
                JOBZ,
                UPLO_ptr,
                N,
                A_copy.ctypes,
                LDA,
                B_copy.ctypes,
                LDB,
                W.ctypes,
                WORK.ctypes,
                LWORK,
                IWORK.ctypes,
                LIWORK,
                INFO,
            )

            if int_ptr_to_val(INFO) != 0:
                W[:] = np.nan

            return W

    return impl
