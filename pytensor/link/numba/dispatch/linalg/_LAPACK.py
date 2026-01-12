import numba
import numpy as np
from numba.core import cgutils, types
from numba.core.extending import get_cython_function_address, intrinsic
from numba.core.registry import CPUDispatcher
from numba.core.types import Complex
from numba.np.linalg import ensure_lapack, get_blas_kind

from pytensor.link.numba.cache import _call_cached_ptr
from pytensor.link.numba.dispatch import basic as numba_basic


nb_i32 = types.int32
nb_i32p = types.CPointer(nb_i32)

nb_f32 = types.float32
nb_f32p = types.CPointer(nb_f32)

nb_f64 = types.float64
nb_f64p = types.CPointer(nb_f64)

nb_c64 = types.complex64
nb_c64p = types.CPointer(nb_c64)

nb_c128 = types.complex128
nb_c128p = types.CPointer(nb_c128)


def get_lapack_ptr(dtype, name):
    d = get_blas_kind(dtype)
    func_name = f"{d}{name}"
    lapack_ptr = get_cython_function_address("scipy.linalg.cython_lapack", func_name)
    return lapack_ptr


def _get_underlying_float(dtype):
    s_dtype = str(dtype)
    out_type = s_dtype
    if s_dtype == "complex64":
        out_type = "float32"
    elif s_dtype == "complex128":
        out_type = "float64"

    return np.dtype(out_type)


def _get_nb_float_from_dtype(blas_dtype, return_pointer=True):
    match blas_dtype:
        case "s":
            return nb_f32p if return_pointer else nb_f32
        case "d":
            return nb_f64p if return_pointer else nb_f64
        case "c":
            return nb_c64p if return_pointer else nb_c64
        case "z":
            return nb_c128p if return_pointer else nb_c128
        case _:
            raise ValueError(f"Unsupported BLAS dtype: {blas_dtype}")


@intrinsic
def sptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.float32(types.CPointer(types.float32))
    return sig, impl


@intrinsic
def dptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.float64(types.CPointer(types.float64))
    return sig, impl


@intrinsic
def int_ptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.int32(types.CPointer(types.int32))
    return sig, impl


@intrinsic
def val_to_int_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.int32)(types.int32)
    return sig, impl


@intrinsic
def val_to_sptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.float32)(types.float32)
    return sig, impl


@intrinsic
def val_to_zptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.complex128)(types.complex128)
    return sig, impl


@intrinsic
def val_to_dptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.float64)(types.float64)
    return sig, impl


class _LAPACK:
    """
    Functions to return type signatures for wrapped LAPACK functions.

    Patterned after https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L74
    """

    def __init__(self):
        ensure_lapack()

    @classmethod
    def numba_xtrtrs(cls, dtype) -> CPUDispatcher:
        """
        Solve a triangular system of equations of the form A @ X = B or A.T @ X = B.

        Called by scipy.linalg.solve_triangular
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}trtrs"

        @numba_basic.numba_njit
        def get_trtrs_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "trtrs")
            return ptr

        trtrs_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # TRANS
                nb_i32p,  # DIAG
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_ptr,  # A
                nb_i32p,  # LDA
                float_ptr,  # B
                nb_i32p,  # LDB
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def trtrs(UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_trtrs_pointer,
                func_type_ref=trtrs_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB, INFO)

        return trtrs

    @classmethod
    def numba_xpotrf(cls, dtype) -> CPUDispatcher:
        """
        Compute the Cholesky factorization of a real symmetric positive definite matrix.

        Called by scipy.linalg.cholesky
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}potrf"

        @numba_basic.numba_njit
        def get_potrf_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "potrf")
            return ptr

        potrf_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                float_ptr,  # A
                nb_i32p,  # LDA
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def potrf(UPLO, N, A, LDA, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_potrf_pointer,
                func_type_ref=potrf_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(UPLO, N, A, LDA, INFO)

        return potrf

    @classmethod
    def numba_xpotrs(cls, dtype) -> CPUDispatcher:
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by numba_potrf.

        Called by scipy.linalg.cho_solve
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}potrs"

        @numba_basic.numba_njit
        def get_potrs_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "potrs")
            return ptr

        potrs_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # B
                nb_i32p,  # LDB
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def potrs(UPLO, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_potrs_pointer,
                func_type_ref=potrs_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(UPLO, N, NRHS, A, LDA, B, LDB, INFO)

        return potrs

    @classmethod
    def numba_xgetrf(cls, dtype) -> CPUDispatcher:
        """
        Compute partial pivoting LU factorization of a general M-by-N matrix A using row interchanges.

        Called by scipy.linalg.lu_factor
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}getrf"

        @numba_basic.numba_njit
        def get_getrf_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "getrf")
            return ptr

        getrf_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # M
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                nb_i32p,  # IPIV
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def getrf(M, N, A, LDA, IPIV, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_getrf_pointer,
                func_type_ref=getrf_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(M, N, A, LDA, IPIV, INFO)

        return getrf

    @classmethod
    def numba_xgetrs(cls, dtype) -> CPUDispatcher:
        """
        Solve a system of linear equations A @ X = B or A.T @ X = B with a general N-by-N matrix A using the LU
        factorization computed by GETRF.

        Called by scipy.linalg.lu_solve
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}getrs"

        @numba_basic.numba_njit
        def get_getrs_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "getrs")
            return ptr

        getrs_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # TRANS
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_pointer,  # A
                nb_i32p,  # LDA
                nb_i32p,  # IPIV
                float_pointer,  # B
                nb_i32p,  # LDB
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def getrs(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_getrs_pointer,
                func_type_ref=getrs_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO)

        return getrs

    @classmethod
    def numba_xsysv(cls, dtype) -> CPUDispatcher:
        """
        Solve a system of linear equations A @ X = B with a symmetric matrix A using the diagonal pivoting method,
        factorizing A into LDL^T or UDU^T form, depending on the value of UPLO

        Called by scipy.linalg.solve when assume_a == "sym"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}sysv"

        @numba_basic.numba_njit
        def get_sysv_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "sysv")
            return ptr

        sysv_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_pointer,  # A
                nb_i32p,  # LDA
                nb_i32p,  # IPIV
                float_pointer,  # B
                nb_i32p,  # LDB
                float_pointer,  # WORK
                nb_i32p,  # LWORK
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def sysv(UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_sysv_pointer,
                func_type_ref=sysv_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO)

        return sysv

    @classmethod
    def numba_xposv(cls, dtype) -> CPUDispatcher:
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by potrf.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}posv"

        @numba_basic.numba_njit
        def get_posv_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "posv")
            return ptr

        posv_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # B
                nb_i32p,  # LDB
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def posv(UPLO, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_posv_pointer,
                func_type_ref=posv_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(UPLO, N, NRHS, A, LDA, B, LDB, INFO)

        return posv

    @classmethod
    def numba_xgttrf(cls, dtype) -> CPUDispatcher:
        """
        Compute the LU factorization of a tridiagonal matrix A using row interchanges.

        Called by scipy.linalg.solve when assume_a == "tri"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}gttrf"

        @numba_basic.numba_njit
        def get_gttrf_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "gttrf")
            return ptr

        gttrf_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # N
                float_pointer,  # DL
                float_pointer,  # D
                float_pointer,  # DU
                float_pointer,  # DU2
                nb_i32p,  # IPIV
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def gttrf(N, DL, D, DU, DU2, IPIV, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gttrf_pointer,
                func_type_ref=gttrf_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(N, DL, D, DU, DU2, IPIV, INFO)

        return gttrf

    @classmethod
    def numba_xgttrs(cls, dtype) -> CPUDispatcher:
        """
        Solve a system of linear equations A @ X = B with a tridiagonal matrix A using the LU factorization computed by numba_gttrf.

        Called by scipy.linalg.solve, when assume_a == "tri"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}gttrs"

        @numba_basic.numba_njit
        def get_gttrs_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "gttrs")
            return ptr

        gttrs_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # TRANS
                nb_i32p,  # N
                nb_i32p,  # NRHS
                float_pointer,  # DL
                float_pointer,  # D
                float_pointer,  # DU
                float_pointer,  # DU2
                nb_i32p,  # IPIV
                float_pointer,  # B
                nb_i32p,  # LDB
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def gttrs(TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gttrs_pointer,
                func_type_ref=gttrs_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB, INFO)

        return gttrs

    @classmethod
    def numba_xgtcon(cls, dtype) -> CPUDispatcher:
        """
        Estimate the reciprocal of the condition number of a tridiagonal matrix A using the LU factorization computed by numba_gttrf.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}gtcon"

        @numba_basic.numba_njit
        def get_gtcon_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "gtcon")
            return ptr

        gtcon_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # NORM
                nb_i32p,  # N
                float_pointer,  # DL
                float_pointer,  # D
                float_pointer,  # DU
                float_pointer,  # DU2
                nb_i32p,  # IPIV
                float_pointer,  # ANORM
                float_pointer,  # RCOND
                float_pointer,  # WORK
                nb_i32p,  # IWORK
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def gtcon(NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, IWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gtcon_pointer,
                func_type_ref=gtcon_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, IWORK, INFO)

        return gtcon

    @classmethod
    def numba_xgeqrf(cls, dtype) -> CPUDispatcher:
        """
        Compute the QR factorization of a general M-by-N matrix A.

        Used in QR decomposition (no pivoting).
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}geqrf"

        @numba_basic.numba_njit
        def get_geqrf_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "geqrf")
            return ptr

        geqrf_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # M
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # TAU
                float_pointer,  # WORK
                nb_i32p,  # LWORK
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def geqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_geqrf_pointer,
                func_type_ref=geqrf_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(M, N, A, LDA, TAU, WORK, LWORK, INFO)

        return geqrf

    @classmethod
    def numba_xgeqp3(cls, dtype) -> CPUDispatcher:
        """
        Compute the QR factorization with column pivoting of a general M-by-N matrix A.

        Used in QR decomposition with pivoting.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}geqp3"

        @numba_basic.numba_njit
        def get_geqp3_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "geqp3")
            return ptr

        if isinstance(dtype, Complex):
            real_pointer = nb_f64p if dtype is nb_c128 else nb_f32p
            geqp3_function_type = types.FunctionType(
                types.void(
                    nb_i32p,  # M
                    nb_i32p,  # N
                    float_pointer,  # A
                    nb_i32p,  # LDA
                    nb_i32p,  # JPVT
                    float_pointer,  # TAU
                    float_pointer,  # WORK
                    nb_i32p,  # LWORK
                    real_pointer,  # RWORK
                    nb_i32p,  # INFO
                )
            )

            @numba_basic.numba_njit
            def geqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, RWORK, INFO):
                fn = _call_cached_ptr(
                    get_ptr_func=get_geqp3_pointer,
                    func_type_ref=geqp3_function_type,
                    unique_func_name_lit=unique_func_name,
                )
                fn(M, N, A, LDA, JPVT, TAU, WORK, LWORK, RWORK, INFO)

        else:
            geqp3_function_type = types.FunctionType(
                types.void(
                    nb_i32p,  # M
                    nb_i32p,  # N
                    float_pointer,  # A
                    nb_i32p,  # LDA
                    nb_i32p,  # JPVT
                    float_pointer,  # TAU
                    float_pointer,  # WORK
                    nb_i32p,  # LWORK
                    nb_i32p,  # INFO
                )
            )

            @numba_basic.numba_njit
            def geqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO):
                fn = _call_cached_ptr(
                    get_ptr_func=get_geqp3_pointer,
                    func_type_ref=geqp3_function_type,
                    unique_func_name_lit=unique_func_name,
                )
                fn(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)

        return geqp3

    @classmethod
    def numba_xorgqr(cls, dtype) -> CPUDispatcher:
        """
        Generate the orthogonal matrix Q from a QR factorization (real types).

        Used in QR decomposition to form Q.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}orgqr"

        @numba_basic.numba_njit
        def get_orgqr_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "orgqr")
            return ptr

        orgqr_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # M
                nb_i32p,  # N
                nb_i32p,  # K
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # TAU
                float_pointer,  # WORK
                nb_i32p,  # LWORK
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def orgqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_orgqr_pointer,
                func_type_ref=orgqr_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(M, N, K, A, LDA, TAU, WORK, LWORK, INFO)

        return orgqr

    @classmethod
    def numba_xungqr(cls, dtype) -> CPUDispatcher:
        """
        Generate the unitary matrix Q from a QR factorization (complex types).

        Used in QR decomposition to form Q for complex types.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}ungqr"

        @numba_basic.numba_njit
        def get_ungqr_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "ungqr")
            return ptr

        ungqr_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # M
                nb_i32p,  # N
                nb_i32p,  # K
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # TAU
                float_pointer,  # WORK
                nb_i32p,  # LWORK
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def ungqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_ungqr_pointer,
                func_type_ref=ungqr_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(M, N, K, A, LDA, TAU, WORK, LWORK, INFO)

        return ungqr

    @classmethod
    def numba_xgees(cls, dtype):
        """
        Compute the eigenvalues and, optionally, the right Schur vectors of a real nonsymmetric matrix A.

        Called by scipy.linalg.schur
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.lapack.{kind}gees"

        @numba_basic.numba_njit
        def get_gees_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "gees")
            return ptr

        if isinstance(dtype, Complex):
            real_pointer = nb_f64p if dtype is nb_c128 else nb_f32p
            gees_function_type = types.FunctionType(
                types.void(
                    nb_i32p,  # JOBVS
                    nb_i32p,  # SORT
                    nb_i32p,  # SELECT
                    nb_i32p,  # N
                    float_pointer,  # A
                    nb_i32p,  # LDA
                    nb_i32p,  # SDIM
                    float_pointer,  # W
                    float_pointer,  # VS
                    nb_i32p,  # LDVS
                    float_pointer,  # WORK
                    nb_i32p,  # LWORK
                    real_pointer,  # RWORK
                    nb_i32p,  # BWORK
                    nb_i32p,  # INFO
                )
            )

            @numba_basic.numba_njit
            def gees(
                JOBVS,
                SORT,
                SELECT,
                N,
                A,
                LDA,
                SDIM,
                W,
                VS,
                LDVS,
                WORK,
                LWORK,
                RWORK,
                BWORK,
                INFO,
            ):
                fn = _call_cached_ptr(
                    get_ptr_func=get_gees_pointer,
                    func_type_ref=gees_function_type,
                    unique_func_name_lit=unique_func_name,
                )
                fn(
                    JOBVS,
                    SORT,
                    SELECT,
                    N,
                    A,
                    LDA,
                    SDIM,
                    W,
                    VS,
                    LDVS,
                    WORK,
                    LWORK,
                    RWORK,
                    BWORK,
                    INFO,
                )

        else:  # Real case
            gees_function_type = types.FunctionType(
                types.void(
                    nb_i32p,  # JOBVS
                    nb_i32p,  # SORT
                    nb_i32p,  # SELECT
                    nb_i32p,  # N
                    float_pointer,  # A
                    nb_i32p,  # LDA
                    nb_i32p,  # SDIM
                    float_pointer,  # WR
                    float_pointer,  # WI
                    float_pointer,  # VS
                    nb_i32p,  # LDVS
                    float_pointer,  # WORK
                    nb_i32p,  # LWORK
                    nb_i32p,  # BWORK
                    nb_i32p,  # INFO
                )
            )

            @numba_basic.numba_njit
            def gees(
                JOBVS,
                SORT,
                SELECT,
                N,
                A,
                LDA,
                SDIM,
                WR,
                WI,
                VS,
                LDVS,
                WORK,
                LWORK,
                BWORK,
                INFO,
            ):
                fn = _call_cached_ptr(
                    get_ptr_func=get_gees_pointer,
                    func_type_ref=gees_function_type,
                    unique_func_name_lit=unique_func_name,
                )
                fn(
                    JOBVS,
                    SORT,
                    SELECT,
                    N,
                    A,
                    LDA,
                    SDIM,
                    WR,
                    WI,
                    VS,
                    LDVS,
                    WORK,
                    LWORK,
                    BWORK,
                    INFO,
                )

        return gees

    @classmethod
    def numba_xtrsyl(cls, dtype):
        """
        Solve the Sylvester equation A*X + ISGN*X*B = C or A**T*X + ISGN*X*B**T = C.

        Called by scipy.linalg.solve_sylvester and scipy.linalg.solve_continuous_lyapunov.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)

        if kind in "ld":
            real_pointer = float_pointer
        else:
            real_pointer = nb_f64p if dtype is nb_c128 else nb_f32p

        unique_func_name = f"scipy.lapack.{kind}trsyl"

        @numba_basic.numba_njit
        def get_trsyl_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "trsyl")
            return ptr

        trsyl_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # TRANA
                nb_i32p,  # TRANB
                nb_i32p,  # ISGN
                nb_i32p,  # M
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # B
                nb_i32p,  # LDB
                float_pointer,  # C
                nb_i32p,  # LDC
                real_pointer,  # SCALE
                nb_i32p,  # INFO
            )
        )

        @numba_basic.numba_njit
        def trsyl(TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C, LDC, SCALE, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_trsyl_pointer,
                func_type_ref=trsyl_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C, LDC, SCALE, INFO)

        return trsyl
