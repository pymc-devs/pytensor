import ctypes

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

_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_float = ctypes.c_float
_char = ctypes.c_char
_int = ctypes.c_int

_ptr_float = _PTR(_float)
_ptr_dbl = _PTR(_dbl)
_ptr_char = _PTR(_char)
_ptr_int = _PTR(_int)


def _get_lapack_ptr_and_ptr_type(dtype, name):
    d = get_blas_kind(dtype)
    func_name = f"{d}{name}"
    float_pointer = _get_float_pointer_for_dtype(d)
    lapack_ptr = get_cython_function_address("scipy.linalg.cython_lapack", func_name)

    return lapack_ptr, float_pointer


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


def _get_float_pointer_for_dtype(blas_dtype):
    if blas_dtype in ["s", "c"]:
        return _ptr_float
    elif blas_dtype in ["d", "z"]:
        return _ptr_dbl


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


def _get_output_ctype(dtype):
    s_dtype = str(dtype)
    if s_dtype in ["float32", "complex64"]:
        return _float
    elif s_dtype in ["float64", "complex128"]:
        return _dbl


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
    def numba_xtrtrs(cls, dtype):
        """
        Solve a triangular system of equations of the form A @ X = B or A.T @ X = B.

        Called by scipy.linalg.solve_triangular
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}trtrs"

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

        def _trtrs_py(UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_trtrs_pointer,
                func_type_ref=trtrs_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB, INFO)

        trtrs: CPUDispatcher = numba_basic.numba_njit(cache=True)(_trtrs_py)

        return trtrs

    @classmethod
    def numba_xpotrf(cls, dtype):
        """
        Compute the Cholesky factorization of a real symmetric positive definite matrix.

        Called by scipy.linalg.cholesky
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}potrf"

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

        def _potrf_py(UPLO, N, A, LDA, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_potrf_pointer,
                func_type_ref=potrf_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, A, LDA, INFO)

        potrf: CPUDispatcher = numba_basic.numba_njit(cache=True)(_potrf_py)

        return potrf

    @classmethod
    def numba_xpotrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by numba_potrf.

        Called by scipy.linalg.cho_solve
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}potrs"

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

        def _potrs_py(UPLO, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_potrs_pointer,
                func_type_ref=potrs_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, NRHS, A, LDA, B, LDB, INFO)

        potrs: CPUDispatcher = numba_basic.numba_njit(cache=True)(_potrs_py)

        return potrs

    @classmethod
    def numba_xlange(cls, dtype):
        """
        Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any element of
        a general M-by-N matrix A.

        Called by scipy.linalg.solve, but doesn't correspond to any Op in pytensor.
        """
        kind = get_blas_kind(dtype)
        float_type = _get_nb_float_from_dtype(kind, return_pointer=False)
        float_pointer = _get_nb_float_from_dtype(kind, return_pointer=True)
        cache_key = f"{kind}lange"

        @numba_basic.numba_njit
        def get_lange_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "lange")
            return ptr

        lange_function_type = types.FunctionType(
            float_type(
                nb_i32p,  # NORM
                nb_i32p,  # M
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # WORK
            )
        )

        def _lange_py(NORM, M, N, A, LDA, WORK):
            fn = _call_cached_ptr(
                get_ptr_func=get_lange_pointer,
                func_type_ref=lange_function_type,
                cache_key_lit=cache_key,
            )
            return fn(NORM, M, N, A, LDA, WORK)

        lange: CPUDispatcher = numba_basic.numba_njit(cache=True)(_lange_py)
        return lange

    @classmethod
    def numba_xlamch(cls, dtype):
        """
        Determine machine precision for floating point arithmetic.
        """
        kind = get_blas_kind(dtype)
        float_type = _get_nb_float_from_dtype(kind, return_pointer=False)
        cache_key = f"{kind}lamch"

        @numba_basic.numba_njit
        def get_lamch_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "lamch")
            return ptr

        lamch_function_type = types.FunctionType(
            float_type(  # Return type
                nb_i32p,  # CMACH
            )
        )

        def _lamch_py(CMACH):
            fn = _call_cached_ptr(
                get_ptr_func=get_lamch_pointer,
                func_type_ref=lamch_function_type,
                cache_key_lit=cache_key,
            )
            res = fn(CMACH)
            return res

        lamch: CPUDispatcher = numba_basic.numba_njit(cache=True)(_lamch_py)

        return lamch

    @classmethod
    def numba_xgecon(cls, dtype):
        """
        Estimates the condition number of a matrix A, using the LU factorization computed by numba_getrf.

        Called by scipy.linalg.solve when assume_a == "gen"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}gecon"

        @numba_basic.numba_njit
        def get_gecon_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "gecon")
            return ptr

        gecon_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # NORM
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # ANORM
                float_pointer,  # RCOND
                float_pointer,  # WORK
                nb_i32p,  # IWORK
                nb_i32p,  # INFO
            )
        )

        def _gecon_py(NORM, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gecon_pointer,
                func_type_ref=gecon_function_type,
                cache_key_lit=cache_key,
            )
            fn(NORM, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO)

        gecon: CPUDispatcher = numba_basic.numba_njit(cache=True)(_gecon_py)
        return gecon

    @classmethod
    def numba_xgetrf(cls, dtype):
        """
        Compute partial pivoting LU factorization of a general M-by-N matrix A using row interchanges.

        Called by scipy.linalg.lu_factor
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}getrf"

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

        def _getrf_py(M, N, A, LDA, IPIV, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_getrf_pointer,
                func_type_ref=getrf_function_type,
                cache_key_lit=cache_key,
            )
            fn(M, N, A, LDA, IPIV, INFO)

        getrf: CPUDispatcher = numba_basic.numba_njit(cache=True)(_getrf_py)
        return getrf

    @classmethod
    def numba_xgetrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B or A.T @ X = B with a general N-by-N matrix A using the LU
        factorization computed by GETRF.

        Called by scipy.linalg.lu_solve
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}getrs"

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

        def _getrs_py(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_getrs_pointer,
                func_type_ref=getrs_function_type,
                cache_key_lit=cache_key,
            )
            fn(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO)

        getrs: CPUDispatcher = numba_basic.numba_njit(cache=True)(_getrs_py)
        return getrs

    @classmethod
    def numba_xsysv(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric matrix A using the diagonal pivoting method,
        factorizing A into LDL^T or UDU^T form, depending on the value of UPLO

        Called by scipy.linalg.solve when assume_a == "sym"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}sysv"

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

        def _sysv_py(UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_sysv_pointer,
                func_type_ref=sysv_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO)

        sysv: CPUDispatcher = numba_basic.numba_njit(cache=True)(_sysv_py)
        return sysv

    @classmethod
    def numba_xsycon(cls, dtype):
        """
        Estimate the reciprocal of the condition number of a symmetric matrix A using the UDU or LDL factorization
        computed by xSYTRF.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}sycon"

        @numba_basic.numba_njit
        def get_sycon_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "sycon")
            return ptr

        sycon_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                nb_i32p,  # IPIV
                float_pointer,  # ANORM
                float_pointer,  # RCOND
                float_pointer,  # WORK
                nb_i32p,  # IWORK
                nb_i32p,  # INFO
            )
        )

        def _sycon_py(UPLO, N, A, LDA, IPIV, ANORM, RCOND, WORK, IWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_sycon_pointer,
                func_type_ref=sycon_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, A, LDA, IPIV, ANORM, RCOND, WORK, IWORK, INFO)

        sycon: CPUDispatcher = numba_basic.numba_njit(cache=True)(_sycon_py)

        return sycon

    @classmethod
    def numba_xpocon(cls, dtype):
        """
        Estimates the reciprocal of the condition number of a positive definite matrix A using the Cholesky factorization
        computed by potrf.

        Called by scipy.linalg.solve when assume_a == "pos"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}pocon"

        @numba_basic.numba_njit
        def get_pocon_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_lapack_ptr(dtype, "pocon")
            return ptr

        pocon_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # UPLO
                nb_i32p,  # N
                float_pointer,  # A
                nb_i32p,  # LDA
                float_pointer,  # ANORM
                float_pointer,  # RCOND
                float_pointer,  # WORK
                nb_i32p,  # IWORK
                nb_i32p,  # INFO
            )
        )

        def _pocon_py(UPLO, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_pocon_pointer,
                func_type_ref=pocon_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO)

        pocon: CPUDispatcher = numba_basic.numba_njit(cache=True)(_pocon_py)
        return pocon

    @classmethod
    def numba_xposv(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by potrf.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}posv"

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

        def _posv_py(UPLO, N, NRHS, A, LDA, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_posv_pointer,
                func_type_ref=posv_function_type,
                cache_key_lit=cache_key,
            )
            fn(UPLO, N, NRHS, A, LDA, B, LDB, INFO)

        posv: CPUDispatcher = numba_basic.numba_njit(cache=True)(_posv_py)
        return posv

    @classmethod
    def numba_xgttrf(cls, dtype):
        """
        Compute the LU factorization of a tridiagonal matrix A using row interchanges.

        Called by scipy.linalg.solve when assume_a == "tri"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}gttrf"

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

        def _gttrf_py(N, DL, D, DU, DU2, IPIV, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gttrf_pointer,
                func_type_ref=gttrf_function_type,
                cache_key_lit=cache_key,
            )
            fn(N, DL, D, DU, DU2, IPIV, INFO)

        gttrf: CPUDispatcher = numba_basic.numba_njit(cache=True)(_gttrf_py)
        return gttrf

    @classmethod
    def numba_xgttrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a tridiagonal matrix A using the LU factorization computed by numba_gttrf.

        Called by scipy.linalg.solve, when assume_a == "tri"
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}gttrs"

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

        def _gttrs_py(TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gttrs_pointer,
                func_type_ref=gttrs_function_type,
                cache_key_lit=cache_key,
            )
            fn(TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB, INFO)

        gttrs: CPUDispatcher = numba_basic.numba_njit(cache=True)(_gttrs_py)
        return gttrs

    @classmethod
    def numba_xgtcon(cls, dtype):
        """
        Estimate the reciprocal of the condition number of a tridiagonal matrix A using the LU factorization computed by numba_gttrf.
        """
        kind = get_blas_kind(dtype)
        float_pointer = _get_nb_float_from_dtype(kind)
        cache_key = f"{kind}gtcon"

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

        def _gtcon_py(NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, IWORK, INFO):
            fn = _call_cached_ptr(
                get_ptr_func=get_gtcon_pointer,
                func_type_ref=gtcon_function_type,
                cache_key_lit=cache_key,
            )
            fn(NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, IWORK, INFO)

        gtcon: CPUDispatcher = numba_basic.numba_njit(cache=True)(_gtcon_py)
        return gtcon

    @classmethod
    def numba_xgeqrf(cls, dtype):
        """
        Compute the QR factorization of a general M-by-N matrix A.

        Used in QR decomposition (no pivoting).
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "geqrf")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # M
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # TAU
            float_pointer,  # WORK
            _ptr_int,  # LWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgeqp3(cls, dtype):
        """
        Compute the QR factorization with column pivoting of a general M-by-N matrix A.

        Used in QR decomposition with pivoting.
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "geqp3")
        ctype_args = (
            _ptr_int,  # M
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # JPVT
            float_pointer,  # TAU
            float_pointer,  # WORK
            _ptr_int,  # LWORK
        )

        if isinstance(dtype, Complex):
            ctype_args = (
                *ctype_args,
                float_pointer,  # RWORK)
            )

        functype = ctypes.CFUNCTYPE(
            None,
            *ctype_args,
            _ptr_int,  # INFO
        )

        return functype(lapack_ptr)

    @classmethod
    def numba_xorgqr(cls, dtype):
        """
        Generate the orthogonal matrix Q from a QR factorization (real types).

        Used in QR decomposition to form Q.
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "orgqr")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # M
            _ptr_int,  # N
            _ptr_int,  # K
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # TAU
            float_pointer,  # WORK
            _ptr_int,  # LWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xungqr(cls, dtype):
        """
        Generate the unitary matrix Q from a QR factorization (complex types).

        Used in QR decomposition to form Q for complex types.
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "ungqr")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # M
            _ptr_int,  # N
            _ptr_int,  # K
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # TAU
            float_pointer,  # WORK
            _ptr_int,  # LWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)
