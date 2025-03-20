import ctypes

import numpy as np
from numba.core import cgutils, types
from numba.core.extending import get_cython_function_address, intrinsic
from numba.np.linalg import ensure_lapack, get_blas_kind


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
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "trtrs")

        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # TRANS
            _ptr_int,  # DIAG
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # B
            _ptr_int,  # LDB
            _ptr_int,  # INFO
        )

        return functype(lapack_ptr)

    @classmethod
    def numba_xpotrf(cls, dtype):
        """
        Compute the Cholesky factorization of a real symmetric positive definite matrix.

        Called by scipy.linalg.cholesky
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "potrf")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO,
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xpotrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by numba_potrf.

        Called by scipy.linalg.cho_solve
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "potrs")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # B
            _ptr_int,  # LDB
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xlange(cls, dtype):
        """
        Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any element of
        a general M-by-N matrix A.

        Called by scipy.linalg.solve
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "lange")
        output_ctype = _get_output_ctype(dtype)
        functype = ctypes.CFUNCTYPE(
            output_ctype,  # Output
            _ptr_int,  # NORM
            _ptr_int,  # M
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # WORK
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xlamch(cls, dtype):
        """
        Determine machine precision for floating point arithmetic.
        """

        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "lamch")
        output_dtype = _get_output_ctype(dtype)
        functype = ctypes.CFUNCTYPE(
            output_dtype,  # Output
            _ptr_int,  # CMACH
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgecon(cls, dtype):
        """
        Estimates the condition number of a matrix A, using the LU factorization computed by numba_getrf.

        Called by scipy.linalg.solve when assume_a == "gen"
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "gecon")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # NORM
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # ANORM
            float_pointer,  # RCOND
            float_pointer,  # WORK
            _ptr_int,  # IWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgetrf(cls, dtype):
        """
        Compute partial pivoting LU factorization of a general M-by-N matrix A using row interchanges.

        Called by scipy.linalg.lu_factor
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "getrf")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # M
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # IPIV
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgetrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B or A.T @ X = B with a general N-by-N matrix A using the LU
        factorization computed by GETRF.

        Called by scipy.linalg.lu_solve
        """
        ...
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "getrs")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # TRANS
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # IPIV
            float_pointer,  # B
            _ptr_int,  # LDB
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xsysv(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric matrix A using the diagonal pivoting method,
        factorizing A into LDL^T or UDU^T form, depending on the value of UPLO

        Called by scipy.linalg.solve when assume_a == "sym"
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "sysv")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # IPIV
            float_pointer,  # B
            _ptr_int,  # LDB
            float_pointer,  # WORK
            _ptr_int,  # LWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xsycon(cls, dtype):
        """
        Estimate the reciprocal of the condition number of a symmetric matrix A using the UDU or LDL factorization
        computed by xSYTRF.
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "sycon")

        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            _ptr_int,  # IPIV
            float_pointer,  # ANORM
            float_pointer,  # RCOND
            float_pointer,  # WORK
            _ptr_int,  # IWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xpocon(cls, dtype):
        """
        Estimates the reciprocal of the condition number of a positive definite matrix A using the Cholesky factorization
        computed by potrf.

        Called by scipy.linalg.solve when assume_a == "pos"
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "pocon")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # N
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # ANORM
            float_pointer,  # RCOND
            float_pointer,  # WORK
            _ptr_int,  # IWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xposv(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a symmetric positive definite matrix A using the Cholesky
        factorization computed by potrf.
        """

        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "posv")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # UPLO
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # A
            _ptr_int,  # LDA
            float_pointer,  # B
            _ptr_int,  # LDB
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgttrf(cls, dtype):
        """
        Compute the LU factorization of a tridiagonal matrix A using row interchanges.

        Called by scipy.linalg.lu_factor
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "gttrf")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # N
            float_pointer,  # DL
            float_pointer,  # D
            float_pointer,  # DU
            float_pointer,  # DU2
            _ptr_int,  # IPIV
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgttrs(cls, dtype):
        """
        Solve a system of linear equations A @ X = B with a tridiagonal matrix A using the LU factorization computed by numba_gttrf.

        Called by scipy.linalg.lu_solve
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "gttrs")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # TRANS
            _ptr_int,  # N
            _ptr_int,  # NRHS
            float_pointer,  # DL
            float_pointer,  # D
            float_pointer,  # DU
            float_pointer,  # DU2
            _ptr_int,  # IPIV
            float_pointer,  # B
            _ptr_int,  # LDB
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)

    @classmethod
    def numba_xgtcon(cls, dtype):
        """
        Estimate the reciprocal of the condition number of a tridiagonal matrix A using the LU factorization computed by numba_gttrf.
        """
        lapack_ptr, float_pointer = _get_lapack_ptr_and_ptr_type(dtype, "gtcon")
        functype = ctypes.CFUNCTYPE(
            None,
            _ptr_int,  # NORM
            _ptr_int,  # N
            float_pointer,  # DL
            float_pointer,  # D
            float_pointer,  # DU
            float_pointer,  # DU2
            _ptr_int,  # IPIV
            float_pointer,  # ANORM
            float_pointer,  # RCOND
            float_pointer,  # WORK
            _ptr_int,  # IWORK
            _ptr_int,  # INFO
        )
        return functype(lapack_ptr)
