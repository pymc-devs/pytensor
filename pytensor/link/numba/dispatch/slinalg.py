import ctypes

import numba
import numpy as np
from numba.core import cgutils, types
from numba.extending import get_cython_function_address, intrinsic, overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack, get_blas_kind
from scipy import linalg

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.tensor.slinalg import SolveTriangular


_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_float = ctypes.c_float
_char = ctypes.c_char
_int = ctypes.c_int

_ptr_float = _PTR(_float)
_ptr_dbl = _PTR(_dbl)
_ptr_char = _PTR(_char)
_ptr_int = _PTR(_int)


@intrinsic
def val_to_dptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.float64)(types.float64)
    return sig, impl


@intrinsic
def val_to_zptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.complex128)(types.complex128)
    return sig, impl


@intrinsic
def val_to_sptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.float32)(types.float32)
    return sig, impl


@intrinsic
def val_to_int_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(types.int32)(types.int32)
    return sig, impl


@intrinsic
def int_ptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.int32(types.CPointer(types.int32))
    return sig, impl


@intrinsic
def dptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.float64(types.CPointer(types.float64))
    return sig, impl


@intrinsic
def sptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = types.float32(types.CPointer(types.float32))
    return sig, impl


def _get_float_pointer_for_dtype(blas_dtype):
    if blas_dtype in ["s", "c"]:
        return _ptr_float
    elif blas_dtype in ["d", "z"]:
        return _ptr_dbl


def _get_underlying_float(dtype):
    s_dtype = str(dtype)
    out_type = s_dtype
    if s_dtype == "complex64":
        out_type = "float32"
    elif s_dtype == "complex128":
        out_type = "float64"

    return np.dtype(out_type)


def _get_lapack_ptr_and_ptr_type(dtype, name):
    d = get_blas_kind(dtype)
    func_name = f"{d}{name}"
    float_pointer = _get_float_pointer_for_dtype(d)
    lapack_ptr = get_cython_function_address("scipy.linalg.cython_lapack", func_name)

    return lapack_ptr, float_pointer


def _check_scipy_linalg_matrix(a, func_name):
    """
    Adapted from https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L831
    """
    prefix = "scipy.linalg"
    interp = (prefix, func_name)
    # Unpack optional type
    if isinstance(a, types.Optional):
        a = a.type
    if not isinstance(a, types.Array):
        msg = "%s.%s() only supported for array types" % interp
        raise numba.TypingError(msg, highlighting=False)
    if a.ndim not in [1, 2]:
        msg = "%s.%s() only supported on 1d or 2d arrays, found %s." % (
            interp + (a.ndim,)
        )
        raise numba.TypingError(msg, highlighting=False)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        msg = "%s.%s() only supported on " "float and complex arrays." % interp
        raise numba.TypingError(msg, highlighting=False)


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


def _solve_triangular(A, B, trans=0, lower=False, unit_diagonal=False):
    return linalg.solve_triangular(
        A, B, trans=trans, lower=lower, unit_diagonal=unit_diagonal
    )


@overload(_solve_triangular)
def solve_triangular_impl(A, B, trans=0, lower=False, unit_diagonal=False):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_triangular")
    _check_scipy_linalg_matrix(B, "solve_triangular")

    dtype = A.dtype
    if str(dtype).startswith("complex"):
        raise ValueError(
            "Complex inputs not currently supported by solve_triangular in Numba mode"
        )

    w_type = _get_underlying_float(dtype)
    numba_trtrs = _LAPACK().numba_xtrtrs(dtype)

    def impl(A, B, trans=0, lower=False, unit_diagonal=False):
        B_is_1d = B.ndim == 1

        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")

        if A.shape[0] != B.shape[0]:
            raise linalg.LinAlgError("Dimensions of A and B do not conform")

        if B_is_1d:
            B_copy = np.asfortranarray(np.expand_dims(B, -1))
        else:
            B_copy = _copy_to_fortran_order(B)

        if trans not in [0, 1, 2]:
            raise ValueError('Parameter "trans" should be one of N, C, T or 0, 1, 2')
        if trans == 0:
            transval = ord("N")
        elif trans == 1:
            transval = ord("T")
        else:
            transval = ord("C")

        B_NDIM = 1 if B_is_1d else int(B.shape[1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        TRANS = val_to_int_ptr(transval)
        DIAG = val_to_int_ptr(ord("U") if unit_diagonal else ord("N"))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(B_NDIM)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_trtrs(
            UPLO,
            TRANS,
            DIAG,
            N,
            NRHS,
            np.asfortranarray(A).T.view(w_type).ctypes,
            LDA,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            return B_copy[..., 0]
        return B_copy

    return impl


@numba_funcify.register(SolveTriangular)
def numba_funcify_SolveTriangular(op, node, **kwargs):
    trans = op.trans
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite

    @numba_basic.numba_njit(inline="always")
    def solve_triangular(a, b):
        res = _solve_triangular(a, b, trans, lower, unit_diagonal)
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(res), np.isnan(res))):
                raise ValueError(
                    "Non-numeric values (nan or inf) returned by solve_triangular"
                )
        return res

    return solve_triangular
