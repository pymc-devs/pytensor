import numba
from numba.core import types
from numba.core.extending import get_cython_function_address
from numba.core.registry import CPUDispatcher
from numba.np.linalg import ensure_blas, get_blas_kind

from pytensor.link.numba.cache import _call_cached_ptr
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _get_nb_float_from_dtype,
    nb_i32p,
)


def get_blas_ptr(dtype, name):
    d = get_blas_kind(dtype)
    func_name = f"{d}{name}"
    blas_ptr = get_cython_function_address("scipy.linalg.cython_blas", func_name)
    return blas_ptr


class _BLAS:
    """
    Functions to return type signatures for wrapped BLAS functions.

    Patterned after https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L74
    """

    def __init__(self):
        ensure_blas()

    @classmethod
    def numba_xtrsm(cls, dtype) -> CPUDispatcher:
        r"""
        Solve a triangular matrix equation of the form :math:`op(A) X = \alpha B` (``side="L"``) or
        :math:`X op(A) = \alpha B` (``side="R"``), overwriting ``B`` with the solution.
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.blas.{kind}trsm"

        @numba_basic.numba_njit
        def get_trsm_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_blas_ptr(dtype, "trsm")
            return ptr

        trsm_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # SIDE
                nb_i32p,  # UPLO
                nb_i32p,  # TRANSA
                nb_i32p,  # DIAG
                nb_i32p,  # M
                nb_i32p,  # N
                float_ptr,  # ALPHA
                float_ptr,  # A
                nb_i32p,  # LDA
                float_ptr,  # B
                nb_i32p,  # LDB
            )
        )

        @numba_basic.numba_njit
        def trsm(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB):
            fn = _call_cached_ptr(
                get_ptr_func=get_trsm_pointer,
                func_type_ref=trsm_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB)

        return trsm

    @classmethod
    def numba_xgemm(cls, dtype) -> CPUDispatcher:
        r"""
        Compute a general matrix-matrix product, overwriting :math:`C`.

        .. math::

            C \leftarrow \alpha \, op(A) \, op(B) + \beta C

        where :math:`op(X)` is :math:`X`, :math:`X^T` or :math:`X^H` according to
        ``TRANSA`` and ``TRANSB``. Taking the transposes as flags is the point of
        binding this directly: BLAS reads either operand in transposed order at no
        cost, so a caller never has to materialize one.
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.blas.{kind}gemm"

        @numba_basic.numba_njit
        def get_gemm_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_blas_ptr(dtype, "gemm")
            return ptr

        gemm_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # TRANSA
                nb_i32p,  # TRANSB
                nb_i32p,  # M
                nb_i32p,  # N
                nb_i32p,  # K
                float_ptr,  # ALPHA
                float_ptr,  # A
                nb_i32p,  # LDA
                float_ptr,  # B
                nb_i32p,  # LDB
                float_ptr,  # BETA
                float_ptr,  # C
                nb_i32p,  # LDC
            )
        )

        @numba_basic.numba_njit
        def gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC):
            fn = _call_cached_ptr(
                get_ptr_func=get_gemm_pointer,
                func_type_ref=gemm_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

        return gemm

    @classmethod
    def numba_xgemv(cls, dtype) -> CPUDispatcher:
        """Matrix-vector product with signed increments for both vectors."""
        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        unique_func_name = f"scipy.blas.{kind}gemv"

        @numba_basic.numba_njit
        def get_gemv_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_blas_ptr(dtype, "gemv")
            return ptr

        gemv_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # TRANS
                nb_i32p,  # M
                nb_i32p,  # N
                float_ptr,  # ALPHA
                float_ptr,  # A
                nb_i32p,  # LDA
                float_ptr,  # X
                nb_i32p,  # INCX
                float_ptr,  # BETA
                float_ptr,  # Y
                nb_i32p,  # INCY
            )
        )

        @numba_basic.numba_njit
        def gemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY):
            fn = _call_cached_ptr(
                get_ptr_func=get_gemv_pointer,
                func_type_ref=gemv_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)

        return gemv

    @classmethod
    def numba_xdot(cls, dtype) -> CPUDispatcher:
        """Real vector product with signed increments."""
        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        scalar_type = _get_nb_float_from_dtype(kind, return_pointer=False)
        unique_func_name = f"scipy.blas.{kind}dot"

        @numba_basic.numba_njit
        def get_dot_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_blas_ptr(dtype, "dot")
            return ptr

        dot_function_type = types.FunctionType(
            scalar_type(nb_i32p, float_ptr, nb_i32p, float_ptr, nb_i32p)
        )

        @numba_basic.numba_njit
        def dot(N, X, INCX, Y, INCY):
            fn = _call_cached_ptr(
                get_ptr_func=get_dot_pointer,
                func_type_ref=dot_function_type,
                unique_func_name_lit=unique_func_name,
            )
            return fn(N, X, INCX, Y, INCY)

        return dot
