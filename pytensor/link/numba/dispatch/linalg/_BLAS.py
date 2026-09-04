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
    def numba_xger(cls, dtype) -> CPUDispatcher:
        r"""
        Add a rank-1 update to a general matrix, overwriting :math:`A`.

        .. math::

            A \leftarrow \alpha \, x \, y^T + A

        BLAS names this ``ger`` only for the real kinds; the complex ones split it
        into an unconjugated ``geru`` and a conjugated ``gerc``, and this binds the
        unconjugated one so every dtype computes the same :math:`x y^T`.
        """

        kind = get_blas_kind(dtype)
        float_ptr = _get_nb_float_from_dtype(kind)
        name = "ger" if kind in "sd" else "geru"
        unique_func_name = f"scipy.blas.{kind}{name}"

        @numba_basic.numba_njit
        def get_ger_pointer():
            with numba.objmode(ptr=types.intp):
                ptr = get_blas_ptr(dtype, name)
            return ptr

        ger_function_type = types.FunctionType(
            types.void(
                nb_i32p,  # M
                nb_i32p,  # N
                float_ptr,  # ALPHA
                float_ptr,  # X
                nb_i32p,  # INCX
                float_ptr,  # Y
                nb_i32p,  # INCY
                float_ptr,  # A
                nb_i32p,  # LDA
            )
        )

        @numba_basic.numba_njit
        def ger(M, N, ALPHA, X, INCX, Y, INCY, A, LDA):
            fn = _call_cached_ptr(
                get_ptr_func=get_ger_pointer,
                func_type_ref=ger_function_type,
                unique_func_name_lit=unique_func_name,
            )
            fn(M, N, ALPHA, X, INCX, Y, INCY, A, LDA)

        return ger
