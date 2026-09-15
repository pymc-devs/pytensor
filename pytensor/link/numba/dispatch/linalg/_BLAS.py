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
