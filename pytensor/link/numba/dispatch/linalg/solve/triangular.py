import numpy as np
from numba.core import types
from numba.core.extending import overload
from numba.core.types import Float
from numba.np.linalg import ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_dtypes_match,
    _check_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _trans_char_to_int,
)


def _solve_triangular(
    A, B, trans=0, lower=False, unit_diagonal=False, overwrite_b=False
):
    """
    Thin wrapper around scipy.linalg.solve_triangular.

    This function is overloaded instead of the original scipy function to avoid unexpected side-effects to users who
    import pytensor.

    The signature must be the same as solve_triangular_impl, so b_ndim is included, although this argument is not
    used by scipy.linalg.solve_triangular.
    """
    return linalg.solve_triangular(
        A,
        B,
        trans=trans,
        lower=lower,
        unit_diagonal=unit_diagonal,
        overwrite_b=overwrite_b,
        check_finite=False,
    )


@overload(_solve_triangular)
def solve_triangular_impl(A, B, trans, lower, unit_diagonal, overwrite_b):
    ensure_lapack()

    _check_linalg_matrix(A, ndim=2, dtype=Float, func_name="solve_triangular")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="solve_triangular")
    _check_dtypes_match((A, B), func_name="solve_triangular")
    dtype = A.dtype
    numba_trtrs = _LAPACK().numba_xtrtrs(dtype)
    if isinstance(dtype, types.Complex):
        # If you want to make this work with complex numbers make sure you handle the c_contiguous trick correctly
        raise TypeError(
            "This function is not expected to work with complex numbers yet"
        )

    def impl(A, B, trans, lower, unit_diagonal, overwrite_b):
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        B_is_1d = B.ndim == 1

        if A.flags.f_contiguous or (A.flags.c_contiguous and trans in (0, 1)):
            A_f = A
            if A.flags.c_contiguous:
                # An upper/lower triangular c_contiguous is the same as a lower/upper triangular f_contiguous
                # Is this valid for complex matrices that were .conj().mT by PyTensor?
                lower = not lower
                trans = 1 - trans
        else:
            A_f = np.asfortranarray(A)

        if overwrite_b and B.flags.f_contiguous:
            B_copy = B
        else:
            B_copy = _copy_to_fortran_order_even_if_1d(B)

        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        NRHS = 1 if B_is_1d else int(B_copy.shape[-1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        TRANS = val_to_int_ptr(_trans_char_to_int(trans))
        DIAG = val_to_int_ptr(ord("U") if unit_diagonal else ord("N"))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_trtrs(
            UPLO,
            TRANS,
            DIAG,
            N,
            NRHS,
            A_f.ctypes,
            LDA,
            B_copy.ctypes,
            LDB,
            INFO,
        )
        if int_ptr_to_val(INFO) != 0:
            B_copy = np.full_like(B_copy, np.nan)

        if B_is_1d:
            return B_copy[..., 0]

        return B_copy

    return impl
