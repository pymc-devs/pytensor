from collections.abc import Sequence

import numba
from numba.core import types
from numba.np.linalg import _copy_to_fortran_order

from pytensor.link.numba.dispatch import basic as numba_basic


@numba_basic.numba_njit(inline="always")
def _copy_to_fortran_order_even_if_1d(x):
    # Numba's _copy_to_fortran_order doesn't do anything for vectors
    return x.copy() if x.ndim == 1 else _copy_to_fortran_order(x)


@numba_basic.numba_njit(inline="always")
def _trans_char_to_int(trans):
    if trans not in [0, 1, 2]:
        raise ValueError('Parameter "trans" should be one of 0, 1, 2')
    if trans == 0:
        return ord("N")
    elif trans == 1:
        return ord("T")
    else:
        return ord("C")


def _check_linalg_matrix(a, *, ndim: int | Sequence[int], dtype, func_name):
    """
    Adapted from https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L831
    """
    if not isinstance(a, types.Array):
        msg = f"{func_name} only supported for array types"
        raise numba.TypingError(msg, highlighting=False)
    ndim_msg = f"{func_name} only supported on {ndim}d arrays, got {a.ndim}."
    if isinstance(ndim, int):
        if a.ndim != ndim:
            raise numba.TypingError(ndim_msg, highlighting=False)
    elif a.ndim not in ndim:
        raise numba.TypingError(ndim_msg, highlighting=False)

    dtype_msg = f"{func_name} only supported for {dtype}, got {a.dtype}."
    if isinstance(dtype, type | tuple):
        if not isinstance(a.dtype, dtype):
            raise numba.TypingError(dtype_msg, highlighting=False)
    elif a.dtype != dtype:
        raise numba.TypingError(dtype_msg, highlighting=False)


def _check_dtypes_match(arrays: Sequence, func_name="cho_solve"):
    dtypes = [a.dtype for a in arrays]
    first_dtype = dtypes[0]
    for other_dtype in dtypes[1:]:
        if first_dtype != other_dtype:
            msg = f"{func_name} only supported for matching dtypes, got {dtypes}"
            raise numba.TypingError(msg, highlighting=False)
