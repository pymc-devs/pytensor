from collections.abc import Sequence

import numba
import numpy as np
from numba.core import types
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order

from pytensor.link.numba.dispatch import basic as numba_basic


# A LAPACK kernel chooses its working layout from the input types at typing time, so
# every return path has one layout and the output is typed "C" or "F", never "A". Where
# the math allows, a C-typed input is worked on as its transpose, in a buffer from
# _c_work_array. Every other input goes through _fortran_work_array. A runtime choice
# between layouts would type both, and Numba unifies them to "A", which every consumer
# of the result then inherits.


def _fortran_work_array(x: np.ndarray, overwrite: bool) -> np.ndarray:
    """Return ``x`` in LAPACK's layout, reusing its buffer if ``overwrite`` allows."""
    return np.asfortranarray(x) if overwrite else np.array(x, order="F")


@overload(_fortran_work_array)
def _fortran_work_array_impl(x, overwrite):
    if not x.mutable:
        # Never reuse a readonly buffer. Numba's _copy_to_fortran_order would also type
        # the copy of an "A" array readonly, so copy into a fresh Fortran buffer instead.
        if x.ndim == 1:
            return lambda x, overwrite: x.copy()

        def impl(x, overwrite):
            out = np.empty((x.shape[1], x.shape[0]), dtype=x.dtype).T
            for j in range(x.shape[1]):
                for i in range(x.shape[0]):
                    out[i, j] = x[i, j]
            return out

        return impl

    if x.ndim == 1:

        def impl(x, overwrite):
            return np.ascontiguousarray(x) if overwrite else x.copy()

        return impl

    def impl(x, overwrite):
        if overwrite:
            # The "A" view makes asfortranarray check contiguity at runtime, so a C-typed
            # array that is also F-contiguous, such as one column, is reused.
            return np.asfortranarray(x[:, ::1])
        return _copy_to_fortran_order(x)

    return impl


def _c_work_array(x: np.ndarray, overwrite: bool) -> np.ndarray:
    """Return C-typed ``x`` as a C-ordered buffer, reusing it if ``overwrite`` allows."""
    return x if overwrite else x.copy()


@overload(_c_work_array)
def _c_work_array_impl(x, overwrite):
    if not x.mutable:
        return lambda x, overwrite: x.copy()
    return lambda x, overwrite: x if overwrite else x.copy()


def _full_fortran_order(shape, fill_value, dtype) -> np.ndarray:
    """Return a new array of ``shape`` filled with ``fill_value``, in LAPACK's layout."""
    return np.full(shape, fill_value, dtype=dtype, order="F")


@overload(_full_fortran_order)
def _full_fortran_order_impl(shape, fill_value, dtype):
    if len(shape) == 1:

        def impl(shape, fill_value, dtype):
            return np.full(shape, fill_value, dtype)

        return impl

    def impl(shape, fill_value, dtype):
        out = np.empty((shape[1], shape[0]), dtype).T
        out.fill(fill_value)
        return out

    return impl


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
