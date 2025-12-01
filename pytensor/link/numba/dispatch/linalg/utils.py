from collections.abc import Callable, Sequence

import numba
from numba.core import types
from numba.core.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from numpy.linalg import LinAlgError

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    val_to_int_ptr,
)


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


@numba_basic.numba_njit(inline="always")
def _solve_check(n, info, lamch=False, rcond=None):
    """
    Check arguments during the different steps of the solution phase
    Adapted from https://github.com/scipy/scipy/blob/7f7f04caa4a55306a9c6613c89eef91fedbd72d4/scipy/linalg/_basic.py#L38
    """
    if info < 0:
        # TODO: figure out how to do an fstring here
        msg = "LAPACK reported an illegal value in input"
        raise ValueError(msg)
    elif 0 < info:
        raise LinAlgError("Matrix is singular.")

    if lamch:
        E = _xlamch("E")
        if rcond < E:
            # TODO: This should be a warning, but we can't raise warnings in numba mode
            print(  # noqa: T201
                "Ill-conditioned matrix, rcond=", rcond, ", result may not be accurate."
            )


def _xlamch(kind: str = "E"):
    """
    Placeholder for getting machine precision; used by linalg.solve. Not used by pytensor to numbify graphs.
    """
    pass


@overload(_xlamch)
def xlamch_impl(kind: str = "E") -> Callable[[str], float]:
    """
    Compute the machine precision for a given floating point type.
    """
    from pytensor import config

    ensure_lapack()
    w_type = _get_underlying_float(config.floatX)

    if w_type == "float32":
        dtype = types.float32
    elif w_type == "float64":
        dtype = types.float64
    else:
        raise NotImplementedError("Unsupported dtype")

    numba_lamch = _LAPACK().numba_xlamch(dtype)

    def impl(kind: str = "E") -> float:
        KIND = val_to_int_ptr(ord(kind))
        return numba_lamch(KIND)  # type: ignore

    return impl
