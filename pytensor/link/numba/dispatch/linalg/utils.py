from collections.abc import Callable

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


def _check_scipy_linalg_matrix(a, func_name):
    """
    Adapted from https://github.com/numba/numba/blob/bd7ebcfd4b850208b627a3f75d4706000be36275/numba/np/linalg.py#L831
    """
    prefix = "scipy.linalg"
    # Unpack optional type
    if isinstance(a, types.Optional):
        a = a.type
    if not isinstance(a, types.Array):
        msg = f"{prefix}.{func_name}() only supported for array types"
        raise numba.TypingError(msg, highlighting=False)
    if a.ndim not in [1, 2]:
        msg = (
            f"{prefix}.{func_name}() only supported on 1d or 2d arrays, found {a.ndim}."
        )
        raise numba.TypingError(msg, highlighting=False)
    if not isinstance(a.dtype, types.Float | types.Complex):
        msg = f"{prefix}.{func_name}() only supported on float and complex arrays."
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
