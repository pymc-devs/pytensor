from collections.abc import Callable
from typing import cast as typing_cast

import numpy as np
from numba import njit as numba_njit
from numba.core.extending import overload
from numba.np.linalg import ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _getrf
from pytensor.link.numba.dispatch.linalg.utils import _check_scipy_linalg_matrix


@numba_njit
def _pivot_to_permutation(p, dtype):
    p_inv = np.arange(len(p)).astype(dtype)
    for i in range(len(p)):
        p_inv[i], p_inv[p[i]] = p_inv[p[i]], p_inv[i]
    return p_inv


@numba_njit
def _lu_factor_to_lu(a, dtype, overwrite_a):
    A_copy, IPIV, INFO = _getrf(a, overwrite_a=overwrite_a)

    L = np.eye(A_copy.shape[-1], dtype=dtype)
    L += np.tril(A_copy, k=-1)
    U = np.triu(A_copy)

    # Fortran is 1 indexed, so we need to subtract 1 from the IPIV array
    IPIV = IPIV - 1
    p_inv = _pivot_to_permutation(IPIV, dtype=dtype)
    perm = np.argsort(p_inv).astype("int32")

    return perm, L, U


def _lu_1(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin wrapper around scipy.linalg.lu. Used as an overload target to avoid side-effects on users to import Pytensor.

    Called when permute_l is True and p_indices is False, and returns a tuple of (perm, L, U), where perm an integer
    array of row swaps, such that L[perm] @ U = A.
    """
    return typing_cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        linalg.lu(
            a,
            permute_l=permute_l,
            check_finite=check_finite,
            p_indices=p_indices,
            overwrite_a=overwrite_a,
        ),
    )


def _lu_2(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scipy.linalg.lu. Used as an overload target to avoid side-effects on users to import Pytensor.

    Called when permute_l is False and p_indices is True, and returns a tuple of (PL, U), where PL is the
    permuted L matrix, PL = P @ L.
    """
    return typing_cast(
        tuple[np.ndarray, np.ndarray],
        linalg.lu(
            a,
            permute_l=permute_l,
            check_finite=check_finite,
            p_indices=p_indices,
            overwrite_a=overwrite_a,
        ),
    )


def _lu_3(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin wrapper around scipy.linalg.lu. Used as an overload target to avoid side-effects on users to import Pytensor.

    Called when permute_l is False and p_indices is False, and returns a tuple of (P, L, U), where P is the permutation
    matrix, P @ L @ U = A.
    """
    return typing_cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        linalg.lu(
            a,
            permute_l=permute_l,
            check_finite=check_finite,
            p_indices=p_indices,
            overwrite_a=overwrite_a,
        ),
    )


@overload(_lu_1)
def lu_impl_1(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> Callable[
    [np.ndarray, bool, bool, bool, bool], tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Overload scipy.linalg.lu with a numba function. This function is called when permute_l is True and p_indices is
    False. Returns a tuple of (perm, L, U), where perm an integer array of row swaps, such that L[perm] @ U = A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(a, "lu")
    dtype = a.dtype

    def impl(
        a: np.ndarray,
        permute_l: bool,
        check_finite: bool,
        p_indices: bool,
        overwrite_a: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm, L, U = _lu_factor_to_lu(a, dtype, overwrite_a)
        return perm, L, U

    return impl


@overload(_lu_2)
def lu_impl_2(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> Callable[[np.ndarray, bool, bool, bool, bool], tuple[np.ndarray, np.ndarray]]:
    """
    Overload scipy.linalg.lu with a numba function. This function is called when permute_l is False and p_indices is
    True. Returns a tuple of (PL, U), where PL is the permuted L matrix, PL = P @ L.
    """

    ensure_lapack()
    _check_scipy_linalg_matrix(a, "lu")
    dtype = a.dtype

    def impl(
        a: np.ndarray,
        permute_l: bool,
        check_finite: bool,
        p_indices: bool,
        overwrite_a: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        perm, L, U = _lu_factor_to_lu(a, dtype, overwrite_a)
        PL = L[perm]

        return PL, U

    return impl


@overload(_lu_3)
def lu_impl_3(
    a: np.ndarray,
    permute_l: bool,
    check_finite: bool,
    p_indices: bool,
    overwrite_a: bool,
) -> Callable[
    [np.ndarray, bool, bool, bool, bool], tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Overload scipy.linalg.lu with a numba function. This function is called when permute_l is True and p_indices is
    False. Returns a tuple of (P, L, U), such that P @ L @ U = A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(a, "lu")
    dtype = a.dtype

    def impl(
        a: np.ndarray,
        permute_l: bool,
        check_finite: bool,
        p_indices: bool,
        overwrite_a: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm, L, U = _lu_factor_to_lu(a, dtype, overwrite_a)
        P = np.eye(a.shape[-1], dtype=dtype)[perm]

        return P, L, U

    return impl
