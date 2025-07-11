from collections.abc import Callable

import numpy as np
from numba.core.extending import overload
from numba.np.linalg import ensure_lapack
from numpy import ndarray
from scipy import linalg

from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_scipy_linalg_matrix,
    _copy_to_fortran_order_even_if_1d,
    _solve_check,
    _trans_char_to_int,
)
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)


@numba_njit
def tridiagonal_norm(du, d, dl):
    # Adapted from scipy _matrix_norm_tridiagonal:
    # https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/linalg/_basic.py#L356-L367
    anorm = np.abs(d)
    anorm[1:] += np.abs(du)
    anorm[:-1] += np.abs(dl)
    anorm = anorm.max()
    return anorm


def _gttrf(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    overwrite_dl: bool,
    overwrite_d: bool,
    overwrite_du: bool,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
    """Placeholder for LU factorization of tridiagonal matrix."""
    return  # type: ignore


@overload(_gttrf)
def gttrf_impl(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    overwrite_dl: bool,
    overwrite_d: bool,
    overwrite_du: bool,
) -> Callable[
    [ndarray, ndarray, ndarray, bool, bool, bool],
    tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int],
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(dl, "gttrf")
    _check_scipy_linalg_matrix(d, "gttrf")
    _check_scipy_linalg_matrix(du, "gttrf")
    dtype = d.dtype
    w_type = _get_underlying_float(dtype)
    numba_gttrf = _LAPACK().numba_xgttrf(dtype)

    def impl(
        dl: ndarray,
        d: ndarray,
        du: ndarray,
        overwrite_dl: bool,
        overwrite_d: bool,
        overwrite_du: bool,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
        n = np.int32(d.shape[-1])
        ipiv = np.empty(n, dtype=np.int32)
        du2 = np.empty(n - 2, dtype=dtype)
        info = val_to_int_ptr(0)

        if not overwrite_dl or not dl.flags.f_contiguous:
            dl = dl.copy()

        if not overwrite_d or not d.flags.f_contiguous:
            d = d.copy()

        if not overwrite_du or not du.flags.f_contiguous:
            du = du.copy()

        numba_gttrf(
            val_to_int_ptr(n),
            dl.view(w_type).ctypes,
            d.view(w_type).ctypes,
            du.view(w_type).ctypes,
            du2.view(w_type).ctypes,
            ipiv.ctypes,
            info,
        )

        return dl, d, du, du2, ipiv, int_ptr_to_val(info)

    return impl


def _gttrs(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    du2: ndarray,
    ipiv: ndarray,
    b: ndarray,
    overwrite_b: bool,
    trans: bool,
) -> tuple[ndarray, int]:
    """Placeholder for solving an LU-decomposed tridiagonal system."""
    return  # type: ignore


@overload(_gttrs)
def gttrs_impl(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    du2: ndarray,
    ipiv: ndarray,
    b: ndarray,
    overwrite_b: bool,
    trans: bool,
) -> Callable[
    [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, bool, bool],
    tuple[ndarray, int],
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(dl, "gttrs")
    _check_scipy_linalg_matrix(d, "gttrs")
    _check_scipy_linalg_matrix(du, "gttrs")
    _check_scipy_linalg_matrix(du2, "gttrs")
    _check_scipy_linalg_matrix(b, "gttrs")
    dtype = d.dtype
    w_type = _get_underlying_float(dtype)
    numba_gttrs = _LAPACK().numba_xgttrs(dtype)

    def impl(
        dl: ndarray,
        d: ndarray,
        du: ndarray,
        du2: ndarray,
        ipiv: ndarray,
        b: ndarray,
        overwrite_b: bool,
        trans: bool,
    ) -> tuple[ndarray, int]:
        n = np.int32(d.shape[-1])
        nrhs = 1 if b.ndim == 1 else int(b.shape[-1])
        info = val_to_int_ptr(0)

        if not overwrite_b or not b.flags.f_contiguous:
            b = _copy_to_fortran_order_even_if_1d(b)

        if not dl.flags.f_contiguous:
            dl = dl.copy()

        if not d.flags.f_contiguous:
            d = d.copy()

        if not du.flags.f_contiguous:
            du = du.copy()

        if not du2.flags.f_contiguous:
            du2 = du2.copy()

        if not ipiv.flags.f_contiguous:
            ipiv = ipiv.copy()

        numba_gttrs(
            val_to_int_ptr(_trans_char_to_int(trans)),
            val_to_int_ptr(n),
            val_to_int_ptr(nrhs),
            dl.view(w_type).ctypes,
            d.view(w_type).ctypes,
            du.view(w_type).ctypes,
            du2.view(w_type).ctypes,
            ipiv.ctypes,
            b.view(w_type).ctypes,
            val_to_int_ptr(n),
            info,
        )

        return b, int_ptr_to_val(info)

    return impl


def _gtcon(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    du2: ndarray,
    ipiv: ndarray,
    anorm: float,
    norm: str,
) -> tuple[ndarray, int]:
    """Placeholder for computing the condition number of a tridiagonal system."""
    return  # type: ignore


@overload(_gtcon)
def gtcon_impl(
    dl: ndarray,
    d: ndarray,
    du: ndarray,
    du2: ndarray,
    ipiv: ndarray,
    anorm: float,
    norm: str,
) -> Callable[
    [ndarray, ndarray, ndarray, ndarray, ndarray, float, str], tuple[ndarray, int]
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(dl, "gtcon")
    _check_scipy_linalg_matrix(d, "gtcon")
    _check_scipy_linalg_matrix(du, "gtcon")
    _check_scipy_linalg_matrix(du2, "gtcon")
    dtype = d.dtype
    w_type = _get_underlying_float(dtype)
    numba_gtcon = _LAPACK().numba_xgtcon(dtype)

    def impl(
        dl: ndarray,
        d: ndarray,
        du: ndarray,
        du2: ndarray,
        ipiv: ndarray,
        anorm: float,
        norm: str,
    ) -> tuple[ndarray, int]:
        n = np.int32(d.shape[-1])
        rcond = np.empty(1, dtype=dtype)
        work = np.empty(2 * n, dtype=dtype)
        iwork = np.empty(n, dtype=np.int32)
        info = val_to_int_ptr(0)

        numba_gtcon(
            val_to_int_ptr(ord(norm)),
            val_to_int_ptr(n),
            dl.view(w_type).ctypes,
            d.view(w_type).ctypes,
            du.view(w_type).ctypes,
            du2.view(w_type).ctypes,
            ipiv.ctypes,
            np.array(anorm, dtype=dtype).view(w_type).ctypes,
            rcond.view(w_type).ctypes,
            work.view(w_type).ctypes,
            iwork.ctypes,
            info,
        )

        return rcond, int_ptr_to_val(info)

    return impl


def _solve_tridiagonal(
    a: ndarray,
    b: ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """
    Solve a positive-definite linear system using the Cholesky decomposition.
    """
    return linalg.solve(
        a=a,
        b=b,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        transposed=transposed,
        assume_a="tridiagonal",
    )


@overload(_solve_tridiagonal)
def _tridiagonal_solve_impl(
    A: ndarray,
    B: ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[[ndarray, ndarray, bool, bool, bool, bool, bool], ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")

    def impl(
        A: ndarray,
        B: ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> ndarray:
        n = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)
        norm = "1"

        if transposed:
            A = A.T
        dl, d, du = np.diag(A, -1), np.diag(A, 0), np.diag(A, 1)

        anorm = tridiagonal_norm(du, d, dl)

        dl, d, du, du2, IPIV, INFO = _gttrf(
            dl, d, du, overwrite_dl=True, overwrite_d=True, overwrite_du=True
        )
        _solve_check(n, INFO)

        X, INFO = _gttrs(
            dl, d, du, du2, IPIV, B, trans=transposed, overwrite_b=overwrite_b
        )
        _solve_check(n, INFO)

        RCOND, INFO = _gtcon(dl, d, du, du2, IPIV, anorm, norm)
        _solve_check(n, INFO, True, RCOND)

        return X

    return impl


@numba_funcify.register(LUFactorTridiagonal)
def numba_funcify_LUFactorTridiagonal(op: LUFactorTridiagonal, node, **kwargs):
    overwrite_dl = op.overwrite_dl
    overwrite_d = op.overwrite_d
    overwrite_du = op.overwrite_du

    @numba_njit(cache=False)
    def lu_factor_tridiagonal(dl, d, du):
        dl, d, du, du2, ipiv, _ = _gttrf(
            dl,
            d,
            du,
            overwrite_dl=overwrite_dl,
            overwrite_d=overwrite_d,
            overwrite_du=overwrite_du,
        )
        return dl, d, du, du2, ipiv

    return lu_factor_tridiagonal


@numba_funcify.register(SolveLUFactorTridiagonal)
def numba_funcify_SolveLUFactorTridiagonal(
    op: SolveLUFactorTridiagonal, node, **kwargs
):
    overwrite_b = op.overwrite_b
    transposed = op.transposed

    @numba_njit(cache=False)
    def solve_lu_factor_tridiagonal(dl, d, du, du2, ipiv, b):
        x, _ = _gttrs(
            dl,
            d,
            du,
            du2,
            ipiv,
            b,
            overwrite_b=overwrite_b,
            trans=transposed,
        )
        return x

    return solve_lu_factor_tridiagonal
