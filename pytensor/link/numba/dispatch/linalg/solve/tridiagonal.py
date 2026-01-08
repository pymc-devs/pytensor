from collections.abc import Callable

import numpy as np
from numba.core.extending import overload
from numba.core.types import Float, int32
from numba.np.linalg import ensure_lapack
from numpy import ndarray
from scipy import linalg

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    register_funcify_default_op_cache_key,
)
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
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)


@numba_basic.numba_njit
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
    _check_linalg_matrix(dl, ndim=1, dtype=Float, func_name="gttrf")
    _check_linalg_matrix(d, ndim=1, dtype=Float, func_name="gttrf")
    _check_linalg_matrix(du, ndim=1, dtype=Float, func_name="gttrf")
    _check_dtypes_match((dl, d, du), func_name="gttrf")
    dtype = d.dtype
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
            dl.ctypes,
            d.ctypes,
            du.ctypes,
            du2.ctypes,
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
    _check_linalg_matrix(dl, ndim=1, dtype=Float, func_name="gttrs")
    _check_linalg_matrix(d, ndim=1, dtype=Float, func_name="gttrs")
    _check_linalg_matrix(du, ndim=1, dtype=Float, func_name="gttrs")
    _check_linalg_matrix(du2, ndim=1, dtype=Float, func_name="gttrs")
    _check_linalg_matrix(b, ndim=(1, 2), dtype=Float, func_name="gttrs")
    _check_dtypes_match((dl, d, du, du2, b), func_name="gttrs")
    _check_linalg_matrix(ipiv, ndim=1, dtype=int32, func_name="gttrs")
    dtype = d.dtype
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
            dl.ctypes,
            d.ctypes,
            du.ctypes,
            du2.ctypes,
            ipiv.ctypes,
            b.ctypes,
            val_to_int_ptr(n),
            info,
        )

        return b, int_ptr_to_val(info)

    return impl


def _solve_tridiagonal(
    a: ndarray,
    b: ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
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
        check_finite=False,
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
    transposed: bool,
) -> Callable[[ndarray, ndarray, bool, bool, bool, bool], ndarray]:
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=Float, func_name="solve")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="solve")
    _check_dtypes_match((A, B), func_name="solve")

    def impl(
        A: ndarray,
        B: ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        transposed: bool,
    ) -> ndarray:
        _solve_check_input_shapes(A, B)

        if transposed:
            A = A.T
        dl, d, du = np.diag(A, -1), np.diag(A, 0), np.diag(A, 1)

        dl, d, du, du2, ipiv, info1 = _gttrf(
            dl, d, du, overwrite_dl=True, overwrite_d=True, overwrite_du=True
        )

        X, info2 = _gttrs(
            dl, d, du, du2, ipiv, B, trans=transposed, overwrite_b=overwrite_b
        )

        if info1 != 0 or info2 != 0:
            X = np.full_like(X, np.nan)

        return X

    return impl


@register_funcify_default_op_cache_key(LUFactorTridiagonal)
def numba_funcify_LUFactorTridiagonal(op: LUFactorTridiagonal, node, **kwargs):
    if any(i.type.numpy_dtype.kind == "c" for i in node.inputs):
        return generate_fallback_impl(op, node=node)

    overwrite_dl = op.overwrite_dl
    overwrite_d = op.overwrite_d
    overwrite_du = op.overwrite_du
    out_dtype = node.outputs[1].type.numpy_dtype

    cast_inputs = (cast_dl, cast_d, cast_du) = tuple(
        inp.type.numpy_dtype != out_dtype for inp in node.inputs
    )
    if any(cast_inputs) and config.compiler_verbose:
        print("LUFactorTridiagonal requires casting at least one input")  # noqa: T201

    @numba_basic.numba_njit(cache=False)
    def lu_factor_tridiagonal(dl, d, du):
        if d.size == 0:
            return (
                np.zeros(dl.shape, dtype=out_dtype),
                np.zeros(d.shape, dtype=out_dtype),
                np.zeros(du.shape, dtype=out_dtype),
                np.zeros(d.shape, dtype=out_dtype),
                np.zeros(d.shape, dtype="int32"),
            )

        if cast_d:
            d = d.astype(out_dtype)
        if cast_dl:
            dl = dl.astype(out_dtype)
        if cast_du:
            du = du.astype(out_dtype)
        dl, d, du, du2, ipiv, _ = _gttrf(
            dl,
            d,
            du,
            overwrite_dl=overwrite_dl,
            overwrite_d=overwrite_d,
            overwrite_du=overwrite_du,
        )
        return dl, d, du, du2, ipiv

    cache_version = 2
    return lu_factor_tridiagonal, cache_version


@register_funcify_default_op_cache_key(SolveLUFactorTridiagonal)
def numba_funcify_SolveLUFactorTridiagonal(
    op: SolveLUFactorTridiagonal, node, **kwargs
):
    if any(i.type.numpy_dtype.kind == "c" for i in node.inputs):
        return generate_fallback_impl(op, node=node)
    out_dtype = node.outputs[0].type.numpy_dtype

    b_ndim = op.b_ndim
    overwrite_b = op.overwrite_b
    transposed = op.transposed

    must_cast_inputs = (cast_dl, cast_d, cast_du, cast_du2, cast_ipiv, cast_b) = tuple(
        inp.type.numpy_dtype != (np.int32 if i == 4 else out_dtype)
        for i, inp in enumerate(node.inputs)
    )
    if any(must_cast_inputs) and config.compiler_verbose:
        print("SolveLUFactorTridiagonal requires casting at least one input")  # noqa: T201

    @numba_basic.numba_njit(cache=False)
    def solve_lu_factor_tridiagonal(dl, d, du, du2, ipiv, b):
        if d.size == 0:
            if b_ndim == 1:
                return np.zeros(d.shape, dtype=out_dtype)
            else:
                return np.zeros((d.shape[0], b.shape[1]), dtype=out_dtype)

        if cast_dl:
            dl = dl.astype(out_dtype)
        if cast_d:
            d = d.astype(out_dtype)
        if cast_du:
            du = du.astype(out_dtype)
        if cast_du2:
            du2 = du2.astype(out_dtype)
        if cast_ipiv:
            ipiv = ipiv.astype(np.int32)
        if cast_b:
            b = b.astype(out_dtype)
        x, info = _gttrs(
            dl,
            d,
            du,
            du2,
            ipiv,
            b,
            overwrite_b=overwrite_b,
            trans=transposed,
        )

        if info != 0:
            x = np.full_like(x, np.nan)

        return x

    cache_version = 2
    return solve_lu_factor_tridiagonal, cache_version
