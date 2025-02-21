from collections.abc import Callable

import numba
import numpy as np
from numba.core import types
from numba.extending import overload
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from numpy.linalg import LinAlgError
from scipy import linalg

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.tensor.slinalg import (
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    Solve,
    SolveTriangular,
)


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


def _solve_triangular(
    A, B, trans=0, lower=False, unit_diagonal=False, b_ndim=1, overwrite_b=False
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
    )


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


@numba_basic.numba_njit(inline="always")
def _solve_check_input_shapes(A, B):
    if A.shape[0] != B.shape[0]:
        raise linalg.LinAlgError("Dimensions of A and B do not conform")
    if A.shape[-2] != A.shape[-1]:
        raise linalg.LinAlgError("Last 2 dimensions of A must be square")


@overload(_solve_triangular)
def solve_triangular_impl(A, B, trans, lower, unit_diagonal, b_ndim, overwrite_b):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_triangular")
    _check_scipy_linalg_matrix(B, "solve_triangular")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_trtrs = _LAPACK().numba_xtrtrs(dtype)

    def impl(A, B, trans, lower, unit_diagonal, b_ndim, overwrite_b):
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        # Seems weird to not use the b_ndim input directly, but when I did that Numba complained that the output type
        # could potentially be 3d (it didn't understand b_ndim was always equal to B.ndim)
        B_is_1d = B.ndim == 1

        # This will only copy if A is not already fortran contiguous
        A_f = np.asfortranarray(A)

        if overwrite_b:
            if B_is_1d:
                B_copy = np.expand_dims(B, -1)
            else:
                # This *will* allow inplace destruction of B, but only if it is already fortran contiguous.
                # Otherwise, there's no way to get around the need to copy the data before going into TRTRS
                B_copy = np.asfortranarray(B)
        else:
            if B_is_1d:
                B_copy = np.copy(np.expand_dims(B, -1))
            else:
                B_copy = _copy_to_fortran_order(B)

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
            A_f.view(w_type).ctypes,
            LDA,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        _solve_check(int_ptr_to_val(LDA), int_ptr_to_val(INFO))

        if B_is_1d:
            return B_copy[..., 0]

        return B_copy

    return impl


@numba_funcify.register(SolveTriangular)
def numba_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite
    overwrite_b = op.overwrite_b
    b_ndim = op.b_ndim

    dtype = node.inputs[0].dtype
    if str(dtype).startswith("complex"):
        raise NotImplementedError(
            "Complex inputs not currently supported by solve_triangular in Numba mode"
        )

    @numba_basic.numba_njit(inline="always")
    def solve_triangular(a, b):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input A to solve_triangular"
                )
            if np.any(np.bitwise_or(np.isinf(b), np.isnan(b))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input b to solve_triangular"
                )

        res = _solve_triangular(
            a,
            b,
            trans=0,  # transposing is handled explicitly on the graph, so we never use this argument
            lower=lower,
            unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b,
            b_ndim=b_ndim,
        )

        return res

    return solve_triangular


def _cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    return (
        linalg.cholesky(
            a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite
        ),
        0,
    )


@overload(_cholesky)
def cholesky_impl(A, lower=0, overwrite_a=False, check_finite=True):
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "cholesky")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_potrf = _LAPACK().numba_xpotrf(dtype)

    def impl(A, lower=0, overwrite_a=False, check_finite=True):
        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        if not overwrite_a:
            A_copy = _copy_to_fortran_order(A)
        else:
            A_copy = A

        numba_potrf(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            INFO,
        )

        if lower:
            for j in range(1, _N):
                for i in range(j):
                    A_copy[i, j] = 0.0
        else:
            for j in range(_N):
                for i in range(j + 1, _N):
                    A_copy[i, j] = 0.0

        return A_copy, int_ptr_to_val(INFO)

    return impl


@numba_funcify.register(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    """
    Overload scipy.linalg.cholesky with a numba function.

    Note that np.linalg.cholesky is already implemented in numba, but it does not support additional keyword arguments.
    In particular, the `inplace` argument is not supported, which is why we choose to implement our own version.
    """
    lower = op.lower
    overwrite_a = False
    check_finite = op.check_finite
    on_error = op.on_error

    dtype = node.inputs[0].dtype
    if str(dtype).startswith("complex"):
        raise NotImplementedError(
            "Complex inputs not currently supported by cholesky in Numba mode"
        )

    @numba_basic.numba_njit(inline="always")
    def nb_cholesky(a):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) found in input to cholesky"
                )
        res, info = _cholesky(a, lower, overwrite_a, check_finite)

        if on_error == "raise":
            if info > 0:
                raise np.linalg.LinAlgError(
                    "Input to cholesky is not positive definite"
                )
            if info < 0:
                raise ValueError(
                    'LAPACK reported an illegal value in input on entry to "POTRF."'
                )
        else:
            if info != 0:
                res = np.full_like(res, np.nan)

        return res

    return nb_cholesky


@numba_funcify.register(BlockDiagonal)
def numba_funcify_BlockDiagonal(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    # TODO: Why do we always inline all functions? It doesn't work with starred args, so can't use it in this case.
    @numba_basic.numba_njit(inline="never")
    def block_diag(*arrs):
        shapes = np.array([a.shape for a in arrs], dtype="int")
        out_shape = [int(s) for s in np.sum(shapes, axis=0)]
        out = np.zeros((out_shape[0], out_shape[1]), dtype=dtype)

        r, c = 0, 0
        # no strict argument because it is incompatible with numba
        for arr, shape in zip(arrs, shapes):  # noqa: B905
            rr, cc = shape
            out[r : r + rr, c : c + cc] = arr
            r += rr
            c += cc
        return out

    return block_diag


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


def _xlange(A: np.ndarray, order: str | None = None) -> float:
    """
    Placeholder for computing the norm of a matrix; used by linalg.solve. Will never be called in python mode.
    """
    return  # type: ignore


@overload(_xlange)
def xlange_impl(
    A: np.ndarray, order: str | None = None
) -> Callable[[np.ndarray, str], float]:
    """
    xLANGE returns the value of the one norm, or the Frobenius norm, or the infinity norm, or the  element of
    largest absolute value of a matrix A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "norm")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_lange = _LAPACK().numba_xlange(dtype)

    def impl(A: np.ndarray, order: str | None = None):
        _M, _N = np.int32(A.shape[-2:])  # type: ignore

        A_copy = _copy_to_fortran_order(A)

        M = val_to_int_ptr(_M)  # type: ignore
        N = val_to_int_ptr(_N)  # type: ignore
        LDA = val_to_int_ptr(_M)  # type: ignore

        NORM = (
            val_to_int_ptr(ord(order))
            if order is not None
            else val_to_int_ptr(ord("1"))
        )
        WORK = np.empty(_M, dtype=dtype)  # type: ignore

        result = numba_lange(
            NORM, M, N, A_copy.view(w_type).ctypes, LDA, WORK.view(w_type).ctypes
        )

        return result

    return impl


def _xgecon(A: np.ndarray, A_norm: float, norm: str) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a matrix; used by linalg.solve. Not used by pytensor to numbify
    graphs.
    """
    return  # type: ignore


@overload(_xgecon)
def xgecon_impl(
    A: np.ndarray, A_norm: float, norm: str
) -> Callable[[np.ndarray, float, str], tuple[np.ndarray, int]]:
    """
    Compute the condition number of a matrix A.
    """
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "gecon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_gecon = _LAPACK().numba_xgecon(dtype)

    def impl(A: np.ndarray, A_norm: float, norm: str) -> tuple[np.ndarray, int]:
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        A_NORM = np.array(A_norm, dtype=dtype)
        NORM = val_to_int_ptr(ord(norm))
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(4 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(1)

        numba_gecon(
            NORM,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            A_NORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _getrf(A, overwrite_a=False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Placeholder for LU factorization; used by linalg.solve.

    # TODO: Implement an LU_factor Op, then dispatch to this function in numba mode.
    """
    return  # type: ignore


@overload(_getrf)
def getrf_impl(
    A: np.ndarray, overwrite_a: bool = False
) -> Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "getrf")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_getrf = _LAPACK().numba_xgetrf(dtype)

    def impl(
        A: np.ndarray, overwrite_a: bool = False
    ) -> tuple[np.ndarray, np.ndarray, int]:
        _M, _N = np.int32(A.shape[-2:])  # type: ignore

        if not overwrite_a:
            A_copy = _copy_to_fortran_order(A)
        else:
            A_copy = A

        M = val_to_int_ptr(_M)  # type: ignore
        N = val_to_int_ptr(_N)  # type: ignore
        LDA = val_to_int_ptr(_M)  # type: ignore
        IPIV = np.empty(_N, dtype=np.int32)  # type: ignore
        INFO = val_to_int_ptr(0)

        numba_getrf(M, N, A_copy.view(w_type).ctypes, LDA, IPIV.ctypes, INFO)

        return A_copy, IPIV, int_ptr_to_val(INFO)

    return impl


def _getrs(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
) -> tuple[np.ndarray, int]:
    """
    Placeholder for solving a linear system with a matrix that has been LU-factored; used by linalg.solve.

    # TODO: Implement an LU_solve Op, then dispatch to this function in numba mode.
    """
    return  # type: ignore


@overload(_getrs)
def getrs_impl(
    LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int, bool], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(LU, "getrs")
    _check_scipy_linalg_matrix(B, "getrs")
    dtype = LU.dtype
    w_type = _get_underlying_float(dtype)
    numba_getrs = _LAPACK().numba_xgetrs(dtype)

    def impl(
        LU: np.ndarray, B: np.ndarray, IPIV: np.ndarray, trans: int, overwrite_b: bool
    ) -> tuple[np.ndarray, int]:
        _N = np.int32(LU.shape[-1])
        _solve_check_input_shapes(LU, B)

        B_is_1d = B.ndim == 1

        if not overwrite_b:
            B_copy = _copy_to_fortran_order(B)
        else:
            B_copy = B

        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        NRHS = 1 if B_is_1d else int(B_copy.shape[-1])

        TRANS = val_to_int_ptr(_trans_char_to_int(trans))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        IPIV = _copy_to_fortran_order(IPIV)
        INFO = val_to_int_ptr(0)

        numba_getrs(
            TRANS,
            N,
            NRHS,
            LU.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            return B_copy[..., 0], int_ptr_to_val(INFO)

        return B_copy, int_ptr_to_val(INFO)

    return impl


def _solve_gen(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve. Used as an overload target for numba to avoid unexpected side-effects
    for users who import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a="gen",
        transposed=transposed,
    )


@overload(_solve_gen)
def solve_gen_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> np.ndarray:
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        order = "I" if transposed else "1"
        norm = _xlange(A, order=order)

        N = A.shape[1]
        LU, IPIV, INFO = _getrf(A, overwrite_a=overwrite_a)
        _solve_check(N, INFO)

        X, INFO = _getrs(
            LU=LU, B=B, IPIV=IPIV, trans=transposed, overwrite_b=overwrite_b
        )
        _solve_check(N, INFO)

        RCOND, INFO = _xgecon(LU, norm, "1")
        _solve_check(N, INFO, True, RCOND)

        return X

    return impl


def _sysv(
    A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Placeholder for solving a linear system with a symmetric matrix; used by linalg.solve.
    """
    return  # type: ignore


@overload(_sysv)
def sysv_impl(
    A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
) -> Callable[
    [np.ndarray, np.ndarray, bool, bool, bool], tuple[np.ndarray, np.ndarray, int]
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "sysv")
    _check_scipy_linalg_matrix(B, "sysv")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_sysv = _LAPACK().numba_xsysv(dtype)

    def impl(
        A: np.ndarray, B: np.ndarray, lower: bool, overwrite_a: bool, overwrite_b: bool
    ):
        _LDA, _N = np.int32(A.shape[-2:])  # type: ignore
        _solve_check_input_shapes(A, B)

        if not overwrite_a:
            A_copy = _copy_to_fortran_order(A)
        else:
            A_copy = A

        B_is_1d = B.ndim == 1

        if not overwrite_b:
            B_copy = _copy_to_fortran_order(B)
        else:
            B_copy = B
        if B_is_1d:
            B_copy = np.asfortranarray(np.expand_dims(B_copy, -1))

        NRHS = 1 if B_is_1d else int(B.shape[-1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)  # type: ignore
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_LDA)  # type: ignore
        IPIV = np.empty(_N, dtype=np.int32)  # type: ignore
        LDB = val_to_int_ptr(_N)  # type: ignore
        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        INFO = val_to_int_ptr(0)

        # Workspace query
        numba_sysv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual solve
        numba_sysv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            IPIV.ctypes,
            B_copy.view(w_type).ctypes,
            LDB,
            WORK.view(w_type).ctypes,
            LWORK,
            INFO,
        )

        if B_is_1d:
            return B_copy[..., 0], IPIV, int_ptr_to_val(INFO)
        return B_copy, IPIV, int_ptr_to_val(INFO)

    return impl


def _sycon(A: np.ndarray, ipiv: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a symmetric matrix; used by linalg.solve. Never called in
    python mode.
    """
    return  # type: ignore


@overload(_sycon)
def sycon_impl(
    A: np.ndarray, ipiv: np.ndarray, anorm: float
) -> Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "sycon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_sycon = _LAPACK().numba_xsycon(dtype)

    def impl(A: np.ndarray, ipiv: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        UPLO = val_to_int_ptr(ord("L"))
        ANORM = np.array(anorm, dtype=dtype)
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(2 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_sycon(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            ipiv.ctypes,
            ANORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _solve_symmetric(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve for symmetric matrices. Used as an overload target for numba to avoid
    unexpected side-effects when users import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a="sym",
        transposed=transposed,
    )


@overload(_solve_symmetric)
def solve_symmetric_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> np.ndarray:
        _solve_check_input_shapes(A, B)

        x, ipiv, info = _sysv(A, B, lower, overwrite_a, overwrite_b)
        _solve_check(A.shape[-1], info)

        rcond, info = _sycon(A, ipiv, _xlange(A, order="I"))
        _solve_check(A.shape[-1], info, True, rcond)

        return x

    return impl


def _posv(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> tuple[np.ndarray, int]:
    """
    Placeholder for solving a linear system with a positive-definite matrix; used by linalg.solve.
    """
    return  # type: ignore


@overload(_posv)
def posv_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[
    [np.ndarray, np.ndarray, bool, bool, bool, bool, bool], tuple[np.ndarray, int]
]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_posv = _LAPACK().numba_xposv(dtype)

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> tuple[np.ndarray, int]:
        _solve_check_input_shapes(A, B)

        _N = np.int32(A.shape[-1])

        if not overwrite_a:
            A_copy = _copy_to_fortran_order(A)
        else:
            A_copy = A

        B_is_1d = B.ndim == 1

        if not overwrite_b:
            B_copy = _copy_to_fortran_order(B)
        else:
            B_copy = B

        if B_is_1d:
            B_copy = np.expand_dims(B_copy, -1)

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        NRHS = 1 if B_is_1d else int(B.shape[-1])

        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_posv(
            UPLO,
            N,
            NRHS,
            A_copy.view(w_type).ctypes,
            LDA,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            return B_copy[..., 0], int_ptr_to_val(INFO)
        return B_copy, int_ptr_to_val(INFO)

    return impl


def _pocon(A: np.ndarray, anorm: float) -> tuple[np.ndarray, int]:
    """
    Placeholder for computing the condition number of a cholesky-factorized positive-definite matrix. Used by
    linalg.solve when assume_a = "pos".
    """
    return  # type: ignore


@overload(_pocon)
def pocon_impl(
    A: np.ndarray, anorm: float
) -> Callable[[np.ndarray, float], tuple[np.ndarray, int]]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "pocon")
    dtype = A.dtype
    w_type = _get_underlying_float(dtype)
    numba_pocon = _LAPACK().numba_xpocon(dtype)

    def impl(A: np.ndarray, anorm: float):
        _N = np.int32(A.shape[-1])
        A_copy = _copy_to_fortran_order(A)

        UPLO = val_to_int_ptr(ord("L"))
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        ANORM = np.array(anorm, dtype=dtype)
        RCOND = np.empty(1, dtype=dtype)
        WORK = np.empty(3 * _N, dtype=dtype)
        IWORK = np.empty(_N, dtype=np.int32)
        INFO = val_to_int_ptr(0)

        numba_pocon(
            UPLO,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            ANORM.view(w_type).ctypes,
            RCOND.view(w_type).ctypes,
            WORK.view(w_type).ctypes,
            IWORK.ctypes,
            INFO,
        )

        return RCOND, int_ptr_to_val(INFO)

    return impl


def _solve_psd(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve for positive-definite matrices. Used as an overload target for numba to
    avoid unexpected side-effects when users import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        transposed=transposed,
        assume_a="pos",
    )


@overload(_solve_psd)
def solve_psd_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    check_finite: bool,
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_scipy_linalg_matrix(A, "solve")
    _check_scipy_linalg_matrix(B, "solve")

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        check_finite: bool,
        transposed: bool,
    ) -> np.ndarray:
        _solve_check_input_shapes(A, B)

        x, info = _posv(A, B, lower, overwrite_a, overwrite_b, check_finite, transposed)
        _solve_check(A.shape[-1], info)

        rcond, info = _pocon(x, _xlange(A))
        _solve_check(A.shape[-1], info=info, lamch=True, rcond=rcond)

        return x

    return impl


@numba_funcify.register(Solve)
def numba_funcify_Solve(op, node, **kwargs):
    assume_a = op.assume_a
    lower = op.lower
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b
    transposed = False  # TODO: Solve doesnt currently allow the transposed argument

    dtype = node.inputs[0].dtype
    if str(dtype).startswith("complex"):
        raise NotImplementedError(
            "Complex inputs not currently supported by solve in Numba mode"
        )

    if assume_a == "gen":
        solve_fn = _solve_gen
    elif assume_a == "sym":
        solve_fn = _solve_symmetric
    elif assume_a == "her":
        raise NotImplementedError(
            'Use assume_a = "sym" for symmetric real matrices. If you need compelx support, '
            "please open an issue on github."
        )
    elif assume_a == "pos":
        solve_fn = _solve_psd
    else:
        raise NotImplementedError(f"Assumption {assume_a} not supported in Numba mode")

    @numba_basic.numba_njit(inline="always")
    def solve(a, b):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input A to solve"
                )
            if np.any(np.bitwise_or(np.isinf(b), np.isnan(b))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input b to solve"
                )

        res = solve_fn(a, b, lower, overwrite_a, overwrite_b, check_finite, transposed)
        return res

    return solve


def _cho_solve(A_and_lower, B, overwrite_a=False, overwrite_b=False, check_finite=True):
    """
    Solve a positive-definite linear system using the Cholesky decomposition.
    """
    A, lower = A_and_lower
    return linalg.cho_solve((A, lower), B)


@overload(_cho_solve)
def cho_solve_impl(C, B, lower=False, overwrite_b=False, check_finite=True):
    ensure_lapack()
    _check_scipy_linalg_matrix(C, "cho_solve")
    _check_scipy_linalg_matrix(B, "cho_solve")
    dtype = C.dtype
    w_type = _get_underlying_float(dtype)
    numba_potrs = _LAPACK().numba_xpotrs(dtype)

    def impl(C, B, lower=False, overwrite_b=False, check_finite=True):
        _solve_check_input_shapes(C, B)

        _N = np.int32(C.shape[-1])
        C_copy = _copy_to_fortran_order(C)

        B_is_1d = B.ndim == 1
        if B_is_1d:
            B_copy = np.asfortranarray(np.expand_dims(B, -1))
        else:
            B_copy = _copy_to_fortran_order(B)

        NRHS = 1 if B_is_1d else int(B.shape[-1])

        UPLO = val_to_int_ptr(ord("L") if lower else ord("U"))
        N = val_to_int_ptr(_N)
        NRHS = val_to_int_ptr(NRHS)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        INFO = val_to_int_ptr(0)

        numba_potrs(
            UPLO,
            N,
            NRHS,
            C_copy.view(w_type).ctypes,
            LDA,
            B_copy.view(w_type).ctypes,
            LDB,
            INFO,
        )

        if B_is_1d:
            return B_copy[..., 0], int_ptr_to_val(INFO)
        return B_copy, int_ptr_to_val(INFO)

    return impl


@numba_funcify.register(CholeskySolve)
def numba_funcify_CholeskySolve(op, node, **kwargs):
    lower = op.lower
    overwrite_b = op.overwrite_b
    check_finite = op.check_finite

    dtype = node.inputs[0].dtype
    if str(dtype).startswith("complex"):
        raise NotImplementedError(
            "Complex inputs not currently supported by cho_solve in Numba mode"
        )

    @numba_basic.numba_njit(inline="always")
    def cho_solve(c, b):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(c), np.isnan(c))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input A to cho_solve"
                )
            if np.any(np.bitwise_or(np.isinf(b), np.isnan(b))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input b to cho_solve"
                )

        res, info = _cho_solve(
            c, b, lower=lower, overwrite_b=overwrite_b, check_finite=check_finite
        )

        if info < 0:
            raise np.linalg.LinAlgError("Illegal values found in input to cho_solve")
        elif info > 0:
            raise np.linalg.LinAlgError(
                "Matrix is not positive definite in input to cho_solve"
            )
        return res

    return cho_solve
