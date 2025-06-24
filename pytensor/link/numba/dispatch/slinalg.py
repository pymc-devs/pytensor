import warnings

import numpy as np

from pytensor.link.numba.dispatch.basic import numba_funcify, numba_njit
from pytensor.link.numba.dispatch.linalg.decomposition.cholesky import _cholesky
from pytensor.link.numba.dispatch.linalg.decomposition.lu import (
    _lu_1,
    _lu_2,
    _lu_3,
    _pivot_to_permutation,
)
from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _lu_factor
from pytensor.link.numba.dispatch.linalg.solve.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solve.general import _solve_gen
from pytensor.link.numba.dispatch.linalg.solve.posdef import _solve_psd
from pytensor.link.numba.dispatch.linalg.solve.symmetric import _solve_symmetric
from pytensor.link.numba.dispatch.linalg.solve.triangular import _solve_triangular
from pytensor.link.numba.dispatch.linalg.solve.tridiagonal import _solve_tridiagonal
from pytensor.tensor.slinalg import (
    LU,
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    LUFactor,
    PivotToPermutations,
    Solve,
    SolveTriangular,
)
from pytensor.tensor.type import complex_dtypes


_COMPLEX_DTYPE_NOT_SUPPORTED_MSG = (
    "Complex dtype for {op} not supported in numba mode. "
    "If you need this functionality, please open an issue at: https://github.com/pymc-devs/pytensor"
)


@numba_funcify.register(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    """
    Overload scipy.linalg.cholesky with a numba function.

    Note that np.linalg.cholesky is already implemented in numba, but it does not support additional keyword arguments.
    In particular, the `inplace` argument is not supported, which is why we choose to implement our own version.
    """
    lower = op.lower
    overwrite_a = op.overwrite_a
    check_finite = op.check_finite
    on_error = op.on_error

    dtype = node.inputs[0].dtype
    if dtype in complex_dtypes:
        raise NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    @numba_njit
    def cholesky(a):
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

    return cholesky


@numba_funcify.register(PivotToPermutations)
def pivot_to_permutation(op, node, **kwargs):
    inverse = op.inverse
    dtype = node.outputs[0].dtype

    @numba_njit
    def numba_pivot_to_permutation(piv):
        p_inv = _pivot_to_permutation(piv, dtype)

        if inverse:
            return p_inv

        return np.argsort(p_inv)

    return numba_pivot_to_permutation


@numba_funcify.register(LU)
def numba_funcify_LU(op, node, **kwargs):
    permute_l = op.permute_l
    check_finite = op.check_finite
    p_indices = op.p_indices
    overwrite_a = op.overwrite_a

    dtype = node.inputs[0].dtype
    if dtype in complex_dtypes:
        NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    @numba_njit(inline="always")
    def lu(a):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) found in input to lu"
                )

        if p_indices:
            res = _lu_1(
                a,
                permute_l=permute_l,
                check_finite=check_finite,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        elif permute_l:
            res = _lu_2(
                a,
                permute_l=permute_l,
                check_finite=check_finite,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        else:
            res = _lu_3(
                a,
                permute_l=permute_l,
                check_finite=check_finite,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )

        return res

    return lu


@numba_funcify.register(LUFactor)
def numba_funcify_LUFactor(op, node, **kwargs):
    dtype = node.inputs[0].dtype
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a

    if dtype in complex_dtypes:
        NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    @numba_njit
    def lu_factor(a):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) found in input to cholesky"
                )

        LU, piv = _lu_factor(a, overwrite_a)

        return LU, piv

    return lu_factor


@numba_funcify.register(BlockDiagonal)
def numba_funcify_BlockDiagonal(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    # TODO: Why do we always inline all functions? It doesn't work with starred args, so can't use it in this case.
    @numba_njit
    def block_diag(*arrs):
        shapes = np.array([a.shape for a in arrs], dtype="int")
        out_shape = [int(s) for s in np.sum(shapes, axis=0)]
        out = np.zeros((out_shape[0], out_shape[1]), dtype=dtype)

        r, c = 0, 0
        # no strict argument because it is incompatible with numba
        for arr, shape in zip(arrs, shapes):
            rr, cc = shape
            out[r : r + rr, c : c + cc] = arr
            r += rr
            c += cc
        return out

    return block_diag


@numba_funcify.register(Solve)
def numba_funcify_Solve(op, node, **kwargs):
    assume_a = op.assume_a
    lower = op.lower
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b
    transposed = False  # TODO: Solve doesnt currently allow the transposed argument

    dtype = node.inputs[0].dtype
    if dtype in complex_dtypes:
        raise NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    if assume_a == "gen":
        solve_fn = _solve_gen
    elif assume_a == "sym":
        solve_fn = _solve_symmetric
    elif assume_a == "her":
        # We already ruled out complex inputs
        solve_fn = _solve_symmetric
    elif assume_a == "pos":
        solve_fn = _solve_psd
    elif assume_a == "tridiagonal":
        solve_fn = _solve_tridiagonal
    else:
        warnings.warn(
            f"Numba assume_a={assume_a} not implemented. Falling back to general solve.\n"
            f"If appropriate, you may want to set assume_a to one of 'sym', 'pos', 'her', 'triangular' or 'tridiagonal' to improve performance.",
            UserWarning,
        )
        solve_fn = _solve_gen

    @numba_njit
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


@numba_funcify.register(SolveTriangular)
def numba_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite
    overwrite_b = op.overwrite_b
    b_ndim = op.b_ndim

    dtype = node.inputs[0].dtype
    if dtype in complex_dtypes:
        raise NotImplementedError(
            _COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op="Solve Triangular")
        )

    @numba_njit
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


@numba_funcify.register(CholeskySolve)
def numba_funcify_CholeskySolve(op, node, **kwargs):
    lower = op.lower
    overwrite_b = op.overwrite_b
    check_finite = op.check_finite

    dtype = node.inputs[0].dtype
    if dtype in complex_dtypes:
        raise NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    @numba_njit
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

        return _cho_solve(
            c, b, lower=lower, overwrite_b=overwrite_b, check_finite=check_finite
        )

    return cho_solve
