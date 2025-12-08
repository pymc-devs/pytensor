import warnings

import numpy as np

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.linalg.decomposition.cholesky import _cholesky
from pytensor.link.numba.dispatch.linalg.decomposition.lu import (
    _lu_1,
    _lu_2,
    _lu_3,
    _pivot_to_permutation,
)
from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _lu_factor
from pytensor.link.numba.dispatch.linalg.decomposition.qr import (
    _qr_full_no_pivot,
    _qr_full_pivot,
    _qr_r_no_pivot,
    _qr_r_pivot,
    _qr_raw_no_pivot,
    _qr_raw_pivot,
)
from pytensor.link.numba.dispatch.linalg.solve.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solve.general import _solve_gen
from pytensor.link.numba.dispatch.linalg.solve.posdef import _solve_psd
from pytensor.link.numba.dispatch.linalg.solve.symmetric import _solve_symmetric
from pytensor.link.numba.dispatch.linalg.solve.triangular import _solve_triangular
from pytensor.link.numba.dispatch.linalg.solve.tridiagonal import _solve_tridiagonal
from pytensor.tensor.slinalg import (
    LU,
    QR,
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    LUFactor,
    PivotToPermutations,
    Solve,
    SolveTriangular,
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

    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("Cholesky requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def cholesky(a):
        if a.size == 0:
            return np.zeros(a.shape, dtype=out_dtype)

        if discrete_inp:
            a = a.astype(out_dtype)
        elif check_finite:
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


@register_funcify_default_op_cache_key(PivotToPermutations)
def pivot_to_permutation(op, node, **kwargs):
    inverse = op.inverse

    @numba_basic.numba_njit
    def numba_pivot_to_permutation(piv):
        p_inv = _pivot_to_permutation(piv)

        if inverse:
            return p_inv

        return np.argsort(p_inv)

    cache_key = 1
    return numba_pivot_to_permutation, cache_key


@numba_funcify.register(LU)
def numba_funcify_LU(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LU requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    permute_l = op.permute_l
    check_finite = op.check_finite
    p_indices = op.p_indices
    overwrite_a = op.overwrite_a

    @numba_basic.numba_njit
    def lu(a):
        if a.size == 0:
            L = np.zeros(a.shape, dtype=a.dtype)
            U = np.zeros(a.shape, dtype=a.dtype)
            if permute_l:
                return L, U
            elif p_indices:
                P = np.zeros(a.shape[0], dtype="int32")
                return P, L, U
            else:
                P = np.zeros(a.shape, dtype=a.dtype)
                return P, L, U

        if discrete_inp:
            a = a.astype(out_dtype)
        elif check_finite:
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
    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LUFactor requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a

    @numba_basic.numba_njit
    def lu_factor(a):
        if a.size == 0:
            return (
                np.zeros(a.shape, dtype=out_dtype),
                np.zeros(a.shape[0], dtype="int32"),
            )

        if discrete_inp:
            a = a.astype(out_dtype)
        elif check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) found in input to cholesky"
                )

        LU, piv = _lu_factor(a, overwrite_a)

        return LU, piv

    return lu_factor


@register_funcify_default_op_cache_key(BlockDiagonal)
def numba_funcify_BlockDiagonal(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    @numba_basic.numba_njit
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
    A_dtype, b_dtype = (i.type.numpy_dtype for i in node.inputs)
    out_dtype = node.outputs[0].type.numpy_dtype

    if A_dtype.kind == "c" or b_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    must_cast_A = A_dtype != out_dtype
    if must_cast_A and config.compiler_verbose:
        print("Solve requires casting first input `A`")  # noqa: T201
    must_cast_B = b_dtype != out_dtype
    if must_cast_B and config.compiler_verbose:
        print("Solve requires casting second input `b`")  # noqa: T201

    check_finite = op.check_finite
    overwrite_a = op.overwrite_a

    assume_a = op.assume_a
    lower = op.lower
    check_finite = op.check_finite
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b
    transposed = False  # TODO: Solve doesnt currently allow the transposed argument

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

    @numba_basic.numba_njit
    def solve(a, b):
        if b.size == 0:
            return np.zeros(b.shape, dtype=out_dtype)

        if must_cast_A:
            a = a.astype(out_dtype)
        if must_cast_B:
            b = b.astype(out_dtype)
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

    A_dtype, b_dtype = (i.type.numpy_dtype for i in node.inputs)
    out_dtype = node.outputs[0].type.numpy_dtype

    if A_dtype.kind == "c" or b_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    must_cast_A = A_dtype != out_dtype
    if must_cast_A and config.compiler_verbose:
        print("SolveTriangular requires casting first input `A`")  # noqa: T201
    must_cast_B = b_dtype != out_dtype
    if must_cast_B and config.compiler_verbose:
        print("SolveTriangular requires casting second input `b`")  # noqa: T201

    @numba_basic.numba_njit
    def solve_triangular(a, b):
        if b.size == 0:
            return np.zeros(b.shape, dtype=out_dtype)
        if must_cast_A:
            a = a.astype(out_dtype)
        if must_cast_B:
            b = b.astype(out_dtype)
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

    c_dtype, b_dtype = (i.type.numpy_dtype for i in node.inputs)
    out_dtype = node.outputs[0].type.numpy_dtype

    if c_dtype.kind == "c" or b_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    must_cast_c = c_dtype != out_dtype
    if must_cast_c and config.compiler_verbose:
        print("CholeskySolve requires casting first input `c`")  # noqa: T201
    must_cast_b = b_dtype != out_dtype
    if must_cast_b and config.compiler_verbose:
        print("CholeskySolve requires casting second input `b`")  # noqa: T201

    @numba_basic.numba_njit
    def cho_solve(c, b):
        if b.size == 0:
            return np.zeros(b.shape, dtype=out_dtype)
        if must_cast_c:
            c = c.astype(out_dtype)
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(c), np.isnan(c))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input A to cho_solve"
                )

        if must_cast_b:
            b = b.astype(out_dtype)
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(b), np.isnan(b))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) in input b to cho_solve"
                )

        return _cho_solve(
            c,
            b,
            lower=lower,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )

    return cho_solve


@numba_funcify.register(QR)
def numba_funcify_QR(op, node, **kwargs):
    mode = op.mode
    check_finite = op.check_finite
    pivoting = op.pivoting
    overwrite_a = op.overwrite_a

    in_dtype = node.inputs[0].type.numpy_dtype
    integer_input = in_dtype.kind in "ibu"
    if integer_input and config.compiler_verbose:
        print("QR requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def qr(a):
        if check_finite:
            if np.any(np.bitwise_or(np.isinf(a), np.isnan(a))):
                raise np.linalg.LinAlgError(
                    "Non-numeric values (nan or inf) found in input to qr"
                )

        if integer_input:
            a = a.astype(out_dtype)

        if (mode == "full" or mode == "economic") and pivoting:
            Q, R, P = _qr_full_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return Q, R, P

        elif (mode == "full" or mode == "economic") and not pivoting:
            Q, R = _qr_full_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return Q, R

        elif mode == "r" and pivoting:
            R, P = _qr_r_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return R, P

        elif mode == "r" and not pivoting:
            (R,) = _qr_r_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return R

        elif mode == "raw" and pivoting:
            H, tau, R, P = _qr_raw_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return H, tau, R, P

        elif mode == "raw" and not pivoting:
            H, tau, R = _qr_raw_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )
            return H, tau, R

        else:
            raise NotImplementedError(
                f"QR mode={mode}, pivoting={pivoting} not supported in numba mode."
            )

    return qr
