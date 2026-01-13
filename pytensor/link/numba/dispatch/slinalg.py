import warnings

import numpy as np

from pytensor import config
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
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
from pytensor.link.numba.dispatch.linalg.decomposition.schur import (
    schur_complex,
    schur_real,
)
from pytensor.link.numba.dispatch.linalg.solve.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solve.general import _solve_gen
from pytensor.link.numba.dispatch.linalg.solve.linear_control import (
    _trsyl,
)
from pytensor.link.numba.dispatch.linalg.solve.posdef import _solve_psd
from pytensor.link.numba.dispatch.linalg.solve.symmetric import _solve_symmetric
from pytensor.link.numba.dispatch.linalg.solve.triangular import _solve_triangular
from pytensor.link.numba.dispatch.linalg.solve.tridiagonal import _solve_tridiagonal
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
)
from pytensor.tensor._linalg.solve.linear_control import TRSYL
from pytensor.tensor.slinalg import (
    LU,
    QR,
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    LUFactor,
    PivotToPermutations,
    Schur,
    Solve,
    SolveTriangular,
)


@register_funcify_default_op_cache_key(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    """
    Overload scipy.linalg.cholesky with a numba function.

    Note that np.linalg.cholesky is already implemented in numba, but it does not support additional keyword arguments.
    In particular, the `inplace` argument is not supported, which is why we choose to implement our own version.
    """
    lower = op.lower
    overwrite_a = op.overwrite_a

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

        return _cholesky(a, lower, overwrite_a)

    cache_version = 2
    return cholesky, cache_version


@register_funcify_default_op_cache_key(PivotToPermutations)
def pivot_to_permutation(op, node, **kwargs):
    inverse = op.inverse

    @numba_basic.numba_njit
    def numba_pivot_to_permutation(piv):
        p_inv = _pivot_to_permutation(piv)

        if inverse:
            return p_inv

        return np.argsort(p_inv)

    cache_version = 2
    return numba_pivot_to_permutation, cache_version


@register_funcify_default_op_cache_key(LU)
def numba_funcify_LU(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LU requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    permute_l = op.permute_l
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

        if p_indices:
            res = _lu_1(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        elif permute_l:
            res = _lu_2(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        else:
            res = _lu_3(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )

        return res

    cache_version = 2
    return lu, cache_version


@register_funcify_default_op_cache_key(LUFactor)
def numba_funcify_LUFactor(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LUFactor requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
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

        LU, piv = _lu_factor(a, overwrite_a)

        return LU, piv

    cache_version = 2
    return lu_factor, cache_version


@register_funcify_default_op_cache_key(BlockDiagonal)
def numba_funcify_BlockDiagonal(op, node, **kwargs):
    """

    Because we have variadic arguments we need to use codegen.

    The generated code looks something like:

    def block_diagonal(arr0, arr1, arr2):
        out_r = arr0.shape[0] + arr1.shape[0] + arr2.shape[0]
        out_c = arr0.shape[1] + arr1.shape[1] + arr2.shape[1]
        out = np.zeros((out_r, out_c), dtype=np.float64)

        r, c = 0, 0
        rr, cc = arr0.shape
        out[r: r + rr, c: c + cc] = arr0
        r += rr
        c += cc

        rr, cc = arr1.shape
        out[r: r + rr, c: c + cc] = arr1
        r += rr
        c += cc

        rr, cc = arr2.shape
        out[r: r + rr, c: c + cc] = arr2
        r += rr
        c += cc

        return out
    """
    dtype = node.outputs[0].dtype
    n_inp = len(node.inputs)

    arg_names = [f"arr{i}" for i in range(n_inp)]
    code = [
        f"def block_diagonal({', '.join(arg_names)}):",
        CODE_TOKEN.INDENT,
        f"out_r = {' + '.join(f'{a}.shape[0]' for a in arg_names)}",
        f"out_c = {' + '.join(f'{a}.shape[1]' for a in arg_names)}",
        f"out = np.zeros((out_r, out_c), dtype=np.{dtype})",
        CODE_TOKEN.EMPTY_LINE,
        "r, c = 0, 0",
    ]
    for i, arg_name in enumerate(arg_names):
        code.extend(
            [
                f"rr, cc = {arg_name}.shape",
                f"out[r: r + rr, c: c + cc] = {arg_name}",
                "r += rr",
                "c += cc",
                CODE_TOKEN.EMPTY_LINE,
            ]
        )
    code.append("return out")

    code_txt = build_source_code(code)
    block_diag = compile_numba_function_src(
        code_txt,
        "block_diagonal",
        globals() | {"np": np},
    )

    cache_version = 1
    return numba_basic.numba_njit(block_diag), cache_version


@register_funcify_default_op_cache_key(Solve)
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

    overwrite_a = op.overwrite_a
    assume_a = op.assume_a
    lower = op.lower
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

        return solve_fn(a, b, lower, overwrite_a, overwrite_b, transposed)

    cache_version = 2
    return solve, cache_version


@register_funcify_default_op_cache_key(SolveTriangular)
def numba_funcify_SolveTriangular(op, node, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    overwrite_b = op.overwrite_b

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

        return _solve_triangular(
            a,
            b,
            trans=0,  # transposing is handled explicitly on the graph, so we never use this argument
            lower=lower,
            unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b,
        )

    cache_version = 2
    return solve_triangular, cache_version


@register_funcify_default_op_cache_key(CholeskySolve)
def numba_funcify_CholeskySolve(op, node, **kwargs):
    lower = op.lower
    overwrite_b = op.overwrite_b

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

        if must_cast_b:
            b = b.astype(out_dtype)

        return _cho_solve(
            c,
            b,
            lower=lower,
            overwrite_b=overwrite_b,
        )

    cache_version = 2
    return cho_solve, cache_version


@register_funcify_default_op_cache_key(QR)
def numba_funcify_QR(op, node, **kwargs):
    mode = op.mode
    pivoting = op.pivoting
    overwrite_a = op.overwrite_a

    in_dtype = node.inputs[0].type.numpy_dtype
    integer_input = in_dtype.kind in "ibu"
    if integer_input and config.compiler_verbose:
        print("QR requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def qr(a):
        if integer_input:
            a = a.astype(out_dtype)

        if (mode == "full" or mode == "economic") and pivoting:
            Q, R, P = _qr_full_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return Q, R, P

        elif (mode == "full" or mode == "economic") and not pivoting:
            Q, R = _qr_full_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return Q, R

        elif mode == "r" and pivoting:
            R, P = _qr_r_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return R, P

        elif mode == "r" and not pivoting:
            (R,) = _qr_r_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return R

        elif mode == "raw" and pivoting:
            H, tau, R, P = _qr_raw_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return H, tau, R, P

        elif mode == "raw" and not pivoting:
            H, tau, R = _qr_raw_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return H, tau, R

        else:
            raise NotImplementedError(
                f"QR mode={mode}, pivoting={pivoting} not supported in numba mode."
            )

    cache_version = 2
    return qr, cache_version


@register_funcify_default_op_cache_key(Schur)
def numba_funcify_Schur(op, node, **kwargs):
    output = op.output
    overwrite_a = op.overwrite_a
    sort = op.sort

    if sort is not None:
        if config.compiler_verbose:
            print(  # noqa: T201
                "Schur is not implemented in numba mode when `sort` is not None, "
                "falling back to object mode"
            )
        return generate_fallback_impl(op, node=node, **kwargs)

    in_dtype = node.inputs[0].type.numpy_dtype
    out_dtype = node.outputs[0].type.numpy_dtype
    integer_input = in_dtype.kind in "ibu"
    complex_input = in_dtype.kind in "cz"
    needs_complex_cast = in_dtype.kind in "fd" and output == "complex"

    # Disable overwrite_a for dtype conversion (real->complex upcast)
    if needs_complex_cast:
        overwrite_a = False
        if config.compiler_verbose:
            print(  # noqa: T201
                "Schur: disabling overwrite_a due to dtype conversion (casting prevents in-place operation)"
            )

    if integer_input and config.compiler_verbose:
        print("Schur requires casting discrete input to float")  # noqa: T201

    # Complex input always produces complex output, and output == "complex" forces complex output
    if complex_input or output == "complex":

        @numba_basic.numba_njit
        def schur(a):
            if integer_input:
                a = a.astype(out_dtype)
            elif needs_complex_cast:
                a = a.astype(out_dtype)
            T, Z = schur_complex(a, lwork=None, overwrite_a=overwrite_a)
            return T, Z
    else:
        # Real input with real output
        @numba_basic.numba_njit
        def schur(a):
            if integer_input:
                a = a.astype(out_dtype)
            T, Z = schur_real(a, lwork=None, overwrite_a=overwrite_a)
            return T, Z

    cache_version = 1
    return schur, cache_version


@register_funcify_default_op_cache_key(TRSYL)
def numba_funcify_TRSYL(op, node, **kwargs):
    in_dtype_a = node.inputs[0].type.numpy_dtype
    in_dtype_b = node.inputs[1].type.numpy_dtype
    in_dtype_c = node.inputs[2].type.numpy_dtype
    out_dtype = node.outputs[0].type.numpy_dtype

    overwrite_c = op.overwrite_c

    must_cast_a = in_dtype_a != out_dtype
    if must_cast_a and config.compiler_verbose:
        print("TRSYL requires casting first input `A`")  # noqa: T201
    must_cast_b = in_dtype_b != out_dtype
    if must_cast_b and config.compiler_verbose:
        print("TRSYL requires casting second input `B`")  # noqa: T201
    must_cast_c = in_dtype_c != out_dtype
    if must_cast_c and config.compiler_verbose:
        print("TRSYL requires casting third input `C`")  # noqa: T201

    @numba_basic.numba_njit
    def trsyl(a, b, c):
        if must_cast_a:
            a = a.astype(out_dtype)
        if must_cast_b:
            b = b.astype(out_dtype)
        if must_cast_c:
            c = c.astype(out_dtype)

        x = _trsyl(a, b, c, overwrite_c=overwrite_c)
        return x

    cache_version = 1
    return trsyl, cache_version
