import warnings

import numpy as np

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg.solvers.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solvers.general import _solve_gen
from pytensor.link.numba.dispatch.linalg.solvers.hermitian import _solve_hermitian
from pytensor.link.numba.dispatch.linalg.solvers.linear_control import (
    _trsyl,
)
from pytensor.link.numba.dispatch.linalg.solvers.posdef import _solve_psd
from pytensor.link.numba.dispatch.linalg.solvers.symmetric import _solve_symmetric
from pytensor.link.numba.dispatch.linalg.solvers.triangular import _solve_triangular
from pytensor.link.numba.dispatch.linalg.solvers.tridiagonal import _solve_tridiagonal
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.linear_control import TRSYL
from pytensor.tensor.linalg.solvers.psd import CholeskySolve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


@register_funcify_default_op_cache_key(Solve)
def numba_funcify_Solve(op, node, **kwargs):
    A_dtype, b_dtype = (i.type.numpy_dtype for i in node.inputs)
    out_dtype = node.outputs[0].type.numpy_dtype

    assume_a = op.assume_a

    must_cast_A = A_dtype != out_dtype
    if must_cast_A and config.compiler_verbose:
        print("Solve requires casting first input `A`")  # noqa: T201
    must_cast_B = b_dtype != out_dtype
    if must_cast_B and config.compiler_verbose:
        print("Solve requires casting second input `b`")  # noqa: T201

    overwrite_a = op.overwrite_a
    lower = op.lower
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b
    is_complex = out_dtype.kind == "c"
    transposed = False  # TODO: Solve doesnt currently allow the transposed argument

    if assume_a == "gen":
        solve_fn = _solve_gen
    elif assume_a == "sym":
        solve_fn = _solve_symmetric
    elif assume_a == "her":
        # For real inputs, Hermitian == symmetric
        solve_fn = _solve_hermitian if is_complex else _solve_symmetric
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
