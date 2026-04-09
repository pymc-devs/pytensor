import warnings

import jax

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor._linalg.solve.general import Solve
from pytensor.tensor._linalg.solve.linear_control import SolveSylvester
from pytensor.tensor._linalg.solve.psd import CholeskySolve
from pytensor.tensor._linalg.solve.triangular import SolveTriangular


@jax_funcify.register(Solve)
def jax_funcify_Solve(op, **kwargs):
    assume_a = op.assume_a
    lower = op.lower
    b_is_vec = op.b_ndim == 1

    if assume_a == "tridiagonal":

        def solve(a, b):
            dl = jax.numpy.diagonal(a, offset=-1, axis1=-2, axis2=-1)
            d = jax.numpy.diagonal(a, offset=0, axis1=-2, axis2=-1)
            du = jax.numpy.diagonal(a, offset=1, axis1=-2, axis2=-1)

            dl = jax.numpy.pad(dl, (1, 0))
            du = jax.numpy.pad(du, (0, 1))

            if b_is_vec:
                b = jax.numpy.expand_dims(b, -1)

            res = jax.lax.linalg.tridiagonal_solve(dl, d, du, b)

            if b_is_vec:
                return jax.numpy.squeeze(res, -1)

            return res

    else:
        if assume_a not in ("gen", "sym", "her", "pos"):
            warnings.warn(
                f"JAX solve does not support assume_a={op.assume_a}. Defaulting to assume_a='gen'.\n"
                f"If appropriate, you may want to set assume_a to one of 'sym', 'pos', 'her' or 'tridiagonal' to improve performance.",
                UserWarning,
            )
            assume_a = "gen"

        def solve(a, b):
            return jax.scipy.linalg.solve(a, b, lower=lower, assume_a=assume_a)

    return solve


@jax_funcify.register(SolveTriangular)
def jax_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal

    def solve_triangular(A, b):
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            lower=lower,
            trans=0,
            unit_diagonal=unit_diagonal,
            check_finite=False,
        )

    return solve_triangular


@jax_funcify.register(CholeskySolve)
def jax_funcify_ChoSolve(op, **kwargs):
    lower = op.lower
    overwrite_b = op.overwrite_b

    def cho_solve(c, b):
        return jax.scipy.linalg.cho_solve(
            (c, lower), b, check_finite=False, overwrite_b=overwrite_b
        )

    return cho_solve


@jax_funcify.register(SolveSylvester)
def jax_funcify_SolveSylsterer(op, **kwargs):
    def solve_sylvester(a, b, c):
        return jax.scipy.linalg.solve_sylvester(a, b, c)

    return solve_sylvester
