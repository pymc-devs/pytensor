import numpy as np
import pytest
from scipy.optimize import root as scipy_root

import pytensor
import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import root


def rosenbrock(x):
    return pt.stack([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def broyden_tridiagonal(x):
    left = pt.concatenate([[0.0], x[:-1]])
    right = pt.concatenate([x[1:], [0.0]])
    return (3 - 2 * x) * x - left - 2 * right + 1


def box_3d(x):
    t = 0.1 * np.arange(1, 11)
    return pt.exp(-t * x[0]) - pt.exp(-t * x[1]) - x[2] * (np.exp(-t) - np.exp(-10 * t))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "equations, starts",
    [
        (rosenbrock, [[-1.2, 1.0], [-12.0, 10.0]]),
        (broyden_tridiagonal, [[-1.0] * 10, [-10.0] * 10]),
        (box_3d, [[0.0, 10.0, 20.0], [0.0, 10.0, 5.0]]),
    ],
    ids=["rosenbrock", "broyden_tridiagonal", "box_3d"],
)
@pytest.mark.parametrize(
    "optimizer_kwargs",
    [
        {},
        {"tol": 1e-12},
        {"options": {"ftol": 1e-12}},
        {"options": {"gtol": 1e-3}},
        {"options": {"maxiter": 5}},
        {"options": {"maxiter": 0}},
        {"options": {"factor": 1.0}},
    ],
    ids=["default", "tol", "ftol", "gtol", "maxiter", "maxiter_unset", "factor"],
)
def test_lm(equations, starts, optimizer_kwargs, native_only):
    # SciPy's compiled MINPACK may fuse multiply-adds that the Numba port does
    # not, so the two agree to rounding rather than bit for bit.
    x = pt.dvector("x")
    system = equations(x)
    solution, success = root(system, x, method="lm", optimizer_kwargs=optimizer_kwargs)
    fn = function([x], [solution, success], mode="NUMBA")
    residual = function(
        [x], [system, pytensor.gradient.jacobian(system, x)], mode="NUMBA"
    )
    for x0 in map(np.array, starts):
        actual, converged = fn(x0)
        expected = scipy_root(
            lambda z: tuple(residual(z)),
            x0,
            jac=True,
            method="lm",
            **optimizer_kwargs,
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, rtol=0, atol=1e-10)


def test_lm_jacobian_view_of_x(native_only):
    # The Jacobian of x**2 / 2 compiles to a view of x, and lm reuses the
    # Jacobian of each accepted point.
    x = pt.dscalar("x")
    system = x**2 / 2 + 2
    solution, success = root(system, x, method="lm")
    fn = function([x], [solution, success], mode="NUMBA")
    residual = function(
        [x],
        [pt.atleast_1d(system), pt.atleast_2d(pytensor.gradient.jacobian(system, x))],
        mode="NUMBA",
    )
    for x0 in [4.0, 100.0]:
        actual, converged = fn(np.array(x0))
        expected = scipy_root(
            lambda z: tuple(residual(z[0])),
            [x0],
            jac=True,
            method="lm",
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x[0], rtol=0, atol=1e-10)


def test_lm_rank_deficient(native_only):
    # MGH 33: the Jacobian has rank one, so the minimizers form a subspace and
    # rounding decides which one lm reaches. The minimum sum of squares is
    # m (m - 1) / (2 (2 m + 1)) for m equations.
    x = pt.dvector("x")
    m = 10
    residuals = np.arange(1, m + 1) * (pt.arange(1, 6) * x).sum() - 1
    solution, success = root(residuals, x, method="lm")
    fn = function([x], [solution, success], mode="NUMBA")
    residual = function([x], residuals, mode="NUMBA")
    for x0 in [np.ones(5), np.array([-3.0, 1.0, 2.0, 0.5, 4.0])]:
        actual, converged = fn(x0)
        at_solution = residual(actual)
        assert converged
        np.testing.assert_allclose(
            at_solution @ at_solution, m * (m - 1) / (2 * (2 * m + 1)), rtol=1e-10
        )
