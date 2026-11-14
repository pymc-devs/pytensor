import numpy as np
import pytest
from scipy.optimize import root as scipy_root

import pytensor
import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import root


pytestmark = pytest.mark.filterwarnings("error")


def rosenbrock(x):
    return pt.stack([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def broyden_tridiagonal(x):
    left = pt.concatenate([[0.0], x[:-1]])
    right = pt.concatenate([x[1:], [0.0]])
    return (3 - 2 * x) * x - left - 2 * right + 1


BROYDEN_STARTS = [[-1.0] * 10, [-10.0] * 10]


@pytest.mark.parametrize(
    "equations, starts, optimizer_kwargs",
    [
        (rosenbrock, [[-1.2, 1.0], [-12.0, 10.0]], {}),
        (broyden_tridiagonal, BROYDEN_STARTS, {}),
        (broyden_tridiagonal, BROYDEN_STARTS, {"tol": 1e-12}),
        (broyden_tridiagonal, BROYDEN_STARTS, {"options": {"maxfev": 5}}),
        (broyden_tridiagonal, BROYDEN_STARTS, {"options": {"factor": 1.0}}),
    ],
    ids=["rosenbrock", "broyden", "broyden-tol", "broyden-maxfev", "broyden-factor"],
)
def test_hybr(equations, starts, optimizer_kwargs, native_only):
    # SciPy's compiled MINPACK may fuse multiply-adds that the Numba port does
    # not, so the two agree to rounding rather than bit for bit.
    x = pt.dvector("x")
    system = equations(x)
    solution, success = root(system, x, optimizer_kwargs=optimizer_kwargs)
    fn = function([x], [solution, success], mode="NUMBA")
    residuals_and_jacobian = function(
        [x], [system, pytensor.gradient.jacobian(system, x)], mode="NUMBA"
    )
    for x0 in map(np.array, starts):
        actual, converged = fn(x0)
        expected = scipy_root(
            lambda z: tuple(residuals_and_jacobian(z)),
            x0,
            jac=True,
            method="hybr",
            **optimizer_kwargs,
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, rtol=0, atol=1e-10)


def test_hybr_jacobian_view_of_x(native_only):
    # The Jacobian of x**2 / 2 compiles to a view of x. Without a root, hybr
    # takes poor accepted steps and then reuses the Jacobian of its last point.
    x = pt.dscalar("x")
    system = x**2 / 2 + 2
    solution, success = root(system, x)
    fn = function([x], [solution, success], mode="NUMBA")
    residuals_and_jacobian = function(
        [x],
        [pt.atleast_1d(system), pt.atleast_2d(pytensor.gradient.jacobian(system, x))],
        mode="NUMBA",
    )
    for x0 in [4.0, 100.0]:
        actual, converged = fn(np.array(x0))
        expected = scipy_root(
            lambda z: tuple(residuals_and_jacobian(z[0])),
            [x0],
            jac=True,
            method="hybr",
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x[0], rtol=0, atol=1e-10)
