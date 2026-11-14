import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import minimize


pytestmark = pytest.mark.filterwarnings("error")


def test_bfgs_restart(native_only):
    # The kink at 0 makes every other line search fail. Each restart from
    # steepest descent keeps the iterates shrinking toward 0 (SciPy stops at the
    # first failure, near [-0.15, -0.59]). Only the 50 accepted steps count
    # against maxiter; counting restarts too would stop near |x| = 1e-17.
    x = pt.dvector("x")
    solution, success = minimize(
        (pt.abs(x) + 0.01 * x**2).sum(),
        x,
        optimizer_kwargs={"options": {"maxiter": 50}},
    )
    fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.array([1.987, -2.75]))
    assert not converged
    assert np.max(np.abs(actual)) < 1e-30


def test_bfgs_rosenbrock(native_only):
    x = pt.dvector("x")
    scale = pt.dscalar("scale")
    objective = (scale * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    solution, success = minimize(objective, x, optimizer_kwargs={"tol": 1e-8})
    fn = function([x, scale], [solution, success], mode="NUMBA")
    for x0, scale_value in [(np.zeros(5), 0.5), (np.array([-1.2, 1.0]), 100.0)]:
        actual, converged = fn(x0, scale_value)
        assert converged
        np.testing.assert_allclose(actual, 1, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "options",
    [
        {"maxiter": 0},
        {"maxiter": 1},
        # xrtol=1 stops after the first step, which matches SciPy exactly.
        {"xrtol": 1, "c1": 1e-3, "c2": 0.5},
    ],
    ids=["maxiter-0", "maxiter-1", "xrtol-wolfe-constants"],
)
def test_bfgs_options(options, native_only):
    x = pt.dvector("x")
    weights = pt.dvector("weights")
    solution, success = minimize(
        (weights * x**2).sum(), x, optimizer_kwargs={"options": options}
    )
    fn = function([x, weights], [solution, success], mode="NUMBA")
    x0 = np.array([2.0, -1.0])
    weights_value = np.array([0.1, 1.0])
    actual, converged = fn(x0, weights_value)
    expected = scipy_minimize(
        lambda x: (np.sum(weights_value * x**2), 2 * weights_value * x),
        x0,
        jac=True,
        method="BFGS",
        options=options,
    )
    assert converged == expected.success
    np.testing.assert_allclose(actual, expected.x)

    # An empty x has a zero gradient norm, which must not count as convergence.
    _, converged = fn(np.zeros(0), np.zeros(0))
    assert not converged


@pytest.mark.parametrize(
    "objective",
    [lambda x: -x.sum(), lambda x: pt.log(x).sum()],
    ids=["unbounded", "nan-start"],
)
def test_bfgs_failure(objective, native_only):
    x = pt.dvector("x")
    solution, success = minimize(objective(x), x)
    fn = function([x], [solution, success], mode="NUMBA")
    _, converged = fn(np.array([-1.0]))
    assert not converged


@pytest.mark.parametrize("norm", [2, np.inf])
def test_bfgs_gradient_norm(norm, native_only):
    # The inf-norm run stops with the gradient's 2-norm still above gtol, so
    # each run passes only if it stops on the norm it was given.
    x = pt.dvector("x")
    weights = np.array([0.1, 1.0, 10.0])
    gtol = 0.37
    solution, success = minimize(
        (weights * x**2).sum(),
        x,
        optimizer_kwargs={"options": {"norm": norm, "gtol": gtol}},
    )
    fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.array([20.0, -5.0, 3.0]))
    assert converged
    assert np.linalg.norm(2 * weights * actual, ord=norm) <= gtol
