import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import rosen

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import minimize


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "jac, optimizer_kwargs",
    [
        (False, {}),
        (False, {"options": {"adaptive": True}, "tol": 1e-8}),
        (False, {"options": {"maxfev": 50}}),
        (False, {"options": {"maxiter": 20}}),
        # SciPy fails at once on a negative limit; unset limits are not -1.
        (False, {"options": {"maxiter": -1}}),
        # Runs natively on the objective output alone.
        (True, {}),
    ],
    ids=["default", "adaptive-tol", "maxfev", "maxiter", "negative-maxiter", "jac"],
)
def test_nelder_mead(jac, optimizer_kwargs, native_only):
    x = pt.dvector("x")
    objective = (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    solution, success = minimize(
        objective,
        x,
        method="Nelder-Mead",
        jac=jac,
        optimizer_kwargs=optimizer_kwargs,
    )
    fn = function([x], [solution, success], mode="NUMBA")
    for x0 in (np.array([-1.2, 1.0]), np.zeros(4)):
        actual, converged = fn(x0)
        expected = scipy_minimize(rosen, x0, method="Nelder-Mead", **optimizer_kwargs)
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("error")
def test_nelder_mead_nan_objective(native_only):
    # Starting at x[0] = 0.99, the initial simplex steps x[0] to 1.04, where log
    # is NaN. That vertex must sort last, as NumPy sorts NaN.
    x = pt.dvector("x")
    solution, success = minimize(
        ((x - 0.5) ** 2).sum() - pt.log(1 - x[0]),
        x,
        method="Nelder-Mead",
        jac=False,
    )
    fn = function([x], [solution, success], mode="NUMBA")
    for x0 in (np.array([0.99, 0.9]), np.array([0.99, 0.2, 0.7])):
        actual, converged = fn(x0)
        with np.errstate(invalid="ignore"):
            expected = scipy_minimize(
                lambda z: ((z - 0.5) ** 2).sum() - np.log(1 - z[0]),
                x0,
                method="Nelder-Mead",
            )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, rtol=0, atol=1e-12)
