import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import rosen, rosen_der

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import minimize


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "options",
    [
        {"maxiter": 0},
        {"maxiter": 1},
        {"maxfun": 1},
        {"maxls": 1},
        {"maxcor": 2, "ftol": 1e-12, "gtol": 1e-8},
    ],
)
def test_lbfgsb_options(options, native_only):
    x = pt.dvector("x")
    weights = pt.dvector("weights")
    solution, success = minimize(
        (weights * x**2).sum(),
        x,
        method="L-BFGS-B",
        optimizer_kwargs={"options": options},
    )
    fn = function([x, weights], [solution, success], mode="NUMBA")
    x0 = np.array([2.0, -1.0])
    w = np.array([0.1, 1.0])
    actual, converged = fn(x0, w)
    expected = scipy_minimize(
        lambda x: ((w * x**2).sum(), 2 * w * x),
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    assert converged == expected.success
    np.testing.assert_allclose(actual, expected.x, atol=1e-12)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "optimizer_kwargs",
    [
        {"options": {"ftol": 0, "gtol": 1e-6}},
        # Exhausts a line search mid-run from [-1.2, 1], which restarts
        # L-BFGS-B with an empty memory.
        {"options": {"ftol": 0, "gtol": 1e-6, "maxls": 2}},
        # Sets both ftol and gtol. Either one alone stops at a different point.
        {"tol": 1e-12},
    ],
    ids=["default", "restart", "tol"],
)
def test_lbfgsb_rosenbrock(optimizer_kwargs, native_only):
    x = pt.dvector("x")
    objective = (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    solution, success = minimize(
        objective, x, method="L-BFGS-B", optimizer_kwargs=optimizer_kwargs
    )
    fn = function([x], [solution, success], mode="NUMBA")
    for x0 in (np.array([-1.2, 1.0]), np.zeros(5)):
        actual, converged = fn(x0)
        expected = scipy_minimize(
            rosen, x0, jac=rosen_der, method="L-BFGS-B", **optimizer_kwargs
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, rtol=0, atol=1e-12)
