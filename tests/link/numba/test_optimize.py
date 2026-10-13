import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import MinimizeOp, minimize


@pytest.fixture
def native_only(monkeypatch):
    def perform(*args):
        raise AssertionError("BFGS used the Python fallback")

    monkeypatch.setattr(MinimizeOp, "perform", perform)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(), (4,)])
def test_bfgs_quadratic(dtype, shape, native_only):
    x = pt.tensor("x", shape=shape, dtype=dtype)
    target = x.type("target")
    weights = x.type("weights")
    gtol = 1e-4 if dtype == "float32" else 1e-8
    solution, success = minimize(
        (weights * (x - target) ** 2).sum(),
        x,
        optimizer_kwargs={"options": {"gtol": gtol}},
    )
    fn = function([x, target, weights], [solution, success], mode="NUMBA")
    x0 = np.zeros(shape, dtype=dtype)
    target_value = np.full(shape, 2, dtype=dtype)
    weights_value = np.full(shape, 0.25, dtype=dtype)
    for shift in (0, 1):
        expected = target_value + shift
        actual, converged = fn(x0, expected, weights_value)
        assert converged
        assert actual.shape == shape
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=0, atol=gtol)
    np.testing.assert_array_equal(x0, 0)
    np.testing.assert_array_equal(weights_value, 0.25)


@pytest.mark.filterwarnings("error")
def test_bfgs_objective_scale(native_only):
    x = pt.dvector("x")
    scale = pt.dscalar("scale")
    solution, success = minimize(scale * (x**2).sum(), x)
    fn = function([x, scale], [solution, success], mode="NUMBA")
    for weight, x0 in [
        (1e-8, np.array([10000.0])),
        (1e-12, np.array([1e8])),
        (1e-20, np.array([1e16])),
        (1e16, np.array([2.0])),
        (1e16, np.array([1.0, 2.0])),
        (1e16, np.array([2.0, -1.0])),
        (1e16, np.arange(1.0, 6.0)),
    ]:
        actual, converged = fn(x0, weight)
        assert converged, (weight, x0, actual)
        assert np.max(np.abs(2 * weight * actual)) <= 1e-5


@pytest.mark.filterwarnings("error")
def test_bfgs_finite_domain(native_only):
    x = pt.dvector("x")
    objective = ((x - 0.01) ** 2 - 1e-4 * pt.log(x)).sum()
    solution, success = minimize(objective, x)
    fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.array([0.1]))
    assert converged
    expected = (0.01 + np.sqrt(0.01**2 + 2e-4)) / 2
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.filterwarnings("error")
def test_bfgs_rosenbrock(native_only):
    x = pt.dvector("x")
    scale = pt.dscalar("scale")
    objective = (scale * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    solution, success = minimize(objective, x, optimizer_kwargs={"tol": 1e-8})
    fn = function([x, scale], [solution, success], mode="NUMBA")
    for x0, a in [(np.zeros(5), 0.5), (np.array([-1.2, 1.0]), 100.0)]:
        actual, converged = fn(x0, a)
        assert converged
        np.testing.assert_allclose(actual, 1, rtol=0, atol=1e-7)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("options", [{"maxiter": 0}, {"maxiter": 1}, {"maxiter": -1}])
def test_bfgs_iteration_limit(options, native_only):
    x = pt.dvector("x")
    solution, success = minimize((x**2).sum(), x, optimizer_kwargs={"options": options})
    fn = function([x], [solution, success], mode="NUMBA")
    x0 = np.array([2.0, -1.0])
    actual, converged = fn(x0)
    expected = scipy_minimize(
        lambda x: (np.sum(x**2), 2 * x), x0, jac=True, method="BFGS", options=options
    )
    assert converged == expected.success
    np.testing.assert_allclose(actual, expected.x)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("objective", [lambda x: -x.sum(), lambda x: pt.log(x).sum()])
def test_bfgs_failure(objective, native_only):
    x = pt.dvector("x")
    solution, success = minimize(objective(x), x)
    fn = function([x], [solution, success], mode="NUMBA")
    x0 = np.array([-1.0])
    actual, converged = fn(x0)
    assert not converged
    np.testing.assert_array_equal(actual, x0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("norm", [1, 2, np.inf, -np.inf])
def test_bfgs_stopping_options(norm, native_only):
    x = pt.dvector("x")
    weights = pt.dvector("weights")
    options = {"norm": norm, "xrtol": 1, "c1": 1e-3, "c2": 0.5}
    solution, success = minimize(
        (weights * x**2).sum(), x, optimizer_kwargs={"options": options}
    )
    fn = function([x, weights], [solution, success], mode="NUMBA")
    x0 = np.array([2.0, -1.0])
    w = np.array([0.1, 1.0])
    actual, converged = fn(x0, w)
    expected = scipy_minimize(
        lambda x: (np.sum(w * x**2), 2 * w * x),
        x0,
        jac=True,
        method="BFGS",
        options=options,
    )
    assert converged == expected.success
    np.testing.assert_allclose(actual, expected.x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "L-BFGS-B"},
        {"jac": False},
        {"optimizer_kwargs": {"options": {"hess_inv0": np.eye(2)}}},
    ],
)
def test_minimize_fallback(kwargs):
    x = pt.dvector("x")
    solution, success = minimize((x**2).sum(), x, **kwargs)
    with pytest.warns(UserWarning, match="object mode.*MinimizeOp"):
        fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.array([2.0, -1.0]))
    assert converged
    np.testing.assert_allclose(actual, 0, rtol=0, atol=1e-6)
