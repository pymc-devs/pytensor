import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import rosen, rosen_der

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import MinimizeOp, minimize


@pytest.fixture
def native_only(monkeypatch):
    def perform(*args):
        raise AssertionError("Numba minimize used the full Python fallback")

    monkeypatch.setattr(MinimizeOp, "perform", perform)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(), (4,)])
def test_minimize_quadratic(method, dtype, shape, native_only):
    x = pt.tensor("x", shape=shape, dtype=dtype)
    target = x.type("target")
    weights = x.type("weights")
    gtol = 1e-4 if dtype == "float32" else 1e-8
    solution, success = minimize(
        (weights * (x - target) ** 2).sum(),
        x,
        method=method,
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
        {"method": "CG"},
        {"method": "L-BFGS-B", "optimizer_kwargs": {"options": {"eps": 1e-8}}},
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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "bounds",
    [
        [(None, 0.5), (0, None), (1, 1), (None, None)],
        pytest.param("broadcast", id="Bounds-broadcast"),
    ],
)
def test_lbfgsb_bounds_fallback(bounds):
    from scipy.optimize import Bounds

    if bounds == "broadcast":
        bounds = Bounds(0, 0.5)
    x = pt.dvector("x")
    target = pt.dvector("target")
    solution, success = minimize(
        ((x - target) ** 2).sum(),
        x,
        method="L-BFGS-B",
        optimizer_kwargs={"bounds": bounds, "tol": 1e-8},
    )
    with pytest.warns(UserWarning, match="object mode.*MinimizeOp"):
        fn = function([x, target], [solution, success], mode="NUMBA")
    x0 = np.array([-2.0, 3.0, 0.0, -1.0])
    for shift in (0, 1):
        t = np.array([1.0, -1.0, 2.0, 0.0]) + shift
        actual, converged = fn(x0, t)
        expected = scipy_minimize(
            lambda x: (((x - t) ** 2).sum(), 2 * (x - t)),
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            tol=1e-8,
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, atol=1e-10)
    np.testing.assert_array_equal(x0, [-2.0, 3.0, 0.0, -1.0])


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
def test_lbfgsb_rosenbrock(native_only):
    x = pt.dvector("x")
    objective = (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    options = {"ftol": 0, "gtol": 1e-6}
    solution, success = minimize(
        objective, x, method="L-BFGS-B", optimizer_kwargs={"options": options}
    )
    fn = function([x], [solution, success], mode="NUMBA")
    for x0 in (np.array([-1.2, 1.0]), np.zeros(5)):
        actual, converged = fn(x0)
        expected = scipy_minimize(
            rosen, x0, jac=rosen_der, method="L-BFGS-B", options=options
        )
        assert converged == expected.success
        np.testing.assert_allclose(actual, expected.x, atol=1e-6)


@pytest.mark.filterwarnings("error")
def test_lbfgs_native(native_only, monkeypatch):
    import scipy.optimize

    def forbidden(*args, **kwargs):
        raise AssertionError("Unbounded L-BFGS called SciPy minimize")

    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    x, scale = pt.dvector("x"), pt.dscalar("scale")
    solution, success = minimize(
        scale * (x**2).sum(),
        x,
        method="L-BFGS-B",
        optimizer_kwargs={"options": {"ftol": 0, "gtol": 1e-8}},
    )
    fn = function([x, scale], [solution, success], mode="NUMBA")
    for weight, x0 in (
        (1e-8, np.array([10000.0])),
        (1e16, np.array([1.0, 2.0])),
        (1.0, np.arange(1.0, 6.0)),
    ):
        actual, converged = fn(x0, weight)
        assert converged
        assert np.max(np.abs(2 * weight * actual)) <= 1e-8
