import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph.traversal import ancestors
from pytensor.link.numba.dispatch.basic import numba_funcify_and_cache_key
from pytensor.tensor.optimize import MinimizeOp, minimize


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "Nelder-Mead"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(), (4,)])
def test_minimize_quadratic(method, dtype, shape, native_only):
    x = pt.tensor("x", shape=shape, dtype=dtype)
    target = x.type("target")
    weights = x.type("weights")
    tol = 1e-4 if dtype == "float32" else 1e-8
    solution, success = minimize(
        (weights * (x - target) ** 2).sum(),
        x,
        method=method,
        optimizer_kwargs={"tol": tol},
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
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tol)
    np.testing.assert_array_equal(x0, 0)
    np.testing.assert_array_equal(weights_value, 0.25)


@pytest.mark.filterwarnings("error")
def test_minimize_inner_inplace(native_only):
    # The inner graph may work in place on its own buffers, never on its inputs.
    x = pt.dvector("x")
    a = pt.dvector("a")
    residual = pt.cumsum(x) - a
    solution, success = minimize(residual.dot(residual), x)
    fn = function([x, a], [solution, success], mode="NUMBA")
    [node] = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, MinimizeOp)
    ]
    assert any(inner.op.destroy_map for inner in node.op.fgraph.apply_nodes)
    x0 = np.zeros(3)
    a_value = np.array([1.0, 2.0, 3.0])
    actual, converged = fn(x0, a_value)
    assert converged
    np.testing.assert_allclose(actual, 1, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(x0, 0)
    np.testing.assert_array_equal(a_value, [1.0, 2.0, 3.0])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "method, optimizer_kwargs",
    [("BFGS", {}), ("L-BFGS-B", {"options": {"ftol": 0}})],
    ids=["BFGS", "L-BFGS-B"],
)
def test_minimize_objective_scale(method, optimizer_kwargs, native_only):
    x = pt.dvector("x")
    scale = pt.dscalar("scale")
    solution, success = minimize(
        scale * (x**2).sum(),
        x,
        method=method,
        optimizer_kwargs=optimizer_kwargs,
    )
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
@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
def test_minimize_finite_domain(method, native_only):
    # The first trial step leaves the domain of log, where f is NaN.
    x = pt.dvector("x")
    objective = ((x - 0.01) ** 2 - 1e-4 * pt.log(x)).sum()
    solution, success = minimize(objective, x, method=method)
    fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.array([0.1]))
    assert converged
    expected = (0.01 + np.sqrt(0.01**2 + 2e-4)) / 2
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "CG"},
        {"method": "L-BFGS-B", "optimizer_kwargs": {"options": {"eps": 1e-8}}},
        {"method": "L-BFGS-B", "optimizer_kwargs": {"bounds": [(-1, 1), (-2, None)]}},
        {
            "method": "Nelder-Mead",
            "jac": False,
            "optimizer_kwargs": {
                "bounds": [(-3, 3), (-2, None)],
                "options": {"xatol": 1e-8, "fatol": 1e-16},
            },
        },
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


@pytest.mark.parametrize(
    "method, options, message",
    [
        ("BFGS", {"c1": 0.9, "c2": 0.5}, "'c1' and 'c2' must satisfy"),
        ("L-BFGS-B", {"maxcor": 0}, "'maxcor' must be positive"),
        ("L-BFGS-B", {"maxls": 0}, "'maxls' must be positive"),
        ("L-BFGS-B", {"ftol": -1.0}, "'ftol' must be non-negative"),
        ("L-BFGS-B", {"gtol": -1.0}, "'gtol' must be non-negative"),
    ],
)
def test_minimize_invalid_options(method, options, message):
    x = pt.dvector("x")
    solution, _ = minimize(
        (x**2).sum(),
        x,
        method=method,
        optimizer_kwargs={"options": options},
    )
    with pytest.raises(ValueError, match=message):
        function([x], solution, mode="NUMBA")


def test_minimize_cache_key():
    x = pt.dvector("x")

    def cache_key(method, options):
        solution, _ = minimize(
            (x**2).sum(),
            x,
            method=method,
            optimizer_kwargs={"options": options},
        )
        [node] = {
            variable.owner
            for variable in ancestors([solution])
            if isinstance(variable.owner_op, MinimizeOp)
        }
        _, key = numba_funcify_and_cache_key(node.op, node)
        assert key is not None
        return key

    keys = [
        cache_key("BFGS", {"gtol": 1e-5}),
        cache_key("BFGS", {"gtol": 1e-6}),
        cache_key("L-BFGS-B", {"gtol": 1e-5}),
        cache_key("L-BFGS-B", {"gtol": 1e-6}),
    ]
    assert len(set(keys)) == len(keys)
    assert cache_key("BFGS", {"gtol": 1e-5}) == keys[0]


@pytest.mark.filterwarnings("error")
def test_minimize_nested(native_only):
    # The inner solution x*(theta) solves 2 (x - theta) + 0.4 x^3 = 0, so the
    # outer problem x*(theta) = target has theta = target + 0.2 target^3.
    x = pt.dvector("x")
    theta = pt.dvector("theta")
    target = pt.dvector("target")
    inner, _ = minimize(
        ((x - theta) ** 2 + 0.1 * x**4).sum(),
        x,
        optimizer_kwargs={"tol": 1e-12},
    )
    solution, success = minimize(
        ((inner - target) ** 2).sum(),
        theta,
        optimizer_kwargs={"tol": 1e-12},
    )
    fn = function([theta, x, target], [solution, success], mode="NUMBA")
    target_value = np.array([0.5, -1.0, 2.0])
    actual, converged = fn(np.zeros(3), np.zeros(3), target_value)
    assert converged
    np.testing.assert_allclose(actual, target_value + 0.2 * target_value**3, rtol=1e-8)


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
def test_minimize_grad(method, native_only):
    x = pt.dvector("x")
    a = pt.dvector("a")
    b = pt.dscalar("b")
    solution, _ = minimize(
        ((x - a) ** 2 + b * x**2).sum(),
        x,
        method=method,
        optimizer_kwargs={"tol": 1e-12},
    )
    grads = pytensor.grad(solution.sum(), [a, b])
    fn = function([x, a, b], grads, mode="NUMBA")
    a_value = np.array([1.0, -2.0, 3.0])
    b_value = 0.5
    grad_a, grad_b = fn(np.zeros(3), a_value, b_value)
    np.testing.assert_allclose(grad_a, 1 / (1 + b_value), rtol=1e-8)
    np.testing.assert_allclose(grad_b, -a_value.sum() / (1 + b_value) ** 2, rtol=1e-8)
