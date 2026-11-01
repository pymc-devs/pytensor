import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph.traversal import ancestors
from pytensor.link.numba.dispatch.basic import numba_funcify_and_cache_key
from pytensor.tensor.optimize import RootOp, root


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("method", ["hybr", "lm"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(), (3,)])
def test_root_quadratic(method, dtype, shape, native_only):
    x = pt.tensor("x", shape=shape, dtype=dtype)
    target = x.type("target")
    solution, success = root(x**2 - target, x, method=method)
    fn = function([x, target], [solution, success], mode="NUMBA")
    x0 = np.ones(shape, dtype=dtype)
    target_value = np.full(shape, 2.0, dtype=dtype)
    actual, converged = fn(x0, target_value)
    assert converged
    assert actual.shape == shape
    assert actual.dtype == dtype
    atol = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(actual, np.sqrt(2.0), rtol=0, atol=atol)
    np.testing.assert_array_equal(x0, 1)
    np.testing.assert_array_equal(target_value, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer_kwargs": {"options": {"eps": 1e-8}}},
        {"optimizer_kwargs": {"options": {"diag": [1.0, 2.0]}}},
        {"method": "lm", "optimizer_kwargs": {"options": {"diag": [1.0, 2.0]}}},
    ],
)
def test_root_fallback(kwargs):
    x = pt.dvector("x")
    solution, success = root(x**2 - 2.0, x, **kwargs)
    with pytest.warns(UserWarning, match="object mode.*RootOp"):
        fn = function([x], [solution, success], mode="NUMBA")
    actual, converged = fn(np.ones(2))
    assert converged
    np.testing.assert_allclose(actual, np.sqrt(2.0), rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "method, options, message",
    [
        ("hybr", {"xtol": -1.0}, "'xtol' must be non-negative"),
        ("hybr", {"factor": 0.0}, "'factor' must be positive"),
        ("lm", {"ftol": -1.0}, "'ftol' must be non-negative"),
        ("lm", {"gtol": -1.0}, "'gtol' must be non-negative"),
        ("lm", {"factor": 0.0}, "'factor' must be positive"),
    ],
)
def test_root_invalid_options(method, options, message):
    x = pt.dvector("x")
    solution, _ = root(
        x**2 - 2.0, x, method=method, optimizer_kwargs={"options": options}
    )
    with pytest.raises(ValueError, match=message):
        function([x], solution, mode="NUMBA")


@pytest.mark.parametrize(
    "method, equations, message",
    [
        ("hybr", lambda x: pt.concatenate([x, x[:1]]) - 1.0, "as many equations as"),
        ("lm", lambda x: x[:1] - 1.0, "at least as many equations as"),
    ],
)
def test_root_wrong_equation_count(method, equations, message, native_only):
    x = pt.dvector("x")
    solution, _ = root(equations(x), x, method=method)
    fn = function([x], solution, mode="NUMBA")
    with pytest.raises(ValueError, match=message):
        fn(np.ones(2))


def test_root_grad(native_only):
    # The root of x**2 - a is sqrt(a), so its gradient is 1 / (2 sqrt(a)).
    x = pt.dvector("x")
    a = pt.dvector("a")
    solution, _ = root(x**2 - a, x, optimizer_kwargs={"tol": 1e-12})
    fn = function([x, a], pytensor.grad(solution.sum(), a), mode="NUMBA")
    a_value = np.array([2.0, 3.0, 5.0])
    np.testing.assert_allclose(
        fn(np.ones(3), a_value), 1 / (2 * np.sqrt(a_value)), rtol=1e-8
    )


def test_root_cache_key():
    x = pt.dvector("x")

    def cache_key(options, method="hybr"):
        solution, _ = root(
            x**2 - 2.0, x, method=method, optimizer_kwargs={"options": options}
        )
        [node] = {
            variable.owner
            for variable in ancestors([solution])
            if isinstance(variable.owner_op, RootOp)
        }
        _, key = numba_funcify_and_cache_key(node.op, node)
        assert key is not None
        return key

    keys = [
        cache_key({"xtol": 1e-8}),
        cache_key({"xtol": 1e-10}),
        cache_key({"maxfev": 50}),
        cache_key({"factor": 1.0}),
        cache_key({}, method="lm"),
        cache_key({"ftol": 1e-10}, method="lm"),
        cache_key({"gtol": 1e-3}, method="lm"),
    ]
    assert len(set(keys)) == len(keys)
    assert cache_key({"xtol": 1e-8}) == keys[0]
