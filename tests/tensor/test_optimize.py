import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.graph import Apply, Op
from pytensor.tensor import scalar
from pytensor.tensor.optimize import minimize, minimize_scalar, root, root_scalar
from tests import unittest_tools as utt


floatX = config.floatX


def test_minimize_scalar():
    x = pt.scalar("x")
    a = pt.scalar("a")
    c = pt.scalar("c")

    b = a * 2
    b.name = "b"
    out = (x - b * c) ** 2

    minimized_x, success = minimize_scalar(out, x)

    a_val = 2.0
    c_val = 3.0

    f = function([a, c, x], [minimized_x, success])

    minimized_x_val, success_val = f(a_val, c_val, 0.0)

    assert success_val
    np.testing.assert_allclose(minimized_x_val, (2 * a_val * c_val))

    def f(x, a, b):
        objective = (x - a * b) ** 2
        out = minimize_scalar(objective, x)[0]
        return out

    utt.verify_grad(f, [0.0, a_val, c_val], eps=1e-6)


def test_simple_minimize():
    x = pt.scalar("x")
    a = pt.scalar("a")
    c = pt.scalar("c")

    b = a * 2
    b.name = "b"
    out = (x - b * c) ** 2

    minimized_x, success = minimize(out, x)

    a_val = 2.0
    c_val = 3.0

    f = function([a, c, x], [minimized_x, success])

    minimized_x_val, success_val = f(a_val, c_val, 0.0)

    assert success_val
    np.testing.assert_allclose(
        minimized_x_val,
        2 * a_val * c_val,
        atol=1e-8 if config.floatX == "float64" else 1e-6,
        rtol=1e-8 if config.floatX == "float64" else 1e-6,
    )

    def f(x, a, b):
        objective = (x - a * b) ** 2
        out = minimize(objective, x)[0]
        return out

    utt.verify_grad(f, [0.0, a_val, c_val], eps=1e-6)


@pytest.mark.parametrize(
    "method, jac, hess",
    [
        ("Newton-CG", True, True),
        ("L-BFGS-B", True, False),
        ("powell", False, False),
    ],
    ids=["Newton-CG", "L-BFGS-B", "powell"],
)
def test_minimize_vector_x(method, jac, hess):
    def rosenbrock_shifted_scaled(x, a, b):
        return (a * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum() + b

    x = pt.tensor("x", shape=(None,))
    a = pt.scalar("a")
    b = pt.scalar("b")

    objective = rosenbrock_shifted_scaled(x, a, b)
    minimized_x, success = minimize(
        objective, x, method=method, jac=jac, hess=hess, optimizer_kwargs={"tol": 1e-16}
    )

    fn = pytensor.function([x, a, b], [minimized_x, success])

    a_val = np.array(0.5, dtype=floatX)
    b_val = np.array(1.0, dtype=floatX)
    x0 = np.zeros((5,)).astype(floatX)
    x_star_val, success = fn(x0, a_val, b_val)

    assert success

    np.testing.assert_allclose(
        x_star_val,
        np.ones_like(x_star_val),
        atol=1e-8 if config.floatX == "float64" else 1e-3,
        rtol=1e-8 if config.floatX == "float64" else 1e-3,
    )

    assert x_star_val.dtype == floatX

    def f(x, a, b):
        objective = rosenbrock_shifted_scaled(x, a, b)
        out = minimize(objective, x)[0]
        return out

    utt.verify_grad(f, [x0, a_val, b_val], eps=1e-3 if floatX == "float32" else 1e-6)


@pytest.mark.parametrize(
    "method, jac, hess",
    [("secant", False, False), ("newton", True, False), ("halley", True, True)],
)
def test_root_scalar(method, jac, hess):
    x = pt.scalar("x")
    a = pt.scalar("a")

    def fn(x, a):
        return x + 2 * a * pt.cos(x)

    f = fn(x, a)
    root_f, success = root_scalar(f, x, method=method, jac=jac, hess=hess)
    func = pytensor.function([x, a], [root_f, success])

    x0 = 0.0
    a_val = 1.0
    solution, success = func(x0, a_val)

    assert success
    np.testing.assert_allclose(
        solution,
        -1.02986653,
        atol=1e-8 if config.floatX == "float64" else 1e-6,
        rtol=1e-8 if config.floatX == "float64" else 1e-6,
    )

    def root_fn(x, a):
        f = fn(x, a)
        return root_scalar(f, x, method=method, jac=jac, hess=hess)[0]

    utt.verify_grad(root_fn, [x0, a_val], eps=1e-6)


def test_root_simple():
    x = pt.scalar("x")
    a = pt.scalar("a")

    def fn(x, a):
        return x + 2 * a * pt.cos(x)

    f = fn(x, a)
    root_f, success = root(f, x, method="lm", optimizer_kwargs={"tol": 1e-8})
    func = pytensor.function([x, a], [root_f, success])

    x0 = 0.0
    a_val = 1.0
    solution, success = func(x0, a_val)

    assert success
    np.testing.assert_allclose(
        solution,
        -1.02986653,
        atol=1e-8 if config.floatX == "float64" else 1e-6,
        rtol=1e-8 if config.floatX == "float64" else 1e-6,
    )

    def root_fn(x, a):
        f = fn(x, a)
        return root(f, x)[0]

    utt.verify_grad(root_fn, [x0, a_val], eps=1e-6)


def test_root_system_of_equations():
    x = pt.tensor("x", shape=(None,))
    a = pt.tensor("a", shape=(None,))
    b = pt.tensor("b", shape=(None,))

    f = pt.stack([a[0] * x[0] * pt.cos(x[1]) - b[0], x[0] * x[1] - a[1] * x[1] - b[1]])

    root_f, success = root(f, x, method="lm", optimizer_kwargs={"tol": 1e-8})
    func = pytensor.function([x, a, b], [root_f, success])

    x0 = np.array([1.0, 1.0], dtype=floatX)
    a_val = np.array([1.0, 1.0], dtype=floatX)
    b_val = np.array([4.0, 5.0], dtype=floatX)
    solution, success = func(x0, a_val, b_val)

    assert success

    np.testing.assert_allclose(
        solution,
        np.array([6.50409711, 0.90841421]),
        atol=1e-8 if config.floatX == "float64" else 1e-6,
        rtol=1e-8 if config.floatX == "float64" else 1e-6,
    )

    def root_fn(x, a, b):
        f = pt.stack(
            [a[0] * x[0] * pt.cos(x[1]) - b[0], x[0] * x[1] - a[1] * x[1] - b[1]]
        )
        return root(f, x)[0]

    utt.verify_grad(
        root_fn, [x0, a_val, b_val], eps=1e-6 if floatX == "float64" else 1e-3
    )


@pytest.mark.parametrize("optimize_op", (minimize, root))
def test_minimize_0d(optimize_op):
    # Scipy vector minimizers upcast 0d x to 1d. We need to work-around this

    class AssertScalar(Op):
        view_map = {0: [0]}

        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, output_storage):
            [x] = inputs
            assert x.ndim == 0
            output_storage[0][0] = x

        def L_op(self, inputs, outputs, out_grads):
            return out_grads

    x = scalar("x")
    x_check = AssertScalar()(x)
    opt_x, _ = optimize_op(x_check**2, x)
    opt_x_res = opt_x.eval({x: np.array(5, dtype=x.type.dtype)})
    np.testing.assert_allclose(
        opt_x_res, 0, atol=1e-15 if floatX == "float64" else 1e-6
    )
