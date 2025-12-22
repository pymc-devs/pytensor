import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import Variable, config, function
from pytensor.gradient import NullTypeGradError, disconnected_type
from pytensor.graph import Apply, Op, Type
from pytensor.tensor import alloc, scalar, scalar_from_tensor, tensor_from_scalar
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
def test_optimize_0d(optimize_op):
    # Scipy vector minimizers upcast 0d x to 1d. We need to work-around this

    class AssertScalar(Op):
        view_map = {0: [0]}

        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, output_storage):
            [x] = inputs
            assert x.ndim == 0
            output_storage[0][0] = x

        def L_op(self, inputs, outputs, output_grads):
            return output_grads

    x = scalar("x")
    x_check = AssertScalar()(x)
    opt_x, _ = optimize_op(x_check**2, x)
    opt_x_res = opt_x.eval({x: np.array(5, dtype=x.type.dtype)})
    np.testing.assert_allclose(
        opt_x_res, 0, atol=1e-15 if floatX == "float64" else 1e-6
    )


@pytest.mark.parametrize("optimize_op", (minimize, minimize_scalar, root, root_scalar))
def test_optimize_grad_scalar_arg(optimize_op):
    # Regression test for https://github.com/pymc-devs/pytensor/pull/1744
    x = scalar("x")
    theta = scalar("theta")
    theta_scalar = scalar_from_tensor(theta)
    obj = tensor_from_scalar((scalar_from_tensor(x) + theta_scalar) ** 2)
    x0, _ = optimize_op(obj, x)

    # Confirm theta is a direct input to the node
    assert x0.owner.inputs[1] is theta_scalar

    grad_wrt_theta = pt.grad(x0, theta)
    np.testing.assert_allclose(grad_wrt_theta.eval({x: np.pi, theta: np.e}), -1)


@pytest.mark.parametrize("optimize_op", (minimize, minimize_scalar, root, root_scalar))
def test_optimize_grad_disconnected_numerical_inp(optimize_op):
    x = scalar("x", dtype="float64")
    theta = scalar("theta", dtype="int64")
    obj = alloc(x**2, theta).sum()  # repeat theta times and sum
    x0, _ = optimize_op(obj, x)

    # Confirm theta is a direct input to the node
    assert x0.owner.inputs[1] is theta

    # This should technically raise, but does not right now
    grad_wrt_theta = pt.grad(x0, theta, disconnected_inputs="raise")
    np.testing.assert_allclose(grad_wrt_theta.eval({x: np.pi, theta: 5}), 0)

    # This should work even if the previous one raised
    grad_wrt_theta = pt.grad(x0, theta, disconnected_inputs="ignore")
    np.testing.assert_allclose(grad_wrt_theta.eval({x: np.pi, theta: 5}), 0)


@pytest.mark.parametrize("optimize_op", (minimize, minimize_scalar, root, root_scalar))
def test_optimize_grad_disconnected_non_numerical_inp(optimize_op):
    class StrType(Type):
        def filter(self, data, strict=False, allow_downcast=None):
            if isinstance(data, str):
                return data
            raise TypeError

    class SmileOrFrown(Op):
        def make_node(self, x, str_emoji):
            return Apply(self, [x, str_emoji], [x.type()])

        def perform(self, node, inputs, output_storage):
            [x, str_emoji] = inputs
            match str_emoji:
                case ":)":
                    out = np.array(x)
                case ":(":
                    out = np.array(-x)
                case _:
                    ValueError("str_emoji must be a smile or a frown")
            output_storage[0][0] = out

        def connection_pattern(self, node):
            # Gradient connected only to first input
            return [[True], [False]]

        def L_op(self, inputs, outputs, output_grads):
            [_x, str_emoji] = inputs
            [g] = output_grads
            return [
                self(g, str_emoji),
                disconnected_type(),
            ]

    # We could try to use real types like NoneTypeT or SliceType, but this is more robust to future API changes
    str_type = StrType()
    smile_or_frown = SmileOrFrown()

    x = scalar("x", dtype="float64")
    num_theta = pt.scalar("num_theta", dtype="float64")
    str_theta = Variable(str_type, None, None, name="str_theta")
    obj = (smile_or_frown(x, str_theta) + num_theta) ** 2
    x_star, _ = optimize_op(obj, x)

    # Confirm thetas are direct inputs to the node
    assert set(x_star.owner.inputs[1:]) == {num_theta, str_theta}

    # Confirm forward pass works, no point in worrying about gradient otherwise
    np.testing.assert_allclose(
        x_star.eval({x: np.pi, num_theta: np.e, str_theta: ":)"}),
        -np.e,
    )
    np.testing.assert_allclose(
        x_star.eval({x: np.pi, num_theta: np.e, str_theta: ":("}),
        np.e,
    )

    with pytest.raises(NullTypeGradError):
        pt.grad(x_star, str_theta, disconnected_inputs="raise")

    # This could be supported, but it is not right now.
    with pytest.raises(NullTypeGradError):
        _grad_wrt_num_theta = pt.grad(x_star, num_theta, disconnected_inputs="raise")
    # np.testing.assert_allclose(grad_wrt_num_theta.eval({x: np.pi, num_theta: np.e, str_theta: ":)"}), -1)
    # np.testing.assert_allclose(grad_wrt_num_theta.eval({x: np.pi, num_theta: np.e, str_theta: ":("}), 1)
