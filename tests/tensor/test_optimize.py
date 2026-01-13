import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import Variable, config, function
from pytensor.gradient import (
    DisconnectedInputError,
    disconnected_type,
)
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

    def f(x, a, c):
        objective = (x - a * c) ** 2
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
        objective,
        x,
        method=method,
        jac=jac,
        hess=hess,
        optimizer_kwargs={"tol": 1e-16},
        use_vectorized_jac=True,
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
        out = minimize(objective, x, use_vectorized_jac=False)[0]
        return out

    utt.verify_grad(f, [x0, a_val, b_val], eps=1e-3 if floatX == "float32" else 1e-6)


def test_optimize_multiple_minimands():
    """
    Test optimization with many input variables of different shapes, as occurs in a PyMC model.
    """
    x0, x1, x2 = pt.dvectors("x1", "x2", "d3")
    x3 = pt.dmatrix("x3")
    b0, b1, b2 = pt.dscalars("b0", "b1", "b2")
    b3 = pt.dvector("b3")

    y = pt.dvector("y")

    y_hat = x0 * b0 + x1 * b1 + x2 * b2 + x3 @ b3
    objective = ((y - y_hat) ** 2).sum()

    minimized_x, success = minimize(
        objective,
        [b0, b1, b2, b3],
        jac=True,
        hess=True,
        method="Newton-CG",
        use_vectorized_jac=True,
    )

    assert len(minimized_x) == 4

    fn = pytensor.function([b0, b1, b2, b3, x0, x1, x2, x3, y], [*minimized_x, success])

    rng = np.random.default_rng()
    X = rng.normal(size=(100, 3)).astype(floatX)
    X3 = rng.normal(size=(100, 5)).astype(floatX)
    b_vec = rng.normal(size=(8,)).astype(floatX)
    true_b = [b_vec[0], b_vec[1], b_vec[2], b_vec[3:]]
    true_y = X @ b_vec[0:3] + X3 @ b_vec[3:]
    init_b = np.zeros((8,)).astype(floatX)

    inputs = (
        init_b[0],
        init_b[1],
        init_b[2],
        init_b[3:],
        X[:, 0],
        X[:, 1],
        X[:, 2],
        X3,
        true_y,
    )

    *estimated_b, success = fn(*inputs)
    assert success
    for est, true in zip(estimated_b, true_b):
        np.testing.assert_allclose(
            est,
            true,
            atol=1e-5,
            rtol=1e-5,
        )

    def f(b0, b1, b2, b3, x0, x1, x2, x3, y):
        y_hat = x0 * b0 + x1 * b1 + x2 * b2 + x3 @ b3
        objective = ((y - y_hat) ** 2).sum()
        result = minimize(
            objective,
            [b0, b1, b2, b3],
            jac=True,
            hess=True,
            method="trust-ncg",
            use_vectorized_jac=True,
        )[0]
        return pt.sum([x.sum() for x in result])

    utt.verify_grad(f, inputs, eps=1e-6)


def test_minimize_mvn_logp_mu_and_cov():
    """Regression test for https://github.com/pymc-devs/pytensor/issues/1550"""
    d = 3

    def objective(mu, cov, data):
        L = pt.linalg.cholesky(cov)
        _, logdet = pt.linalg.slogdet(cov)

        v = mu - data
        y = pt.linalg.solve_triangular(L, v, lower=True)
        quad_term = (y**2).sum()

        return 0.5 * (d * pt.log(2 * np.pi) + logdet + quad_term)

    data = pt.vector("data", shape=(d,))
    mu = pt.vector("mu", shape=(d,))
    cov = pt.dmatrix("cov", shape=(d, d))

    neg_logp = objective(mu, cov, data)
    mu_star, success = minimize(
        objective=neg_logp,
        x=mu,
        method="BFGS",
        jac=True,
        hess=False,
        use_vectorized_jac=True,
    )

    # This replace + gradient was the original source of the error in #1550, check that no longer raises
    y_star = pytensor.graph_replace(neg_logp, {mu: mu_star})
    _ = pt.grad(y_star, [mu, cov, data])

    rng = np.random.default_rng()
    data_val = rng.normal(size=(d,)).astype(floatX)

    L = rng.normal(size=(d, d)).astype(floatX)
    cov_val = L @ L.T
    mu0_val = rng.normal(size=(d,)).astype(floatX)

    fn = pytensor.function([mu, cov, data], [mu_star, success])
    _, success_flag = fn(mu0_val, cov_val, data_val)
    assert success_flag

    def min_fn(mu, cov, data):
        mu_star, _ = minimize(
            objective=objective(mu, cov, data),
            x=mu,
            method="BFGS",
            jac=True,
            hess=False,
            use_vectorized_jac=True,
        )
        return mu_star.sum()

    utt.verify_grad(
        min_fn, [mu0_val, cov_val, data_val], eps=1e-3 if floatX == "float32" else 1e-6
    )


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


def test_root_system_multiple_inputs():
    # Example problem from Notes and Problems from Applied General Equilibrium Economics, Chapter 3

    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    v3 = pt.dscalar("v3")
    equations = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])

    def f_analytic(v3):
        v1 = 1 / np.sqrt(v3)
        v2 = 2 - v1
        return np.array([v1, v2])

    solution, success = root(equations=equations, variables=variables)
    fn = pytensor.function([v1, v2, v3], [*solution, success])

    v1_val = 1.0
    v2_val = 1.0
    v3_val = 1.0

    *solution_vals, success_flag = fn(v1_val, v2_val, v3_val)
    assert success_flag
    np.testing.assert_allclose(np.array(solution_vals), f_analytic(v3_val))

    def root_fn(v1, v2, v3):
        equations = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])
        [v1_solution, v2_solution], _ = root(equations=equations, variables=[v1, v2])
        return v1_solution + v2_solution

    utt.verify_grad(
        root_fn, [v1_val, v2_val, v3_val], eps=1e-6 if floatX == "float64" else 1e-3
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

        def L_op(self, inputs, outputs, out_grads):
            return out_grads

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


@pytest.mark.parametrize(
    "optimize_op",
    (minimize, minimize_scalar, root, root_scalar),
    ids=["minimize", "minimize_scalar", "root", "root_scalar"],
)
def test_optimize_grad_disconnected_numerical_inp(optimize_op):
    x = scalar("x", dtype="float64")
    theta = scalar("theta", dtype="int64")
    obj = alloc(x**2, theta).sum()  # repeat theta times and sum
    x0, _ = optimize_op(obj, x)

    # Confirm theta is a direct input to the node
    assert x0.owner.inputs[1] is theta

    with pytest.raises(DisconnectedInputError):
        pt.grad(x0, theta, disconnected_inputs="raise")

    # This should work even if the previous one raised
    grad_wrt_theta = pt.grad(x0, theta, disconnected_inputs="ignore")
    np.testing.assert_allclose(
        grad_wrt_theta.eval({x: np.pi, theta: 5}, on_unused_input="ignore"), 0
    )


@pytest.mark.parametrize("optimize_op", (minimize, minimize_scalar, root, root_scalar))
def test_optimize_grad_disconnected_non_numerical_inp(optimize_op):
    class StrType(Type):
        def filter(self, x, **kwargs):
            if isinstance(x, str):
                return x
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

        def L_op(self, inputs, outputs, output_gradients):
            [_x, str_emoji] = inputs
            [g] = output_gradients
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

    with pytest.raises(DisconnectedInputError):
        pt.grad(x_star, str_theta, disconnected_inputs="raise")

    grad_wrt_num_theta = pt.grad(x_star, num_theta, disconnected_inputs="raise")
    np.testing.assert_allclose(
        grad_wrt_num_theta.eval({x: np.pi, num_theta: np.e, str_theta: ":)"}), -1
    )
    np.testing.assert_allclose(
        grad_wrt_num_theta.eval({x: np.pi, num_theta: np.e, str_theta: ":("}), 1
    )


def test_vectorize_root_gradients():
    """Regression test for https://github.com/pymc-devs/pytensor/issues/1586"""
    a, x, y = pt.dscalars("a", "x", "y")

    eq_1 = a * x**2 - y - 1
    eq_2 = x - a * y**2 + 1

    [x_star, y_star], _ = pt.optimize.root(
        equations=pt.stack([eq_1, eq_2]),
        variables=[x, y],
        method="hybr",
        optimizer_kwargs={"tol": 1e-8},
    )
    solution = pt.stack([x_star, y_star])
    a_grad = pt.grad(solution.sum(), a)
    a_grid = pt.dmatrix("a_grid")

    solution_grid, a_grad_grid = pytensor.graph.vectorize_graph(
        [solution, a_grad], {a: a_grid}
    )

    def analytical_roots_and_grad(
        a_vals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        a = a_vals
        # There are 4 roots to this equation, but we're always starting the optimizer at (1, 1) to control which one
        # we get. For a > 0, this is the root with both x and y positive.
        solution_grid = np.array(
            (
                -1 + (np.sqrt(4 * a + 1) + 1) ** 2 / (4 * a),
                (np.sqrt(4 * a + 1) + 1) / (2 * a),
            )
        )

        # Derivative of the sum of the two solutions w.r.t. a
        dx_da = (np.sqrt(4 * a + 1) + 1) / (a * np.sqrt(4 * a + 1)) + 1 / (
            a * np.sqrt(4 * a + 1)
        )
        dy_da = -((np.sqrt(4 * a + 1) + 1) ** 2) / (4 * a**2) - (
            np.sqrt(4 * a + 1) + 1
        ) / (2 * a**2)
        solution_a_grad = dx_da + dy_da

        return solution_grid.transpose((1, 2, 0)), solution_a_grad

    fn = pytensor.function(
        [a_grid, x, y],
        [solution_grid, a_grad_grid],
        on_unused_input="ignore",
    )

    a_grid = np.linspace(1, 10, 9).reshape((3, 3))
    solution_grid_val, a_grad_grid_val = fn(a_grid=a_grid, x=1.0, y=1.0)

    analytical_solution_grid, analytical_a_grad_grid = analytical_roots_and_grad(a_grid)

    np.testing.assert_allclose(solution_grid_val, analytical_solution_grid)
    np.testing.assert_allclose(a_grad_grid_val, analytical_a_grad_grid)
