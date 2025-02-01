import numpy as np

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.optimize import minimize
from tests import unittest_tools as utt


floatX = config.floatX


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

    assert success
    assert minimized_x.eval({a: a_val, c: c_val, x: 0.0}) == (2 * a_val * c_val)

    def f(x, a, b):
        objective = (x - a * b) ** 2
        out = minimize(objective, x)[0]
        return out

    utt.verify_grad(f, [0.0, a_val, c_val], eps=1e-6)


def test_minimize_vector_x():
    def rosenbrock_shifted_scaled(x, a, b):
        return (a * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum() + b

    x = pt.dvector("x")
    a = pt.scalar("a")
    b = pt.scalar("b")

    objective = rosenbrock_shifted_scaled(x, a, b)

    minimized_x, success = minimize(objective, x, method="BFGS")

    a_val = 0.5
    b_val = 1.0
    x0 = np.zeros(5).astype(floatX)
    x_star_val = minimized_x.eval({a: a_val, b: b_val, x: x0})

    assert success
    np.testing.assert_allclose(
        x_star_val, np.ones_like(x_star_val), atol=1e-6, rtol=1e-6
    )

    utt.verify_grad(rosenbrock_shifted_scaled, [x0, a_val, b_val], eps=1e-6)
