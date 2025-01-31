import pytensor.tensor as pt
from pytensor.tensor.optimize import minimize


def test_minimize():
    x = pt.scalar("x")
    a = pt.scalar("a")
    c = pt.scalar("c")

    b = a * 2
    b.name = "b"
    out = (x - b * c) ** 2

    minimized_x, success = minimize(out, x, debug=False)

    a_val = 2
    c_val = 3

    assert success
    assert minimized_x.eval({a: a_val, c: c_val, x: 0.0}) == (2 * a_val * c_val)

    x_grad, a_grad, c_grad = pt.grad(minimized_x, [x, a, c])

    assert x_grad.eval({x: 0.0}) == 0.0
    assert a_grad.eval({a: a_val, c: c_val, x: 0.0}) == 2 * c_val
    assert c_grad.eval({a: a_val, c: c_val, x: 0.0}) == 2 * a_val
