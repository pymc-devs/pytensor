import numpy as np

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.optimize import MinimizeOp, ScipyWrapperOp


def test_inline_constants():
    """Constants passed as args should be inlined into the inner graph."""
    x = pt.scalar("x")
    a = pt.scalar("a")
    b = pt.scalar("b", dtype=int)
    c = pt.scalar("c")

    objective = (x - c * a) ** b
    minimize_op = MinimizeOp(
        x,
        a,
        b,
        c,
        objective=objective,
        method="BFGS",
    )

    two_float = pt.full((), 2.0, dtype=a.dtype)
    two_int = two_float.astype(b.dtype)
    minimize_node = minimize_op.make_node(x, two_float, two_int, c)
    assert len(minimize_node.inputs) == 4

    f = function([x, c], minimize_node.outputs)

    # Check the two constants are inlined
    [minimize_node] = [
        node
        for node in f.maker.fgraph.apply_nodes
        if isinstance(node.op, ScipyWrapperOp)
    ]
    assert len(minimize_node.inputs) == 2

    # Check correctness
    c_val = 3.0
    minimized_x_val, success_val = f(np.pi, c_val)
    assert success_val
    np.testing.assert_allclose(
        minimized_x_val,
        2 * c_val,
    )


def test_remove_duplicate_inputs(new_minimize_outs=None):
    """Duplicate outer inputs should be deduplicated."""
    x = pt.scalar("x")
    a = pt.scalar("a")
    b = pt.scalar("b")

    objective = (x + a) ** 2 + (x - b) ** 2
    minimize_op = MinimizeOp(
        x,
        a,
        b,
        objective=objective,
        method="BFGS",
    )

    # Use same outer variable for both a, b
    c = pt.scalar("c")
    minimized_node = minimize_op.make_node(x, c, c)
    assert len(minimized_node.inputs) == 3

    f = function([x, c], minimized_node.outputs)

    [minimize_node] = [
        node
        for node in f.maker.fgraph.apply_nodes
        if isinstance(node.op, ScipyWrapperOp)
    ]
    assert len(minimize_node.inputs) == 2

    # Check correctness: minimum of (x+a)^2 + (x-a)^2 = 2x^2 + 2a^2 is at x=0
    minimized_x_val, success_val = f(np.pi, np.e)
    assert success_val
    np.testing.assert_allclose(minimized_x_val, 0.0, atol=1e-8)
