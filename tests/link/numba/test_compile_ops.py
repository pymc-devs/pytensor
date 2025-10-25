import numpy as np
import pytest

from pytensor import OpFromGraph, config, function, ifelse
from pytensor import tensor as pt
from pytensor.compile import ViewOp
from pytensor.raise_op import assert_op
from tests.link.numba.test_basic import compare_numba_and_py


def test_ViewOp():
    v = pt.vector()
    v_test_value = np.arange(4, dtype=config.floatX)
    g = ViewOp()(v)

    compare_numba_and_py(
        [v],
        [g],
        [v_test_value],
    )


# We were seeing some weird results in CI where the following two almost
# sign-swapped results were being return from Numba and Python, respectively.
# The issue might be related to https://github.com/numba/numba/issues/4519.
# Regardless, I was not able to reproduce anything like it locally after
# extensive testing.
x = np.array(
    [
        [-0.60407637, -0.71177603, -0.35842241],
        [-0.07735968, 0.50000561, -0.86256007],
        [-0.7931628, 0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)

y = np.array(
    [
        [0.60407637, 0.71177603, -0.35842241],
        [0.07735968, -0.50000561, -0.86256007],
        [0.7931628, -0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "inputs, cond_fn, true_vals, false_vals",
    [
        ([], lambda: np.array(True), np.r_[1, 2, 3], np.r_[-1, -2, -3]),
        (
            [(pt.dscalar(), np.array(0.2, dtype=np.float64))],
            lambda x: x < 0.5,
            np.r_[1, 2, 3],
            np.r_[-1, -2, -3],
        ),
        (
            [
                (pt.dscalar(), np.array(0.3, dtype=np.float64)),
                (pt.dscalar(), np.array(0.5, dtype=np.float64)),
            ],
            lambda x, y: x > y,
            x,
            y,
        ),
        (
            [
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            x,
            y,
        ),
        (
            [
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
        (
            [
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
    ],
)
def test_IfElse(inputs, cond_fn, true_vals, false_vals):
    inputs, test_values = zip(*inputs, strict=True) if inputs else ([], [])
    out = ifelse(cond_fn(*inputs), true_vals, false_vals)
    compare_numba_and_py(inputs, out, test_values)


def test_OpFromGraph():
    x, y, z = pt.matrices("xyz")
    ofg_1 = OpFromGraph([x, y], [x + y], inline=False)
    ofg_2 = OpFromGraph([x, y], [x * y, x - y], inline=False)

    o1, o2 = ofg_2(y, z)
    out = ofg_1(x, o1) + o2

    xv = np.ones((2, 2), dtype=config.floatX)
    yv = np.ones((2, 2), dtype=config.floatX) * 3
    zv = np.ones((2, 2), dtype=config.floatX) * 5

    compare_numba_and_py([x, y, z], [out], [xv, yv, zv])


@pytest.mark.filterwarnings("error")
def test_ofg_inner_inplace():
    x = pt.vector("x")
    set0 = x[0].set(1)  # SetSubtensor should not inplace on x
    exp_x = pt.exp(x)
    set1 = exp_x[0].set(1)  # SetSubtensor should inplace on exp_x
    ofg0 = OpFromGraph([x], [set0])
    ofg1 = OpFromGraph([x], [set1])

    y, z = pt.vectors("y", "z")
    fn = function([y, z], [ofg0(y), ofg1(z)], mode="NUMBA")

    fn_ofg0 = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(fn_ofg0, OpFromGraph)
    fn_set0 = fn_ofg0.fgraph.outputs[0]
    assert fn_set0.owner.op.destroy_map == {}

    fn_ofg1 = fn.maker.fgraph.outputs[1].owner.op
    assert isinstance(fn_ofg1, OpFromGraph)
    fn_set1 = fn_ofg1.fgraph.outputs[0]
    assert fn_set1.owner.op.destroy_map == {0: [0]}

    x_test = np.array([0, 1, 1], dtype=config.floatX)
    y_test = np.array([0, 1, 1], dtype=config.floatX)
    res0, res1 = fn(x_test, y_test)
    # Check inputs were not mutated
    np.testing.assert_allclose(x_test, [0, 1, 1])
    np.testing.assert_allclose(y_test, [0, 1, 1])
    # Check outputs are correct
    np.testing.assert_allclose(res0, [1, 1, 1])
    np.testing.assert_allclose(res1, [1, np.e, np.e])


def test_check_and_raise():
    x = pt.vector()
    x_test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x.sum(), np.array(True))

    compare_numba_and_py([x], out, [x_test_value])
