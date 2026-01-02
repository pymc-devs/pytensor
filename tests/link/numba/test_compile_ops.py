import numpy as np
import pytest

from pytensor import Mode, OpFromGraph, config, function, ifelse, scan
from pytensor import tensor as pt
from pytensor.compile import ViewOp
from pytensor.graph import vectorize_graph
from pytensor.raise_op import assert_op
from pytensor.scalar import Add
from pytensor.scan.op import Scan
from pytensor.tensor import dmatrix, dtensor3, matrix
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.slinalg import Cholesky
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


def test_ofg_aliased_outputs():
    x = matrix("x")
    # Create multiple views of x
    outs = OpFromGraph([x], [x, x.T, x[::-1]])(x)
    # Add one to each x, which when inplace shouldn't propagate across outputs
    bumped_outs = [o + 1 for o in outs]
    fn = function([x], bumped_outs, mode="NUMBA")
    fn.dprint(print_destroy_map=True)
    # Check our outputs are indeed inplace adds
    assert all(
        (
            isinstance(o.owner.op, Elemwise)
            and isinstance(o.owner.op.scalar_op, Add)
            and o.owner.op.destroy_map
        )
        for o in fn.maker.fgraph.outputs
    )
    x_test = np.zeros((2, 2))
    for res in fn(x_test):
        np.testing.assert_allclose(res, np.ones((2, 2)))


def test_ofg_elemwise_regression():
    # Regression bug for https://github.com/pymc-devs/pytensor/issues/1507
    x = dmatrix("x", shape=(None, None))
    z = OpFromGraph(
        inputs=[x],
        outputs=[x + 1],
    )(x)

    x_batched = dtensor3("X_batched", shape=(None, None, None))
    z_batched = vectorize_graph(z, {x: x_batched})
    compare_numba_and_py(
        [x_batched],
        [z_batched],
        [np.random.normal(size=(3, 2, 4))],
        eval_obj_mode=False,
    )


def test_check_and_raise():
    x = pt.vector()
    x_test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x.sum(), np.array(True))

    compare_numba_and_py([x], out, [x_test_value])


def test_ofg_with_inner_scan_rewrite():
    # Regression test where inner scan would be mutated when compiling outer OFG
    ys = pt.tensor("ys", shape=(5, 3, 3))
    xs = scan(
        lambda y: pt.linalg.cholesky(y),
        sequences=[ys],
        return_updates=False,
        mode=Mode(optimizer=None),
    )
    xs_ofg = OpFromGraph([ys], [xs])(ys)
    fn = function([ys], xs_ofg, mode="NUMBA")

    # Check that we have a BlockwiseWithCoreShape in the inner Scan
    fn_ofg_op = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(fn_ofg_op, OpFromGraph)
    fn_scan_op = fn_ofg_op.fgraph.outputs[0].owner.op
    assert isinstance(fn_scan_op, Scan)
    fn_cholesky_op = fn_scan_op.fgraph.outputs[0].owner.op
    assert isinstance(fn_cholesky_op, BlockwiseWithCoreShape)
    assert isinstance(fn_cholesky_op.core_op, Cholesky)

    # Check original Ops aren't modified
    ofg_op = xs_ofg.owner.op
    assert isinstance(ofg_op, OpFromGraph)
    scan_op = ofg_op.fgraph.outputs[0].owner.op
    assert isinstance(scan_op, Scan)
    cholesky_op = scan_op.fgraph.outputs[0].owner.op
    assert isinstance(cholesky_op, Blockwise)
    assert isinstance(cholesky_op.core_op, Cholesky)
