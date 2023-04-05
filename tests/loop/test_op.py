import numpy as np

import pytensor
from pytensor import config, function, shared
from pytensor.compile import DeepCopyOp
from pytensor.graph import FunctionGraph
from pytensor.graph.rewriting.basic import in2out
from pytensor.loop.op import Loop, Scan, scan_view_last_state
from pytensor.tensor import constant, empty, lscalar, scalar, vector
from pytensor.tensor.random import normal
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor.tensor.subtensor import Subtensor
from pytensor.typed_list import TypedListType


def test_loop_basic():
    i = lscalar("i")
    x = scalar("x")
    update_fg = FunctionGraph([i, x], [(i + 1) < 10, i + 1, x + 2])

    loop_op = Loop(update_fg=update_fg)
    assert len(loop_op.state_types) == 2
    assert len(loop_op.const_types) == 0
    _, y = loop_op(np.array(0, dtype="int64"), x)
    assert y.eval({x: 0}) == 20


def test_loop_with_constant():
    i = lscalar("i")
    x = scalar("x")
    const = scalar("const")
    update_fg = FunctionGraph([i, x, const], [(i + 1) < 10, i + 1, x + const])

    loop_op = Loop(update_fg=update_fg)
    assert len(loop_op.state_types) == 2
    assert len(loop_op.const_types) == 1
    _, y = loop_op(np.array(0, dtype="int64"), x, const)
    assert y.eval({x: 0, const: 2}) == 20


def test_fori_scan():
    x = scalar("x")
    update_fg = FunctionGraph([x], [constant(np.array(True)), x + 2])

    n_iters = 10
    y, ys = Scan(update_fg=update_fg)(n_iters, x)

    fn = function([x], [y, ys])

    subtensor_nodes = tuple(
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Subtensor)
    )
    assert len(subtensor_nodes) == 0
    loop_nodes = tuple(
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Loop)
    )
    assert len(loop_nodes) == 1
    (loop_node,) = loop_nodes
    assert len(loop_node.outputs) == 3
    assert loop_node.outputs[0].type.shape == ()
    assert loop_node.outputs[1].type.shape == ()
    assert loop_node.outputs[2].type.shape == (10,)

    y_eval, ys_eval = fn(0)
    np.testing.assert_array_equal(ys_eval, np.arange(2, 22, 2))
    np.testing.assert_array_equal(ys_eval[-1], y_eval)


def test_fori_scan_shape():
    x = scalar("x")
    update_fg = FunctionGraph([x], [constant(np.array(True)), x + 2])

    n_iters = 10
    _, ys = Scan(update_fg=update_fg)(n_iters, x)

    fn = function([x], ys.shape, on_unused_input="ignore")
    nodes = tuple(fn.maker.fgraph.apply_nodes)
    assert len(nodes) == 1
    assert isinstance(nodes[0].op, DeepCopyOp)
    assert fn(0) == 10


def test_while_scan():
    i = lscalar("i")
    x = scalar("x")
    update_fg = FunctionGraph([i, x], [(i + 1) < 10, i + 1, x + 2])

    max_iters = 1000
    _, y, _, ys = Scan(update_fg=update_fg)(max_iters, np.array(0, dtype="int64"), x)

    fn = function([x], [y, ys])

    subtensor_nodes = tuple(
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Subtensor)
    )
    assert len(subtensor_nodes) == 1
    loop_nodes = tuple(
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Loop)
    )
    assert len(loop_nodes) == 1
    (loop_node,) = loop_nodes
    assert len(loop_node.outputs) == 3
    assert loop_node.outputs[0].type.shape == ()
    assert loop_node.outputs[1].type.shape == ()
    assert loop_node.outputs[2].type.shape == (1000,)

    y_eval, ys_eval = fn(0)
    np.testing.assert_array_equal(ys_eval, np.arange(2, 22, 2))
    np.testing.assert_array_equal(ys_eval[-1], y_eval)


def test_while_scan_shape():
    i = lscalar("i")
    x = scalar("x")
    update_fg = FunctionGraph([i, x], [(i + 1) < 10, i + 1, x + 2])

    max_iters = 1000
    _, _, _, ys = Scan(update_fg=update_fg)(max_iters, np.array(0, dtype="int64"), x)

    fn = function([x], ys.shape)
    loop_nodes = tuple(
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Loop)
    )
    assert len(loop_nodes) == 1
    assert fn(0) == 10


def test_foreach_scan():
    idx = scalar("idx", dtype="int64")
    dummy_x0 = empty(())
    xs = vector("xs")
    const = scalar("const")
    update_fg = FunctionGraph(
        [idx, dummy_x0, xs, const], [constant(np.array(True)), idx + 1, xs[idx] * const]
    )

    n_steps = xs.shape[0]
    _, _, _, ys = Scan(update_fg=update_fg)(n_steps, 0, dummy_x0, xs, const)

    fn = pytensor.function([xs, const], ys)

    np.testing.assert_almost_equal(
        fn(np.arange(10, dtype=config.floatX), 100), np.arange(10) * 100
    )


def test_fori_random_scan():
    rng_test = np.random.default_rng(123)
    rng_shared = shared(np.random.default_rng(123))
    n_iters = 5

    dummy_init = empty(())
    rng = rng_shared.type()
    update_fg = FunctionGraph(
        [dummy_init, rng],
        [constant(np.array(True)), *normal(rng=rng).owner.outputs[::-1]],
    )

    last_y, last_rng, ys, rngs = Scan(update_fg=update_fg)(
        n_iters, dummy_init, rng_shared
    )
    assert isinstance(last_rng.type, RandomGeneratorType)
    assert isinstance(rngs.type, TypedListType)
    assert isinstance(rngs.type.ttype, RandomGeneratorType)

    fn = function([], [ys, rngs], updates={rng_shared: last_rng})
    for i in range(2):
        ys_res, rngs_res = fn()
        for y_res, rng_res in zip(ys_res, rngs_res):
            np.testing.assert_almost_equal(y_res, rng_test.normal())
            assert rng_res.__getstate__() == rng_test.__getstate__()


def test_scan_view_last_state():
    x = scalar("x")
    update_fg = FunctionGraph([x], [x > 5, x + 2])

    n_iters = 10
    y1, ys = Scan(update_fg=update_fg)(n_iters, x)

    y2 = ys[-1]
    fgraph = FunctionGraph(outputs=[y2, ys], clone=False)
    assert fgraph.outputs[0] is not y1
    in2out(scan_view_last_state).apply(fgraph)
    assert fgraph.outputs[0] is y1
    assert fgraph.outputs[1] is ys
