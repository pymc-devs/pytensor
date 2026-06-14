import numpy as np
import pytest

import pytensor
from pytensor import config, function, shared
from pytensor.compile import DeepCopyOp
from pytensor.graph import FunctionGraph
from pytensor.graph.fg import FrozenFunctionGraph
from pytensor.graph.rewriting.basic import in2out
from pytensor.loop.op import Scan, scan_view_last_state
from pytensor.scan.op import Scan as LegacyScan
from pytensor.tensor import constant, empty, lscalar, scalar, vector
from pytensor.tensor.random import normal
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor.typed_list import TypedListType


def test_fori_scan():
    x = scalar("x")
    update_fg = FunctionGraph([x], [constant(np.array(True)), x + 2])

    n_iters = 10
    y, ys = Scan(update_fg=update_fg)(n_iters, x)

    fn = function([x], [y, ys])

    [scan_node] = (
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, LegacyScan)
    )
    assert not scan_node.op.info.as_while
    assert fn.maker.fgraph.outputs[1].type.shape == (10,)

    y_eval, ys_eval = fn(0)
    np.testing.assert_array_equal(ys_eval, np.arange(2, 22, 2))
    np.testing.assert_array_equal(ys_eval[-1], y_eval)


def test_scan_immutable_and_mergeable():
    # The inner graph is stored frozen, so the Op is hashable and structurally
    # identical Scans compare equal (and merge into a single node).
    def make_scan():
        x = scalar("x")
        return Scan(update_fg=FunctionGraph([x], [constant(np.array(True)), x + 2]))

    op1, op2 = make_scan(), make_scan()
    assert isinstance(op1.update_fg, FrozenFunctionGraph)
    assert op1.update_fg.clone(check_integrity=False) is op1.update_fg
    assert op1 == op2 and hash(op1) == hash(op2)

    x = scalar("x")
    y1, _ = op1(10, x)
    y2, _ = op2(10, x)
    fn = function([x], [y1, y2])
    scan_nodes = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, LegacyScan)
    ]
    assert len(scan_nodes) == 1


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

    [scan_node] = (
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, LegacyScan)
    )
    assert scan_node.op.info.as_while

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
    [scan_node] = (
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, LegacyScan)
    )
    assert scan_node.op.info.as_while
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
    # Non-tensor states (e.g. RandomGenerators) can be carried but not traced:
    # their TypedList trace has no legacy-Scan representation.
    rng_test = np.random.default_rng(123)
    rng_shared = shared(np.random.default_rng(123))
    n_iters = 5

    dummy_init = empty(())
    rng = rng_shared.type()
    update_fg = FunctionGraph(
        [dummy_init, rng],
        [constant(np.array(True)), *normal(rng=rng).owner.outputs[::-1]],
    )

    _last_y, last_rng, ys, rngs = Scan(update_fg=update_fg)(
        n_iters, dummy_init, rng_shared
    )
    assert isinstance(last_rng.type, RandomGeneratorType)
    assert isinstance(rngs.type, TypedListType)

    # Using the RNG trace cannot be lowered to the legacy Scan
    with pytest.raises(NotImplementedError, match="can be carried"):
        function([], [ys, rngs], updates={rng_shared: last_rng}, mode="CVM")

    # Carrying the RNG without tracing it lowers fine
    fn = function([], ys, updates={rng_shared: last_rng}, mode="CVM")
    for _ in range(2):
        for y_res in fn():
            np.testing.assert_almost_equal(y_res, rng_test.normal())


def test_scan_nit_sot_output():
    # An inner output beyond the carries becomes a nit-sot: no input, no final,
    # and it lowers to a legacy nit_sot rather than a (fake) sit-sot.
    c = scalar("c")
    x = scalar("x")
    update_fg = FunctionGraph([c, x], [constant(np.array(True)), c + x, c * x])
    scan_op = Scan(update_fg=update_fg, n_carries=1)
    assert scan_op.n_states == 1
    assert scan_op.n_outputs == 1

    # Output layout: carry final, carry trace, output trace
    final_c, _c_trace, out_trace = scan_op(5, c, x)
    fn = function([c, x], [final_c, out_trace])
    [scan_node] = (
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, LegacyScan)
    )
    assert scan_node.op.info.n_nit_sot == 1

    final_val, out_val = fn(0.0, 2.0)
    np.testing.assert_array_equal(out_val, np.array([0.0, 4.0, 8.0, 12.0, 16.0]))
    assert final_val == 10.0


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
