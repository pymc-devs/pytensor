import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, scan
from pytensor.compile.mode import get_mode
from pytensor.raise_op import CheckAndRaise
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.constant_folding import scan_constant_folding
from pytensor.scan.rewriting.db import scan_inner_graph
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.rewriting.basic import topo_alloc_constant_folding
from tests.unittest_tools import RewriteTester


def test_constant_scan():
    out = scan(
        lambda previous: previous + 1,
        outputs_info=pt.constant(0.0),
        n_steps=4,
        return_updates=False,
    )
    result = RewriteTester([], [out], include="o3")
    result.assert_graph(pt.constant(np.arange(1, 5, dtype=out.dtype)))
    result.assert_eval()

    result = RewriteTester([], [out], include="o3", exclude=["topo_constant_folding"])
    assert any(isinstance(node.op, Scan) for node in result.rewr_fg.apply_nodes)
    result.assert_eval()


@pytest.mark.parametrize(
    "buffer_kind", ["alloc", "alloc_write", "empty_write", "alloc_view"]
)
def test_runtime_scan_buffers(buffer_kind):
    delta = pt.scalar("delta")
    out = scan(
        lambda previous, delta: previous + delta,
        outputs_info=pt.constant(0.0, dtype=delta.dtype),
        non_sequences=[delta],
        n_steps=4,
        return_updates=False,
    )
    scan_node = out.owner.inputs[0].owner
    if buffer_kind == "alloc_view":
        buffer = pt.zeros((1, 5), dtype=delta.dtype).reshape((5,))
    elif buffer_kind == "empty_write":
        buffer = AllocEmpty(delta.dtype)(5)[:1].set(1.0)
    else:
        buffer = pt.zeros(5, dtype=delta.dtype)
        if buffer_kind == "alloc_write":
            buffer = buffer[:1].set(1.0)
    inputs = list(scan_node.inputs)
    inputs[1] = buffer
    out = scan_node.op(*inputs)[1:]
    result = RewriteTester(
        [delta], [out], include=None, custom_rewrite=topo_alloc_constant_folding
    )
    result.assert_graph(out)
    result.assert_eval(2.0)


def test_scan_with_symbolic_initial_value():
    initial = pt.scalar("initial")
    out = scan(
        lambda previous: previous + 1,
        outputs_info=initial,
        n_steps=4,
        return_updates=False,
    )
    result = RewriteTester(
        [initial], [out], include=None, custom_rewrite=scan_constant_folding
    )
    result.assert_graph(out)
    result.assert_eval(2.0)


@pytest.mark.parametrize("location", ["outside", "inside"])
@config.change_flags(on_opt_error="raise")
def test_scan_that_opts_out_of_folding(location):
    check = CheckAndRaise(ValueError, "invalid value")
    initial = pt.constant(0.0)
    if location == "outside":
        initial = check(initial, pt.constant(False))

    def step(previous):
        value = previous + 1
        return check(value, pt.constant(False)) if location == "inside" else value

    out = scan(
        step,
        outputs_info=initial,
        n_steps=4,
        return_updates=False,
    )
    result = RewriteTester(
        [], [out], include=None, custom_rewrite=scan_constant_folding
    )
    result.assert_graph(out)
    with pytest.raises(ValueError, match="invalid value"):
        result.rewr_fn()


def test_constant_nested_scan():
    initial = pt.scalar("initial")

    def step(previous):
        constant_sequence = scan(
            lambda value: value + 1,
            outputs_info=pt.constant(0.0),
            n_steps=4,
            return_updates=False,
        )
        return previous + constant_sequence[-1]

    out = scan(step, outputs_info=initial, n_steps=3, return_updates=False)
    result = RewriteTester(
        [initial], [out], include="o3", custom_rewrite=scan_inner_graph
    )
    [outer_scan] = [
        node.op for node in result.rewr_fg.apply_nodes if isinstance(node.op, Scan)
    ]
    assert not any(isinstance(node.op, Scan) for node in outer_scan.fgraph.apply_nodes)
    result.assert_eval(2.0)


@pytest.mark.parametrize("output_kind", ["last", "full"])
@pytest.mark.parametrize(
    "mode",
    [
        "FAST_RUN",
        "FAST_COMPILE",
        pytest.param(
            get_mode("FAST_RUN").excluding("eigh_of_diag"),
            id="FAST_RUN_no_eigh_of_diag",
        ),
    ],
)
def test_constant_scan_rewrites_inner_graph(mode, output_kind):
    """Constant-folded Scan should rewrite its inner graph like the outer graph."""
    _, eigenvectors = pt.linalg.eigh(25.0 * pt.eye(2))
    corner = eigenvectors[0, 0]
    trace = scan(
        lambda total: total + corner,
        outputs_info=[pt.zeros(())],
        n_steps=2,
        return_updates=False,
    )
    target = trace[-1] if output_kind == "last" else trace
    fn = function([], [target, corner], mode=mode)
    assert not any(isinstance(n.op, Scan) for n in fn.maker.fgraph.apply_nodes)
    res_trace, res_corner = fn()
    expected = (
        2.0 * res_corner
        if output_kind == "last"
        else np.array([res_corner, 2.0 * res_corner])
    )
    np.testing.assert_allclose(res_trace, expected)


@config.change_flags(on_opt_error="raise")
def test_untraced_scan_that_opts_out_of_folding():
    """Generic constant_folding must not bypass Scan's inner opt-out checks."""
    check = CheckAndRaise(ValueError, "invalid value")
    out = scan(
        lambda previous: check(previous + 1, pt.constant(False)),
        outputs_info=pt.constant(0.0),
        n_steps=4,
        return_updates=False,
    )
    fn = function([], out[-1], mode="FAST_RUN")
    assert any(isinstance(n.op, Scan) for n in fn.maker.fgraph.apply_nodes)
    with pytest.raises(ValueError, match="invalid value"):
        fn()


@pytest.mark.parametrize("opt_out", [False, True])
@config.change_flags(on_opt_error="raise")
def test_constant_nested_scan_inside_constant_scan(opt_out):
    """Nested Scans inside a constant Scan fold unless an inner op opts out."""
    check = CheckAndRaise(ValueError, "nested opt-out")
    _, eigenvectors = pt.linalg.eigh(25.0 * pt.eye(2))
    corner = eigenvectors[0, 0]

    def outer_step(previous):
        def inner_step(value):
            next_val = value + corner
            return check(next_val, pt.constant(False)) if opt_out else next_val

        inner = scan(
            inner_step,
            outputs_info=pt.zeros(()),
            n_steps=3,
            return_updates=False,
        )
        return previous + inner[-1]

    out = scan(
        outer_step,
        outputs_info=pt.zeros(()),
        n_steps=2,
        return_updates=False,
    )
    fn = function([], [out[-1], corner], mode="FAST_RUN")
    if opt_out:
        assert any(isinstance(n.op, Scan) for n in fn.maker.fgraph.apply_nodes)
        with pytest.raises(ValueError, match="nested opt-out"):
            fn()
    else:
        assert not any(isinstance(n.op, Scan) for n in fn.maker.fgraph.apply_nodes)
        res_out, res_corner = fn()
        np.testing.assert_allclose(res_out, 6.0 * res_corner)
