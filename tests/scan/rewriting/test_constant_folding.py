import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, scan
from pytensor.raise_op import CheckAndRaise
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.constant_folding import scan_constant_folding
from pytensor.scan.rewriting.db import scan_inner_graph
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
