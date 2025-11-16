import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad, shared
from pytensor.compile.mode import FAST_RUN
from pytensor.scan.views import filter as pt_filter
from pytensor.scan.views import foldl, foldr
from pytensor.scan.views import map as pt_map
from pytensor.scan.views import reduce as pt_reduce
from pytensor.tensor.type import scalar, vector
from tests import unittest_tools as utt
from tests.scan.test_basic import clone_optimized_graph, grab_scan_node


@pytest.mark.parametrize("return_updates", [True, False])
def test_reduce(return_updates):
    v = vector("v")
    s = scalar("s")
    result_raw = pt_reduce(lambda x, y: x + y, v, s, return_updates=return_updates)
    if return_updates:
        result, updates = result_raw
        assert not updates
    else:
        result = result_raw

    f = function([v, s], result, allow_input_downcast=True)
    rng = np.random.default_rng(utt.fetch_seed())
    v_v = rng.uniform(-5.0, 5.0, size=(5,))
    assert abs(np.sum(v_v) - f(v_v, 0.0)) < 1e-3


@pytest.mark.parametrize("return_updates", [True, False])
def test_map(return_updates):
    v = vector("v")
    abs_expr_raw = pt_map(
        lambda x: abs(x),
        v,
        [],
        truncate_gradient=-1,
        go_backwards=False,
        return_updates=return_updates,
    )
    if return_updates:
        abs_expr, abs_updates = abs_expr_raw
        assert not abs_updates
    else:
        abs_expr = abs_expr_raw

    f = function([v], abs_expr, allow_input_downcast=True)

    rng = np.random.default_rng(utt.fetch_seed())
    vals = rng.uniform(-5.0, 5.0, size=(10,))
    abs_vals = abs(vals)
    pytensor_vals = f(vals)
    utt.assert_allclose(abs_vals, pytensor_vals)


def test_reduce_memory_consumption():
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o = pt_reduce(
        lambda v, acc: acc + v,
        x,
        pt.constant(np.asarray(0.0, dtype=config.floatX)),
        return_updates=False,
    )
    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f1 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f1)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))


@pytest.mark.parametrize("return_updates", [True, False])
def test_foldl_memory_consumption(return_updates):
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o_raw = foldl(
        lambda v, acc: acc + v,
        x,
        pt.constant(np.asarray(0.0, dtype=config.floatX)),
        return_updates=return_updates,
    )
    if return_updates:
        o, updates = o_raw
        assert not updates
    else:
        o = o_raw

    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f0 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f0)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))


@pytest.mark.parametrize("return_updates", [True, False])
def test_foldr_memory_consumption(return_updates):
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o_raw = foldr(
        lambda v, acc: acc + v,
        x,
        pt.constant(np.asarray(0.0, dtype=config.floatX)),
        return_updates=return_updates,
    )
    if return_updates:
        o, updates = o_raw
        assert not updates
    else:
        o = o_raw

    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f1 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f1)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))


def test_filter():
    v = pt.vector("v")

    def fn(x):
        return pt.eq(x % 2, 0)

    filtered = pt_filter(fn, v)
    f = function([v], filtered, allow_input_downcast=True)

    rng = np.random.default_rng(utt.fetch_seed())
    vals = rng.integers(0, 10, size=(10,))
    expected = vals[vals % 2 == 0]
    result = f(vals)
    utt.assert_allclose(expected, result)


def test_filter_multiple_masks():
    v1 = pt.vector("v1")
    v2 = pt.vector("v2")

    def fn(x1, x2):
        # Mask v1 for even numbers, mask v2 for numbers > 5
        return pt.eq(x1 % 2, 0), pt.gt(x2, 5)

    filtered_v1, filtered_v2 = pt_filter(fn, [v1, v2])
    f = function([v1, v2], [filtered_v1, filtered_v2], allow_input_downcast=True)

    vals1 = np.arange(10)
    vals2 = np.arange(10)

    expected_v1 = vals1[vals1 % 2 == 0]
    expected_v2 = vals2[vals2 > 5]

    result_v1, result_v2 = f(vals1, vals2)

    utt.assert_allclose(expected_v1, result_v1)
    utt.assert_allclose(expected_v2, result_v2)
