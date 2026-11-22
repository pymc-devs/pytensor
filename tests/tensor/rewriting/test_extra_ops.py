import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import OPT_NONE, Mode, get_default_mode
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.basic import Alloc, alloc, as_tensor_variable, second
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.extra_ops import CumOp, Repeat, Unique, repeat, unique
from pytensor.tensor.rewriting.extra_ops import (
    local_cumop_of_alloc,
    local_useless_cumop,
)
from pytensor.tensor.type import dscalar
from tests.unittest_tools import RewriteTester


@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_scalar(return_index, return_counts, return_inverse):
    x = dscalar()
    y = unique(
        x,
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=None,
    )

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg, clone=False, include=["canonicalize", "local_Unique_scalar"]
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, DimShuffle)
    assert y_rewritten_start.owner.inputs[0] == x

    default_mode = get_default_mode()
    rewrite_mode = default_mode.excluding("local_Unique_scalar")
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)

    x_val = np.array(-10.0, dtype=np.float64)
    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Alloc_lift(
    x_val, axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        alloc(x, *new_shape),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)
    assert y_rewritten_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Alloc) for node in y_rewritten_fg.apply_nodes)

    default_mode = get_default_mode()
    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    # The remaining exclusions simply allow us to perform the check below that
    # makes sure the original `Alloc` is present in our reference (sub)graph.
    rewrite_mode = default_mode.excluding(
        "local_useless_alloc", "local_alloc_sink_dimshuffle", "local_Unique_lift"
    )
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)
    # Make sure that the original `Alloc` is used to compute the reference `y`
    # result
    assert any(isinstance(node.op, Alloc) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, repeats, repeat_axis",
    [
        (np.array([[-10, -3], [-10, 2]], dtype=np.int64), None, (1, 2), 0),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Repeat(
    x_val,
    unique_axis,
    repeats,
    repeat_axis,
    return_index,
    return_counts,
    return_inverse,
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        repeat(x, tuple(repeats), axis=repeat_axis),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)
    assert y_rewritten_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Repeat) for node in y_rewritten_fg.apply_nodes)

    default_mode = get_default_mode()
    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    rewrite_mode = default_mode.excluding("local_Unique_lift")
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)
    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(isinstance(node.op, Repeat) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_second(
    x_val, unique_axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    a = np.zeros(tuple(new_shape), dtype=x.dtype)
    y = unique(
        second(a, x),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_lift"],
        exclude=["local_Unique_scalar", "topo_constant_folding"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)

    y_rewritten_start = y_rewritten_start.owner.inputs[0]

    if y_rewritten_start.owner and isinstance(y_rewritten_start.owner.op, DimShuffle):
        y_rewritten_start = y_rewritten_start.owner.inputs[0]

    assert y_rewritten_start == x
    assert not any(
        isinstance(node.op.scalar_op, ps.Second)
        for node in y_rewritten_fg.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    y_fn = function([x], [y, y_rewritten], mode=Mode(optimizer=OPT_NONE))

    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(
        isinstance(node.op.scalar_op, ps.Second)
        for node in y_fn.maker.fgraph.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize("mode", ["add", "mul"])
@pytest.mark.parametrize(
    "value_shape, axis, value_axis",
    [
        ((), 0, None),
        ((3,), 0, None),
        ((3,), 1, 0),
        ((2, 1), 0, 0),
        ((2, 1), 1, None),
    ],
)
def test_local_cumop_of_alloc(mode, value_shape, axis, value_axis):
    value = pt.tensor("value", shape=value_shape)
    allocated = pt.alloc(value, 2, 3)
    out = CumOp(axis=axis, mode=mode)(allocated)
    result = RewriteTester(
        [value], [out], include=None, custom_rewrite=local_cumop_of_alloc
    )
    if value_axis is not None:
        expected = pt.alloc(CumOp(axis=value_axis, mode=mode)(value), 2, 3)
    elif mode == "add":
        pattern = ["x", "x"]
        pattern[axis] = 0
        n = allocated.owner.inputs[1 + axis]
        k = pt.arange(1, n + 1, dtype=value.dtype).dimshuffle(pattern)
        expected = pt.alloc(value * k, 2, 3)
    else:
        # cumprod along a broadcast axis is left alone
        expected = out
    result.assert_graph(expected)
    rng = np.random.default_rng(0)
    value_test = np.asarray(rng.uniform(0.5, 1.5, size=value_shape), dtype=value.dtype)
    result.assert_eval(value_test)


@pytest.mark.parametrize("mode", ["add", "mul"])
def test_local_useless_cumop(mode):
    x = pt.tensor("x", shape=(3, 1))
    out = CumOp(axis=1, mode=mode)(x)
    result = RewriteTester([x], [out], include=None, custom_rewrite=local_useless_cumop)
    result.assert_graph(x)
