import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import exp, prod
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.sort import local_sort_of_alloc, local_sort_under_reduce
from pytensor.tensor.type import matrix, vector
from tests.unittest_tools import RewriteTester


@pytest.mark.parametrize(
    "value_shape, shape, axis, value_axis",
    [
        ((), (3, 4), 0, None),
        ((), (3, 4), 1, None),
        ((3,), (2, 3), 0, None),
        ((3,), (2, 3), 1, 0),
        ((3, 1), (3, 4), 1, None),
        ((3, 1), (3, 4), 0, 0),
    ],
)
def test_local_sort_of_alloc(value_shape, shape, axis, value_axis):
    value = pt.tensor("value", shape=value_shape)
    allocated = pt.alloc(value, *shape)
    out = pt.sort(allocated, axis=axis)
    result = RewriteTester(
        [value], [out], include=None, custom_rewrite=local_sort_of_alloc
    )
    expected = (
        allocated
        if value_axis is None
        else pt.alloc(pt.sort(value, axis=value_axis), *shape)
    )
    result.assert_graph(expected)
    rng = np.random.default_rng(0)
    value_test = np.asarray(rng.normal(size=value_shape), dtype=value.dtype)
    result.assert_eval(value_test)


@pytest.mark.parametrize("value_shape, axis", [((), 0), ((), 1), ((4,), 1)])
def test_local_argsort_of_alloc(value_shape, axis):
    value = pt.tensor("value", shape=value_shape)
    allocated = pt.alloc(value, 3, 4)
    out = pt.argsort(allocated, axis=axis, stable=True)
    result = RewriteTester(
        [value], [out], include=None, custom_rewrite=local_sort_of_alloc
    )
    if value_shape:
        expected = pt.alloc(pt.argsort(value, axis=0, stable=True), 3, 4)
    else:
        indices = pt.arange((3, 4)[axis], dtype="int64")
        expected = pt.alloc(pt.expand_dims(indices, axis=1 - axis), 3, 4)
    result.assert_graph(expected)
    rng = np.random.default_rng(0)
    value_test = np.asarray(rng.normal(size=value_shape), dtype=value.dtype)
    result.assert_eval(value_test)


@pytest.mark.parametrize(
    "out, expected",
    [
        (
            lambda x, v: pt_max(pt.sort(x, axis=1)[:, ::-1], axis=1),
            lambda x, v: pt_max(x, axis=1),
        ),
        (lambda x, v: pt_sum(pt.sort(x, axis=0)), lambda x, v: pt_sum(x)),
        (
            lambda x, v: prod(pt.sort(x, axis=1)[::-1, ::-1]),
            lambda x, v: prod(x),
        ),
        (
            lambda x, v: pt_sum(pt.sort(x, axis=1) * v[:, None], axis=1),
            lambda x, v: pt_sum(x * v[:, None], axis=1),
        ),
        (
            lambda x, v: pt_sum(pt.sort(x, axis=1) * v[:, None]),
            lambda x, v: pt_sum(x * v[:, None]),
        ),
        (
            lambda x, v: pt_sum(exp(2 * pt.sort(x, axis=1)), axis=1),
            lambda x, v: pt_sum(exp(2 * x), axis=1),
        ),
        (
            lambda x, v: (
                pt_sum(y := exp(pt.sort(x, axis=1)), axis=1) + pt_max(y, axis=1)
            ),
            lambda x, v: pt_sum(y := exp(x), axis=1) + pt_max(y, axis=1),
        ),
        (
            lambda x, v: pt_sum(exp(pt.sort(x, axis=1)[:, ::-1])[::-1]),
            lambda x, v: pt_sum(exp(x)),
        ),
        (
            lambda x, v: pt_sum(pt.sort(x, axis=1).T, axis=0),
            lambda x, v: pt_sum(x.T, axis=0),
        ),
        (
            lambda x, v: pt_sum(
                pt.sort(x, axis=1)[:, None, :] * v[None, :, None], axis=2
            ),
            lambda x, v: pt_sum(x[:, None, :] * v[None, :, None], axis=2),
        ),
        (lambda x, v: pt_sum(pt.sort(x, axis=0), axis=1), None),
        (lambda x, v: pt_sum(pt.sort(x, axis=1).T, axis=1), None),
        (lambda x, v: pt_sum(pt.sort(x, axis=1)[::-1], axis=1), None),
        (lambda x, v: pt_sum(pt.sort(x, axis=1) * v[None, :], axis=1), None),
        (lambda x, v: pt_sum(pt.sort(x, axis=0)[::2], axis=0), None),
    ],
    ids=[
        "max_of_reversed_sort",
        "sum_all_of_sort",
        "prod_of_sort_reversed_on_both_axes",
        "elemwise_with_broadcast_operand",
        "full_reduce_with_broadcast_operand",
        "elemwise_chain_over_sort",
        "elemwise_shared_by_reductions",
        "reversals_around_elemwise",
        "transpose",
        "expand_dims_with_broadcast_operand",
        "sort_on_other_axis",
        "transpose_onto_other_axis",
        "reversal_on_other_axis",
        "elemwise_varying_along_sort",
        "strided_slice",
    ],
)
def test_local_sort_under_reduce(out, expected):
    x = matrix("x")
    v = vector("v")
    if expected is None:
        expected_graph = rewrite_graph(
            out(x, v),
            include=("canonicalize",),
            exclude=("local_sort_under_reduce",),
        )
    else:
        expected_graph = rewrite_graph(expected(x, v), include=("canonicalize",))

    result = RewriteTester([x, v], [out(x, v)], include=("canonicalize",))
    result.assert_graph(expected_graph)

    rng = np.random.default_rng(0)
    result.assert_eval(
        rng.normal(size=(4, 4)).astype(config.floatX),
        rng.normal(size=4).astype(config.floatX),
    )


def test_local_sort_under_reduce_skips_multi_output_elemwise():
    x = matrix("x")
    composite = ps.Composite([a := ps.float64("a")], [ps.exp(a), ps.log(a)])
    out = pt_sum(Elemwise(composite)(pt.sort(x, axis=1))[1], axis=1)

    result = RewriteTester(
        [x], [out], include=None, custom_rewrite=local_sort_under_reduce
    )
    result.assert_graph(out)


def test_local_sort_under_reduce_drops_chained_reversals():
    x = matrix("x")
    out = pt_sum(pt.sort(x, axis=1)[:, ::-1][::-1])

    result = RewriteTester(
        [x], [out], include=None, custom_rewrite=local_sort_under_reduce
    )
    result.assert_graph(pt_sum(x))


def test_local_sort_under_reduce_keeps_sort_seen_by_output():
    x = vector("x")
    y = exp(pt.sort(x))

    result = RewriteTester([x], [y, pt_sum(y)])
    result.assert_graph(
        *rewrite_graph(
            [y, pt_sum(y)],
            include=("canonicalize",),
            exclude=("local_sort_under_reduce",),
        )
    )
