import copy

import numpy as np
import pytest
import scipy.special

import pytensor
import pytensor.scalar as ps
from pytensor import shared
from pytensor.compile import optdb
from pytensor.compile.mode import get_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Max, exp, log
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.special import (
    local_exp_log_softmax,
    local_log_softmax_from_logsumexp,
)
from pytensor.tensor.special import LogSoftmax, Softmax, log_softmax, logsumexp, softmax
from pytensor.tensor.type import TensorType, dvector, matrix, tensor3, vector
from tests import unittest_tools as utt
from tests.unittest_tools import RewriteTester


# Fusion inlines `Softmax` and `LogSoftmax`, which would hide the ops these tests check for
_fast_run_rewrites = RewriteDatabaseQuery(
    include=["fast_run"], exclude=["inline_symbolic_for_fusion"]
)
_fast_run_rewrites = optdb.query(_fast_run_rewrites)


class TestLogSoftmaxRewrites:
    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_logsoftmax_rewrite(self, axis):
        """Test the `Logsoftmax` substitution.

        Check that ``Log(Softmax(x))`` is substituted with ``Logsoftmax(x)``. Note that
        only the forward pass is checked (i.e., doesn't check the gradient)
        """

        x = matrix("x")
        sm = softmax(x, axis=axis)
        logsm = log(sm)
        fgraph = FunctionGraph([x], [logsm])
        _fast_run_rewrites.rewrite(fgraph)
        assert isinstance(fgraph.outputs[0].owner.op, LogSoftmax)
        assert check_stack_trace(fgraph, ops_to_check=LogSoftmax)
        assert check_stack_trace(fgraph, ops_to_check="all")

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_exp_log_softmax_rewrite(self, axis):
        """Check that ``Exp(LogSoftmax(x)) -> Softmax(x)``."""
        x = matrix("x")
        test_val = np.array([[1000.0, 1001.0], [1.0, 2.0]])

        result = RewriteTester(
            [x], [exp(log_softmax(x, axis=axis))], custom_rewrite=local_exp_log_softmax
        )
        result.assert_graph(softmax(x, axis=axis))
        result.assert_eval(test_val)

        # When the LogSoftmax is needed anyway, reusing it beats repeating the reduction
        log_sm = log_softmax(x, axis=axis)
        result = RewriteTester(
            [x], [log_sm, exp(log_sm)], custom_rewrite=local_exp_log_softmax
        )
        result.assert_graph(log_sm, exp(log_sm))

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_log_softmax_from_logsumexp(self, axis):
        """Check that ``x - logsumexp(x, axis, keepdims=True) -> LogSoftmax(x)``."""
        x = matrix("x")
        test_val = np.array([[1000.0, 1001.0], [1.0, 2.0]])

        result = RewriteTester(
            [x],
            [x - logsumexp(x, axis=axis, keepdims=True)],
            custom_rewrite=local_log_softmax_from_logsumexp,
        )
        result.assert_graph(log_softmax(x, axis=axis))
        result.assert_eval(test_val)

        # When the logsumexp is needed anyway, reusing it beats repeating the reduction
        lse = logsumexp(x, axis=axis, keepdims=True)
        result = RewriteTester(
            [x], [x - lse, lse], custom_rewrite=local_log_softmax_from_logsumexp
        )
        result.assert_graph(x - lse, lse)

    @pytest.mark.parametrize("axis", [None, 0, -1])
    @pytest.mark.parametrize("idx0", [0, slice(1, None), slice(None)])
    @pytest.mark.parametrize("idx1", [None, [0, 1, 1, -1]])
    def test_logsoftmax_subtensor_dimshuffle(self, axis, idx0, idx1):
        """Test that stabilization is introduced even when subtensor or dimshuffle operations
        are present between log and softmax.
        """
        logit_p = matrix("logit_p")
        p = softmax(logit_p, axis=axis)
        p_indexed = p[(idx0, idx1)]
        out = log(p_indexed)

        # Don't waste time with C compilation
        with config.change_flags(cxx=""):
            mode = get_mode(None).including("stabilize")
            fn = pytensor.function([logit_p], out, mode=mode)

        assert not any(
            isinstance(node.op, Softmax) for node in fn.maker.fgraph.apply_nodes
        )

        # This range would lead to underflow to -inf without the stabilization
        test_logit_p = np.array(
            [[-10.0, -10.0, 999.0], [999.0, 990.0, -10.0]], dtype=config.floatX
        )
        np.testing.assert_allclose(
            fn(logit_p=test_logit_p),
            scipy.special.log_softmax(test_logit_p, axis=axis)[(idx0, idx1)],
        )

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_logsoftmax_grad_rewrite(self, axis):
        """Test the `Logsoftmax`'s grad substitution.

        Check that ``Log(Softmax(x))``'s grad is substituted with ``Logsoftmax(x)``'s
        grad and that the new operation does not explode for big inputs.
        Note that only the grad is checked.
        """

        m = config.mode
        m = get_mode(m).including("stabilize")
        m.check_isfinite = False
        # some inputs that are large to make the gradient explode in the non
        # rewritten case
        rng = np.random.default_rng(utt.fetch_seed())
        a = np.exp(10 * rng.random((5, 10)).astype(config.floatX))

        def myfunc(x):
            sm = softmax(x, axis=axis)
            logsm = log(sm)
            return logsm

        # We set step to 0.1 because for big values we need a big epsilon
        utt.verify_grad(myfunc, [a], eps=0.1, mode=m)
        sa = shared(a)
        f = FunctionGraph([sa], [myfunc(sa)])
        _fast_run_rewrites(f)
        assert check_stack_trace(f, ops_to_check="all")


def test_softmax_graph():
    """Make sure that sotfmax expressions are turned into
    a softmax Op.

    """
    rng = np.random.default_rng(utt.fetch_seed())
    x = pytensor.shared(rng.normal(size=(3, 4)))

    def softmax_graph(c):
        return exp(c) / exp(c).sum(axis=-1, keepdims=True)

    def f(inputs):
        y = softmax_graph(x)
        return pytensor.grad(None, x, known_grads={y: inputs})

    utt.verify_grad(f, [rng.random((3, 4))])


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN"])
def test_local_log_add_exp(mode):
    m = get_mode(mode).excluding("fusion")
    m = copy.copy(m)
    # No need to put them back as we have a new object
    m.check_isfinite = False

    # check some basic cases
    x = dvector()
    y = dvector()
    f = pytensor.function([x, y], log(exp(x) + exp(y)), mode=m)

    # test that it gives the correct result when it doesn't overflow
    f([10], [10])  # doesn't causes overflow
    utt.assert_allclose(f([10], [10]), 10 + np.log1p(1))

    assert np.isfinite(f([10000], [10000]))  # causes overflow if handled incorrectly
    utt.assert_allclose(f([10000], [10000]), 10000 + np.log1p(1))

    # test that when max = +-inf, rewritten output still works correctly
    assert f([-np.inf], [-np.inf]) == -np.inf
    assert f([np.inf], [np.inf]) == np.inf
    assert f([np.inf], [-np.inf]) == np.inf

    # test that it also works with more than two args
    x = dvector()
    y = dvector()
    f = pytensor.function(
        [x, y], log(exp(x) + exp(y) + exp(x - y) + exp(x + y)), mode=m
    )

    assert np.isfinite(f([10000], [10000]))  # causes overflow if handled incorrectly
    utt.assert_allclose(f([10000], [10000]), 20000)

    # TODO: test that the rewrite works in the presence of broadcasting.


def compile_graph_log_sum_exp(x, axis, dimshuffle_op=None, mode="FAST_RUN"):
    sum_exp = pt_sum(exp(x), axis=axis)
    if dimshuffle_op:
        sum_exp = dimshuffle_op(sum_exp)
    y = log(sum_exp)
    return pytensor.function([x], y, mode=mode)


def check_max_log_sum_exp(x, axis, dimshuffle_op=None):
    f = compile_graph_log_sum_exp(x, axis, dimshuffle_op)

    fgraph = f.maker.fgraph.toposort()
    for node in fgraph:
        if hasattr(node.op, "scalar_op") and node.op.scalar_op == ps.basic.maximum:
            return

        if isinstance(node.op, Max):
            return

    # TODO FIXME: Refactor this test so that it makes a direct assertion and
    # nothing more.
    raise AssertionError("No maximum detected after log_sum_exp rewrite")


def test_local_log_sum_exp_maximum():
    """Test that the rewrite is applied by checking the presence of the maximum."""
    x = tensor3("x")
    check_max_log_sum_exp(x, axis=(0,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(1,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(2,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1, 2), dimshuffle_op=None)

    # If a transpose is applied to the sum
    transpose_op = DimShuffle(input_ndim=2, new_order=(1, 0))
    check_max_log_sum_exp(x, axis=2, dimshuffle_op=transpose_op)

    # If the sum is performed with keepdims=True
    x = TensorType(dtype="floatX", shape=(None, 1, None))("x")
    sum_keepdims_op = x.sum(axis=(0, 1), keepdims=True).owner.op
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=sum_keepdims_op)


def test_local_log_sum_exp_near_one():
    """Test that the rewritten result is correct around 1.0."""

    x = tensor3("x")
    x_val = 1.0 + np.random.random((4, 3, 2)).astype(config.floatX) / 10.0

    f = compile_graph_log_sum_exp(x, axis=(1,))
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1))
    rewritten_ret = f(x_val)
    assert np.allclose(naive_ret, rewritten_ret)

    # If a transpose is applied
    transpose_op = DimShuffle(input_ndim=2, new_order=(1, 0))
    f = compile_graph_log_sum_exp(x, axis=(1,), dimshuffle_op=transpose_op)
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1).T)
    rewritten_ret = f(x_val)
    assert np.allclose(naive_ret, rewritten_ret)


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN"])
@pytest.mark.parametrize(
    "x_val", ([-800.0, 800.0], [-800.0, -805.0]), ids=["overflow", "underflow"]
)
def test_local_log_sum_exp_large(x_val, mode):
    """Test that the rewrite result is correct for values the naive graph can't represent."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0, mode=mode)

    x_val = np.array(x_val, dtype=config.floatX)

    rewritten_ret = f(x_val)
    np.testing.assert_allclose(rewritten_ret, scipy.special.logsumexp(x_val), rtol=1e-5)


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN"])
def test_local_log_sum_exp_inf(mode):
    """Test that when max = +-inf, the rewritten output still works correctly."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0, mode=mode)

    assert f([-np.inf, -np.inf]) == -np.inf
    assert f([np.inf, np.inf]) == np.inf
    assert f([-np.inf, np.inf]) == np.inf
