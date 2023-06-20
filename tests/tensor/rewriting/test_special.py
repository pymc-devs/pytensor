import numpy as np
import pytest

import pytensor
from pytensor import shared
import pytensor.scalar as aes
from pytensor.compile import optdb
from pytensor.compile.mode import get_mode, get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor.math import add, exp, log, true_div, MaxAndArgmax
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad, softmax
from pytensor.tensor.type import matrix, tensor3, TensorType, vector
from tests import unittest_tools as utt
from pytensor.tensor.math import sum as at_sum
from pytensor.tensor.elemwise import DimShuffle



_fast_run_rewrites = RewriteDatabaseQuery(include=["fast_run"])
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

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_logsoftmax_grad_rewrite(self, axis):
        """Test the `Logsoftmax`'s grad substitution.

        Check that ``Log(Softmax(x))``'s grad is substituted with ``Logsoftmax(x)``'s
        grad and that the new operation does not explode for big inputs.
        Note that only the grad is checked.
        """

        m = config.mode
        m = get_mode(m)
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

    def test_logsoftmax_grad_true_div_elemwise(self):
        """
        Checks that the gradient of an expression similar to a ``log(softmax)`` but
        with a different elemwise operation than true_div is not rewritten.
        """

        x = matrix("x")
        y = log(softmax(x))
        g = pytensor.tensor.grad(y.sum(), x)

        softmax_grad_node = g.owner
        assert softmax_grad_node.op == SoftmaxGrad(axis=-1)
        true_div_node = softmax_grad_node.inputs[0].owner
        assert true_div_node.op == true_div

        # We replace thk elemwise true_div op by an elemwise add.
        new_g = SoftmaxGrad(axis=-1)(
            add(*true_div_node.inputs), softmax_grad_node.inputs[1]
        )

        fgraph = FunctionGraph([x], [new_g])
        _fast_run_rewrites.rewrite(fgraph)

        assert SoftmaxGrad(axis=-1) in [n.op for n in fgraph.toposort()]


def test_log_softmax_stabilization():
    mode = pytensor.compile.mode.get_default_mode()
    mode = mode.including("local_log_softmax", "specialize")

    x = matrix()
    y = softmax(x)
    z = log(y)

    fgraph = FunctionGraph([x], [z])
    _fast_run_rewrites(fgraph)
    assert check_stack_trace(fgraph, ops_to_check="all")

    # Check that the softmax has been rewritten
    for node in fgraph.toposort():
        assert not isinstance(node.op, Softmax)

    # Call the function so debug mode can verify the rewritten version matches
    # the un-rewritten version
    f = pytensor.function([x], z, mode=mode)
    rng = np.random.default_rng(utt.fetch_seed())
    f(np.cast[config.floatX](rng.random((2, 3))))


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

def compile_graph_log_sum_exp(x, axis, dimshuffle_op=None):
    sum_exp = at_sum(exp(x), axis=axis)
    if dimshuffle_op:
        sum_exp = dimshuffle_op(sum_exp)
    y = log(sum_exp)
    MODE = get_default_mode().including("local_log_sum_exp")
    return function([x], y, mode=MODE)


def check_max_log_sum_exp(x, axis, dimshuffle_op=None):
    f = compile_graph_log_sum_exp(x, axis, dimshuffle_op)

    fgraph = f.maker.fgraph.toposort()
    for node in fgraph:
        if (
            hasattr(node.op, "scalar_op")
            and node.op.scalar_op == aes.basic.scalar_maximum
        ):
            return

        # In mode FAST_COMPILE, the rewrites don't replace the
        # `MaxAndArgmax` `Op`.
        if isinstance(node.op, MaxAndArgmax):
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
    transpose_op = DimShuffle((False, False), (1, 0))
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
    transpose_op = DimShuffle((False, False), (1, 0))
    f = compile_graph_log_sum_exp(x, axis=(1,), dimshuffle_op=transpose_op)
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1).T)
    rewritten_ret = f(x_val)
    assert np.allclose(naive_ret, rewritten_ret)

def test_local_log_sum_exp_large():
    """Test that the rewrite result is correct for extreme value 100."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0)

    x_val = np.array([-100.0, 100.0]).astype(config.floatX)

    rewritten_ret = f(x_val)
    assert np.allclose(rewritten_ret, 100.0)


def test_local_log_sum_exp_inf():
    """Test that when max = +-inf, the rewritten output still works correctly."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0)

    assert f([-np.inf, -np.inf]) == -np.inf
    assert f([np.inf, np.inf]) == np.inf
    assert f([-np.inf, np.inf]) == np.inf
