import numpy as np
import pytest
import scipy.special

import pytensor
from pytensor import shared
from pytensor.compile import optdb
from pytensor.compile.mode import get_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor.math import add, exp, log, true_div
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad, softmax
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


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
        assert check_stack_trace(fgraph, ops_to_check="all")

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

    def test_logsoftmax_grad_true_div_elemwise(self):
        """
        Checks that the gradient of an expression similar to a ``log(softmax)`` but
        with a different elemwise operation than true_div is not rewritten.
        """

        x = matrix("x")
        y = log(softmax(x, axis=-1))
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
