import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import function, scan, shared
from pytensor.compile.io import In
from pytensor.compile.mode import get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import ScanInplaceOptimizer
from pytensor.tensor.random.op import RandomVariableWithCoreShape
from pytensor.tensor.type import scalar, vector
from tests import unittest_tools as utt
from tests.scan.test_basic import asarrayX


class TestScanInplaceOptimizer:
    mode = get_default_mode().including("scan_make_inplace", "inplace")

    def test_no_inplace(self):
        """Make sure the rewrite doesn't make unnecessary replacements."""

        x = pt.vector("x")

        scan_out = pytensor.scan(
            lambda x: (x + 1) / 2 + 1, sequences=[x], return_updates=False
        )

        fgraph = FunctionGraph(
            outputs=[scan_out], clone=True, copy_inputs=False, copy_orphans=False
        )

        _ = ScanInplaceOptimizer().apply(fgraph)

        fgraph_op = fgraph.outputs[0].owner.inputs[0].owner.op
        assert not fgraph_op.destroy_map
        assert equal_computations([scan_out], fgraph.outputs)

    def test_inplace_basic(self):
        scan_out = pytensor.scan(
            lambda x: x + 1, outputs_info=[pt.zeros(1)], n_steps=3, return_updates=False
        )

        fgraph = FunctionGraph(
            outputs=[scan_out], clone=True, copy_inputs=False, copy_orphans=False
        )

        assert equal_computations([scan_out], fgraph.outputs)

        _ = ScanInplaceOptimizer().apply(fgraph)

        # The graphs shouldn't change; only the `Op.destroy_map`s
        assert equal_computations([scan_out], fgraph.outputs)

        fgraph_op = fgraph.outputs[0].owner.inputs[0].owner.op
        assert fgraph_op.destroy_map == {0: [1]}
        assert not scan_out.owner.inputs[0].owner.op.destroy_map

    @utt.assertFailure_fast
    def test_simple_rnn(self):
        """Simple RNN; compute inplace version 1."""
        rng = np.random.default_rng(utt.fetch_seed())
        vW = asarrayX(np.random.uniform())
        vW_in = asarrayX(np.random.uniform())
        vu0 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu1 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu2 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = vector("u0")
        u1 = vector("u1")
        u2 = vector("u2")
        mu0 = In(u0, mutable=False)
        mu1 = In(u1, mutable=True)
        mu2 = In(u2, mutable=True)
        x0 = scalar("x0")
        x1 = scalar("y0")
        W_in = shared(vW_in, "Win")
        W = shared(vW, "W")

        def f_rnn_shared(u0_t, u1_t, u2_t, x0_tm1, x1_tm1):
            return [
                u0_t * W_in + x0_tm1 * W + u1_t * u2_t,
                u0_t * W_in + x1_tm1 * W + u1_t + u2_t,
            ]

        outputs = scan(
            f_rnn_shared,
            [u0, u1, u2],
            [dict(initial=x0, inplace=u2), dict(initial=x1, inplace=u1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode,
            return_updates=False,
        )

        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            mode=self.mode,
            allow_input_downcast=True,
        )
        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 in scan_node[0].op.destroy_map
        assert 1 in scan_node[0].op.destroy_map
        # compute output in numpy
        numpy_x0 = np.zeros((3,))
        numpy_x1 = np.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu2[0]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu1[0] + vu2[0]
        for i in range(1, 3):
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + vu1[i] * vu2[i]
            numpy_x1[i] = vu0[i] * vW_in + numpy_x1[i - 1] * vW + vu1[i] + vu2[i]

        # note pytensor computes inplace, so call function after numpy
        # equivalent is done
        (pytensor_x0, pytensor_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that pytensor does what it should
        utt.assert_allclose(pytensor_x0, numpy_x0)
        utt.assert_allclose(pytensor_x1, numpy_x1)

    @utt.assertFailure_fast
    def test_simple_rnn_2(self):
        """Simple RNN; compute inplace version 2."""
        rng = np.random.default_rng(utt.fetch_seed())
        vW = asarrayX(np.random.uniform())
        vW_in = asarrayX(np.random.uniform())
        vu0 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu1 = asarrayX(rng.uniform(-5.0, 5.0, size=(4,)))
        vu2 = asarrayX(rng.uniform(-5.0, 5.0, size=(5,)))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = vector("u0")
        u1 = vector("u1")
        u2 = vector("u2")
        mu0 = In(u0, mutable=True)
        mu1 = In(u1, mutable=True)
        mu2 = In(u2, mutable=True)
        x0 = scalar("x0")
        x1 = scalar("y0")
        W_in = shared(vW_in, "Win")
        W = shared(vW, "W")

        def f_rnn_shared(u0_t, u1_t, u1_tp1, u2_tm1, u2_t, u2_tp1, x0_tm1, x1_tm1):
            return [
                u0_t * W_in + x0_tm1 * W + u1_t * u1_tp1,
                u0_t * W_in + x1_tm1 * W + u2_tm1 + u2_t + u2_tp1,
            ]

        outputs = scan(
            f_rnn_shared,
            [u0, dict(input=u1, taps=[0, 1]), dict(input=u2, taps=[-1, 0, +1])],
            [dict(initial=x0), dict(initial=x1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode,
            return_updates=False,
        )
        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            mode=self.mode,
            allow_input_downcast=True,
        )

        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 in scan_node[0].op.destroy_map
        assert 1 in scan_node[0].op.destroy_map
        # compute output in numpy
        numpy_x0 = np.zeros((3,))
        numpy_x1 = np.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu1[1]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu2[0] + vu2[1] + vu2[2]
        for i in range(1, 3):
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + vu1[i] * vu1[i + 1]
            numpy_x1[i] = (
                vu0[i] * vW_in + numpy_x1[i - 1] * vW + vu2[i] + vu2[i + 1] + vu2[i + 2]
            )

        # note pytensor computes inplace, so call function after numpy
        # equivalent is done
        (pytensor_x0, pytensor_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that pytensor does what it should
        utt.assert_allclose(pytensor_x0, numpy_x0)
        utt.assert_allclose(pytensor_x1, numpy_x1)

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE",
        reason="FAST_COMPILE does not trigger inplace optimizations",
    )
    def test_inplace_untraced_sit_sot(self):
        rng = shared(np.random.default_rng())
        next_rng, x0 = pt.random.normal(rng=rng).owner.outputs

        final_rng, xs = scan(
            fn=lambda rng_tm1, x_tm1: (
                pt.random.normal(x_tm1, rng=rng_tm1).owner.outputs
            ),
            outputs_info=[next_rng, x0],
            n_steps=5,
            return_updates=False,
        )

        f = function([], xs[-1], updates={rng: final_rng})
        [scan_node] = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, Scan)]
        op = scan_node.op

        # Both outputs (sit_sot, and untraced_sit_sot), should be in the destory map.
        # Respective inputs are shifted by one, since input 0 is n_steps
        assert op.destroy_map == {0: [1], 1: [2]}

        # The view_map should not contain the untraced_sit_sot entry anymore
        assert op.view_map == {}

        # The inner function should have inplace RNG operations
        from pytensor.tensor.random.op import RandomVariable

        [inner_rv_node] = [
            n
            for n in op.fgraph.toposort()
            if isinstance(n.op, RandomVariable | RandomVariableWithCoreShape)
        ]
        # Rng is first input and output, should be inplace
        # If it's a RandomVariableWithCoreShape, the rng input is shifted
        assert inner_rv_node.op.destroy_map in ({0: [0]}, {0: [1]})

        # Evaluate and check non equality
        assert f() != f()

    def test_inplace3(self):
        rng = np.random.default_rng(utt.fetch_seed())

        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())
        x0 = shared(vx0)
        x1 = shared(vx1)
        outputs = scan(
            lambda x, y: (x + asarrayX(1), y + asarrayX(1)),
            [],
            [x0, x1],
            n_steps=3,
            return_updates=False,
        )
        x0 = asarrayX(np.zeros((4,)))
        x0[0] = vx0
        x0 = pt.constant(x0)

        to_replace = outputs[0].owner.inputs[0].owner.inputs[1]
        outputs = clone_replace(outputs, replace=[(to_replace, x0)])

        f9 = function([], outputs, mode=self.mode)
        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 not in scan_node[0].op.destroy_map
        assert 1 in scan_node[0].op.destroy_map
