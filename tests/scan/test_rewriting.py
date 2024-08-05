import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import function, scan, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.io import In
from pytensor.compile.mode import get_default_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad, jacobian
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import ScanInplaceOptimizer, ScanMerge
from pytensor.scan.utils import until
from pytensor.tensor import stack
from pytensor.tensor.blas import Dot22
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Dot, dot, sigmoid, tanh
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import reshape, shape, specify_shape
from pytensor.tensor.type import (
    dmatrix,
    dvector,
    iscalar,
    ivector,
    matrix,
    scalar,
    tensor3,
    vector,
)
from tests import unittest_tools as utt
from tests.scan.test_basic import asarrayX, scan_nodes_from_fct


mode = pytensor.compile.mode.get_mode(config.mode)


class TestRemoveConstantsAndUnusedInputsScan:
    mode = get_default_mode().including("scan")

    def test_remove_constants_and_unused_inputs_scan_non_seqs(self):
        """Test the rewrite `remove_constants_and_unused_inputs_scan` for non-sequences."""
        W = matrix(name="W")
        v = ivector(name="v")
        y1, _ = scan(
            lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=[W]
        )
        y2, _ = scan(
            lambda i, _, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W],
        )
        y3, _ = scan(
            lambda i, W, _: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0]],
        )
        y4, _ = scan(
            lambda i, _, _2, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W[0], W],
        )
        y5, _ = scan(
            lambda i, _, W, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W, W[0]],
        )
        y6, _ = scan(
            lambda i, W, _, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
        )
        # TODO: y7 have problem during run time. I think it should
        # raise an error during the scan construction.
        # y7, _ = scan(lambda i, W, _, _2: W[i], sequences=v,
        #                    outputs_info=None, non_sequences=[v, W[0], W])

        W_val = np.random.normal(size=(3, 3)).astype(config.floatX)
        exp_val = W_val[np.r_[1, 2]]

        for out in [y1, y2, y3, y4, y5, y6]:
            f = function([W, v], out, mode=self.mode)

            res = f(W_val, [1, 2])
            assert np.array_equal(res, exp_val)

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1

            scan_node = scan_nodes[0]
            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_non_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))

            inp = scan_node.op.outer_non_seqs(scan_node.inputs)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))

    def test_remove_constants_and_unused_inputs_scan_seqs(self):
        """Test the opt remove_constants_and_unused_inputs_scan for sequences."""
        W = matrix(name="W")
        v = ivector(name="v")
        vv = matrix(name="vv")
        y1, _ = scan(
            lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=[W]
        )
        y2, _ = scan(
            lambda i, _, W: W[i], sequences=[v, v], outputs_info=None, non_sequences=W
        )
        y3, _ = scan(
            lambda i, _, W: W[i],
            sequences=[v, vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y4, _ = scan(
            lambda _, i, W: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=W,
        )
        y5, _ = scan(
            lambda _, i, _2, W: W[i],
            sequences=[vv, v, vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y6, _ = scan(
            lambda _, _2, i, W: W[i],
            sequences=[vv[0], vv, v],
            outputs_info=None,
            non_sequences=W,
        )
        y7, _ = scan(
            lambda i, _, _2, W: W[i],
            sequences=[v, vv[0], vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y8, _ = scan(
            lambda _, i, W, _2, _3: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
        )

        W_val = np.random.normal(size=(3, 3)).astype(config.floatX)
        exp_val = W_val[np.r_[1, 2]]

        for out in [y1, y2, y3, y4, y5, y6, y7, y8]:
            f = function(
                [W, v, vv],
                out,
                on_unused_input="ignore",
                mode=self.mode,
            )

            res = f(W_val, [1, 2], W_val)
            assert np.array_equal(res, exp_val)

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_seqs(scan_node.inputs)
            assert len(inp) == 1
            inp = scan_node.op.inner_non_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_non_seqs(scan_node.inputs)
            assert len(inp) == 1


class TestPushOutDot:
    mode = get_default_mode().including("scan")

    def test_pushout_all(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 + W2)

        o, _ = scan(lambda_fn, non_sequences=[h0, W1, W2], n_steps=5)

        f = function([h0, W1, W2], o, mode=self.mode)

        scan_nodes = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(scan_nodes) == 0

        seed = utt.fetch_seed()
        rng = np.random.default_rng(seed)
        floatX = config.floatX
        v_h = np.array(rng.uniform(size=(2,)), dtype=floatX)
        v_W1 = np.array(rng.uniform(size=(2, 2)), dtype=floatX)
        v_W2 = np.array(rng.uniform(size=(2, 2)), dtype=floatX)

        v_out = np.dot(v_h, v_W1 + v_W2)
        sol = np.zeros((5, 2))
        # This line is here to make sol have the same shape as the output of
        # pytensor. Note that what we ask pytensor to do is to repeat the 2
        # elements vector v_out 5 times
        sol[:, :] = v_out
        utt.assert_allclose(sol, f(v_h, v_W1, v_W2))

    def test_pushout_while(self):
        """
        Ensure that the optimizations for Scan that push computation out of
        the Scan don't alter the result for 'as_while' scans.
        """

        W1 = matrix("W1")
        W2 = matrix("W2")
        step_indices = vector("step_indices")

        def lambda_fn(step_idx, W1, W2):
            until_condition = until(step_idx > 2)
            return dot(W1, W2), until_condition

        # Compile a function with the optimization
        o, _ = scan(
            lambda_fn, sequences=[step_indices, W1], non_sequences=[W2], n_steps=5
        )

        f = function([W1, W2, step_indices], o, mode=self.mode)

        # Compule an pytensor function without the optimization
        o, _ = scan(
            lambda_fn,
            sequences=[step_indices, W1],
            non_sequences=[W2],
            n_steps=5,
            mode="FAST_COMPILE",
        )

        f_ref = function([W1, W2, step_indices], o, mode=self.mode)

        # Compare the results of the two implementations
        input_values = [
            np.random.default_rng(utt.fetch_seed()).random((5, 5)).astype("float32"),
            np.random.default_rng(utt.fetch_seed()).random((5, 5)).astype("float32"),
            np.arange(5).astype("float32"),
        ]

        out = f(*input_values)
        out_ref = f_ref(*input_values)
        utt.assert_allclose(out, out_ref)

    def test_pushout(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 + W2)

        o, _ = scan(lambda_fn, outputs_info=h0, non_sequences=[W1, W2], n_steps=5)

        f = function([h0, W1, W2], o, mode=self.mode)

        scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))
        assert (
            len(
                [
                    x
                    for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, Elemwise)
                ]
            )
            == 0
        )

    def test_pushout_nomodif(self):
        inp = matrix("inp")

        def fn(i, i_tm1):
            return i + 10, i_tm1

        ([i_t, i_tm1], _) = scan(
            fn,
            sequences=[inp],
            outputs_info=[np.asarray([0.0, 0.0], config.floatX), None],
        )
        f = function([inp], [i_t, i_tm1])
        val = np.arange(10).reshape(5, 2).astype(config.floatX)
        ret = f(val)
        utt.assert_allclose(ret[0], val + 10)
        utt.assert_allclose(
            ret[1], [[0.0, 0.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
        )


class TestPushOutNonSeqScan:
    """
    Tests for the `scan_push_out_non_seq` optimization in the case where the inner
    function of a `Scan` `Op` has an output which is the result of a `Dot` product
    on a non-sequence matrix input to `Scan` and a vector that is the result of
    computation in the inner function.
    """

    def test_pushout_seqs(self):
        def init_predictive_output(inputs, targets, hyp, x_star, s_star):
            E = hyp.shape[0]

            def init_K(i, X, Y):
                XX = X.sum(1).reshape((X.shape[0], 1))
                K = XX + XX.T
                return K.sum()

            beta, K_updts = scan(
                init_K, sequences=pt.arange(E), non_sequences=[inputs, targets]
            )

            # mean
            def predict_mean_i(i, x_star, s_star, X, beta, h):
                n, D = shape(X)
                # rescale every dimension by the corresponding inverse lengthscale
                iL = pt.diag(h[i, :D])
                inp = (X - x_star).dot(iL)

                # compute the mean
                B = iL.dot(s_star).dot(iL)
                t = inp.dot(B)

                lb = (inp * t).sum() + beta.sum()

                Mi = pt_sum(lb) * h[i, D]
                return Mi

            (M), M_updts = scan(
                predict_mean_i,
                sequences=pt.arange(E),
                non_sequences=[x_star, s_star, inputs, beta, hyp],
            )
            return M

        # some initializations
        hypx = np.log(np.tile([1, 1, 1, 1, 1, 1, 0.01], (3, 1)))

        # variables used in the following expressions
        hyp = shared(hypx)
        inputs = dmatrix("X")
        targets = dmatrix("Y")
        x_star = dvector("x_star")
        s_star = dmatrix("s_star")

        M = init_predictive_output(inputs, targets, hyp, x_star, s_star)

        X = np.random.default_rng(utt.fetch_seed()).random((10, 4))
        Y = np.random.default_rng(utt.fetch_seed()).random((10, 3))
        test_m = np.random.default_rng(utt.fetch_seed()).random((4,))
        test_s = np.eye(4)

        # Compute expected outputs (jacobian of M wrt x_star)
        dfdm = function(
            [inputs, targets, x_star, s_star],
            [
                grad(M[0], x_star),
                grad(M[1], x_star),
                grad(M[2], x_star),
            ],
        )
        expected_output = dfdm(X, Y, test_m, test_s)

        # equivalent code for the jacobian using scan
        dMdm, dMdm_updts = scan(
            lambda i, M, x: grad(M[i], x),
            sequences=pt.arange(M.shape[0]),
            non_sequences=[M, x_star],
        )
        dfdm = function([inputs, targets, x_star, s_star], [dMdm[0], dMdm[1], dMdm[2]])
        scan_output = dfdm(X, Y, test_m, test_s)

        dMdm_j = jacobian(M, x_star)
        dfdm_j = function(
            [inputs, targets, x_star, s_star], [dMdm_j[0], dMdm_j[1], dMdm_j[2]]
        )
        jacobian_outputs = dfdm_j(X, Y, test_m, test_s)

        utt.assert_allclose(expected_output, scan_output)
        utt.assert_allclose(expected_output, jacobian_outputs)

    @config.change_flags(on_opt_error="raise")
    def test_pushout_seqs2(self):
        x = matrix()
        outputs, updates = scan(
            lambda x: [x * x, pt.constant(0).copy().copy()],
            n_steps=2,
            sequences=[],
            non_sequences=[],
            outputs_info=[x, None],
        )

        # Compile an PyTensor function where any optimization error will lead to
        # an exception being raised
        function([x], outputs, updates=updates)

    @config.change_flags(on_opt_error="raise")
    def test_pushout_nonseq(self):
        """
        This test was created for a crashed that occurred during the
        optimization `PushOutNonSeqScan` when it attempted to a scan node with
        two outputs but only providing a replacement for one of those
        outputs. This led the optimization to raise an exception.
        """

        outputs, _ = scan(lambda x: (x * x, x), non_sequences=[2], n_steps=2)
        f = function(inputs=[], outputs=outputs)

        outs = f()
        expected_outs = [[4, 4], [2, 2]]
        utt.assert_allclose(outs, expected_outs)

    def test_dot_not_output(self):
        """
        Test the case where the vector input to the dot is not already an
        output of the inner function.
        """

        v = vector()
        m = matrix()
        output = dot(v, m)

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = pytensor.function([v, m], jacobian(output, v), mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        f_no_opt = pytensor.function([v, m], jacobian(output, v), mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = next(
            node for node in f_opt.maker.fgraph.toposort() if isinstance(node.op, Scan)
        )
        assert len(scan_node.op.inner_outputs) == 1
        assert not isinstance(scan_node.op.inner_outputs[0], Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        v_value = np.random.random(4).astype(config.floatX)
        m_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(v_value, m_value)
        output_no_opt = f_no_opt(v_value, m_value)

        utt.assert_allclose(output_opt, output_no_opt)

    def test_dot_nitsot_output(self):
        """
        Test the case where the vector input to the dot is already a nitsot
        output of the inner function.
        """

        a = matrix()
        b = matrix()

        def inner_fct(vect, mat):
            vect_squared = vect**2
            return dot(vect_squared, mat), vect_squared

        outputs, updates = pytensor.scan(
            fn=inner_fct, outputs_info=[None] * 2, sequences=a, non_sequences=b
        )

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = pytensor.function([a, b], outputs, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        f_no_opt = pytensor.function([a, b], outputs, mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = next(
            node for node in f_opt.maker.fgraph.toposort() if isinstance(node.op, Scan)
        )
        # NOTE: WHEN INFER_SHAPE IS RE-ENABLED, BELOW THE SCAN MUST
        # HAVE ONLY 1 OUTPUT.
        assert len(scan_node.op.inner_outputs) == 2
        assert not isinstance(scan_node.op.inner_outputs[0], Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        a_value = np.random.random((3, 4)).astype(config.floatX)
        b_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(a_value, b_value)
        output_no_opt = f_no_opt(a_value, b_value)

        utt.assert_allclose(output_opt[0], output_no_opt[0])
        utt.assert_allclose(output_opt[1], output_no_opt[1])

    def test_dot_sitsot_output(self):
        """
        Test the case where the vector input to the dot is not already a
        non-nitsot (in this case a sitsot) output of the inner function.
        """

        a = matrix()
        b = matrix()

        def inner_fct(seq1, previous_output1, nonseq1):
            output1 = previous_output1 + seq1
            output2 = dot(output1, nonseq1)
            return output1, output2

        outputs, updates = pytensor.scan(
            fn=inner_fct, outputs_info=[a[0], None], sequences=a, non_sequences=b
        )

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = pytensor.function([a, b], outputs, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        f_no_opt = pytensor.function([a, b], outputs, mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = next(
            node for node in f_opt.maker.fgraph.toposort() if isinstance(node.op, Scan)
        )
        assert len(scan_node.op.inner_outputs) == 2
        assert not isinstance(scan_node.op.inner_outputs[0], Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        a_value = np.random.random((3, 4)).astype(config.floatX)
        b_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(a_value, b_value)
        output_no_opt = f_no_opt(a_value, b_value)

        utt.assert_allclose(output_opt[0], output_no_opt[0])
        utt.assert_allclose(output_opt[1], output_no_opt[1])

    def test_OpFromGraph_shared(self):
        """Make sure that a simple `OpFromGraph` with a shared variable can be pushed out."""

        y = shared(1.0, name="y")

        test_ofg = OpFromGraph([], [1 + y])

        def inner_func():
            return test_ofg()

        out, out_updates = pytensor.scan(inner_func, n_steps=10)

        out_fn = function([], out, updates=out_updates)

        res = out_fn()
        assert np.array_equal(res, np.repeat(2.0, 10))

        y.set_value(2.0)

        res = out_fn()
        assert np.array_equal(res, np.repeat(3.0, 10))

    def test_nested_OpFromGraph_shared(self):
        y = pytensor.shared(1.0, name="y")

        test_ofg = OpFromGraph([], [y])

        def inner_func(x):
            out, _ = pytensor.scan(lambda: test_ofg(), n_steps=x)
            return out

        out, _ = pytensor.scan(inner_func, sequences=[pt.arange(1, 2)])

        _ = pytensor.function([], test_ofg())

        out_fn = pytensor.function([], out)

        assert np.array_equal(out_fn(), [[1.0]])


class TestPushOutAddScan:
    """
    Test case for the `scan_push_out_add` optimization in the case where the `Scan`
    is used to compute the sum over the dot products between the corresponding
    elements of two list of matrices.

    TODO FIXME XXX: These aren't real tests; they simply confirm that a few
    graph that could be relevant to the push-out optimizations can be compiled
    and evaluated.  None of them confirm that a push-out optimization has been
    performed.
    """

    def test_sum_dot(self):
        A = matrix("A")
        B = matrix("B")
        S, _ = scan(
            lambda x1, x2, u: u + dot(x1, x2),
            sequences=[A.dimshuffle(0, 1, "x"), B.dimshuffle(0, "x", 1)],
            outputs_info=[pt.zeros_like(A)],
        )
        f = function([A, B], S.owner.inputs[0][-1])
        rng = np.random.default_rng(utt.fetch_seed())
        vA = rng.uniform(size=(5, 5)).astype(config.floatX)
        vB = rng.uniform(size=(5, 5)).astype(config.floatX)
        utt.assert_allclose(f(vA, vB), np.dot(vA.T, vB))

    def test_pregreedy_optimizer(self, benchmark):
        W = pt.zeros((5, 4))
        bv = pt.zeros((5,))
        bh = pt.zeros((4,))
        v = matrix("v")
        (bv_t, bh_t), _ = scan(
            lambda _: [bv, bh], sequences=v, outputs_info=[None, None]
        )
        chain, _ = scan(
            lambda x: dot(dot(x, W) + bh_t, W.T) + bv_t,
            outputs_info=v,
            n_steps=2,
        )
        # TODO FIXME: Make this a real test and assert something.
        chain_fn = function([v], chain)

        benchmark(chain_fn, np.zeros((3, 5), dtype=config.floatX))

    def test_machine_translation(self):
        """
        This test case comes from https://github.com/rizar/scan-grad-speed and
        is an example of actual computation done with scan in the context of
        machine translation.

        `dim` has been reduced from 1000 to 5 to make the test run faster
        """

        # Parameters from an actual machine translation run
        batch_size = 80
        seq_len = 50
        dim = 5

        # Weight matrices
        U = pytensor.shared(
            np.random.normal(size=(dim, dim), scale=0.0001).astype(config.floatX)
        )
        U.name = "U"
        V = pytensor.shared(U.get_value())
        V.name = "V"
        W = pytensor.shared(U.get_value())
        W.name = "W"

        # Variables and their values
        x = tensor3("x")
        x_value = np.random.normal(
            size=(seq_len, batch_size, dim), scale=0.0001
        ).astype(config.floatX)

        ri = tensor3("ri")
        ri_value = x_value

        zi = tensor3("zi")
        zi_value = x_value

        init = pt.alloc(np.cast[config.floatX](0), batch_size, dim)

        def rnn_step1(
            # sequences
            x,
            ri,
            zi,
            # outputs_info
            h,
        ):
            pre_r = ri + h.dot(U)
            pre_z = zi + h.dot(V)
            r = sigmoid(pre_r)
            z = sigmoid(pre_z)

            after_r = r * h
            pre_h = x + after_r.dot(W)
            new_h = tanh(pre_h)

            res_h = z * new_h + (1 - z) * h
            return res_h

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        h, _ = pytensor.scan(
            rnn_step1,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init,
            name="fpass1",
            mode=opt_mode,
        )
        cost = h[-1].sum()
        grad1 = grad(cost, [U, V, W])
        f_opt = pytensor.function(inputs=[x, ri, zi], outputs=grad1, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        h, _ = pytensor.scan(
            rnn_step1,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init,
            name="fpass1",
            mode=no_opt_mode,
        )
        cost = h[-1].sum()
        grad1 = grad(cost, [U, V, W])
        f_no_opt = pytensor.function(
            inputs=[x, ri, zi], outputs=grad1, mode=no_opt_mode
        )

        # Validate that the optimization has been applied
        scan_node_grad = [
            node for node in f_opt.maker.fgraph.toposort() if isinstance(node.op, Scan)
        ][1]

        for output in scan_node_grad.op.inner_outputs:
            assert not (
                isinstance(output.owner.op, Elemwise)
                and any(isinstance(i, Dot) for i in output.owner.inputs)
            )

        # Compare the outputs of the two functions on the same input data.
        f_opt_output = f_opt(x_value, ri_value, zi_value)
        f_no_opt_output = f_no_opt(x_value, ri_value, zi_value)
        utt.assert_allclose(f_opt_output, f_no_opt_output)

    def test_non_zero_init(self):
        """Test the case where the initial value for the nitsot output is non-zero."""

        input1 = tensor3()
        input2 = tensor3()
        input3 = tensor3()

        W = pytensor.shared(np.random.normal(size=(4, 5))).astype(config.floatX)
        U = pytensor.shared(np.random.normal(size=(6, 7))).astype(config.floatX)

        def inner_fct(seq1, seq2, seq3, previous_output):
            temp1 = dot(seq1, W) + seq3
            temp2 = dot(seq2, U)
            dot_output = dot(temp1, temp2)
            return previous_output + dot_output

        init = pt.as_tensor_variable(np.random.normal(size=(3, 7)))

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        h, _ = pytensor.scan(
            inner_fct,
            sequences=[input1, input2, input3],
            outputs_info=init,
            mode=opt_mode,
        )
        output = h[-1]
        f_opt = pytensor.function([input1, input2, input3], output, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        h, _ = pytensor.scan(
            inner_fct,
            sequences=[input1, input2, input3],
            outputs_info=init,
            mode=no_opt_mode,
        )
        output = h[-1]
        f_no_opt = pytensor.function([input1, input2, input3], output, mode=no_opt_mode)

        # Ensure that the optimization has been applied for f_opt
        # TODO

        # Compare the outputs of the 2 functions
        input1_value = np.random.random((2, 3, 4)).astype(config.floatX)
        input2_value = np.random.random((2, 5, 6)).astype(config.floatX)
        input3_value = np.random.random((2, 3, 5)).astype(config.floatX)

        output_opt = f_opt(input1_value, input2_value, input3_value)
        output_no_opt = f_no_opt(input1_value, input2_value, input3_value)

        utt.assert_allclose(output_opt, output_no_opt)


class TestScanMerge:
    mode = get_default_mode().including("scan").excluding("scan_pushout_seqs_ops")

    @staticmethod
    def count_scans(fn):
        nodes = fn.maker.fgraph.apply_nodes
        scans = [node for node in nodes if isinstance(node.op, Scan)]
        return len(scans)

    def test_basic(self):
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[y])

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, upx = scan(sum, sequences=[x], n_steps=2)
        sy, upy = scan(sum, sequences=[y], n_steps=3)

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, upx = scan(sum, sequences=[x], n_steps=4)
        sy, upy = scan(sum, sequences=[y], n_steps=4)

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x], mode="FAST_COMPILE")

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x], truncate_gradient=1)

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

    def test_three_scans(self):
        r"""
        This test checks a case where we have three `Scan`\s, two of them
        cannot be merged together, but the third one can be merged with
        either.
        """
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, upx = scan(sum, sequences=[x], n_steps=4, name="X")
        # We need to use an expression of y rather than y so the toposort
        # comes up with the 'Y' scan last.
        sy, upy = scan(sum, sequences=[2 * y + 2], n_steps=4, name="Y")
        sz, upz = scan(sum, sequences=[sx], n_steps=4, name="Z")

        f = function([x, y], [sy, sz], mode=self.mode)
        assert self.count_scans(f) == 2

        rng = np.random.default_rng(utt.fetch_seed())
        x_val = rng.uniform(size=(4,)).astype(config.floatX)
        y_val = rng.uniform(size=(4,)).astype(config.floatX)
        # Run it so DebugMode can detect optimization problems.
        f(x_val, y_val)

    def test_belongs_to_set(self):
        """
        Test the method belongs_to of this class. Specifically see if it
        detects the two `Scan` nodes as not being similar.
        """
        inps = vector()
        state = scalar()
        y1, _ = scan(lambda x, y: x * y, sequences=inps, outputs_info=state, n_steps=5)

        y2, _ = scan(
            lambda x, y: (x + y, until(x > 0)),
            sequences=inps,
            outputs_info=state,
            n_steps=5,
        )
        scan_node1 = y1.owner.inputs[0].owner
        assert isinstance(scan_node1.op, Scan)
        scan_node2 = y2.owner.inputs[0].owner
        assert isinstance(scan_node2.op, Scan)
        opt_obj = ScanMerge()
        assert not opt_obj.belongs_to_set(scan_node1, [scan_node2])
        assert not opt_obj.belongs_to_set(scan_node2, [scan_node1])

    @config.change_flags(cxx="")  # Just for faster compilation
    def test_while_scan(self):
        x = vector("x")
        y = vector("y")

        def add(s):
            return s + 1, until(s > 5)

        def sub(s):
            return s - 1, until(s > 5)

        def sub_alt(s):
            return s - 1, until(s > 4)

        sx, upx = scan(add, sequences=[x])
        sy, upy = scan(sub, sequences=[y])

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, upx = scan(add, sequences=[x])
        sy, upy = scan(sub, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, upx = scan(add, sequences=[x])
        sy, upy = scan(sub_alt, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

    @config.change_flags(cxx="")  # Just for faster compilation
    def test_while_scan_nominal_dependency(self):
        """Test case where condition depends on nominal variables.

        This is a regression test for #509
        """
        c1 = scalar("c1")
        c2 = scalar("c2")
        x = vector("x", shape=(5,))
        y = vector("y", shape=(5,))
        z = vector("z", shape=(5,))

        def add(s1, s2, const):
            return s1 + 1, until(s2 > const)

        def sub(s1, s2, const):
            return s1 - 1, until(s2 > const)

        sx, _ = scan(add, sequences=[x, z], non_sequences=[c1])
        sy, _ = scan(sub, sequences=[y, -z], non_sequences=[c1])

        f = pytensor.function(inputs=[x, y, z, c1], outputs=[sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2
        res_sx, res_sy = f(
            x=[0, 0, 0, 0, 0],
            y=[0, 0, 0, 0, 0],
            z=[0, 1, 2, 3, 4],
            c1=0,
        )
        np.testing.assert_array_equal(res_sx, [1, 1])
        np.testing.assert_array_equal(res_sy, [-1, -1, -1, -1, -1])

        sx, _ = scan(add, sequences=[x, z], non_sequences=[c1])
        sy, _ = scan(sub, sequences=[y, z], non_sequences=[c2])

        f = pytensor.function(
            inputs=[x, y, z, c1, c2], outputs=[sx, sy], mode=self.mode
        )
        assert self.count_scans(f) == 2
        res_sx, res_sy = f(
            x=[0, 0, 0, 0, 0],
            y=[0, 0, 0, 0, 0],
            z=[0, 1, 2, 3, 4],
            c1=3,
            c2=1,
        )
        np.testing.assert_array_equal(res_sx, [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(res_sy, [-1, -1, -1])

        sx, _ = scan(add, sequences=[x, z], non_sequences=[c1])
        sy, _ = scan(sub, sequences=[y, z], non_sequences=[c1])

        f = pytensor.function(inputs=[x, y, z, c1], outputs=[sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        def nested_scan(c, x, z):
            sx, _ = scan(add, sequences=[x, z], non_sequences=[c])
            sy, _ = scan(sub, sequences=[x, z], non_sequences=[c])
            return sx.sum() + sy.sum()

        sz, _ = scan(
            nested_scan,
            sequences=[stack([c1, c2])],
            non_sequences=[x, z],
            mode=self.mode,
        )

        f = pytensor.function(inputs=[x, z, c1, c2], outputs=sz, mode=mode)
        [scan_node] = [
            node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
        ]
        inner_f = scan_node.op.fn
        assert self.count_scans(inner_f) == 1


class TestScanInplaceOptimizer:
    mode = get_default_mode().including("scan_make_inplace", "inplace")

    def test_no_inplace(self):
        """Make sure the rewrite doesn't make unnecessary replacements."""

        x = pt.vector("x")

        scan_out, _ = pytensor.scan(
            lambda x: (x + 1) / 2 + 1,
            sequences=[x],
        )

        fgraph = FunctionGraph(
            outputs=[scan_out], clone=True, copy_inputs=False, copy_orphans=False
        )

        _ = ScanInplaceOptimizer().apply(fgraph)

        fgraph_op = fgraph.outputs[0].owner.inputs[0].owner.op
        assert not fgraph_op.destroy_map
        assert equal_computations([scan_out], fgraph.outputs)

    def test_inplace_basic(self):
        scan_out, _ = pytensor.scan(
            lambda x: x + 1,
            outputs_info=[pt.zeros(1)],
            n_steps=3,
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

        outputs, updates = scan(
            f_rnn_shared,
            [u0, u1, u2],
            [dict(initial=x0, inplace=u2), dict(initial=x1, inplace=u1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode,
        )

        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            updates=updates,
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

        outputs, updates = scan(
            f_rnn_shared,
            [u0, dict(input=u1, taps=[0, 1]), dict(input=u2, taps=[-1, 0, +1])],
            [dict(initial=x0), dict(initial=x1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode,
        )
        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            updates=updates,
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

    @utt.assertFailure_fast
    def test_inplace3(self):
        rng = np.random.default_rng(utt.fetch_seed())

        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())
        x0 = shared(vx0)
        x1 = shared(vx1)
        outputs, updates = scan(
            lambda x, y: (x + asarrayX(1), y + asarrayX(1)), [], [x0, x1], n_steps=3
        )
        x0 = asarrayX(np.zeros((4,)))
        x0[0] = vx0
        x0 = pt.constant(x0)

        to_replace = outputs[0].owner.inputs[0].owner.inputs[1]
        outputs = clone_replace(outputs, replace=[(to_replace, x0)])

        f9 = function([], outputs, updates=updates, mode=self.mode)
        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 not in scan_node[0].op.destroy_map
        assert 1 in scan_node[0].op.destroy_map


class TestSaveMem:
    mode = get_default_mode().including("scan_save_mem", "scan_save_mem")

    def test_save_mem(self):
        rng = np.random.default_rng(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.5, 0.5, size=(8, 2)))
        v_u2 = asarrayX(rng.uniform(-0.5, 0.5, size=(8,)))
        v_x0 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                y_tm3 + 1,
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                y_tm1 + dot(x_tm1, W_out),
            ]

        _outputs, updates = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, dict(initial=x0), dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        outputs = [_outputs[0][-1], _outputs[1][-1], _outputs[2][-1]]
        f4 = function(
            [u1, u2, x0, y0, W_in1],
            outputs,
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode,
        )

        # compute the values in numpy
        v_x = np.zeros((8, 2), dtype=config.floatX)
        v_y = np.zeros((8,), dtype=config.floatX)
        v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
        v_y[0] = np.dot(v_x0, vWout) + v_y0[2]

        for i in range(1, 8):
            v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
            v_y[i] = np.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (pytensor_dump, pytensor_x, pytensor_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

        utt.assert_allclose(pytensor_x, v_x[-1:])
        utt.assert_allclose(pytensor_y, v_y[-1:])

    def test_save_mem_reduced_number_of_steps(self):
        def f_rnn(u_t):
            return (
                u_t + 1.0,
                u_t + 2.0,
                u_t + 3.0,
                u_t + 4.0,
                u_t + 5.0,
                u_t + 6.0,
                u_t + 7.0,
            )

        u = vector("u")
        idx = iscalar("idx")
        jdx = iscalar("jdx")
        [x1, x2, x3, x4, x5, x6, x7], updates = scan(
            f_rnn, u, n_steps=None, truncate_gradient=-1, go_backwards=False
        )

        f2 = function(
            [u, idx, jdx],
            [x1[:2], x2[4], x3[idx], x4[:idx], x5[-10], x6[-jdx], x7[:-jdx]],
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode,
        )
        # get random initial values
        rng = np.random.default_rng(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,))

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5, tx6, tx7 = f2(v_u, 3, 15)

        utt.assert_allclose(tx1, v_u[:2] + 1.0)
        utt.assert_allclose(tx2, v_u[4] + 2.0)
        utt.assert_allclose(tx3, v_u[3] + 3.0)
        utt.assert_allclose(tx4, v_u[:3] + 4.0)
        utt.assert_allclose(tx5, v_u[-10] + 5.0)
        utt.assert_allclose(tx6, v_u[-15] + 6.0)
        utt.assert_allclose(tx7, v_u[:-15] + 7.0)

    def test_save_mem_store_steps(self):
        def f_rnn(u_t, x1_tm1, x1_tm3, x2_tm1, x3tm2, x3_tm1, x4_tm1):
            return (
                u_t + 1.0,
                u_t + 2.0,
                u_t + 3.0,
                u_t + 4.0,
                u_t + 5.0,
                u_t + 6.0,
                u_t + 7.0,
            )

        u = vector("u")
        x10 = vector("x10")
        x20 = scalar("x20")
        x30 = vector("x30")
        x40 = scalar("x40")
        [x1, x2, x3, x4, x5, x6, x7], updates = scan(
            f_rnn,
            u,
            [
                None,
                None,
                None,
                dict(initial=x10, taps=[-1, -2]),
                x20,
                dict(initial=x30, taps=[-1, -2]),
                x40,
            ],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f2 = function(
            [u, x10, x20, x30, x40],
            [x1[-7], x2[-3:-1], x3[-6:], x4[-1], x5[-1]],
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode,
        )

        # get random initial values
        rng = np.random.default_rng(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,))

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5 = f2(v_u, [0, 0], 0, [0, 0], 0)

        utt.assert_allclose(tx1, v_u[-7] + 1.0)
        utt.assert_allclose(tx2, v_u[-3:-1] + 2.0)
        utt.assert_allclose(tx3, v_u[-6:] + 3.0)
        utt.assert_allclose(tx4, v_u[-1] + 4.0)
        utt.assert_allclose(tx5, v_u[-1] + 5.0)

    def test_savemem_does_not_duplicate_number_of_scan_nodes(self):
        var = pt.ones(())
        values, _ = scan(
            lambda x: ([x], (), until(x)),
            outputs_info=[var],
            n_steps=2,
        )

        tmp_fn = function([var], values, mode=self.mode)
        scan_nodes = [
            x for x in tmp_fn.maker.fgraph.toposort() if isinstance(x.op, Scan)
        ]
        assert len(scan_nodes) == 1

    def test_savemem_opt(self, benchmark):
        y0 = shared(np.ones((2, 10)))
        [y1, y2], updates = scan(
            lambda y: [y, y],
            outputs_info=[dict(initial=y0, taps=[-2]), None],
            n_steps=5,
        )
        # TODO FIXME: Make this a real test and assert something.
        fn = function([], y2.sum(), mode=self.mode)
        benchmark(fn)

    def test_savemem_opt_0_step(self):
        """
        Test a case where the savemem optimization has the opportunity to
        lower the number of steps of a Scan to 0. It tests that the
        optimization doesn't do so since Scan nodes with 0
        steps are not currently supported and doing so would result in a
        crash during the function execution.
        """

        def inner_scan_step(x_t_t, h_tm1, w):
            return dot(h_tm1, w) + x_t_t

        def outer_scan_step(x_t, w):
            h, _ = scan(
                inner_scan_step,
                sequences=[x_t[1:]],
                outputs_info=[x_t[0]],
                non_sequences=[w],
                strict=True,
                name="the_inner_scan",
            )
            return h

        def get_outputs(x, w):
            features, _ = scan(
                outer_scan_step,
                sequences=[x],
                non_sequences=[w],
                strict=True,
                name="the_outer_scan",
            )

            return_val = grad(features.sum(), w)
            return return_val

        # Compile the pytensor function
        x = tensor3("x")
        w = matrix("w")
        f = function(inputs=[x, w], outputs=get_outputs(x, w), mode=self.mode)

        # Test the function to ensure it returns valid results
        x_value = (
            np.random.default_rng(utt.fetch_seed())
            .random((2, 2, 3))
            .astype(config.floatX)
        )
        w_value = (
            np.random.default_rng(utt.fetch_seed()).random((3, 3)).astype(config.floatX)
        )
        expected_output = np.tile(x_value[:, 0].sum(0), (3, 1)).transpose()

        output = f(x_value, w_value)
        utt.assert_allclose(output, expected_output)

    @pytest.mark.skip(
        reason="The 'assertion' of this test relied on something that no longer exists "
    )
    def test_subtensor_multiple_slices(self):
        r"""
        This addresses a bug that happens when you have multiple subtensors
        on the output of `Scan`.  The bug requires the reshape to be produced,
        and it has something to do with how the `Subtensor`\s overlap.
        """

        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = vector("state")
        n_steps = iscalar("nsteps")
        output, updates = scan(
            f_pow2,
            [],
            state,
            [],
            n_steps=n_steps,
            truncate_gradient=-1,
            go_backwards=False,
        )
        nw_shape = ivector("nw_shape")
        # Note that the output is reshaped to 3 dimensional tensor, and
        my_f = function(
            [state, n_steps, nw_shape],
            [reshape(output, nw_shape, ndim=3)[:-2], output[:-4]],
            updates=updates,
            allow_input_downcast=True,
        )
        nodes = [x for x in my_f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        # This assertion fails if savemem optimization failed on scan
        if config.mode != "FAST_COMPILE":
            assert nodes[0].op._scan_savemem_visited
        rng = np.random.default_rng(utt.fetch_seed())
        my_f(rng.uniform(size=(3,)), 4, np.int64([2, 2, 3]))

    def test_while_scan_taps(self):
        n_steps = scalar("n_steps", dtype="int64")
        x0 = vector("x0")

        ys, _ = pytensor.scan(
            # Fibonacci Sequence
            lambda xtm2, xtm1: (xtm1 + xtm2, {}, until(xtm1 >= 34)),
            outputs_info=[{"initial": x0, "taps": [-2, -1]}],
            n_steps=n_steps,
        )
        # Save memory is triggered by choosing only last value
        y = ys[-1]

        f = pytensor.function(
            [n_steps, x0], y, mode=get_default_mode().including("scan")
        )

        np.testing.assert_equal(f(n_steps=1000, x0=[1, 1]), 55)
        np.testing.assert_equal(f(n_steps=1, x0=[1, 1]), 2)
        with pytest.raises(AssertionError, match="n_steps > 0"):
            f(n_steps=0, x0=[1, 1])

        # ys_trace is an Alloc that controls the size of the inner buffer,
        # it should have shape[0] == 3, with two entries for the taps and one
        # entry for the intermediate output
        [scan_node] = (n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, ys_trace = scan_node.inputs
        debug_fn = pytensor.function(
            [n_steps, x0], ys_trace.shape[0], accept_inplace=True
        )
        assert debug_fn(n_steps=1000, x0=[1, 1]) == 3

    def test_while_scan_map(self):
        xs = vector("xs")
        ys, _ = pytensor.scan(
            lambda x: (x + 1, {}, until(x + 1 >= 10)),
            outputs_info=[None],
            sequences=[xs],
        )
        # Save memory is triggered by choosing only last value
        y = ys[-1]

        f = pytensor.function([xs], y, mode=get_default_mode().including("scan"))
        np.testing.assert_equal(f(xs=np.arange(100, dtype=config.floatX)), 10)
        np.testing.assert_equal(f(xs=[0]), 1)
        with pytest.raises(IndexError):
            f(xs=[])

        # len_ys is a numerical input that controls the shape of the inner buffer
        # It should be 1, as only the last output is needed
        [scan_node] = (n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, _, len_ys = scan_node.inputs
        debug_fn = pytensor.function([xs], len_ys, accept_inplace=True)
        assert debug_fn(xs=np.zeros((100,), dtype=config.floatX)) == 1

    def test_while_scan_taps_and_map(self):
        x0 = scalar("x0")
        seq = vector("seq")
        n_steps = scalar("n_steps", dtype="int64")

        # while loop
        [ys, zs], _ = pytensor.scan(
            lambda s, xtm1: ((xtm1 + 1, xtm1 + 1 + s), {}, until(xtm1 >= 99)),
            sequences=[seq],
            outputs_info=[x0, None],
            n_steps=n_steps,
        )
        # Save memory is triggered by choosing only last value
        y = ys[-1]
        z = zs[-1]

        f = pytensor.function(
            [x0, seq, n_steps], [y, z], mode=get_default_mode().including("scan")
        )
        test_seq = np.zeros(200, dtype=config.floatX)
        np.testing.assert_allclose(f(x0=0, seq=test_seq, n_steps=200), 100)
        np.testing.assert_allclose(f(x0=1, seq=test_seq, n_steps=20), 21)
        np.testing.assert_allclose(f(x0=np.e, seq=test_seq, n_steps=1), np.e + 1)
        with pytest.raises(AssertionError, match="n_steps > 0"):
            f(x0=0, seq=test_seq, n_steps=0)

        # Evaluate the shape of ys_trace and len_zs to confirm the rewrite worked correctly.
        [scan_node] = (n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, _, ys_trace, len_zs = scan_node.inputs
        debug_fn = pytensor.function(
            [x0, n_steps], [ys_trace.shape[0], len_zs], accept_inplace=True
        )
        stored_ys_steps, stored_zs_steps = debug_fn(x0=0, n_steps=200)
        assert stored_ys_steps == 2
        assert stored_zs_steps == 1

    def test_vector_zeros_init(self):
        ys, _ = pytensor.scan(
            fn=lambda ytm2, ytm1: ytm1 + ytm2,
            outputs_info=[{"initial": pt.zeros(2), "taps": range(-2, 0)}],
            n_steps=100,
        )

        fn = pytensor.function([], ys[-50:], mode=self.mode)
        assert tuple(fn().shape) == (50,)

        # Check that rewrite worked
        [scan_node] = (n for n in fn.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, ys_trace = scan_node.inputs
        debug_fn = pytensor.function([], ys_trace.shape[0], accept_inplace=True)
        assert debug_fn() == 50


def test_inner_replace_dot():
    """
    This tests that rewrites are applied to the inner-graph.
    In particular, BLAS-based rewrites that remove the original dot product.

    This was previously a test with a name that implied it was testing the
    `Scan` push-out rewrites, but it wasn't testing that at all, because the
    rewrites were never being applied.
    """
    W = matrix("W")
    h = matrix("h")

    mode = get_default_mode().including("scan")  # .excluding("BlasOpt")

    o, _ = scan(
        lambda hi, him1, W: (hi, dot(hi + him1, W)),
        outputs_info=[pt.zeros([h.shape[1]]), None],
        sequences=[h],
        non_sequences=[W],
        mode=mode,
    )

    f = function([W, h], o, mode=mode)

    scan_nodes = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
    assert len(scan_nodes) == 1
    scan_op = scan_nodes[0].op
    assert not any(isinstance(n.op, Dot) for n in scan_op.fn.maker.fgraph.apply_nodes)


def test_alloc_inputs1():
    W1 = matrix("W1")
    W2 = matrix("W2")
    h0 = vector("h0")

    def lambda_fn(h, W1, W2):
        return dot(h, W1 * W2)

    o, _ = scan(
        lambda_fn,
        outputs_info=h0,
        non_sequences=[W1, pt.zeros_like(W2)],
        n_steps=5,
    )

    f = function([h0, W1, W2], o, mode=get_default_mode().including("scan"))
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))
    assert (
        len(
            [
                x
                for x in scan_node.op.fn.maker.fgraph.toposort()
                if isinstance(x.op, Elemwise)
            ]
        )
        == 0
    )


@pytest.mark.skip(
    reason="This tests depends on an optimization for "
    "scan that has not been implemented yet."
)
def test_alloc_inputs2():
    W1 = matrix()
    W2 = matrix()
    h0 = vector()

    def lambda_fn(W1, h, W2):
        return W1 * dot(h, W2)

    o, _ = scan(
        lambda_fn,
        sequences=pt.zeros_like(W1),
        outputs_info=h0,
        non_sequences=[pt.zeros_like(W2)],
        n_steps=5,
    )

    f = function([h0, W1, W2], o, mode=get_default_mode().including("scan"))
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))

    assert (
        len(
            [
                x
                for x in scan_node.op.fn.maker.fgraph.toposort()
                if isinstance(x.op, Elemwise)
            ]
        )
        == 0
    )


def test_alloc_inputs3():
    _W1 = matrix()
    _W2 = matrix()
    _h0 = vector()

    W1 = specify_shape(_W1, (3, 3))
    W2 = specify_shape(_W2, (3, 3))
    h0 = specify_shape(_h0, (3,))

    def lambda_fn(W1, h, W2):
        return W1 * dot(h, W2)

    o, _ = scan(
        lambda_fn,
        sequences=pt.zeros_like(W1),
        outputs_info=h0,
        non_sequences=[pt.zeros_like(W2)],
        n_steps=5,
    )

    # TODO FIXME: This result depends on unrelated rewrites in the "fast" mode.
    f = function([_h0, _W1, _W2], o, mode="FAST_RUN")
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))

    assert len(scan_node.op.inner_inputs) == 1


def test_opt_order():
    """
    Verify that scan optimizations are applied before blas
    optimizations.

    This is needed as otherwise, the dot won't become a dot22
    so it will be slower and won't get transferred to the gpu.
    """

    x = matrix("x")
    A = matrix("A")

    z, updates = scan(dot, sequences=[], non_sequences=[x, A], n_steps=2)
    f = function([x, A], z, mode="FAST_RUN")
    topo = f.maker.fgraph.toposort()

    assert any(isinstance(node.op, Dot22) for node in topo)

    vx = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=config.floatX)
    vA = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=config.floatX)
    vR = np.array([[[2, 1], [4, 2]], [[2, 1], [4, 2]]], dtype=config.floatX)
    utt.assert_allclose(f(vx, vA), vR)
