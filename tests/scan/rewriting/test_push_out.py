import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import function, scan, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import get_default_mode, get_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad, jacobian
from pytensor.scan.op import Scan, ScanInfo
from pytensor.scan.utils import until
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Dot, dot, sigmoid, tanh
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import shape
from pytensor.tensor.type import (
    dmatrix,
    dvector,
    matrix,
    tensor3,
    vector,
)
from tests import unittest_tools as utt


mode = pytensor.compile.mode.get_mode(config.mode)


class TestPushOutDot:
    mode = get_default_mode().including("scan")

    def test_pushout_all(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 + W2)

        o = scan(lambda_fn, non_sequences=[h0, W1, W2], n_steps=5, return_updates=False)

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
        o = scan(
            lambda_fn,
            sequences=[step_indices, W1],
            non_sequences=[W2],
            n_steps=5,
            return_updates=False,
        )

        f = function([W1, W2, step_indices], o, mode=self.mode)

        # Compule an pytensor function without the optimization
        o = scan(
            lambda_fn,
            sequences=[step_indices, W1],
            non_sequences=[W2],
            n_steps=5,
            mode="FAST_COMPILE",
            return_updates=False,
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

        o = scan(
            lambda_fn,
            outputs_info=h0,
            non_sequences=[W1, W2],
            n_steps=5,
            return_updates=False,
        )

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

        [i_t, i_tm1] = scan(
            fn,
            sequences=[inp],
            outputs_info=[np.asarray([0.0, 0.0], config.floatX), None],
            return_updates=False,
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

            beta, _K_updts = scan(
                init_K, sequences=pt.arange(E), non_sequences=[inputs, targets]
            )

            # mean
            def predict_mean_i(i, x_star, s_star, X, beta, h):
                _n, D = shape(X)
                # rescale every dimension by the corresponding inverse lengthscale
                iL = pt.diag(h[i, :D])
                inp = (X - x_star).dot(iL)

                # compute the mean
                B = iL.dot(s_star).dot(iL)
                t = inp.dot(B)

                lb = (inp * t).sum() + beta.sum()

                Mi = pt_sum(lb) * h[i, D]
                return Mi

            (M), _M_updts = scan(
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
        dMdm, _dMdm_updts = scan(
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
        outputs = scan(
            lambda x: [x * x, pt.constant(0).copy().copy()],
            n_steps=2,
            sequences=[],
            non_sequences=[],
            outputs_info=[x, None],
            return_updates=False,
        )

        # Compile an PyTensor function where any optimization error will lead to
        # an exception being raised
        function([x], outputs)

    @config.change_flags(on_opt_error="raise")
    def test_pushout_nonseq(self):
        """
        This test was created for a crashed that occurred during the
        optimization `PushOutNonSeqScan` when it attempted to a scan node with
        two outputs but only providing a replacement for one of those
        outputs. This led the optimization to raise an exception.
        """

        outputs = scan(
            lambda x: (x * x, x), non_sequences=[2], n_steps=2, return_updates=False
        )
        f = function(inputs=[], outputs=outputs)

        outs = f()
        expected_outs = [[4, 4], [2, 2]]
        utt.assert_allclose(outs, expected_outs)

    def test_pushout_nitsot_buffer_larger_than_nsteps(self):
        """When folding a stateless nit_sot scan into an Elemwise, the folded
        result has length == ``n_steps``, which may be less than the nit_sot's
        declared outer buffer size. Pushout must pad the folded result with
        zeros so the trailing slots match what the un-folded scan would have
        produced (uninitialized-but-conventionally-zero nit_sot slots).

        Reproduced here directly by building a Scan op with distinct
        ``n_steps`` and ``nit_sot_size`` outer inputs -- the same pattern that
        arises in ``Scan.pullback`` when ``truncate_gradient`` is set.
        """
        info = ScanInfo(
            n_seqs=1,
            mit_mot_in_slices=(),
            mit_mot_out_slices=(),
            mit_sot_in_slices=(),
            sit_sot_in_slices=(),
            n_nit_sot=1,
            n_untraced_sit_sot=0,
            n_non_seqs=0,
            as_while=False,
        )

        inner_seq = pt.dscalar("inner_seq")
        inner_out = inner_seq * 2.0
        scan_op = Scan(inputs=[inner_seq], outputs=[inner_out], info=info)

        seq = pt.dvector("seq")
        n_steps = pt.lscalar("n_steps")
        nit_sot_size = pt.lscalar("nit_sot_size")
        out = scan_op(n_steps, seq[:n_steps], nit_sot_size)

        f = function([seq, n_steps, nit_sot_size], out)

        has_scan = any(isinstance(node.op, Scan) for node in f.maker.fgraph.apply_nodes)
        if config.mode == "FAST_COMPILE":
            # Rewrite not fired in fast compile
            assert has_scan
        else:
            # Pushout should have fired: no Scan node left.
            assert not has_scan

        res = f(np.arange(10, dtype="float64"), np.array(3), np.array(7))
        np.testing.assert_array_equal(
            res, np.array([0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        )

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

        outputs, _updates = pytensor.scan(
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

        outputs, _updates = pytensor.scan(
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

        test_ofg = OpFromGraph([y], [1 + y])

        def inner_func():
            return test_ofg(y)

        out, out_updates = pytensor.scan(inner_func, n_steps=10)

        out_fn = function([], out, updates=out_updates)

        res = out_fn()
        assert np.array_equal(res, np.repeat(2.0, 10))

        y.set_value(2.0)

        res = out_fn()
        assert np.array_equal(res, np.repeat(3.0, 10))

    def test_nested_OpFromGraph_shared(self):
        y = pytensor.shared(1.0, name="y")

        test_ofg = OpFromGraph([y], [y])

        def inner_func(x):
            out = pytensor.scan(lambda: test_ofg(y), n_steps=x, return_updates=False)
            return out

        out = pytensor.scan(
            inner_func, sequences=[pt.arange(1, 2)], return_updates=False
        )

        _ = pytensor.function([], test_ofg(y))

        out_fn = pytensor.function([], out)

        assert np.array_equal(out_fn(), [[1.0]])


class TestPushOutAddScan:
    """
    Test case for the `scan_push_out_add` optimization in the case where the `Scan`
    is used to compute the sum over the dot products between the corresponding
    elements of two list of matrices.

    FIXME: These aren't real tests; they simply confirm that a few
    graph that could be relevant to the push-out optimizations can be compiled
    and evaluated.  None of them confirm that a push-out optimization has been
    performed.

    FIXME: The rewrite is indeed broken, probably fro a long while, see FIXME details in the respective rewrite
    """

    def test_sum_dot(self):
        A = matrix("A")
        B = matrix("B")
        S = scan(
            lambda x1, x2, u: u + dot(x1, x2),
            sequences=[A.dimshuffle(0, 1, "x"), B.dimshuffle(0, "x", 1)],
            outputs_info=[pt.zeros_like(A)],
            return_updates=False,
        )
        # FIXME: This `s.owner.inputs[0][-1]` is a hack, users will never do that.
        #  They will do `s[-1]` which the rewrite fails to identify since it explicitly looks for a `scan_out[-1]`
        #  instead of `scan_out[1:][-1]` that the user would define by writing `s[-1]`
        #  It however, tests the only case the rewrite supports now
        f = function([A, B], S.owner.inputs[0][-1])
        has_scan = any(isinstance(node.op, Scan) for node in f.maker.fgraph.apply_nodes)
        # Rewrite is only triggered in fast_run mode
        assert has_scan if (config.mode == "FAST_COMPILE") else (not has_scan)

        rng = np.random.default_rng(utt.fetch_seed())
        vA = rng.uniform(size=(5, 5)).astype(config.floatX)
        vB = rng.uniform(size=(5, 5)).astype(config.floatX)
        utt.assert_allclose(f(vA, vB), np.dot(vA.T, vB))

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

        init = pt.alloc(np.asarray(0, dtype=config.floatX), batch_size, dim)

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
        h = pytensor.scan(
            rnn_step1,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init,
            name="fpass1",
            mode=opt_mode,
            return_updates=False,
        )
        cost = h[-1].sum()
        grad1 = grad(cost, [U, V, W])
        f_opt = pytensor.function(inputs=[x, ri, zi], outputs=grad1, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        h = pytensor.scan(
            rnn_step1,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init,
            name="fpass1",
            mode=no_opt_mode,
            return_updates=False,
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
        """Test the case where the initial value for the sitsot output is non-zero."""

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

        # Compile the function twice, once with the optimization and once without
        opt_mode = mode.including("scan")
        h = pytensor.scan(
            inner_fct,
            sequences=[input1, input2, input3],
            outputs_info=init,
            mode=opt_mode,
            return_updates=False,
        )
        output = h[-1]
        f_opt = pytensor.function([input1, input2, input3], output, mode=opt_mode)

        no_opt_mode = mode.excluding("scan_pushout_add")
        h = pytensor.scan(
            inner_fct,
            sequences=[input1, input2, input3],
            outputs_info=init,
            mode=no_opt_mode,
            return_updates=False,
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

        np.testing.assert_allclose(output_opt, output_no_opt)


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

    mode = get_mode("CVM").including("scan")  # .excluding("BlasOpt")

    o = scan(
        lambda hi, him1, W: (hi, dot(hi + him1, W)),
        outputs_info=[pt.zeros([h.shape[1]]), None],
        sequences=[h],
        non_sequences=[W],
        mode=mode,
        return_updates=False,
    )

    f = function([W, h], o, mode=mode)

    scan_nodes = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
    assert len(scan_nodes) == 1
    scan_op = scan_nodes[0].op
    assert not any(isinstance(n.op, Dot) for n in scan_op.fn.maker.fgraph.apply_nodes)
