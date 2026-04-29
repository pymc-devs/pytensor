import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import function, scan, shared
from pytensor.compile.mode import get_default_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import ancestors
from pytensor.link.basic import JITLinker
from pytensor.scan.op import Scan
from pytensor.scan.utils import until
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.math import dot
from pytensor.tensor.shape import reshape
from pytensor.tensor.type import (
    iscalar,
    ivector,
    matrix,
    scalar,
    tensor3,
    vector,
)
from tests import unittest_tools as utt
from tests.scan.test_basic import asarrayX


class TestSaveMem:
    mode = (
        get_default_mode()
        .including("scan_save_mem", "scan_remove_unused")
        .excluding("scan_pushout")
    )

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

        outs = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, dict(initial=x0), dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            return_updates=False,
        )
        outputs = [outs[0][-1], outs[1][-1], outs[2][-1]]
        f4 = function(
            [u1, u2, x0, y0, W_in1],
            outputs,
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

        (_pytensor_dump, pytensor_x, pytensor_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

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
        [x1, x2, x3, x4, x5, x6, x7] = scan(
            f_rnn,
            u,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            return_updates=False,
        )

        f2 = function(
            [u, idx, jdx],
            [x1[:2], x2[4], x3[idx], x4[:idx], x5[-10], x6[-jdx], x7[:-jdx]],
            allow_input_downcast=True,
            mode=self.mode.excluding("scan_push_out_seq"),
        )
        # Check we actually have a Scan in the compiled function
        [scan_node] = [
            node for node in f2.maker.fgraph.toposort() if isinstance(node.op, Scan)
        ]

        # get random initial values
        rng = np.random.default_rng(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,)).astype(u.type.dtype)

        # Check the number of steps is actually reduced from 20
        n_steps = scan_node.inputs[0]
        n_steps_fn = pytensor.function(
            [u, idx, jdx], n_steps, accept_inplace=True, on_unused_input="ignore"
        )
        assert n_steps_fn(u=v_u, idx=3, jdx=15) == 11  # x5[const=-10] requires 11 steps
        assert n_steps_fn(u=v_u, idx=3, jdx=3) == 18  # x6[jdx=-3] requires 18 steps
        assert n_steps_fn(u=v_u, idx=16, jdx=15) == 17  # x3[idx=16] requires 17 steps
        assert n_steps_fn(u=v_u, idx=-5, jdx=15) == 16  # x3[idx=-5] requires 16 steps
        assert n_steps_fn(u=v_u, idx=19, jdx=15) == 20  # x3[idx=19] requires 20 steps

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5, tx6, tx7 = f2(v_u, 3, 15)

        utt.assert_allclose(tx1, v_u[:2] + 1.0)
        utt.assert_allclose(tx2, v_u[4] + 2.0)
        utt.assert_allclose(tx3, v_u[3] + 3.0)
        utt.assert_allclose(tx4, v_u[:3] + 4.0)
        utt.assert_allclose(tx5, v_u[-10] + 5.0)
        utt.assert_allclose(tx6, v_u[-15] + 6.0)
        utt.assert_allclose(tx7, v_u[:-15] + 7.0)

    def test_save_mem_reduced_number_of_steps_constant(self):
        x0 = pt.scalar("x0")
        xs = scan(
            lambda xtm1: xtm1 + 1, outputs_info=[x0], n_steps=10, return_updates=False
        )

        fn = function([x0], xs[:5], mode=self.mode)
        [scan_node] = [
            node for node in fn.maker.fgraph.toposort() if isinstance(node.op, Scan)
        ]
        n_steps = scan_node.inputs[0]
        assert isinstance(n_steps, Constant) and n_steps.data == 5

        np.testing.assert_allclose(fn(0), np.arange(1, 11)[:5])

    def test_save_mem_cannot_reduce_constant_number_of_steps(self):
        x0 = pt.scalar("x0")
        [xs, ys] = scan(
            lambda xtm1, ytm1: (xtm1 + 1, ytm1 - 1),
            outputs_info=[x0, x0],
            n_steps=10,
            return_updates=False,
        )

        # Because of ys[-1] we need all the steps!
        fn = function([x0], [xs[:5], ys[-1]], mode=self.mode)
        [scan_node] = [
            node for node in fn.maker.fgraph.toposort() if isinstance(node.op, Scan)
        ]
        n_steps = scan_node.inputs[0]
        assert isinstance(n_steps, Constant) and n_steps.data == 10

        res_x, res_y = fn(0)
        np.testing.assert_allclose(
            res_x,
            np.arange(1, 11)[:5],
        )
        np.testing.assert_allclose(
            res_y,
            -np.arange(1, 11)[-1],
        )

    def test_save_mem_store_steps(self):
        def step(u_t, x1_tm1, x1_tm3, x2_tm1, x3tm2, x3_tm1, x4_tm1):
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
        [x1, x2, x3, x4, x5, _x6, _x7] = scan(
            step,
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
            return_updates=False,
        )

        f = function(
            [u, x10, x20, x30, x40],
            [x1[-7], x2[-3:-1], x3[-6:], x4[-1], x5[-1]],
            allow_input_downcast=True,
            mode=self.mode,
        )

        # get random initial values
        rng = np.random.default_rng(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,))

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5 = f(v_u, [0, 0], 0, [0, 0], 0)
        rtol = 1e-7 if config.floatX == "float64" else 1e-6
        np.testing.assert_allclose(tx1, v_u[-7] + 1.0, rtol=rtol)
        np.testing.assert_allclose(tx2, v_u[-3:-1] + 2.0, rtol=rtol)
        np.testing.assert_allclose(tx3, v_u[-6:] + 3.0, rtol=rtol)
        np.testing.assert_allclose(tx4, v_u[-1] + 4.0, rtol=rtol)
        np.testing.assert_allclose(tx5, v_u[-1] + 5.0, rtol=rtol)

        # Confirm reduction in buffer sizes
        [scan_node] = [
            node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
        ]
        # x6 and x7 are dropped because they are not used
        [_n_steps, _seq, x4_buffer, x5_buffer, x1_len, x2_len, x3_len] = (
            scan_node.inputs
        )
        [x4_underlying_alloc] = [
            var
            for var in ancestors([x4_buffer])
            if var.owner and isinstance(var.owner.op, AllocEmpty)
        ]
        [x5_underlying_alloc] = [
            var
            for var in ancestors([x5_buffer])
            if var.owner and isinstance(var.owner.op, AllocEmpty)
        ]
        buffer_lengths = pytensor.function(
            [u, x10, x20, x30, x40],
            [
                x1_len,
                x2_len,
                x3_len,
                x4_underlying_alloc.shape[0],
                x5_underlying_alloc.shape[0],
            ],
            accept_inplace=True,
            on_unused_input="ignore",
            allow_input_downcast=True,
        )(v_u, [0, 0], 0, [0, 0], 0)
        # ScanSaveMem keeps +1 entries to handle taps with preallocated outputs, unless we are using a JITLinker
        maybe_one = 0 if isinstance(f.maker.linker, JITLinker) else 1

        assert [int(i) for i in buffer_lengths] == [
            7,  # entry -7 of a map variable is kept, we need at least that many
            3,  # entries [-3, -2] of a map variable are kept, we need at least 3
            6,  # last six entries of a map variable are kept
            2 + maybe_one,  # last entry of a double tap variable is kept
            1 + maybe_one,  # last entry of a single tap variable is kept
        ]

    def test_savemem_does_not_duplicate_number_of_scan_nodes(self):
        var = pt.ones(())
        values = scan(
            lambda x: ([x], (), until(x)),
            outputs_info=[var],
            n_steps=2,
            return_updates=False,
        )

        tmp_fn = function([var], values, mode=self.mode)
        scan_nodes = [
            x for x in tmp_fn.maker.fgraph.toposort() if isinstance(x.op, Scan)
        ]
        assert len(scan_nodes) == 1

    def test_savemem_opt_0_step(self):
        """
        Test a case where the savemem optimization has the opportunity to
        lower the number of steps of a Scan to 0.
        """

        def inner_scan_step(x_t_t, h_tm1, w):
            return dot(h_tm1, w) + x_t_t

        def outer_scan_step(x_t, w):
            h = scan(
                inner_scan_step,
                sequences=[x_t[1:]],
                outputs_info=[x_t[0]],
                non_sequences=[w],
                strict=True,
                name="the_inner_scan",
                return_updates=False,
            )
            return h

        def get_outputs(x, w):
            features = scan(
                outer_scan_step,
                sequences=[x],
                non_sequences=[w],
                strict=True,
                name="the_outer_scan",
                return_updates=False,
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

    def test_savemem_0_steps_does_not_point_to_unitialized_memory(self):
        # Regression test for https://github.com/pymc-devs/pytensor/issues/1878

        n = pt.tensor("n", shape=(), dtype=int)
        init_state = pt.tensor("init_state", shape=(3,))
        buffer_withot_init = pytensor.scan(
            fn=lambda xtm1: xtm1 * 2,
            outputs_info=[init_state],
            n_steps=n,
            return_updates=False,
        )
        # Access the last state of the Scan output buffer (which includes the initial state)
        # It should never point to unitialized memory
        full_buffer = buffer_withot_init.owner.inputs[0]
        buffer_last_entry = full_buffer[-1]

        fn = pytensor.function([init_state, n], buffer_last_entry)
        init_state_val = np.ones((3,))
        np.testing.assert_allclose(fn(init_state=init_state_val, n=0), init_state_val)
        np.testing.assert_allclose(
            fn(init_state=init_state_val, n=1), init_state_val * 2
        )
        np.testing.assert_allclose(
            fn(init_state=init_state_val, n=2), init_state_val * 4
        )

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
        output = scan(
            f_pow2,
            [],
            state,
            [],
            n_steps=n_steps,
            truncate_gradient=-1,
            go_backwards=False,
            return_updates=False,
        )
        nw_shape = ivector("nw_shape")
        # Note that the output is reshaped to 3 dimensional tensor, and
        my_f = function(
            [state, n_steps, nw_shape],
            [reshape(output, nw_shape, ndim=3)[:-2], output[:-4]],
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

        ys = pytensor.scan(
            # Fibonacci Sequence
            lambda xtm2, xtm1: (xtm1 + xtm2, {}, until(xtm1 >= 34)),
            outputs_info=[{"initial": x0, "taps": [-2, -1]}],
            n_steps=n_steps,
            return_updates=False,
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
        # extra entry to prevent aliasing between the inputs and outputs
        # of the pre-allocation mechanism. JIT linkers don't use pre-allocation
        # so the buffer is one element smaller.
        [scan_node] = (n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, ys_trace = scan_node.inputs
        debug_fn = pytensor.function(
            [n_steps, x0], ys_trace.shape[0], accept_inplace=True
        )
        expected_size = 2 if isinstance(f.maker.linker, JITLinker) else 3
        assert debug_fn(n_steps=1000, x0=[1, 1]) == expected_size

    def test_while_scan_map(self):
        xs = vector("xs")
        ys = pytensor.scan(
            lambda x: (x + 1, {}, until(x + 1 >= 10)),
            outputs_info=[None],
            sequences=[xs],
            return_updates=False,
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
        [ys, zs] = pytensor.scan(
            lambda s, xtm1: ((xtm1 + 1, xtm1 + 1 + s), {}, until(xtm1 >= 99)),
            sequences=[seq],
            outputs_info=[x0, None],
            n_steps=n_steps,
            return_updates=False,
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
        with pytest.raises((AssertionError, IndexError)):
            f(x0=0, seq=test_seq, n_steps=0)

        # Evaluate the shape of ys_trace and len_zs to confirm the rewrite worked correctly.
        # JIT linkers don't use pre-allocation so the buffer is one element smaller.
        [scan_node] = (n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, _, ys_trace, len_zs = scan_node.inputs
        debug_fn = pytensor.function(
            [x0, n_steps], [ys_trace.shape[0], len_zs], accept_inplace=True
        )
        stored_ys_steps, stored_zs_steps = debug_fn(x0=0, n_steps=200)
        expected_y_steps = 1 if isinstance(f.maker.linker, JITLinker) else 2
        assert stored_ys_steps == expected_y_steps
        assert stored_zs_steps == 1

    def test_while_scan_untraced_sit_sot_shape(self):
        # Regression: ``Scan.infer_shape`` for while-scans used to call
        # ``Shape_i(0)`` on every output, including untraced sit_sot ones
        # that are 0-d (no leading time dimension), which crashed
        # compilation under ``on_shape_error=raise``.
        x0 = scalar("x0")
        n_steps = scalar("n_steps", dtype="int64")
        [ys, zs] = pytensor.scan(
            lambda xtm1: ((xtm1 + 1, xtm1 * 2), {}, until(xtm1 >= 10)),
            outputs_info=[x0, None],
            n_steps=n_steps,
            return_updates=False,
        )
        f = pytensor.function([x0, n_steps], [ys[-1], zs[-1]])
        np.testing.assert_allclose(f(x0=0, n_steps=100), [11, 20])

    @pytest.mark.parametrize("val_ndim", (0, 1))
    @pytest.mark.parametrize("keep_beginning", (False, True))
    def test_broadcasted_init(self, keep_beginning, val_ndim):
        # Regression test when the original value is a broadcasted alloc
        # The scan save mem rewrite used to wrongly slice on the unbroadcasted value
        val_shape = (1,) * val_ndim
        val = pt.tensor("val", shape=val_shape)
        val_test = np.zeros(val_shape, dtype=val.dtype)

        init = pt.full((2,), val)
        ys = pytensor.scan(
            fn=lambda *args: pt.add(*args),
            outputs_info=[{"initial": init, "taps": (-2, -1)}],
            n_steps=100,
            return_updates=False,
        )

        out = ys[:-50] if keep_beginning else ys[-50:]
        fn = pytensor.function([val], out, mode=self.mode)
        assert fn(val_test).shape == (50,)

        # Check that rewrite worked
        [scan_node] = (n for n in fn.maker.fgraph.apply_nodes if isinstance(n.op, Scan))
        _, ys_trace = scan_node.inputs
        buffer_size_fn = pytensor.function(
            [val], ys_trace.shape[0], accept_inplace=True
        )
        assert buffer_size_fn(val_test) == 52 if keep_beginning else 50


def test_scan_sit_sot_to_untraced():
    """Test sit_sot to untraced_sit_sot conversion.

    4 outputs: xs (sit_sot, all values used → stays), ys (sit_sot, only last
    → converted), ws (nit_sot, unaffected), rs (sit_sot, required orphan
    → converted). Result: 1 sit_sot, 1 nit_sot, 2 untraced_sit_sot.
    """
    mode = (
        get_default_mode()
        .excluding("scan_save_mem")
        .including("scan_save_mem_no_prealloc", "scan_sit_sot_to_untraced")
    )

    x0 = vector("x0")
    y0 = vector("y0")
    r0 = vector("r0")

    def step(x_tm1, y_tm1, r_tm1):
        r = 1.0 - x_tm1
        x = x_tm1 + 0.5 * r + 0.3 * r_tm1
        y = y_tm1 + 1
        w = x_tm1 * 2
        return x, y, w, r

    [xs, ys, ws, _rs] = scan(
        step, outputs_info=[x0, y0, None, r0], n_steps=10, return_updates=False
    )
    # xs: all values used (stays sit_sot)
    # ys[-1]: only last value (converted)
    # ws[-1]: nit_sot (unaffected)
    # rs: never used externally, required orphan (converted)
    f = function([x0, y0, r0], [xs, ys[-1], ws[-1]], mode=mode)

    [scan_node] = [n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, Scan)]
    assert scan_node.op.info.n_sit_sot == 1
    assert scan_node.op.info.n_nit_sot == 1
    assert scan_node.op.info.n_untraced_sit_sot == 2

    x0_val = np.zeros(3, dtype=config.floatX)
    y0_val = np.zeros(3, dtype=config.floatX)
    r0_val = np.zeros(3, dtype=config.floatX)
    res_xs, res_y, res_w = f(x0_val, y0_val, r0_val)
    np.testing.assert_allclose(res_y, y0_val + 10)
    assert res_xs.shape == (10, 3)
    assert np.all(np.isfinite(res_xs))
    assert np.isfinite(res_w).all()
