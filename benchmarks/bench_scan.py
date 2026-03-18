import numpy as np

import pytensor.tensor as pt
from pytensor import config, function, grad, shared
from pytensor.compile.mode import Mode
from pytensor.gradient import hessian
from pytensor.scan.basic import scan
from pytensor.tensor.math import dot
from pytensor.tensor.type import (
    dmatrix,
    dscalar,
    dvector,
    fvector,
    iscalar,
    matrix,
    vector,
)


class CythonPerformance:
    """Benchmark scan with CVM linker (cython scan_perform)."""

    def setup(self):
        N = 200
        M = -1 / np.arange(1, 11).astype(config.floatX)
        r = np.arange(N * 10).astype(config.floatX).reshape(N, 10)
        s_r = pt.as_tensor_variable(r, dtype=config.floatX)
        s_y = scan(
            fn=lambda ri, rii, M: ri + M * rii,
            sequences=[s_r[1:]],
            non_sequences=[pt.as_tensor_variable(M, dtype=config.floatX)],
            outputs_info=s_r[0],
            mode=Mode(linker="cvm", optimizer="fast_run"),
            return_updates=False,
        )
        self.f_cvm = function([], s_y, mode="FAST_RUN")
        self.f_cvm.trust_input = True

    def time_cython_scan(self):
        self.f_cvm()


class Reordering:
    """Benchmark RNN scan with multiple inputs/outputs and reordering."""

    def setup(self):
        rng = np.random.default_rng(1234)
        vW_in2 = rng.uniform(-0.5, 0.5, size=(2,)).astype(config.floatX)
        vW = rng.uniform(-0.5, 0.5, size=(2, 2)).astype(config.floatX)
        vWout = rng.uniform(-0.5, 0.5, size=(2,)).astype(config.floatX)
        self.vW_in1 = rng.uniform(-0.5, 0.5, size=(2, 2)).astype(config.floatX)
        self.v_u1 = rng.uniform(-0.5, 0.5, size=(3, 2)).astype(config.floatX)
        self.v_u2 = rng.uniform(-0.5, 0.5, size=(3,)).astype(config.floatX)
        self.v_x0 = rng.uniform(-0.5, 0.5, size=(2,)).astype(config.floatX)
        self.v_y0 = rng.uniform(size=(3,)).astype(config.floatX)

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
                y_tm3 + 2,
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                y_tm1 + dot(x_tm1, W_out),
            ]

        outputs = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, None, x0, dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            return_updates=False,
        )
        self.f = function([u1, u2, x0, y0, W_in1], outputs, allow_input_downcast=True)

    def time_reordering(self):
        self.f(self.v_u1, self.v_u2, self.v_x0, self.v_y0, self.vW_in1)


class ScanAsTensorOnGradients:
    """Benchmark compilation of gradient through scan."""

    number = 1
    repeat = 5

    def setup(self):
        to_scan = dvector("to_scan")
        seq = dmatrix("seq")
        f1 = dscalar("f1")

        def scanStep(prev, seq, f1):
            return prev + f1 * seq

        scanned = scan(
            fn=scanStep,
            sequences=[seq],
            outputs_info=[to_scan],
            non_sequences=[f1],
            return_updates=False,
        )
        function(
            inputs=[to_scan, seq, f1],
            outputs=scanned,
            allow_input_downcast=True,
        )
        self.t_grad = grad(scanned.sum(), wrt=[to_scan, f1], consider_constant=[seq])
        self.inputs = [to_scan, seq, f1]

    def time_compile_grad(self):
        function(
            inputs=self.inputs,
            outputs=self.t_grad,
            allow_input_downcast=True,
        )


class HessianBugGradGradTwoScans:
    """Benchmark nested scan with hessian computation."""

    def setup(self):
        W = fvector(name="W")
        n_steps = iscalar(name="Nb_steps")

        def loss_outer(sum_outer, W):
            def loss_inner(sum_inner, W):
                return sum_inner + (W**2).sum()

            result_inner = scan(
                fn=loss_inner,
                outputs_info=pt.as_tensor_variable(np.asarray(0, dtype=np.float32)),
                non_sequences=[W],
                n_steps=1,
                return_updates=False,
            )
            return sum_outer + result_inner[-1]

        result_outer = scan(
            fn=loss_outer,
            outputs_info=pt.as_tensor_variable(np.asarray(0, dtype=np.float32)),
            non_sequences=[W],
            n_steps=n_steps,
            return_list=True,
            return_updates=False,
        )
        cost = result_outer[0][-1]
        H = hessian(cost, W)
        self.f = function([W, n_steps], H)

    def time_hessian_two_scans(self):
        self.f(np.ones((8,), dtype="float32"), 1)


class MultipleOutsTaps:
    """Benchmark complex RNN scan with multiple output taps."""

    def setup(self):
        l = 5
        rng = np.random.default_rng(1234)
        vW_in2 = rng.uniform(-2.0, 2.0, size=(2,)).astype(config.floatX)
        vW = rng.uniform(-2.0, 2.0, size=(2, 2)).astype(config.floatX)
        vWout = rng.uniform(-2.0, 2.0, size=(2,)).astype(config.floatX)
        self.vW_in1 = rng.uniform(-2.0, 2.0, size=(2, 2)).astype(config.floatX)
        self.v_u1 = rng.uniform(-2.0, 2.0, size=(l, 2)).astype(config.floatX)
        self.v_u2 = rng.uniform(-2.0, 2.0, size=(l + 2, 2)).astype(config.floatX)
        self.v_x0 = rng.uniform(-2.0, 2.0, size=(2,)).astype(config.floatX)
        self.v_y0 = rng.uniform(size=(3,)).astype(config.floatX)

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = matrix("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                dot(u1_t, W_in1) + (u2_t + u2_tm1 * u2_tp1) * W_in2 + dot(x_tm1, W),
                (y_tm1 + y_tm3) * dot(x_tm1, W_out),
                dot(u1_t, W_in1),
            ]

        outputs = scan(
            f_rnn_cmpl,
            [u1, dict(input=u2, taps=[-1, 0, 1])],
            [x0, dict(initial=y0, taps=[-1, -3]), None],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            return_updates=False,
        )
        self.f = function([u1, u2, x0, y0, W_in1], outputs, allow_input_downcast=True)

    def time_multiple_outs_taps(self):
        self.f(self.v_u1, self.v_u2, self.v_x0, self.v_y0, self.vW_in1)


class PregreedyOptimizer:
    """Benchmark scan with chained dot products (pregreedy optimizer path)."""

    def setup(self):
        W = pt.zeros((5, 4))
        bv = pt.zeros((5,))
        bh = pt.zeros((4,))
        v = matrix("v")

        (bv_t, bh_t) = scan(
            lambda _: [bv, bh],
            sequences=v,
            outputs_info=[None, None],
            return_updates=False,
        )
        chain = scan(
            lambda x: dot(dot(x, W) + bh_t, W.T) + bv_t,
            outputs_info=v,
            n_steps=2,
            return_updates=False,
        )
        self.chain_fn = function([v], chain)
        self.v_data = np.zeros((3, 5), dtype=config.floatX)

    def time_pregreedy_optimizer(self):
        self.chain_fn(self.v_data)


class SavememOpt:
    """Benchmark scan with save_mem optimization."""

    def setup(self):
        y0 = shared(np.ones((2, 10)))
        [_y1, y2] = scan(
            lambda y: [y, y],
            outputs_info=[dict(initial=y0, taps=[-2]), None],
            n_steps=5,
            return_updates=False,
        )
        self.fn = function([], y2.sum())

    def time_savemem_opt(self):
        self.fn()
