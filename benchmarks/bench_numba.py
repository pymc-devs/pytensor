import itertools

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.compile.io import In, Out
from pytensor.gradient import grad
from pytensor.scan.basic import scan
from pytensor.tensor.signal.conv import convolve1d
from pytensor.tensor.slinalg import cholesky
from pytensor.tensor.type import tensor, tensor3

from .common import create_radon_model


def _check_numba():
    try:
        import numba  # noqa: F401
    except ImportError:
        raise NotImplementedError("Numba not available")


class NumbaRadonCompile:
    """Benchmark compilation and single call of the radon model with Numba."""

    params = [True, False]
    param_names = ["cache"]
    number = 1
    repeat = 5

    def setup(self, cache):
        _check_numba()
        self.joined_inputs, [self.model_logp, self.model_dlogp] = create_radon_model()
        rng = np.random.default_rng(1)
        self.x = rng.normal(size=self.joined_inputs.type.shape).astype(config.floatX)
        self.cache = cache

    def time_compile_and_call(self, cache):
        with config.change_flags(numba__cache=cache):
            fn = function(
                [self.joined_inputs],
                [self.model_logp, self.model_dlogp],
                mode="NUMBA",
                trust_input=True,
            )
            fn(self.x)


class NumbaRadonCall:
    """Benchmark calling a pre-compiled radon model function with Numba."""

    params = [True, False]
    param_names = ["cache"]

    def setup(self, cache):
        _check_numba()
        joined_inputs, [model_logp, model_dlogp] = create_radon_model()
        rng = np.random.default_rng(1)
        self.x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)
        with config.change_flags(numba__cache=cache):
            self.fn = function(
                [joined_inputs],
                [model_logp, model_dlogp],
                mode="NUMBA",
                trust_input=True,
            )
        # Warmup
        self.fn(self.x)

    def time_call(self, cache):
        self.fn(self.x)


class NumbaElemwiseSpeed:
    """Benchmark elemwise expression evaluation with Numba JIT."""

    def setup(self):
        _check_numba()
        x = pt.dmatrix("y")
        y = pt.dvector("z")
        out = pt.exp(2 * x * y + y)
        rng = np.random.default_rng(42)
        self.x_val = rng.normal(size=(200, 500))
        self.y_val = rng.normal(size=500)
        func = function([x, y], out, mode="NUMBA")
        self.func = func.vm.jit_fn
        # Warmup
        self.func(self.x_val, self.y_val)

    def time_elemwise(self):
        self.func(self.x_val, self.y_val)


class NumbaFusedElemwise:
    """Benchmark fused elemwise logp + gradient computation with Numba."""

    def setup(self):
        _check_numba()
        rng = np.random.default_rng(123)
        size = 100_000
        x = pytensor.shared(rng.normal(size=size), name="x")
        mu = pytensor.shared(rng.normal(size=size), name="mu")
        logp = -((x - mu) ** 2) / 2
        grad_logp = grad(logp.sum(), x)
        self.func = pytensor.function([], [logp, grad_logp], mode="NUMBA")
        # JIT compile
        self.func()

    def time_fused_elemwise(self):
        self.func()


class NumbaLogsumexp:
    """Benchmark logsumexp computation with Numba."""

    params = [
        [(10, 10), (1000, 1000), (10000, 10000)],
        [0, 1],
    ]
    param_names = ["size", "axis"]

    def setup(self, size, axis):
        _check_numba()
        X = pt.matrix("X")
        X_max = pt.max(X, axis=axis, keepdims=True)
        X_max = pt.switch(pt.isinf(X_max), 0, X_max)
        X_lse = pt.log(pt.sum(pt.exp(X - X_max), axis=axis, keepdims=True)) + X_max
        rng = np.random.default_rng(23920)
        self.X_val = rng.normal(size=size)
        self.fn = pytensor.function([X], X_lse, mode="NUMBA")
        # JIT compile
        self.fn(self.X_val)

    def time_logsumexp(self, size, axis):
        self.fn(self.X_val)


class NumbaCareduce:
    """Benchmark CAReduce (sum) over various axes and memory layouts with Numba."""

    params = [
        [0, 1, 2, (0, 1), (0, 2), (1, 2), None],
        [True, False],
    ]
    param_names = ["axis", "c_contiguous"]

    def setup(self, axis, c_contiguous):
        _check_numba()
        N = 256
        x_test = np.random.uniform(size=(N, N, N))
        transpose_axis = (0, 1, 2) if c_contiguous else (2, 0, 1)
        x = pytensor.shared(x_test, name="x", shape=x_test.shape)
        out = x.transpose(transpose_axis).sum(axis=axis)
        self.fn = pytensor.function([], out, mode="NUMBA")
        # JIT compile
        self.fn()

    def time_careduce(self, axis, c_contiguous):
        self.fn()


class NumbaDimshuffle:
    """Benchmark DimShuffle operations with Numba."""

    params = [True, False]
    param_names = ["c_contiguous"]

    def setup(self, c_contiguous):
        _check_numba()
        x = tensor3("x")
        if c_contiguous:
            self.x_val = np.random.random((2, 3, 4)).astype(config.floatX)
        else:
            self.x_val = np.random.random((200, 300, 400)).transpose(1, 2, 0)

        ys = [x.transpose(t) for t in itertools.permutations((0, 1, 2))]
        ys += [x[None], x[:, None], x[:, :, None], x[:, :, :, None]]

        self.fn = pytensor.function(
            [In(x, borrow=True)],
            [Out(y, borrow=True) for y in ys],
            mode="NUMBA",
        )
        self.fn.trust_input = True
        # Warmup / JIT compile
        self.fn(self.x_val)

    def time_dimshuffle(self, c_contiguous):
        self.fn(self.x_val)


class NumbaMatVecDot:
    """Benchmark matrix-vector dot product with Numba."""

    params = ["float64", "float32", "mixed"]
    param_names = ["dtype"]

    def setup(self, dtype):
        _check_numba()
        A_dtype = "float64" if dtype == "mixed" else dtype
        x_dtype = "float32" if dtype == "mixed" else dtype
        A = tensor("A", shape=(512, 512), dtype=A_dtype)
        x = tensor("x", shape=(512,), dtype=x_dtype)
        out = pt.dot(A, x)
        self.fn = function([A, x], out, mode="NUMBA", trust_input=True)
        rng = np.random.default_rng(948)
        self.A_test = rng.standard_normal(size=A.type.shape).astype(A.type.dtype)
        self.x_test = rng.standard_normal(size=x.type.shape).astype(x.type.dtype)
        # Warmup
        self.fn(self.A_test, self.x_test)

    def time_matvec_dot(self, dtype):
        self.fn(self.A_test, self.x_test)


class NumbaFunctionOverhead:
    """Benchmark function call overhead with different calling modes in Numba."""

    params = ["default", "trust_input", "direct"]
    param_names = ["mode"]

    def setup(self, mode):
        _check_numba()
        x = pt.vector("x")
        out = pt.exp(x)
        fn = function([x], out, mode="NUMBA")
        if mode == "trust_input":
            fn.trust_input = True
            self.fn = fn
        elif mode == "direct":
            self.fn = fn.vm.jit_fn
        else:
            self.fn = fn
        self.test_x = np.zeros(1000)
        # Warmup
        self.fn(self.test_x)

    def time_function_overhead(self, mode):
        self.fn(self.test_x)


class NumbaScanSEIR:
    """Benchmark SEIR epidemiological model scan with Numba."""

    def setup(self):
        _check_numba()

        def binomln(n, k):
            return pt.exp(n + 1) - pt.exp(k + 1) - pt.exp(n - k + 1)

        def binom_log_prob(n, p, value):
            return binomln(n, value) + value * pt.exp(p) + (n - value) * pt.exp(1 - p)

        pt_C = pt.ivector("C_t")
        pt_D = pt.ivector("D_t")
        st0 = pt.lscalar("s_t0")
        et0 = pt.lscalar("e_t0")
        it0 = pt.lscalar("i_t0")
        logp_c = pt.scalar("logp_c")
        logp_d = pt.scalar("logp_d")
        beta = pt.scalar("beta")
        gamma = pt.scalar("gamma")
        delta = pt.scalar("delta")

        def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
            bt0 = (st0 * beta).astype(st0.dtype)
            logp_c1 = binom_log_prob(et0, gamma, ct0).astype(logp_c.dtype)
            logp_d1 = binom_log_prob(it0, delta, dt0).astype(logp_d.dtype)
            return (
                st0 - bt0,
                et0 + bt0 - ct0,
                it0 + ct0 - dt0,
                logp_c1,
                logp_d1,
            )

        (st, et, it, logp_c_all, logp_d_all) = scan(
            fn=seir_one_step,
            sequences=[pt_C, pt_D],
            outputs_info=[st0, et0, it0, logp_c, logp_d],
            non_sequences=[beta, gamma, delta],
            return_updates=False,
        )
        out = [st, et, it, logp_c_all, logp_d_all]
        self.fn = function(
            [pt_C, pt_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
            out,
            mode="NUMBA",
        )
        self.test_input_vals = [
            np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.int32),
            np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.int32),
            100,
            50,
            25,
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.277792),
            np.float64(0.135330),
            np.float64(0.108753),
        ]
        # Warmup / JIT compile
        self.fn(*self.test_input_vals)

    def time_scan_seir(self):
        self.fn(*self.test_input_vals)


class NumbaVectorTaps:
    """Benchmark scan with multiple input/output taps using Numba."""

    def setup(self):
        _check_numba()
        from pytensor.tensor.type import scalar, vector

        n_steps = 1000
        seq1 = vector("seq1", dtype="float64", shape=(n_steps,))
        seq2 = vector("seq2", dtype="float64", shape=(n_steps,))
        mitsot_init = vector("mitsot_init", dtype="float64", shape=(2,))
        sitsot_init = scalar("sitsot_init", dtype="float64")

        def step(seq1, seq2, mitsot1, mitsot2, sitsot1):
            mitsot3 = (mitsot1 + seq2 + mitsot2 + seq1) / np.sqrt(4)
            sitsot2 = (sitsot1 + mitsot3) / np.sqrt(2)
            return mitsot3, sitsot2

        outs = scan(
            fn=step,
            sequences=[seq1, seq2],
            outputs_info=[
                dict(initial=mitsot_init, taps=[-2, -1]),
                dict(initial=sitsot_init, taps=[-1]),
            ],
            return_updates=False,
        )
        rng = np.random.default_rng(474)
        self.test_vals = [
            rng.normal(size=n_steps),
            rng.normal(size=n_steps),
            rng.normal(size=(2,)),
            rng.normal(),
        ]
        self.fn = pytensor.function(
            [seq1, seq2, mitsot_init, sitsot_init], outs, mode="NUMBA"
        )
        # Warmup
        self.fn(*self.test_vals)

    def time_vector_taps(self):
        self.fn(*self.test_vals)


class NumbaBlockwise:
    """Benchmark blockwise Cholesky decomposition with Numba."""

    def setup(self):
        _check_numba()
        x = tensor(shape=(5, 3, 3))
        out = cholesky(x)
        self.fn = function([x], out, mode="NUMBA")
        self.x_test = np.eye(3) * np.arange(1, 6)[:, None, None]
        # JIT compile
        self.fn(self.x_test)

    def time_blockwise_cholesky(self):
        self.fn(self.x_test)


class NumbaConvolve1d:
    """Benchmark 1D convolution with Numba."""

    params = [
        [True, False],
        ["full", "valid"],
    ]
    param_names = ["batch", "mode"]

    def setup(self, batch, mode):
        _check_numba()
        x = tensor(shape=(7, 183) if batch else (183,))
        y = tensor(shape=(7, 6) if batch else (6,))
        out = convolve1d(x, y, mode=mode)
        self.fn = function([x, y], out, mode="NUMBA", trust_input=True)
        rng = np.random.default_rng()
        self.x_test = rng.normal(size=x.type.shape).astype(x.type.dtype)
        self.y_test = rng.normal(size=y.type.shape).astype(y.type.dtype)
        # Warmup
        self.fn(self.x_test, self.y_test)

    def time_convolve1d(self, batch, mode):
        self.fn(self.x_test, self.y_test)


class NumbaConvolve1dGrad:
    """Benchmark gradient of convolve1d with Numba."""

    params = ["full", "valid"]
    param_names = ["convolve_mode"]

    def setup(self, convolve_mode):
        _check_numba()
        larger = tensor("larger", shape=(8, None))
        smaller = tensor("smaller", shape=(8, None))
        grad_wrt_smaller = grad(
            convolve1d(larger, smaller, mode=convolve_mode).sum(), wrt=smaller
        )
        self.fn = pytensor.function(
            [larger, smaller], grad_wrt_smaller, mode="NUMBA", trust_input=True
        )

        rng = np.random.default_rng([119, convolve_mode == "full"])
        self.test_larger = rng.normal(size=(8, 1024)).astype(larger.type.dtype)
        self.test_smaller = rng.normal(size=(8, 16)).astype(smaller.type.dtype)

        # Warmup
        self.fn(self.test_larger, self.test_smaller)

    def time_convolve1d_grad(self, convolve_mode):
        self.fn(self.test_larger, self.test_smaller)
