import itertools

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import In, Out, config
from pytensor.gradient import grad
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.math import add, log
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.elemwise import FusionOptimizer
from pytensor.tensor.type import dscalar, dvector, tensor3


class DimShuffle:
    """Benchmark DimShuffle operations with various transpositions and expansions."""

    params = [True, False]
    param_names = ["c_contiguous"]

    def setup(self, c_contiguous):
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
            mode="FAST_RUN",
        )
        self.fn.trust_input = True
        # Warmup / JIT compile
        self.fn(self.x_val)

    def time_dimshuffle(self, c_contiguous):
        self.fn(self.x_val)


class CAReduce:
    """Benchmark CAReduce (sum) over various axes and memory layouts."""

    params = [
        [0, 1, 2, (0, 1), (0, 2), (1, 2), None],
        [True, False],
    ]
    param_names = ["axis", "c_contiguous"]

    def setup(self, axis, c_contiguous):
        N = 256
        x_test = np.random.uniform(size=(N, N, N))
        transpose_axis = (0, 1, 2) if c_contiguous else (2, 0, 1)
        x = pytensor.shared(x_test, name="x", shape=x_test.shape)
        out = x.transpose(transpose_axis).sum(axis=axis)
        self.fn = pytensor.function([], out, mode="FAST_RUN")

    def time_careduce(self, axis, c_contiguous):
        self.fn()


class ElemwiseEval:
    """Benchmark evaluation of a fused elemwise logp + gradient computation."""

    def setup(self):
        rng = np.random.default_rng(123)
        size = 100_000
        x = pytensor.shared(rng.normal(size=size), name="x")
        mu = pytensor.shared(rng.normal(size=size), name="mu")
        logp = -((x - mu) ** 2) / 2
        grad_logp = grad(logp.sum(), x)
        self.func = pytensor.function([], [logp, grad_logp], mode="FAST_RUN")

    def time_eval(self):
        self.func()


class FusionRewrite:
    """Benchmark the FusionOptimizer rewrite pass on different graph shapes."""

    params = [
        ["deep_small_kernels", "large_fuseable_graph"],
        [20, 25],
    ]
    param_names = ["graph_fn", "n"]
    number = 5
    repeat = 7

    @staticmethod
    def large_fuseable_graph(n):
        factors = []
        sd = dscalar()
        means = dvector()
        cst_05 = pt.constant(0.5)
        cst_m05 = pt.constant(-0.5)
        cst_2 = pt.constant(2)
        cst_m2 = pt.constant(-2)
        ones = pt.constant(np.ones(10))
        for i in range(n):
            f = cst_m05 * sd**cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * log(
                cst_05 * (sd**cst_m2) / np.pi
            )
            factors.append(pt_sum(f))
        logp = add(*factors)
        vars = [sd, means]
        dlogp = [pytensor.grad(logp, v) for v in vars]
        return vars, dlogp

    @staticmethod
    def deep_small_kernels(n):
        x = pt.matrix("x")
        out = x
        for _ in range(n):
            out = pt.sin(out.T) + pt.cos(out)
        return [x], [out]

    def setup(self, graph_fn, n):
        # Only run matching (graph_fn, n) combinations
        valid = {
            "deep_small_kernels": 20,
            "large_fuseable_graph": 25,
        }
        if valid.get(graph_fn) != n:
            raise NotImplementedError("Skip non-matching parameter combination")

        builder = getattr(self, graph_fn)
        inps, outs = builder(n)
        self.fg = FunctionGraph(inps, outs)
        self.opt = FusionOptimizer()

    def time_rewrite(self, graph_fn, n):
        fg_clone = self.fg.clone()
        self.opt.apply(fg_clone)
