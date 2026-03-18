import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.gradient import grad
from pytensor.tensor.math import gammaincc
from pytensor.tensor.type import vector


class GammainccdkGrad:
    """Benchmark gradient of gammaincc with respect to k."""

    def setup(self):
        k = vector("k")
        x = vector("x")
        out = gammaincc(k, x)
        self.grad_fn = pytensor.function(
            [k, x], grad(out.sum(), wrt=[k]), mode="FAST_RUN", trust_input=True
        )
        self.vals = [
            np.full((1000,), 3.2, dtype=k.dtype),
            np.full((1000,), 0.01, dtype=x.dtype),
        ]

        # Warmup
        self.grad_fn(*self.vals)

    def time_gammaincc_grad(self):
        self.grad_fn(*self.vals)


class Hyp2F1Grad:
    """Benchmark gradient of hyp2f1 with few and many iterations."""

    params = [["few_iters", "many_iters"], ["a", "all"]]
    param_names = ["case", "wrt"]

    _cases = {
        "few_iters": (10.0, -2.0, 7.0, 0.7),
        "many_iters": (3.5, 1.1, 2.0, 0.3),
    }

    def setup(self, case, wrt):
        a1, a2, b1, z = pt.scalars("a1", "a2", "b1", "z")
        hyp2f1_out = pt.hyp2f1(a1, a2, b1, z)
        if wrt == "a":
            hyp2f1_grad = pt.grad(hyp2f1_out, wrt=a1)
        else:
            hyp2f1_grad = pt.grad(hyp2f1_out, wrt=[a1, a2, b1, z])
        self.f_grad = pytensor.function([a1, a2, b1, z], hyp2f1_grad, trust_input=True)

        test_a1, test_a2, test_b1, test_z = self._cases[case]
        self.test_vals = [
            np.float64(test_a1),
            np.float64(test_a2),
            np.float64(test_b1),
            np.float64(test_z),
        ]

        # Warmup
        self.f_grad(*self.test_vals)

    def time_hyp2f1_grad(self, case, wrt):
        self.f_grad(*self.test_vals)
