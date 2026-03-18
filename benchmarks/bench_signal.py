import numpy as np

import pytensor
from pytensor.gradient import grad
from pytensor.tensor.signal.conv import convolve1d
from pytensor.tensor.type import tensor


class Convolve1dGrad:
    """Benchmark gradient of convolve1d with different modes."""

    params = ["full", "valid"]
    param_names = ["convolve_mode"]

    def setup(self, convolve_mode):
        larger = tensor("larger", shape=(8, None))
        smaller = tensor("smaller", shape=(8, None))
        grad_wrt_smaller = grad(
            convolve1d(larger, smaller, mode=convolve_mode).sum(), wrt=smaller
        )
        self.fn = pytensor.function(
            [larger, smaller], grad_wrt_smaller, trust_input=True
        )

        rng = np.random.default_rng([119, convolve_mode == "full"])
        self.test_larger = rng.normal(size=(8, 1024)).astype(larger.type.dtype)
        self.test_smaller = rng.normal(size=(8, 16)).astype(smaller.type.dtype)

        # Warmup
        self.fn(self.test_larger, self.test_smaller)

    def time_convolve1d_grad(self, convolve_mode):
        self.fn(self.test_larger, self.test_smaller)
