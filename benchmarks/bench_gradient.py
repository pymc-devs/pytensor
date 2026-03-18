import numpy as np

import pytensor
from pytensor import function
from pytensor.gradient import jacobian
from pytensor.tensor.math import outer, sqrt
from pytensor.tensor.type import vector


class Jacobian:
    """Benchmark full Jacobian computation."""

    params = [True, False]
    param_names = ["vectorize"]

    def setup(self, vectorize):
        x = vector("x", shape=(3,))
        y = outer(x, x)
        jac_y = jacobian(y, x, vectorize=vectorize)
        self.fn = function([x], jac_y, trust_input=True)
        self.x_val = np.array([0, 1, 2], dtype=x.type.dtype)

        # Warmup
        self.fn(self.x_val)

    def time_jacobian(self, vectorize):
        self.fn(self.x_val)


class PartialJacobian:
    """Benchmark partial Jacobian computation on a large graph."""

    params = [True, False]
    param_names = ["vectorize"]

    def setup(self, vectorize):
        N = 1000
        rng = np.random.default_rng(2025)
        self.x_test = rng.random((N,))
        f_mat = rng.random((N, N))

        x = vector("x", dtype="float64")

        def f(x):
            return sqrt(f_mat @ x / N)

        full_jacobian = jacobian(f(x), x, vectorize=vectorize)
        partial_jacobian = full_jacobian[:5, :5]
        self.fn = pytensor.function([x], partial_jacobian, trust_input=True)

        # Warmup
        self.fn(self.x_test)

    def time_partial_jacobian(self, vectorize):
        self.fn(self.x_test)
