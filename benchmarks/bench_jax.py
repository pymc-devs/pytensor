import numpy as np

import pytensor.tensor as pt
from pytensor import function, grad
from pytensor.scan.basic import scan


class JaxLogsumexp:
    """Benchmark JAX logsumexp-like computation."""

    params = [[(10, 10), (1000, 1000)], [0, 1]]
    param_names = ["size", "axis"]

    def setup(self, size, axis):
        try:
            import jax  # noqa: F401
        except ImportError:
            raise NotImplementedError("JAX not available")

        X = pt.matrix("X")
        X_max = pt.max(X, axis=axis, keepdims=True)
        X_max = pt.switch(pt.isinf(X_max), 0, X_max)
        X_lse = pt.log(pt.sum(pt.exp(X - X_max), axis=axis, keepdims=True)) + X_max

        rng = np.random.default_rng(23920)
        self.X_val = rng.normal(size=size)

        self.fn = function([X], X_lse, mode="JAX")
        self.fn(self.X_val)  # JIT warmup

    def time_logsumexp(self, size, axis):
        self.fn(self.X_val)


class JaxScan:
    """Benchmark JAX scan with forward and backward passes."""

    params = [["forward", "backward", "both"]]
    param_names = ["mode"]

    def setup(self, mode):
        try:
            import jax  # noqa: F401
        except ImportError:
            raise NotImplementedError("JAX not available")

        x0 = pt.vector("x0", shape=(10,), dtype="float64")
        W = pt.matrix("W", shape=(10, 10), dtype="float64")

        def step(x_prev, W):
            return pt.tanh(pt.dot(x_prev, W))

        result = scan(
            fn=step,
            outputs_info=[x0],
            non_sequences=[W],
            n_steps=50,
            return_updates=False,
        )
        loss = result[-1].sum()
        dloss = grad(loss, wrt=[x0, W])

        if mode == "forward":
            self.fn = function([x0, W], result, mode="JAX")
        elif mode == "backward":
            self.fn = function([x0, W], dloss, mode="JAX")
        else:  # both
            self.fn = function([x0, W], [loss, *dloss], mode="JAX")

        rng = np.random.default_rng(42)
        self.x0_val = rng.normal(size=(10,))
        self.W_val = rng.normal(size=(10, 10)) * 0.1
        self.fn(self.x0_val, self.W_val)  # JIT warmup

    def time_scan(self, mode):
        self.fn(self.x0_val, self.W_val)
