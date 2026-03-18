import numpy as np

import pytensor
from pytensor import grad
from pytensor.tensor.math import log
from pytensor.tensor.nlinalg import diagonal
from pytensor.tensor.signal.conv import convolve1d
from pytensor.tensor.slinalg import cholesky, solve_triangular
from pytensor.tensor.type import dmatrix, tensor


class BatchedMVNormalLogpAndDlogp:
    """Benchmark batched multivariate normal log-probability and its gradient."""

    params = [
        [(), (1000,), (4, 1000)],
        [(), (1000,), (4, 1000)],
    ]
    param_names = ["mu_batch_shape", "cov_batch_shape"]

    def setup(self, mu_batch_shape, cov_batch_shape):
        rng = np.random.default_rng(sum(map(ord, "batched_mvnormal")))

        value_batch_shape = mu_batch_shape
        if len(cov_batch_shape) > len(mu_batch_shape):
            value_batch_shape = cov_batch_shape

        value = tensor("value", shape=(*value_batch_shape, 10))
        mu = tensor("mu", shape=(*mu_batch_shape, 10))
        cov = tensor("cov", shape=(*cov_batch_shape, 10, 10))

        self.test_values = [
            rng.normal(size=value.type.shape),
            rng.normal(size=mu.type.shape),
            np.eye(cov.type.shape[-1]) * np.abs(rng.normal(size=cov.type.shape)),
        ]

        chol_cov = cholesky(cov, lower=True, on_error="raise")
        delta_trans = solve_triangular(chol_cov, value - mu, b_ndim=1)
        quaddist = (delta_trans**2).sum(axis=-1)
        diag = diagonal(chol_cov, axis1=-2, axis2=-1)
        logdet = log(diag).sum(axis=-1)
        k = value.shape[-1]
        norm = -0.5 * k * (np.log(2 * np.pi))
        logp = norm - 0.5 * quaddist - logdet
        dlogp = grad(logp.sum(), wrt=[value, mu, cov])

        self.fn = pytensor.function([value, mu, cov], [logp, *dlogp])

    def time_batched_mvnormal_logp_and_dlogp(self, mu_batch_shape, cov_batch_shape):
        self.fn(*self.test_values)


class SmallBlockwisePerformance:
    """Benchmark small blockwise convolution."""

    def setup(self):
        a = dmatrix(shape=(7, 128))
        b = dmatrix(shape=(7, 20))
        out = convolve1d(a, b, mode="valid")
        self.fn = pytensor.function([a, b], out, trust_input=True)

        rng = np.random.default_rng(495)
        self.a_test = rng.normal(size=a.type.shape)
        self.b_test = rng.normal(size=b.type.shape)

    def time_small_blockwise(self):
        self.fn(self.a_test, self.b_test)
