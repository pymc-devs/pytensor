import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import diagonal, grad, log, tensor
from pytensor.tensor.linalg import cholesky, solve_triangular


def _test_blockwise_cholesky_benchmark(mode, benchmark):
    from pytensor.tensor.blockwise import Blockwise

    x = tensor(shape=(5, 3, 3))
    out = cholesky(x)
    assert isinstance(out.owner.op, Blockwise)

    fn = function([x], out, mode=mode, trust_input=True)
    x_test = np.eye(3) * np.arange(1, 6)[:, None, None]
    fn(x_test)  # JIT compile
    benchmark(fn, x_test)


def test_blockwise_cholesky_benchmark_c(benchmark):
    _test_blockwise_cholesky_benchmark("CVM", benchmark)


def test_blockwise_cholesky_benchmark_numba(benchmark):
    _test_blockwise_cholesky_benchmark("NUMBA", benchmark)


def _test_batched_mvnormal_logp_and_dlogp(
    mode, mu_batch_shape, cov_batch_shape, benchmark
):
    rng = np.random.default_rng(sum(map(ord, "batched_mvnormal")))

    value_batch_shape = mu_batch_shape
    if len(cov_batch_shape) > len(mu_batch_shape):
        value_batch_shape = cov_batch_shape

    value = tensor("value", shape=(*value_batch_shape, 10))
    mu = tensor("mu", shape=(*mu_batch_shape, 10))
    cov = tensor("cov", shape=(*cov_batch_shape, 10, 10))

    test_values = [
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

    fn = function([value, mu, cov], [logp, *dlogp], mode=mode, trust_input=True)
    benchmark(fn, *test_values)


@pytest.mark.parametrize(
    "mu_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"mu:{arg}"
)
@pytest.mark.parametrize(
    "cov_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"cov:{arg}"
)
def test_batched_mvnormal_logp_and_dlogp_c(mu_batch_shape, cov_batch_shape, benchmark):
    _test_batched_mvnormal_logp_and_dlogp(
        mode="CVM",
        mu_batch_shape=mu_batch_shape,
        cov_batch_shape=cov_batch_shape,
        benchmark=benchmark,
    )


@pytest.mark.parametrize(
    "mu_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"mu:{arg}"
)
@pytest.mark.parametrize(
    "cov_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"cov:{arg}"
)
def test_batched_mvnormal_logp_and_dlogp_numba(
    mu_batch_shape, cov_batch_shape, benchmark
):
    _test_batched_mvnormal_logp_and_dlogp(
        mode="NUMBA",
        mu_batch_shape=mu_batch_shape,
        cov_batch_shape=cov_batch_shape,
        benchmark=benchmark,
    )
