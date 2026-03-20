import numpy as np
import pytest
import scipy.special

from pytensor import function
from pytensor.tensor import exp, isinf, log, matrix, max, sum, switch


def _test_logsumexp_benchmark(mode, size, axis, benchmark):
    X = matrix("X")
    X_max = max(X, axis=axis, keepdims=True)
    X_max = switch(isinf(X_max), 0, X_max)
    X_lse = log(sum(exp(X - X_max), axis=axis, keepdims=True)) + X_max

    rng = np.random.default_rng(23920)
    X_val = rng.normal(size=size)

    X_lse_fn = function([X], X_lse, mode=mode, trust_input=True)

    # JIT compile first
    res = X_lse_fn(X_val)
    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)

    res = benchmark(X_lse_fn, X_val)


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark_c(size, axis, benchmark):
    _test_logsumexp_benchmark("CVM", size, axis, benchmark)


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark_numba(size, axis, benchmark):
    _test_logsumexp_benchmark("NUMBA", size, axis, benchmark)


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark_jax(size, axis, benchmark):
    pytest.importorskip("jax")
    _test_logsumexp_benchmark("JAX", size, axis, benchmark)


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark_mlx(size, axis, benchmark):
    pytest.importorskip("mlx")
    _test_logsumexp_benchmark("MLX", size, axis, benchmark)
