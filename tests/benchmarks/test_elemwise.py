import numpy as np

from pytensor import function, shared
from pytensor.tensor import exp, grad, matrix, vector


def _test_simple_elemwise_benchmark(mode, benchmark):
    x = matrix("y")
    y = vector("z")
    out = exp(2 * x * y + y)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(200, 500))
    y_val = rng.normal(size=500)
    expected_out = np.exp(2 * x_val * y_val + y_val)

    func = function([x, y], out, mode=mode, trust_input=True)
    np.testing.assert_allclose(func(x_val, y_val), expected_out)

    benchmark(func, x_val, y_val)


def test_simple_elemwise_benchmark_c(benchmark):
    _test_simple_elemwise_benchmark("CVM", benchmark)


def test_simple_elemwise_benchmark_numba(benchmark):
    _test_simple_elemwise_benchmark("NUMBA", benchmark)


def _test_fused_elemwise_benchmark(mode, benchmark):
    rng = np.random.default_rng(123)
    size = 100_000
    x = shared(rng.normal(size=size), name="x")
    mu = shared(rng.normal(size=size), name="mu")

    logp = -((x - mu) ** 2) / 2
    grad_logp = grad(logp.sum(), x)

    func = function([], [logp, grad_logp], mode=mode)
    func()  # JIT compile for JIT backends
    benchmark(func)


def test_fused_elemwise_benchmark_c(benchmark):
    _test_fused_elemwise_benchmark("CVM", benchmark)


def test_fused_elemwise_benchmark_numba(benchmark):
    _test_fused_elemwise_benchmark("NUMBA", benchmark)
