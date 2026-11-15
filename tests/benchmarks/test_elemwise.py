import numpy as np
import pytest
from scipy.special import expit

from pytensor import function, shared
from pytensor.tensor import exp, grad, matrix, scalar, sigmoid, softplus, vector
from pytensor.tensor.special import LogAddExp


@pytest.mark.parametrize("input_kind", ["unary", "typed", "literal"])
def test_elemwise_graph_creation(benchmark, input_kind):
    x = vector("x", dtype="float32")
    if input_kind == "unary":
        benchmark(exp, x)
    else:
        other = vector("y", dtype="float32") if input_kind == "typed" else 2
        benchmark(lambda: x + other)


def test_symbolic_graph_creation(benchmark):
    x = vector("x", dtype="float32")
    y = scalar("y", dtype="float32")
    op = LogAddExp(input_types=[x.type, y.type])
    benchmark(op, x, 2)


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


def test_shared_exp_benchmark(benchmark):
    x = vector("x", dtype="float64")
    x_val = np.random.default_rng(42).normal(size=100_000)
    out = softplus(x) + softplus(-x) + sigmoid(x)
    expected = np.logaddexp(0, x_val) + np.logaddexp(0, -x_val) + expit(x_val)

    func = function([x], out, mode="NUMBA", trust_input=True)
    assert len(func.maker.fgraph.apply_nodes) == 1
    np.testing.assert_allclose(func(x_val), expected)

    benchmark(func, x_val)
