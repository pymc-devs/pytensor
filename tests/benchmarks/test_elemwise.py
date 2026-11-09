import numpy as np
import pytest
from scipy.special import expit, polygamma

from pytensor import function, shared
from pytensor.scalar import BinaryScalarOp, upgrade_to_float_no_complex
from pytensor.tensor import exp, grad, matrix, sigmoid, softplus, vector
from pytensor.tensor.elemwise import Elemwise


def array_only_polygamma(order, x):
    return polygamma(order, x)


class ArrayFallbackScalarOp(BinaryScalarOp):
    # Keep benchmarking fallback even if SciPy gains native polygamma support.
    nfunc_spec = (f"{__name__}.array_only_polygamma", 2, 1)

    @staticmethod
    def output_types_preference(order_type, x_type):
        return upgrade_to_float_no_complex(x_type)

    def impl(self, order, x):
        return array_only_polygamma(order, x)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_array_function_fallback_benchmark_numba(benchmark, dtype):
    x = vector("x", dtype=dtype)
    out = Elemwise(ArrayFallbackScalarOp())(2, x)
    with pytest.warns(UserWarning, match="Numba will use object mode"):
        func = function([x], out, mode="NUMBA", trust_input=True)
    values = np.linspace(0.1, 20, 20_000, dtype=dtype)
    np.testing.assert_allclose(func(values), array_only_polygamma(2, values), rtol=1e-6)
    benchmark(func, values)


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
