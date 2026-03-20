import numpy as np
import pytest

import pytensor
from pytensor import function
from pytensor.gradient import jacobian
from pytensor.tensor.math import outer, sqrt
from pytensor.tensor.type import vector


@pytest.mark.parametrize("vectorize", [False, True], ids=lambda x: f"vectorize={x}")
def test_jacobian_benchmark(vectorize, benchmark):
    x = vector("x", shape=(3,))
    y = outer(x, x)

    jac_y = jacobian(y, x, vectorize=vectorize)

    fn = function([x], jac_y, mode="FAST_RUN", trust_input=True)
    benchmark(fn, np.array([0, 1, 2], dtype=x.type.dtype))


@pytest.mark.parametrize("vectorize", [False, True], ids=lambda x: f"vectorize={x}")
def test_partial_jacobian_benchmark(vectorize, benchmark):
    # Example from https://github.com/jax-ml/jax/discussions/5904#discussioncomment-422956
    N = 1000
    rng = np.random.default_rng(2025)
    x_test = rng.random((N,))

    f_mat = rng.random((N, N))
    x = vector("x", dtype="float64")

    def f(x):
        return sqrt(f_mat @ x / N)

    full_jacobian = jacobian(f(x), x, vectorize=vectorize)
    partial_jacobian = full_jacobian[:5, :5]

    fn = pytensor.function([x], partial_jacobian, mode="FAST_RUN", trust_input=True)
    benchmark(fn, x_test)
