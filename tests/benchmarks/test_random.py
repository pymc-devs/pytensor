import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, shared


@pytest.mark.parametrize("n, size", [(50, 20_000)])
def test_mvnormal_shared_cov_benchmark_numba(n, size, benchmark):
    """MvNormal draws from a single shared covariance.

    Because ``MvNormalRV`` is a ``SymbolicRVOp`` (``mean + cholesky(cov) @ z``),
    an unbatched ``cov`` is factorized once and reused across all ``size`` draws,
    instead of re-factorizing it per draw inside the vectorized loop.
    """
    rng = shared(np.random.default_rng(0))
    mean = pt.zeros(n)
    cov = pt.tensor("cov", shape=(n, n))
    draws = pt.random.multivariate_normal(mean, cov, size=(size,), rng=rng)

    fn = function([cov], draws, mode="NUMBA", trust_input=True)

    test_rng = np.random.default_rng(1)
    a = test_rng.standard_normal((n, n))
    cov_test = a @ a.T + n * np.eye(n)  # symmetric positive-definite

    out = fn(cov_test)
    assert out.shape == (size, n)
    benchmark(fn, cov_test)
