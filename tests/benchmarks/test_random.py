import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, shared


@pytest.mark.parametrize("floatX", ("float64", "float32"))
def test_normal_rv_benchmark_numba(floatX, benchmark):
    # Drawing standard normals through numba. The ``0, 1`` literals are typed int8, and
    # numba's rng.normal samples in float64; without cast_rv_float_params_to_float64
    # upcasting the parameters once, each of the >100k draws pays a per-element cast and
    # the function runs ~3x slower. This benchmark tracks that hot path.
    with config.change_flags(floatX=floatX):
        rng = shared(np.random.default_rng(0))
        next_rng, draws = rng.normal(0, 1, size=(2160, 50))
        fn = function(
            [], draws, updates={rng: next_rng}, mode="NUMBA", trust_input=True
        )
    fn()  # compile / warm up before timing
    benchmark(fn)


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
