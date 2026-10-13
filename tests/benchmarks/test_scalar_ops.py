"""Scalar-favorable validation graph for the scalar-carrying optimizations.

The compiled `logp_dlogp`-shaped graph mirrors what PyMC builds: a single flat
input vector is split by `Subtensor` into scalar parameters plus one vector
parameter, an elementwise expression is reduced to a scalar `logp`, and the
gradient of every parameter is joined back into one raveled `dlogp` vector.

That shape stresses exactly the ops the scalar-carrying work targets:

* scalar params extracted from the joined vector (``Subtensor`` -> scalar),
* a fully-reduced scalar ``logp`` (``CAReduce`` / fused reduction -> scalar),
* a raveled gradient mixing scalar and vector entries (``Join``).

`build_scalar_favorable_graph` is importable so allocation-counting scripts can
reuse the exact same graph the timing benchmark measures.
"""

import numpy as np
import pytest

from pytensor import Out, function, grad
from pytensor.graph.replace import graph_replace
from pytensor.tensor import as_tensor_variable, exp, join, scalar, vector


def _reference_logp(theta, y, group_idx, n_groups):
    """NumPy reference: value and gradient of the graph below, for validation."""
    mu, sigma_log, sigma_group_log = theta[0], theta[1], theta[2]
    offsets = theta[3:]
    sigma = np.exp(sigma_log)
    sigma_group = np.exp(sigma_group_log)
    group_means = mu + sigma_group * offsets
    resid = (y - group_means[group_idx]) / sigma
    logp = -0.5 * (resid**2).sum() - y.size * sigma_log - 0.5 * (offsets**2).sum()
    return logp


def build_scalar_favorable_graph(n_obs, n_groups, seed=490):
    """Return ``(joined, [logp, dlogp], test_value, y, group_idx)``.

    ``joined`` is the single flat parameter vector of length ``n_groups + 3``
    ordered ``[mu, sigma_log, sigma_group_log, *offsets]``. The observations
    ``y`` and the group assignment ``group_idx`` are baked in as constants, as
    the data is in a real ``logp_dlogp_function``.
    """
    rng = np.random.default_rng(seed)
    y_np = rng.standard_normal(n_obs)
    group_idx_np = rng.integers(0, n_groups, size=n_obs)

    y = as_tensor_variable(y_np)
    group_idx = as_tensor_variable(group_idx_np)

    # Parameters as separate variables so `grad` produces one adjoint per
    # parameter and the raveled gradient is a genuine mixed scalar/vector Join.
    mu = scalar("mu")
    sigma_log = scalar("sigma_log")
    sigma_group_log = scalar("sigma_group_log")
    offsets = vector("offsets", shape=(n_groups,))
    params = [mu, sigma_log, sigma_group_log, offsets]

    sigma = exp(sigma_log)
    sigma_group = exp(sigma_group_log)
    group_means = mu + sigma_group * offsets
    resid = (y - group_means[group_idx]) / sigma
    logp = -0.5 * (resid**2).sum() - n_obs * sigma_log - 0.5 * (offsets**2).sum()

    grads = grad(logp, params)
    dlogp = join(0, *[g.reshape((-1,)) for g in grads])

    # Fold the separate parameters into one flat input, as PyMC does when it
    # builds `logp_dlogp_function`: the scalar params become `Subtensor`s of the
    # joined vector, the vector param a basic slice.
    joined = vector("joined", shape=(n_groups + 3,))
    replace = {
        mu: joined[0],
        sigma_log: joined[1],
        sigma_group_log: joined[2],
        offsets: joined[3:],
    }
    logp, dlogp = graph_replace([logp, dlogp], replace)

    test_value = rng.standard_normal(n_groups + 3)
    return joined, [logp, dlogp], test_value, y_np, group_idx_np


def _build_fn(mode, n_obs, n_groups):
    joined, outs, test_value, y_np, group_idx_np = build_scalar_favorable_graph(
        n_obs, n_groups
    )
    fn = function(
        [joined],
        [Out(o, borrow=True) for o in outs],
        mode=mode,
        trust_input=True,
    )
    return fn, test_value, y_np, group_idx_np


def _validate(fn, test_value, y_np, group_idx_np, n_groups):
    logp_val, dlogp_val = fn(test_value)
    ref_logp = _reference_logp(test_value, y_np, group_idx_np, n_groups)
    np.testing.assert_allclose(logp_val, ref_logp, rtol=1e-8)

    # Central finite differences of the reference logp validate the gradient.
    eps = 1e-6
    fd = np.empty_like(test_value)
    for i in range(test_value.size):
        step = np.zeros_like(test_value)
        step[i] = eps
        fd[i] = (
            _reference_logp(test_value + step, y_np, group_idx_np, n_groups)
            - _reference_logp(test_value - step, y_np, group_idx_np, n_groups)
        ) / (2 * eps)
    np.testing.assert_allclose(dlogp_val, fd, rtol=1e-4, atol=1e-4)


# Small stresses fixed per-call overhead (boxing/allocation); large exposes the
# per-element scaling.
SIZES = [(100, 5), (10_000, 100)]


@pytest.mark.parametrize("n_obs, n_groups", SIZES, ids=lambda s: str(s))
def test_scalar_favorable_benchmark_numba(n_obs, n_groups, benchmark):
    fn, test_value, y_np, group_idx_np = _build_fn("NUMBA", n_obs, n_groups)
    _validate(fn, test_value, y_np, group_idx_np, n_groups)
    benchmark(fn, test_value)


@pytest.mark.parametrize("n_obs, n_groups", SIZES, ids=lambda s: str(s))
def test_scalar_favorable_benchmark_c(n_obs, n_groups, benchmark):
    fn, test_value, y_np, group_idx_np = _build_fn("CVM", n_obs, n_groups)
    _validate(fn, test_value, y_np, group_idx_np, n_groups)
    benchmark(fn, test_value)
