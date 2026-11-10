"""Benchmark scalar parameters, reductions and joined gradients."""

import numpy as np

from pytensor import Out, function, grad
from pytensor.graph.replace import graph_replace
from pytensor.tensor import as_tensor_variable, exp, join, scalar, vector


def test_scalar_favorable_benchmark(benchmark):
    n_obs, n_groups = 100, 5
    rng = np.random.default_rng(490)
    y = as_tensor_variable(rng.standard_normal(n_obs))
    group_idx = as_tensor_variable(rng.integers(0, n_groups, size=n_obs))

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

    # Flatten the parameters as in a PyMC logp_dlogp function.
    joined = vector("joined", shape=(n_groups + 3,))
    replace = {
        mu: joined[0],
        sigma_log: joined[1],
        sigma_group_log: joined[2],
        offsets: joined[3:],
    }
    logp, dlogp = graph_replace([logp, dlogp], replace)
    fn = function(
        [joined],
        [Out(logp, borrow=True), Out(dlogp, borrow=True)],
        trust_input=True,
    )
    test_value = rng.standard_normal(n_groups + 3)
    fn(test_value)
    benchmark(fn, test_value)
