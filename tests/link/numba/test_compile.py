import numpy as np

import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import explicit_graph_inputs


def test_radon_model_logp_dlogp():
    def halfnormal(name, *, sigma=1.0, model_logp):
        log_value = pt.scalar(f"{name}_log")
        value = pt.exp(log_value)

        logp = (
            -0.5 * ((value / sigma) ** 2) + pt.log(pt.sqrt(2.0 / np.pi)) - pt.log(sigma)
        )
        logp = pt.switch(value >= 0, logp, -np.inf)
        model_logp.append(logp + value)
        return value

    def normal(name, *, mu=0.0, sigma=1.0, model_logp, observed=None):
        value = pt.scalar(name) if observed is None else pt.as_tensor(observed)

        logp = (
            -0.5 * (((value - mu) / sigma) ** 2)
            - pt.log(pt.sqrt(2.0 * np.pi))
            - pt.log(sigma)
        )
        model_logp.append(logp)
        return value

    def zerosumnormal(name, *, sigma=1.0, size, model_logp):
        raw_value = pt.vector(f"{name}_zerosum", shape=(size - 1,))
        n = raw_value.shape[0] + 1
        sum_vals = raw_value.sum(0, keepdims=True)
        norm = sum_vals / (pt.sqrt(n) + n)
        fill_value = norm - sum_vals / pt.sqrt(n)
        value = pt.concatenate([raw_value, fill_value]) - norm

        shape = value.shape
        _full_size = pt.prod(shape)
        _degrees_of_freedom = pt.prod(shape[-1:].inc(-1))
        logp = pt.sum(
            -0.5 * ((value / sigma) ** 2)
            - (pt.log(pt.sqrt(2.0 * np.pi)) + pt.log(sigma))
            * (_degrees_of_freedom / _full_size)
        )
        model_logp.append(logp)
        return value

    rng = np.random.default_rng(1)
    n_counties = 85
    county_idx = rng.integers(n_counties, size=919)
    county_idx.sort()
    floor = rng.binomial(n=1, p=0.5, size=919).astype(np.float64)
    log_radon = rng.normal(size=919)

    # joined_inputs = pt.vector("joined_inputs")

    model_logp = []
    intercept = normal("intercept", sigma=10, model_logp=model_logp)

    # County effects
    county_raw = zerosumnormal("county_raw", size=n_counties, model_logp=model_logp)
    county_sd = halfnormal("county_sd", model_logp=model_logp)
    county_effect = county_raw * county_sd

    # Global floor effect
    floor_effect = normal("floor_effect", sigma=2, model_logp=model_logp)

    county_floor_raw = zerosumnormal(
        "county_floor_raw", size=n_counties, model_logp=model_logp
    )
    county_floor_sd = halfnormal("county_floor_sd", model_logp=model_logp)
    county_floor_effect = county_floor_raw * county_floor_sd

    mu = (
        intercept
        + county_effect[county_idx]
        + floor_effect * floor
        + county_floor_effect[county_idx] * floor
    )

    sigma = halfnormal("sigma", model_logp=model_logp)
    _ = normal(
        "log_radon",
        mu=mu,
        sigma=sigma,
        observed=log_radon,
        model_logp=model_logp,
    )

    model_logp = pt.sum([logp.sum() for logp in model_logp])
    model_logp = rewrite_graph(
        model_logp, include=("canonicalize", "stabilize"), clone=False
    )
    params = list(explicit_graph_inputs(model_logp))
    model_dlogp = pt.concatenate([term.ravel() for term in pt.grad(model_logp, params)])

    # TODO: Replace inputs by raveled vector

    function(params, [model_logp, model_dlogp]).dprint()
