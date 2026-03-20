from contextlib import nullcontext

import numpy as np
import pytest

from pytensor import config, function
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import (
    as_tensor,
    concatenate,
    exp,
    grad,
    log,
    prod,
    scalar,
    sqrt,
    switch,
    vector,
)
from pytensor.tensor import (
    sum as pt_sum,
)


def create_radon_model(
    intercept_dist="normal", sigma_dist="halfnormal", centered=False
):
    def halfnormal(name, *, sigma=1.0, model_logp):
        log_value = scalar(f"{name}_log")
        value = exp(log_value)

        logp = -0.5 * ((value / sigma) ** 2) + log(sqrt(2.0 / np.pi)) - log(sigma)
        logp = switch(value >= 0, logp, -np.inf)
        model_logp.append(logp + value)
        return value

    def normal(name, *, mu=0.0, sigma=1.0, model_logp, observed=None):
        value = scalar(name) if observed is None else as_tensor(observed)

        logp = (
            -0.5 * (((value - mu) / sigma) ** 2) - log(sqrt(2.0 * np.pi)) - log(sigma)
        )
        model_logp.append(logp)
        return value

    def lognormal(name, *, mu=0.0, sigma=1.0, model_logp):
        value = normal(name, mu=mu, sigma=sigma, model_logp=model_logp)
        return exp(value)

    def zerosumnormal(name, *, sigma=1.0, size, model_logp):
        raw_value = vector(f"{name}_zerosum", shape=(size - 1,))
        n = raw_value.shape[0] + 1
        sum_vals = raw_value.sum(0, keepdims=True)
        norm = sum_vals / (sqrt(n) + n)
        fill_value = norm - sum_vals / sqrt(n)
        value = concatenate([raw_value, fill_value]) - norm

        shape = value.shape
        _full_size = prod(shape)
        _degrees_of_freedom = prod(shape[-1:].inc(-1))
        logp = pt_sum(
            -0.5 * ((value / sigma) ** 2)
            - (log(sqrt(2.0 * np.pi)) + log(sigma)) * (_degrees_of_freedom / _full_size)
        )
        model_logp.append(logp)
        return value

    dist_fn_map = {
        fn.__name__: fn for fn in (halfnormal, normal, lognormal, zerosumnormal)
    }

    rng = np.random.default_rng(1)
    n_counties = 85
    county_idx = rng.integers(n_counties, size=919)
    county_idx.sort()
    floor = rng.binomial(n=1, p=0.5, size=919).astype(np.float64)
    log_radon = rng.normal(size=919)

    model_logp = []
    intercept = dist_fn_map[intercept_dist](
        "intercept", sigma=10, model_logp=model_logp
    )

    # County effects
    county_sd = halfnormal("county_sd", model_logp=model_logp)
    if centered:
        county_effect = zerosumnormal(
            "county_raw", sigma=county_sd, size=n_counties, model_logp=model_logp
        )
    else:
        county_raw = zerosumnormal("county_raw", size=n_counties, model_logp=model_logp)
        county_effect = county_raw * county_sd

    # Global floor effect
    floor_effect = normal("floor_effect", sigma=2, model_logp=model_logp)

    county_floor_sd = halfnormal("county_floor_sd", model_logp=model_logp)
    if centered:
        county_floor_effect = zerosumnormal(
            "county_floor_raw",
            sigma=county_floor_sd,
            size=n_counties,
            model_logp=model_logp,
        )
    else:
        county_floor_raw = zerosumnormal(
            "county_floor_raw", size=n_counties, model_logp=model_logp
        )
        county_floor_effect = county_floor_raw * county_floor_sd

    mu = (
        intercept
        + county_effect[county_idx]
        + floor_effect * floor
        + county_floor_effect[county_idx] * floor
    )

    sigma = dist_fn_map[sigma_dist]("sigma", model_logp=model_logp)
    _ = normal(
        "log_radon",
        mu=mu,
        sigma=sigma,
        observed=log_radon,
        model_logp=model_logp,
    )

    model_logp = pt_sum([logp.sum() for logp in model_logp])
    model_logp = rewrite_graph(
        model_logp, include=("canonicalize", "stabilize"), clone=False
    )
    params = list(explicit_graph_inputs(model_logp))
    model_dlogp = concatenate([term.ravel() for term in grad(model_logp, params)])

    size = sum(int(np.prod(p.type.shape)) for p in params)
    joined_inputs = vector("joined_inputs", shape=(size,))
    idx = 0
    replacement = {}
    for param in params:
        param_shape = param.type.shape
        param_size = int(np.prod(param_shape))
        replacement[param] = joined_inputs[idx : idx + param_size].reshape(param_shape)
        idx += param_size
    assert idx == joined_inputs.type.shape[0]

    model_logp, model_dlogp = graph_replace([model_logp, model_dlogp], replacement)
    return joined_inputs, [model_logp, model_dlogp]


@pytest.fixture(scope="session")
def radon_model():
    return create_radon_model()


@pytest.fixture(scope="session")
def radon_model_variants():
    # Convert to list comp
    return [
        create_radon_model(
            intercept_dist=intercept_dist,
            sigma_dist=sigma_dist,
            centered=centered,
        )
        for centered in (True, False)
        for intercept_dist in ("normal", "lognormal")
        for sigma_dist in ("halfnormal", "lognormal")
    ]


@pytest.mark.parametrize(
    "mode, cache",
    [("C", None), ("CVM", None), ("NUMBA", False), ("NUMBA", True)],
)
def test_radon_model_compile_repeatedly_benchmark(mode, cache, radon_model, benchmark):
    joined_inputs, [model_logp, model_dlogp] = radon_model
    rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)

    def compile_and_call_once():
        fn = function(
            [joined_inputs], [model_logp, model_dlogp], mode=mode, trust_input=True
        )
        fn(x)

    ctx = (
        config.change_flags(numba__cache=cache) if cache is not None else nullcontext()
    )
    with ctx:
        benchmark.pedantic(compile_and_call_once, rounds=5, iterations=1)


@pytest.mark.parametrize(
    "mode, cache",
    [("C", None), ("CVM", None), ("NUMBA", False), ("NUMBA", True)],
)
def test_radon_model_compile_variants_benchmark(
    mode, cache, radon_model, radon_model_variants, benchmark
):
    joined_inputs, [model_logp, model_dlogp] = radon_model
    rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)

    # Compile base function once to populate the cache
    fn = function(
        [joined_inputs], [model_logp, model_dlogp], mode=mode, trust_input=True
    )
    fn(x)

    def compile_and_call_once():
        for joined_inputs, [model_logp, model_dlogp] in radon_model_variants:
            fn = function(
                [joined_inputs], [model_logp, model_dlogp], mode=mode, trust_input=True
            )
            fn(x)

    ctx = (
        config.change_flags(numba__cache=cache) if cache is not None else nullcontext()
    )
    with ctx:
        benchmark.pedantic(compile_and_call_once, rounds=1, iterations=1)


@pytest.mark.parametrize(
    "mode, cache",
    [
        ("C", None),
        ("CVM", None),
        ("CVM_NOGC", None),
        ("NUMBA", False),
        ("NUMBA", True),
    ],
)
def test_radon_model_call_benchmark(mode, cache, radon_model, benchmark):
    joined_inputs, [model_logp, model_dlogp] = radon_model

    real_mode = "CVM" if mode == "CVM_NOGC" else mode
    ctx = (
        config.change_flags(numba__cache=cache) if cache is not None else nullcontext()
    )
    with ctx:
        fn = function(
            [joined_inputs], [model_logp, model_dlogp], mode=real_mode, trust_input=True
        )
    if mode == "CVM_NOGC":
        fn.vm.allow_gc = False

    rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)
    fn(x)  # warmup

    benchmark(fn, x)
