"""Compare the scalar log-CDF with PyMC's helper after pymc#8601."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function


def pymc_normal_logcdf(z):
    z = -z
    u = pt.erfcx(pt.switch(z < 0, -z, z) / np.sqrt(2))
    return pt.switch(
        z < 0,
        pt.log1p(-pt.exp(-(z**2) / 2) * u / 2),
        pt.log(u) - z**2 / 2 - np.log(2),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("implementation", ["pymc", "scalar"])
@pytest.mark.parametrize("wrt", [(), ("mu",), ("log_sigma",), ("mu", "log_sigma")])
def test_normal_logcdf_benchmark(benchmark, dtype, implementation, wrt):
    x = pt.vector("x", dtype=dtype)
    mu = pt.scalar("mu", dtype=dtype)
    log_sigma = pt.scalar("log_sigma", dtype=dtype)
    z = (x - mu) / pt.exp(log_sigma)
    logcdf = pymc_normal_logcdf(z) if implementation == "pymc" else pt.log_ndtr(z)
    total = logcdf.sum()
    parameters = {"mu": mu, "log_sigma": log_sigma}
    outputs = [total, *pt.grad(total, [parameters[name] for name in wrt])]
    fn = function([x, mu, log_sigma], outputs, mode="NUMBA", trust_input=True)
    values = np.random.default_rng(8601).uniform(-10, 10, 52_396).astype(dtype)
    args = (values, np.array(0.3, dtype=dtype), np.array(0.2, dtype=dtype))
    fn(*args)
    benchmark(fn, *args)
