import numpy as np
import pytest

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
