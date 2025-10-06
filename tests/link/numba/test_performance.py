import timeit

import numpy as np
import pytest


pytest.importorskip("numba")

import pytensor.tensor as aet
from pytensor import config
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.numba.linker import NumbaLinker
from pytensor.tensor.math import Max
from tests.fixtures import *  # noqa: F403


opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


@pytest.mark.parametrize(
    "careduce_fn, numpy_fn, axis, inputs, input_vals",
    [
        pytest.param(
            lambda x, axis=None: Max(axis)(x),
            np.max,
            (0, 1),
            [
                aet.matrix(),
            ],
            [np.arange(3000 * 2000, dtype=config.floatX).reshape((3000, 2000))],
        )
    ],
)
def test_careduce_performance(careduce_fn, numpy_fn, axis, inputs, input_vals):
    g = careduce_fn(*inputs, axis=axis)

    pytensor_numba_fn = function(
        inputs,
        g,
        mode=numba_mode,
    )

    # pytensor_c_fn = function(
    #     inputs,
    #     g,
    #     mode=Mode("cvm")
    # )

    numpy_res = numpy_fn(*input_vals)
    numba_res = pytensor_numba_fn(*input_vals)
    # c_res = pytensor_c_fn(*input_vals)

    assert np.array_equal(numba_res, numpy_res)

    # FYI: To test the Numba JITed function directly, use `pytensor_numba_fn.vm.jit_fn`

    numpy_timer = timeit.Timer("numpy_fn(*input_vals)", "pass", globals=locals())
    numba_timer = timeit.Timer(
        "pytensor_numba_fn.vm.jit_fn(*input_vals)", "pass", globals=locals()
    )
    # c_timer = timeit.Timer("pytensor_c_fn(*input_vals)", "pass", globals=locals())

    n_loops, _ = numpy_timer.autorange()

    numpy_times = numpy_timer.repeat(5, n_loops)
    numba_times = numba_timer.repeat(5, n_loops)
    # c_times = c_timer.repeat(5, n_loops)

    mean_numba_time = np.mean(numba_times)
    mean_numpy_time = np.mean(numpy_times)
    # mean_c_time = np.mean(c_times)

    # FIXME: Why are we asserting >=? Numba could be doing worse than numpy!
    assert mean_numba_time / mean_numpy_time >= 0.75


@pytest.mark.parametrize("cache", (False, True))
def test_radon_model_compile_repeatedly_numba_benchmark(cache, radon_model, benchmark):
    joined_inputs, [model_logp, model_dlogp] = radon_model
    rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)

    def compile_and_call_once():
        with config.change_flags(numba__cache=cache):
            fn = function(
                [joined_inputs],
                [model_logp, model_dlogp],
                mode="NUMBA",
                trust_input=True,
            )
            fn(x)

    benchmark.pedantic(compile_and_call_once, rounds=5, iterations=1)


@pytest.mark.parametrize("cache", (False, True))
def test_radon_model_compile_variants_numba_benchmark(
    cache, radon_model, radon_model_variants, benchmark
):
    """Test compilation speed when a slightly variant of a function is compiled each time.

    This test more realistically simulates a use case where a model is recompiled
    multiple times with small changes, such as in an interactive environment.

    NOTE: For this test to be meaningful on subsequent runs, the cache must be cleared
    """
    joined_inputs, [model_logp, model_dlogp] = radon_model
    rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)

    # Compile base function once to populate the cache
    fn = function(
        [joined_inputs], [model_logp, model_dlogp], mode="NUMBA", trust_input=True
    )
    fn(x)

    def compile_and_call_once():
        with config.change_flags(numba__cache=cache):
            for joined_inputs, [model_logp, model_dlogp] in radon_model_variants:
                fn = function(
                    [joined_inputs],
                    [model_logp, model_dlogp],
                    mode="NUMBA",
                    trust_input=True,
                )
                fn(x)

    benchmark.pedantic(compile_and_call_once, rounds=1, iterations=1)


@pytest.mark.parametrize("cache", (False, True))
def test_radon_model_call_numba_benchmark(cache, radon_model, benchmark):
    joined_inputs, [model_logp, model_dlogp] = radon_model

    with config.change_flags(numba__cache=cache):
        fn = function(
            [joined_inputs], [model_logp, model_dlogp], mode="NUMBA", trust_input=True
        )
        rng = np.random.default_rng(1)
    x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)
    fn(x)  # warmup

    benchmark.pedantic(fn, (x,), rounds=10_000, iterations=10)
