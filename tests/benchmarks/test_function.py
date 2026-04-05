import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import In, function
from pytensor.compile.io import Out
from pytensor.compile.mode import Mode
from pytensor.tensor import exp, vector
from pytensor.tensor.random.basic import normal
from pytensor.tensor.random.type import random_generator_type


def _test_minimal_random_function_call_benchmark(mode, trust_input, benchmark):
    rng = random_generator_type()
    x = normal(rng=rng, size=(100,))

    f = function([In(rng, mutable=True)], x, trust_input=trust_input, mode=mode)

    rng_val = np.random.default_rng()
    benchmark(f, rng_val)


@pytest.mark.parametrize("trust_input", [True, False])
def test_minimal_random_function_call_benchmark_c(trust_input, benchmark):
    _test_minimal_random_function_call_benchmark("CVM", trust_input, benchmark)


@pytest.mark.parametrize("trust_input", [True, False])
def test_minimal_random_function_call_benchmark_numba(trust_input, benchmark):
    _test_minimal_random_function_call_benchmark("NUMBA", trust_input, benchmark)


@pytest.mark.parametrize("mode", ("default", "trust_input", "direct"))
def test_numba_function_overhead_benchmark(mode, benchmark):
    x = vector("x")
    out = exp(x)

    fn = function([x], out, mode="NUMBA")
    if mode == "default":
        pass
    elif mode == "trust_input":
        fn.trust_input = True
    elif mode == "direct":
        fn = fn.vm.jit_fn
    else:
        raise ValueError(f"mode {mode} not understod")

    test_x = np.zeros(1000)
    assert np.sum(fn(test_x)) == 1000

    benchmark(fn, test_x)


@pytest.mark.parametrize("trust_input", [True, False], ids=lambda x: f"trust_input={x}")
@pytest.mark.parametrize("linker", ["cvm", "cvm_nogc"])
def test_identity_function_overhead_benchmark(trust_input, linker, benchmark):
    x = pt.vector("x")
    fn = function(
        [In(x, borrow=True)],
        Out(x, borrow=True),
        trust_input=trust_input,
        mode=Mode(linker=linker, optimizer=None),
    )
    fn.dprint(print_memory_map=True)
    x_test = np.zeros(10)
    benchmark(fn, x_test)
