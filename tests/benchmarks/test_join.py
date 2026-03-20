import numpy as np
import pytest

from pytensor import Out, function
from pytensor.configdefaults import config
from pytensor.tensor import join, matrices, vectors


def _test_join_benchmark(mode, ndim, axis, memory_layout, gc, benchmark):
    if ndim == 1 and not (memory_layout == "C-contiguous" and axis == 0):
        pytest.skip("Redundant parametrization")

    n = 64
    inputs = vectors("abcdef") if ndim == 1 else matrices("abcdef")
    out = join(axis, *inputs)
    fn = function(inputs, Out(out, borrow=True), mode=mode, trust_input=True)
    fn.vm.allow_gc = gc
    test_values = [np.zeros((n, n)[:ndim], dtype=inputs[0].dtype) for _ in inputs]
    if memory_layout == "C-contiguous":
        pass
    elif memory_layout == "F-contiguous":
        test_values = [t.T for t in test_values]
    elif memory_layout == "Mixed":
        test_values = [t if i % 2 else t.T for i, t in enumerate(test_values)]
    else:
        raise ValueError

    assert fn(*test_values).shape == ((n * 6, n)[:ndim] if axis == 0 else (n, n * 6))
    benchmark(fn, *test_values)


@pytest.mark.parametrize("memory_layout", ["C-contiguous", "F-contiguous", "Mixed"])
@pytest.mark.parametrize("axis", (0, 1), ids=lambda x: f"axis={x}")
@pytest.mark.parametrize("ndim", (1, 2), ids=["vector", "matrix"])
@pytest.mark.parametrize("gc", [True, False])
@config.change_flags(cmodule__warn_no_version=False)
def test_join_benchmark_c(ndim, axis, memory_layout, gc, benchmark):
    _test_join_benchmark("CVM", ndim, axis, memory_layout, gc, benchmark)


@pytest.mark.parametrize("memory_layout", ["C-contiguous", "F-contiguous", "Mixed"])
@pytest.mark.parametrize("axis", (0, 1), ids=lambda x: f"axis={x}")
@pytest.mark.parametrize("ndim", (1, 2), ids=["vector", "matrix"])
@config.change_flags(cmodule__warn_no_version=False)
def test_join_benchmark_numba(ndim, axis, memory_layout, benchmark):
    _test_join_benchmark("NUMBA", ndim, axis, memory_layout, False, benchmark)
