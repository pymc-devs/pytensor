import numpy as np
import pytest

from pytensor import Out, function
from pytensor.tensor import vector, zeros_like
from pytensor.tensor.subtensor import inc_subtensor, set_subtensor


def _test_advanced_subtensor1_benchmark(mode, static_shape, gc, benchmark):
    x = vector("x", shape=(85 if static_shape else None,))

    x_values = np.random.normal(size=(85,))
    idxs_values = np.arange(85).repeat(11)

    # With static shape and constant indices we know all idxs are valid
    # And can use faster mode in numpy.take
    out = x[idxs_values]

    fn = function(
        [x],
        Out(out, borrow=True),
        mode=mode,
        trust_input=True,
    )
    fn.vm.allow_gc = gc
    benchmark(fn, x_values, idxs_values)


@pytest.mark.parametrize(
    "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
)
@pytest.mark.parametrize("gc", [True, False])
def test_advanced_subtensor1_benchmark_c(static_shape, gc, benchmark):
    _test_advanced_subtensor1_benchmark("CVM", static_shape, gc, benchmark)


@pytest.mark.parametrize(
    "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
)
def test_advanced_subtensor1_benchmark_numba(static_shape, benchmark):
    _test_advanced_subtensor1_benchmark("NUMBA", static_shape, False, benchmark)


def _test_advanced_incsubtensor1_benchmark(mode, func, static_shape, gc, benchmark):
    x = vector("x", shape=(85 if static_shape else None,))
    x_values = np.zeros((85,))
    buffer = zeros_like(x)
    y_values = np.random.normal(size=(85 * 11,))
    idxs_values = np.arange(85).repeat(11)

    # With static shape and constant indices we know all idxs are valid
    # Reuse same buffer of zeros, to check we rather allocate twice than copy inside IncSubtensor
    out1 = func(buffer[idxs_values], y_values)
    out2 = func(buffer[idxs_values[::-1]], y_values)

    fn = function(
        [x],
        [Out(out1, borrow=True), Out(out2, borrow=True)],
        mode=mode,
        trust_input=True,
    )
    fn.vm.allow_gc = gc
    benchmark(fn, x_values)


@pytest.mark.parametrize(
    "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
)
@pytest.mark.parametrize("func", (inc_subtensor, set_subtensor))
@pytest.mark.parametrize("gc", [True, False])
def test_advanced_incsubtensor1_benchmark_c(func, static_shape, gc, benchmark):
    _test_advanced_incsubtensor1_benchmark("CVM", func, static_shape, gc, benchmark)


@pytest.mark.parametrize(
    "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
)
@pytest.mark.parametrize("func", (inc_subtensor, set_subtensor))
def test_advanced_incsubtensor1_benchmark_numba(func, static_shape, benchmark):
    _test_advanced_incsubtensor1_benchmark(
        "NUMBA", func, static_shape, False, benchmark
    )
