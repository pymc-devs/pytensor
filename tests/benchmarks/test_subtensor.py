import numpy as np
import pytest

from pytensor import Out, function
from pytensor.compile.mode import get_default_mode
from pytensor.tensor import dvector, vector, zeros_like
from pytensor.tensor.subtensor import Subtensor, inc_subtensor, set_subtensor


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


@pytest.mark.parametrize("depth", (3, 5, 8), ids=lambda d: f"depth={d}")
def test_local_subtensor_merge_compile_benchmark(depth, benchmark):
    """Compile time for nested constant slices on a symbolically-shaped vector.

    Regression for #112, even if graph simplified by the end, it would take many cycles due to graph blowup
    """

    no_fusion = get_default_mode().excluding("fusion")

    def build():
        x = dvector("x")
        y = x
        for _ in range(depth):
            y = y[1:-1]
        return [x], y

    # Warm caches so we measure rewriting, not import / module init.
    function(*build(), mode=no_fusion)
    fn = benchmark(lambda: function(*build(), mode=no_fusion))

    # Chain collapses to a single Subtensor
    n_subtensor = sum(isinstance(n.op, Subtensor) for n in fn.maker.fgraph.apply_nodes)
    assert n_subtensor == 1
