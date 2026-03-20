import numpy as np

from pytensor import In, Out, function
from pytensor.tensor import tensor3


def _test_reshape_benchmark(mode, benchmark):
    x = tensor3("x")
    x_val = np.random.random((2, 3, 4))
    y1 = x.reshape((6, 4))
    y2 = x.reshape((2, 12))
    y3 = x.reshape((-1,))
    # Borrow to avoid deepcopy overhead
    reshape_fn = function(
        [In(x, borrow=True)],
        [Out(y1, borrow=True), Out(y2, borrow=True), Out(y3, borrow=True)],
        trust_input=True,
        mode=mode,
    )
    reshape_fn(x_val)
    benchmark(reshape_fn, x_val)


def test_reshape_benchmark_c(benchmark):
    _test_reshape_benchmark("CVM", benchmark)


def test_reshape_benchmark_numba(benchmark):
    _test_reshape_benchmark("NUMBA", benchmark)
