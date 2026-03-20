import itertools

import numpy as np
import pytest

from pytensor import In, Out, function
from pytensor.tensor import tensor3


def _test_dimshuffle_benchmark(mode, c_contiguous, benchmark):
    x = tensor3("x")
    if c_contiguous:
        x_val = np.random.random((2, 3, 4))
    else:
        x_val = np.random.random((200, 300, 400)).transpose(1, 2, 0)
    ys = [x.transpose(t) for t in itertools.permutations((0, 1, 2))]
    ys += [
        x[None],
        x[:, None],
        x[:, :, None],
        x[:, :, :, None],
    ]
    # Borrow to avoid deepcopy overhead
    fn = function(
        [In(x, borrow=True)],
        [Out(y, borrow=True) for y in ys],
        mode=mode,
        trust_input=True,
    )
    fn(x_val)  # JIT compile for JIT backends
    benchmark(fn, x_val)


@pytest.mark.parametrize("c_contiguous", [True, False])
def test_dimshuffle_benchmark_c(c_contiguous, benchmark):
    _test_dimshuffle_benchmark(
        mode="CVM", c_contiguous=c_contiguous, benchmark=benchmark
    )


@pytest.mark.parametrize("c_contiguous", [True, False])
def test_dimshuffle_benchmark_numba(c_contiguous, benchmark):
    _test_dimshuffle_benchmark(
        mode="NUMBA", c_contiguous=c_contiguous, benchmark=benchmark
    )
