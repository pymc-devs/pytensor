import numpy as np
import pytest

from pytensor import In, Out, function
from pytensor.tensor import alloc, tensor


# Each case broadcasts a different subset of dimensions, which is what decides
# the fill loop Alloc generates.
# name: (static shape of value, runtime shape of value, output shape)
ALLOC_CASES = {
    "scalar": ((), (), (1000, 1000)),
    "outer": ((1, None), (1, 1000), (1000, 1000)),
    "inner": ((None, 1), (1000, 1), (1000, 1000)),
}


@pytest.mark.parametrize("case", ALLOC_CASES)
def test_alloc_benchmark_numba(case, benchmark):
    static_shape, val_shape, out_shape = ALLOC_CASES[case]
    x = tensor("x", shape=static_shape)
    x_val = np.random.random(val_shape).astype(x.type.dtype)
    y = alloc(x, *out_shape)
    # Borrow to avoid deepcopy overhead
    fn = function(
        [In(x, borrow=True)],
        Out(y, borrow=True),
        mode="NUMBA",
        trust_input=True,
    )
    fn(x_val)  # JIT compile
    benchmark(fn, x_val)
