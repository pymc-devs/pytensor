import numpy as np
import pytest

from pytensor import function, shared


def _test_careduce_benchmark(axis, c_contiguous, mode, benchmark):
    N = 256
    x_test = np.random.uniform(size=(N, N, N))
    transpose_axis = (0, 1, 2) if c_contiguous else (2, 0, 1)

    x = shared(x_test, name="x", shape=x_test.shape)
    out = x.transpose(transpose_axis).sum(axis=axis)
    fn = function([], out, mode=mode)

    np.testing.assert_allclose(
        fn(),
        x_test.transpose(transpose_axis).sum(axis=axis),
    )
    benchmark(fn)


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "c_contiguous",
    (True, False),
    ids=lambda x: f"c_contiguous={x}",
)
def test_careduce_benchmark_c(axis, c_contiguous, benchmark):
    _test_careduce_benchmark(
        axis=axis, c_contiguous=c_contiguous, mode="CVM", benchmark=benchmark
    )


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "c_contiguous",
    (True, False),
    ids=lambda x: f"c_contiguous={x}",
)
def test_careduce_benchmark_numba(axis, c_contiguous, benchmark):
    _test_careduce_benchmark(
        axis=axis, c_contiguous=c_contiguous, mode="NUMBA", benchmark=benchmark
    )
