from functools import partial

import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import grad, matrix, tensor
from pytensor.tensor.signal import convolve1d


def _test_convolve1d_benchmark(mode, batch, convolve_mode, benchmark):
    x = tensor(shape=(7, 183) if batch else (183,))
    y = tensor(shape=(7, 6) if batch else (6,))
    out = convolve1d(x, y, mode=convolve_mode)
    fn = function([x, y], out, mode=mode, trust_input=True)

    rng = np.random.default_rng()
    x_test = rng.normal(size=(x.type.shape)).astype(x.type.dtype)
    y_test = rng.normal(size=(y.type.shape)).astype(y.type.dtype)

    np_convolve1d = np.vectorize(
        partial(np.convolve, mode=convolve_mode), signature="(x),(y)->(z)"
    )

    np.testing.assert_allclose(
        fn(x_test, y_test),
        np_convolve1d(x_test, y_test),
    )
    benchmark(fn, x_test, y_test)


@pytest.mark.parametrize("convolve_mode", ("full", "valid"), ids=lambda x: f"mode={x}")
@pytest.mark.parametrize("batch", (False, True), ids=lambda x: f"batch={x}")
def test_convolve1d_benchmark_c(batch, convolve_mode, benchmark):
    _test_convolve1d_benchmark(
        mode="CVM", batch=batch, convolve_mode=convolve_mode, benchmark=benchmark
    )


@pytest.mark.parametrize("convolve_mode", ("full", "valid"), ids=lambda x: f"mode={x}")
@pytest.mark.parametrize("batch", (False, True), ids=lambda x: f"batch={x}")
def test_convolve1d_benchmark_numba(batch, convolve_mode, benchmark):
    _test_convolve1d_benchmark(
        mode="NUMBA", batch=batch, convolve_mode=convolve_mode, benchmark=benchmark
    )


def _test_convolve1d_grad_benchmark(mode, convolve_mode, benchmark):
    # Use None core shape so PyTensor doesn't know which mode to use until runtime.
    larger = matrix("larger", shape=(8, None))
    smaller = matrix("smaller", shape=(8, None))
    grad_wrt_smaller = grad(
        convolve1d(larger, smaller, mode=convolve_mode).sum(), wrt=smaller
    )

    fn = function([larger, smaller], grad_wrt_smaller, trust_input=True, mode=mode)

    rng = np.random.default_rng([119, mode == "full"])
    test_larger = rng.normal(size=(8, 1024)).astype(larger.type.dtype)
    test_smaller = rng.normal(size=(8, 16)).astype(smaller.type.dtype)
    fn(test_larger, test_smaller)  # JIT compile for JIT backends
    benchmark(fn, test_larger, test_smaller)


@pytest.mark.parametrize("convolve_mode", ["full", "valid"])
def test_convolve1d_grad_benchmark_c(convolve_mode, benchmark):
    _test_convolve1d_grad_benchmark(
        mode="CVM", convolve_mode=convolve_mode, benchmark=benchmark
    )


@pytest.mark.parametrize("convolve_mode", ["full", "valid"])
def test_convolve1d_grad_benchmark_numba(convolve_mode, benchmark):
    _test_convolve1d_grad_benchmark(
        mode="NUMBA", convolve_mode=convolve_mode, benchmark=benchmark
    )
