import numpy as np
import pytest

from pytensor import function
from pytensor.compile.mode import get_default_mode
from pytensor.tensor import matrix, tensor
from pytensor.tensor._linalg.decomposition.cholesky import cholesky
from pytensor.tensor.linalg import block_diag


def _test_cholesky_benchmark(mode, benchmark):
    rng = np.random.default_rng(6)
    r = rng.standard_normal((10, 10))
    pd = np.dot(r, r.T)
    x = matrix()
    chol = cholesky(x)
    ch_f = function([x], chol, mode=mode, trust_input=True)
    benchmark(ch_f, pd)


def test_cholesky_benchmark_c(benchmark):
    _test_cholesky_benchmark("CVM", benchmark)


def test_cholesky_benchmark_numba(benchmark):
    _test_cholesky_benchmark("NUMBA", benchmark)


@pytest.mark.parametrize("rewrite", [True, False], ids=["rewrite", "no_rewrite"])
@pytest.mark.parametrize("size", [10, 100, 1000], ids=["small", "medium", "large"])
def test_block_diag_dot_benchmark(benchmark, size, rewrite):
    rng = np.random.default_rng()
    a_size = int(rng.uniform(1, int(0.8 * size)))
    b_size = int(rng.uniform(1, int(0.8 * (size - a_size))))
    c_size = size - a_size - b_size

    a = tensor("a", shape=(a_size, a_size))
    b = tensor("b", shape=(b_size, b_size))
    c = tensor("c", shape=(c_size, c_size))
    d = tensor("d", shape=(size,))

    x = block_diag(a, b, c)
    out = x @ d

    mode = get_default_mode()
    if not rewrite:
        mode = mode.excluding("local_block_diag_dot_to_dot_block_diag")
    fn = function([a, b, c, d], out, mode=mode, trust_input=True)

    a_val = rng.normal(size=a.type.shape).astype(a.type.dtype)
    b_val = rng.normal(size=b.type.shape).astype(b.type.dtype)
    c_val = rng.normal(size=c.type.shape).astype(c.type.dtype)
    d_val = rng.normal(size=d.type.shape).astype(d.type.dtype)

    benchmark(fn, a_val, b_val, c_val, d_val)
