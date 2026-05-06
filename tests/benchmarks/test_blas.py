import numpy as np
import pytest

from pytensor import In, function
from pytensor.tensor import dot, empty, matrix, outer, scalar, tensor, vector
from pytensor.tensor.blas.blas_c import CGemv


@pytest.mark.parametrize("dtype", ("float64", "float32", "mixed"))
def test_mat_vec_dot_benchmark_numba(dtype, benchmark):
    A = tensor("A", shape=(512, 512), dtype="float64" if dtype == "mixed" else dtype)
    x = tensor("x", shape=(512,), dtype="float32" if dtype == "mixed" else dtype)
    out = dot(A, x)

    fn = function([A, x], out, mode="NUMBA", trust_input=True)

    rng = np.random.default_rng(948)
    A_test = rng.standard_normal(size=A.type.shape).astype(A.type.dtype)
    x_test = rng.standard_normal(size=x.type.shape).astype(x.type.dtype)
    np.testing.assert_allclose(fn(A_test, x_test), np.dot(A_test, x_test), atol=1e-4)
    benchmark(fn, A_test, x_test)


def _test_ger_benchmark(mode, n, inplace, benchmark):
    alpha = scalar("alpha")
    x = vector("x")
    y = vector("y")
    A = matrix("A")

    out = alpha * outer(x, y) + A

    fn = function(
        [alpha, x, y, In(A, mutable=inplace)],
        out,
        mode=mode,
        trust_input=True,
    )

    rng = np.random.default_rng([2274, n])
    alpha_test = rng.normal(size=())
    x_test = rng.normal(size=(n,))
    y_test = rng.normal(size=(n,))
    A_test = rng.normal(size=(n, n))

    benchmark(fn, alpha_test, x_test, y_test, A_test)


@pytest.mark.parametrize("inplace", (True, False), ids=["inplace", "no_inplace"])
@pytest.mark.parametrize("n", [2**7, 2**9, 2**13])
def test_ger_benchmark_c(n, inplace, benchmark):
    _test_ger_benchmark("CVM", n, inplace, benchmark)


@pytest.mark.parametrize("inplace", (True, False), ids=["inplace", "no_inplace"])
@pytest.mark.parametrize("n", [2**7, 2**9, 2**13])
def test_ger_benchmark_numba(n, inplace, benchmark):
    _test_ger_benchmark("NUMBA", n, inplace, benchmark)


def test_cgemv_vector_dot_benchmark(benchmark):
    n = 400_000
    a = vector("A", shape=(n,))
    b = vector("x", shape=(n,))

    out = CGemv(inplace=True)(
        empty((1,)),
        1.0,
        a[None],
        b,
        0.0,
    )
    fn = function([a, b], out, accept_inplace=True, mode="CVM", trust_input=True)

    rng = np.random.default_rng(430)
    test_a = rng.normal(size=n)
    test_b = rng.normal(size=n)

    np.testing.assert_allclose(
        fn(test_a, test_b),
        np.dot(test_a, test_b),
    )

    benchmark(fn, test_a, test_b)


@pytest.mark.parametrize(
    "neg_stride1", (True, False), ids=["neg_stride1", "pos_stride1"]
)
@pytest.mark.parametrize(
    "neg_stride0", (True, False), ids=["neg_stride0", "pos_stride0"]
)
@pytest.mark.parametrize("F_layout", (True, False), ids=["F_layout", "C_layout"])
def test_cgemv_negative_strides_benchmark(
    neg_stride0, neg_stride1, F_layout, benchmark
):
    A = matrix("A", shape=(512, 512))
    x = vector("x", shape=(A.type.shape[-1],))
    y = vector("y", shape=(A.type.shape[0],))

    out = CGemv(inplace=False)(
        y,
        1.0,
        A,
        x,
        1.0,
    )
    fn = function([A, x, y], out, trust_input=True, mode="CVM")

    rng = np.random.default_rng(430)
    test_A = rng.normal(size=A.type.shape)
    test_x = rng.normal(size=x.type.shape)
    test_y = rng.normal(size=y.type.shape)

    if F_layout:
        test_A = test_A.T
    if neg_stride0:
        test_A = test_A[::-1]
    if neg_stride1:
        test_A = test_A[:, ::-1]
    assert (test_A.strides[0] < 0) == neg_stride0
    assert (test_A.strides[1] < 0) == neg_stride1

    # Check result is correct by using a copy of A with positive strides
    res = fn(test_A, test_x, test_y)
    np.testing.assert_allclose(res, fn(test_A.copy(), test_x, test_y))

    benchmark(fn, test_A, test_x, test_y)
