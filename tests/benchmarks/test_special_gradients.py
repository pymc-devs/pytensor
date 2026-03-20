import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import gammaincc, grad, hyp2f1, scalars, vector
from tests.unittest_tools import verify_grad


def _test_gammaincc_ddk_benchmark(mode, benchmark):
    rng = np.random.default_rng(1)
    k = vector("k")
    x = vector("x")

    out = gammaincc(k, x)
    grad_fn = function(
        [k, x],
        grad(out.sum(), wrt=[k]),
        mode=mode,
        trust_input=True,
    )
    vals = [
        # Values that hit the second branch of the gradient
        np.full((1000,), 3.2, dtype=k.dtype),
        np.full((1000,), 0.01, dtype=x.dtype),
    ]

    verify_grad(gammaincc, vals, rng=rng)
    grad_fn(*vals)  # JIT compile for JIT backends
    benchmark(grad_fn, *vals)


def test_gammaincc_ddk_benchmark_c(benchmark):
    _test_gammaincc_ddk_benchmark(mode="CVM", benchmark=benchmark)


def test_gammaincc_ddk_benchmark_numba(benchmark):
    _test_gammaincc_ddk_benchmark(mode="NUMBA", benchmark=benchmark)


_hyp2f1_few_iters_case = (
    2.0,
    1.0,
    2.0,
    0.4,
    0.4617734323582945,
    0.851376039609984,
    -0.4617734323582945,
    2.777777777777778,
)

_hyp2f1_many_iters_case = (
    3.70975,
    1.0,
    2.70975,
    0.999696,
    29369830.002773938200417693317785,
    36347869.41885337,
    -30843032.10697079073015067426929807,
    26278034019.28811,
)


def _test_hyp2f1_grad_benchmark(mode, case, wrt, benchmark):
    a1, a2, b1, z = scalars("a1", "a2", "b1", "z")
    hyp2f1_out = hyp2f1(a1, a2, b1, z)
    hyp2f1_grad = grad(hyp2f1_out, wrt=a1 if wrt == "a" else [a1, a2, b1, z])
    f_grad = function([a1, a2, b1, z], hyp2f1_grad, mode=mode, trust_input=True)

    (test_a1, test_a2, test_b1, test_z, *expected_dds) = case
    test_a1 = np.array(test_a1, dtype=a1.dtype)
    test_a2 = np.array(test_a2, dtype=a2.dtype)
    test_b1 = np.array(test_b1, dtype=b1.dtype)
    test_z = np.array(test_z, dtype=z.dtype)

    result = benchmark(f_grad, test_a1, test_a2, test_b1, test_z)

    expected_result = expected_dds[0] if wrt == "a" else np.array(expected_dds)
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-9,
    )


@pytest.mark.parametrize("case", (_hyp2f1_few_iters_case, _hyp2f1_many_iters_case))
@pytest.mark.parametrize("wrt", ("a", "all"))
def test_hyp2f1_grad_benchmark_c(case, wrt, benchmark):
    _test_hyp2f1_grad_benchmark("CVM", case, wrt, benchmark)


@pytest.mark.parametrize("case", (_hyp2f1_few_iters_case, _hyp2f1_many_iters_case))
@pytest.mark.parametrize("wrt", ("a", "all"))
def test_hyp2f1_grad_benchmark_numba(case, wrt, benchmark):
    _test_hyp2f1_grad_benchmark("NUMBA", case, wrt, benchmark)
