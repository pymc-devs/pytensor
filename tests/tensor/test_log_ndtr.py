import numpy as np
import pytest
from scipy import special

import pytensor.tensor as pt
from pytensor import function
from tests.unittest_tools import verify_grad


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["FAST_COMPILE", "CVM", "NUMBA"])
def test_log_ndtr(dtype, mode):
    x = pt.vector("x", dtype=dtype)
    log_cdf = pt.log_ndtr(x)
    fn = function([x], [log_cdf, pt.grad(log_cdf.sum(), x)], mode=mode)
    scalar_type = np.dtype(dtype).type
    cutoff = scalar_type(-8)
    values = np.array(
        [
            -1e10,
            -40,
            -9,
            np.nextafter(cutoff, scalar_type(-np.inf)),
            cutoff,
            np.nextafter(cutoff, scalar_type(np.inf)),
            -3,
            0,
            3,
            10,
        ],
        dtype=dtype,
    )
    expected_grad = np.sqrt(2 / np.pi) / special.erfcx(
        -values.astype("float64") / np.sqrt(2)
    )
    actual, actual_grad = fn(values)
    rtol = 5e-6 if dtype == "float32" else 2e-13
    np.testing.assert_allclose(actual, special.log_ndtr(values), rtol=rtol, strict=True)
    np.testing.assert_allclose(
        actual_grad, expected_grad.astype(dtype), rtol=rtol, strict=True
    )
    if mode == "NUMBA":
        assert len(fn.maker.fgraph.apply_nodes) == 1


def test_log_ndtr_higher_derivatives():
    values = np.array([-12.0, -9.0, -7.0, 0.0, 3.0])
    verify_grad(pt.log_ndtr, [values], mode="NUMBA")
    verify_grad(lambda x: pt.grad(pt.log_ndtr(x).sum(), x), [values], mode="NUMBA")
