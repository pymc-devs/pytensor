import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

import numpy as np
import xarray as xr

from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_arange_like, xr_assert_allclose, xr_function


@pytest.mark.parametrize(
    "dim", [..., None, "a", ("c", "a")], ids=["Ellipsis", "None", "a", "(a, c)"]
)
@pytest.mark.parametrize(
    "method",
    ["sum", "prod", "all", "any", "max", "min", "mean", "cumsum", "cumprod"],
)
def test_reduction(method, dim):
    x = xtensor("x", dims=("a", "b", "c"), shape=(3, 5, 7))
    out = getattr(x, method)(dim=dim)

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)

    xr_assert_allclose(
        fn(x_test),
        getattr(x_test, method)(dim=dim),
    )


@pytest.mark.parametrize(
    "dim", [..., None, "a", ("c", "a")], ids=["Ellipsis", "None", "a", "(a, c)"]
)
@pytest.mark.parametrize("method", ["std", "var"])
def test_std_var(method, dim):
    x = xtensor("x", dims=("a", "b", "c"), shape=(3, 5, 7))
    out = [
        getattr(x, method)(dim=dim),
        getattr(x, method)(dim=dim, ddof=2),
    ]

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)
    results = fn(x_test)

    xr_assert_allclose(
        results[0],
        getattr(x_test, method)(dim=dim),
    )

    xr_assert_allclose(
        results[1],
        getattr(x_test, method)(dim=dim, ddof=2),
    )


@pytest.mark.parametrize("signed", [True, False])
def test_discrete_reduction_upcasting(signed):
    # Test that sum, prod reductions on discrete inputs are upcast to prevent overflow
    # This is also a regression test for lower_xtensor, which would raise by returning a different dtype
    in_dtype = "int8" if signed else "uint8"
    out_dtype = "int64" if signed else "uint64"
    test_val = 127 if signed else 255  # max value allowed by in_dtype
    x = xtensor("x", dtype=in_dtype, dims=("a",), shape=(2,))
    x_val = xr.DataArray(np.array([test_val, test_val], dtype=in_dtype), dims="a")
    assert x_val.dtype == in_dtype

    # sum
    out = x.sum()
    assert out.dtype == out_dtype
    fn = xr_function([x], out)
    res = fn(x_val)
    assert res == test_val * 2
    xr_assert_allclose(res, x_val.sum())

    # prod
    out = x.prod()
    assert out.dtype == out_dtype
    fn = xr_function([x], out)
    res = fn(x_val)
    assert res == test_val**2
    xr_assert_allclose(res, x_val.prod())

    # cumsum
    out = x.cumsum()
    assert out.dtype == out_dtype
    fn = xr_function([x], out)
    res = fn(x_val)
    np.testing.assert_allclose(res, [test_val, test_val * 2])
    xr_assert_allclose(res, x_val.cumsum())

    # cumprod
    out = x.cumprod()
    assert out.dtype == out_dtype
    fn = xr_function([x], out)
    res = fn(x_val)
    np.testing.assert_allclose(res, [test_val, test_val**2])
    xr_assert_allclose(res, x_val.cumprod())
