import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

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
