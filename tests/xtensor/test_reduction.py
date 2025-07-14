# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

from pytensor.xtensor.type import DimVariable, dim, xtensor
from tests.xtensor.util import xr_arange_like, xr_assert_allclose, xr_function


a = dim("a", size=3)
b = dim("b", size=5)
c = dim("c", size=7)


@pytest.mark.parametrize(
    "reduce_dim", [..., None, a, (c, a)], ids=["Ellipsis", "None", "a", "(a, c)"]
)
@pytest.mark.parametrize(
    "method", ["sum", "prod", "all", "any", "max", "min", "cumsum", "cumprod"][2:]
)
def test_reduction(method, reduce_dim):
    x = xtensor("x", dims=(a, b, c))
    out = getattr(x, method)(dim=reduce_dim)

    out.dprint()

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)

    if reduce_dim == ...:
        reduce_dim_name = ...
    elif reduce_dim is None:
        reduce_dim_name = None
    elif isinstance(reduce_dim, DimVariable):
        reduce_dim_name = reduce_dim.type.name
    elif isinstance(reduce_dim, tuple | list):
        reduce_dim_name = tuple(dim.type.name for dim in reduce_dim)

    xr_assert_allclose(
        fn(x_test),
        getattr(x_test, method)(dim=reduce_dim_name),
    )
