import numpy as np
import pytest
from xarray import DataArray

from pytensor.xtensor import xtensor
from tests.xtensor.util import xr_assert_allclose, xr_function


@pytest.mark.parametrize(
    "indices",
    [
        (0,),
        (slice(1, None),),
        (slice(None, -1),),
        (slice(None, None, -1),),
        (0, slice(None), -1, slice(1, None)),
        (..., 0, -1),
        (0, ..., -1),
        (0, -1, ...),
    ],
)
@pytest.mark.parametrize("labeled", (False, True), ids=["unlabeled", "labeled"])
def test_basic_indexing(labeled, indices):
    if ... in indices and labeled:
        pytest.skip("Ellipsis not supported with labeled indexing")

    dims = ("a", "b", "c", "d")
    x = xtensor(dims=dims, shape=(2, 3, 5, 7))

    if labeled:
        shufled_dims = tuple(np.random.permutation(dims))
        indices = dict(zip(shufled_dims, indices, strict=False))
    out = x[indices]

    fn = xr_function([x], out)
    x_test_values = np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(
        x.type.shape
    )
    x_test = DataArray(x_test_values, dims=x.type.dims)
    res = fn(x_test)
    expected_res = x_test[indices]
    xr_assert_allclose(res, expected_res)
