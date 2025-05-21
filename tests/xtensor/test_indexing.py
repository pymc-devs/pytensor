import numpy as np
import pytest
from xarray import DataArray

from pytensor.tensor import tensor
from pytensor.xtensor import xtensor
from tests.xtensor.util import xr_arange_like, xr_assert_allclose, xr_function


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


def test_single_vector_indexing_on_existing_dim():
    x = xtensor(dims=("a", "b"), shape=(3, 5))
    idx = tensor("idx", dtype=int, shape=(4,))
    xidx = xtensor("idx", dtype=int, shape=(4,), dims=("a",))

    x_test = xr_arange_like(x)
    idx_test = np.array([0, 1, 0, 2], dtype=int)
    xidx_test = DataArray(idx_test, dims=("a",))

    # Three equivalent ways of indexing a->a
    y = x[idx]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[idx_test]
    xr_assert_allclose(res, expected_res)

    y = x[(("a", idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[(("a", idx_test),)]
    xr_assert_allclose(res, expected_res)

    y = x[xidx]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test]
    xr_assert_allclose(res, expected_res)


def test_single_vector_indexing_on_new_dim():
    x = xtensor(dims=("a", "b"), shape=(3, 5))
    idx = tensor("idx", dtype=int, shape=(4,))
    xidx = xtensor("idx", dtype=int, shape=(4,), dims=("a",))

    x_test = xr_arange_like(x)
    idx_test = np.array([0, 1, 0, 2], dtype=int)
    xidx_test = DataArray(idx_test, dims=("a",))

    # Two equvilant ways of indexing a->new_a
    y = x[(("new_a", idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[(("new_a", idx_test),)]
    xr_assert_allclose(res, expected_res)

    y = x[xidx.rename(a="new_a")]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test.rename(a="new_a")]
    xr_assert_allclose(res, expected_res)


def test_single_vector_indexing_interacting_with_exisiting_dim():
    x = xtensor(dims=("a", "b"), shape=(3, 5))
    idx = tensor("idx", dtype=int, shape=(4,))
    xidx = xtensor("idx", dtype=int, shape=(4,), dims=("a",))

    x_test = xr_arange_like(x)
    idx_test = np.array([0, 1, 0, 2], dtype=int)
    xidx_test = DataArray(idx_test, dims=("a",))

    # Two equivalent ways of indexing a->b
    # By labeling the index on a, as "b", we cause pointwise indexing between the two dimensions.
    y = x[(("b", idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[("b", idx_test), 1:]
    xr_assert_allclose(res, expected_res)

    y = x[xidx.rename(a="b")]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test.rename(a="b"), 1:]
    xr_assert_allclose(res, expected_res)
