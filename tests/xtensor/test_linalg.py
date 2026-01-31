import pytest


pytest.importorskip("xarray")
pytest.importorskip("xarray_einstats")
pytestmark = pytest.mark.filterwarnings("error")

import numpy as np
from xarray import DataArray
from xarray_einstats.linalg import (
    cholesky as xr_cholesky,
)
from xarray_einstats.linalg import (
    solve as xr_solve,
)

from pytensor.xtensor.linalg import cholesky, solve
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_assert_allclose, xr_function


def test_cholesky():
    x = xtensor("x", dims=("a", "batch", "b"), shape=(4, 3, 4))
    y = cholesky(x, dims=["b", "a"])
    assert y.type.dims == ("batch", "b", "a")
    assert y.type.shape == (3, 4, 4)

    fn = xr_function([x], y)
    rng = np.random.default_rng(25)
    x_ = rng.random(size=(3, 4, 4))
    x_ = x_ @ x_.mT
    x_test = DataArray(x_.transpose(1, 0, 2), dims=x.type.dims)
    xr_assert_allclose(
        fn(x_test),
        xr_cholesky(x_test, dims=["b", "a"]),
    )


def test_solve_vector_b():
    a = xtensor("a", dims=("city", "country", "galaxy"), shape=(None, 4, 1))
    b = xtensor("b", dims=("city", "planet"), shape=(None, 2))
    x = solve(a, b, dims=["country", "city"])
    assert x.type.dims == ("galaxy", "planet", "country")
    # Core Solve doesn't make use of the fact A must be square in the static shape
    assert x.type.shape == (1, 2, None)

    fn = xr_function([a, b], x)

    rng = np.random.default_rng(25)
    a_test = DataArray(rng.random(size=(4, 4, 1)), dims=a.type.dims)
    b_test = DataArray(rng.random(size=(4, 2)), dims=b.type.dims)

    xr_assert_allclose(
        fn(a_test, b_test),
        xr_solve(a_test, b_test, dims=["country", "city"]),
    )


def test_solve_matrix_b():
    a = xtensor("a", dims=("city", "country", "galaxy"), shape=(None, 4, 1))
    b = xtensor("b", dims=("district", "city", "planet"), shape=(5, None, 2))
    x = solve(a, b, dims=["country", "city", "district"])
    assert x.type.dims == ("galaxy", "planet", "country", "district")
    # Core Solve doesn't make use of the fact A must be square in the static shape
    assert x.type.shape == (1, 2, None, 5)

    fn = xr_function([a, b], x)

    rng = np.random.default_rng(25)
    a_test = DataArray(rng.random(size=(4, 4, 1)), dims=a.type.dims)
    b_test = DataArray(rng.random(size=(5, 4, 2)), dims=b.type.dims)

    xr_assert_allclose(
        fn(a_test, b_test),
        xr_solve(a_test, b_test, dims=["country", "city", "district"]),
    )
