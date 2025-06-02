# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")  #

import numpy as np
from xarray import DataArray

from pytensor import function
from pytensor.xtensor.basic import rename
from pytensor.xtensor.math import add, exp
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_assert_allclose, xr_function


def test_dimension_alignment():
    x = xtensor("x", dims=("city", "country", "planet"), shape=(2, 3, 4))
    y = xtensor(
        "y",
        dims=("galaxy", "country", "city"),
        shape=(5, 3, 2),
    )
    z = xtensor("z", dims=("universe",), shape=(1,))
    out = add(x, y, z)
    assert out.type.dims == ("city", "country", "planet", "galaxy", "universe")

    fn = function([x, y, z], out)

    rng = np.random.default_rng(41)
    test_x, test_y, test_z = (
        DataArray(rng.normal(size=inp.type.shape), dims=inp.type.dims)
        for inp in [x, y, z]
    )
    np.testing.assert_allclose(
        fn(test_x.values, test_y.values, test_z.values),
        (test_x + test_y + test_z).values,
    )


def test_renamed_dimension_alignment():
    x = xtensor("x", dims=("a", "b1", "b2"), shape=(2, 3, 3))
    y = rename(x, b1="b2", b2="b1")
    z = rename(x, b2="b3")
    assert y.type.dims == ("a", "b2", "b1")
    assert z.type.dims == ("a", "b1", "b3")

    out1 = add(x, x)  # self addition
    assert out1.type.dims == ("a", "b1", "b2")
    out2 = add(x, y)  # transposed addition
    assert out2.type.dims == ("a", "b1", "b2")
    out3 = add(x, z)  # outer addition
    assert out3.type.dims == ("a", "b1", "b2", "b3")

    fn = xr_function([x], [out1, out2, out3])
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )
    results = fn(x_test)
    expected_results = [
        x_test + x_test,
        x_test + x_test.rename(b1="b2", b2="b1"),
        x_test + x_test.rename(b2="b3"),
    ]
    for result, expected_result in zip(results, expected_results):
        xr_assert_allclose(result, expected_result)


def test_chained_operations():
    x = xtensor("x", dims=("city",), shape=(None,))
    y = xtensor("y", dims=("country",), shape=(4,))
    z = add(exp(x), exp(y))
    assert z.type.dims == ("city", "country")
    assert z.type.shape == (None, 4)

    fn = function([x, y], z)

    x_test = DataArray(np.zeros(3), dims="city")
    y_test = DataArray(np.ones(4), dims="country")

    np.testing.assert_allclose(
        fn(x_test.values, y_test.values),
        (np.exp(x_test) + np.exp(y_test)).values,
    )


def test_multiple_constant():
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    out = exp(x * 2) + 2

    fn = function([x], out)

    x_test = np.zeros((2, 3), dtype=x.type.dtype)
    res = fn(x_test)
    expected_res = np.exp(x_test * 2) + 2
    np.testing.assert_allclose(res, expected_res)
