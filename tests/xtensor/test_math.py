# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

import inspect

import numpy as np
from xarray import DataArray

import pytensor.scalar as ps
import pytensor.xtensor.math as pxm
from pytensor import function
from pytensor.scalar import ScalarOp
from pytensor.xtensor.basic import rename
from pytensor.xtensor.math import add, exp
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_arange_like, xr_assert_allclose, xr_function


def test_all_scalar_ops_are_wrapped():
    # This ignores wrapper functions
    pxm_members = {name for name, _ in inspect.getmembers(pxm)}
    for name, op in inspect.getmembers(ps):
        if name in {
            "complex_from_polar",
            "inclosedrange",
            "inopenrange",
            "round_half_away_from_zero",
            "round_half_to_even",
            "scalar_abs",
            "scalar_maximum",
            "scalar_minimum",
        } or name.startswith("convert_to_"):
            # These are not regular numpy functions or are unusual alias
            continue
        if isinstance(op, ScalarOp) and name not in pxm_members:
            raise NotImplementedError(f"ScalarOp {name} not wrapped in xtensor.math")


def test_scalar_case():
    x = xtensor("x", dims=(), shape=())
    y = xtensor("y", dims=(), shape=())
    out = add(x, y)

    fn = function([x, y], out)

    x_test = DataArray(2.0, dims=())
    y_test = DataArray(3.0, dims=())
    np.testing.assert_allclose(fn(x_test.values, y_test.values), 5.0)


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


def test_cast():
    x = xtensor("x", shape=(2, 3), dims=("a", "b"), dtype="float32")
    yf64 = x.astype("float64")
    yi16 = x.astype("int16")
    ybool = x.astype("bool")

    fn = xr_function([x], [yf64, yi16, ybool])
    x_test = xr_arange_like(x)
    res_f64, res_i16, res_bool = fn(x_test)
    xr_assert_allclose(res_f64, x_test.astype("float64"))
    xr_assert_allclose(res_i16, x_test.astype("int16"))
    xr_assert_allclose(res_bool, x_test.astype("bool"))

    yc64 = x.astype("complex64")
    with pytest.raises(TypeError, match="Casting from complex to real is ambiguous"):
        yc64.astype("float64")
