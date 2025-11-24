import pytest


pytest.importorskip("xarray")

import inspect

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp
from xarray import DataArray

import pytensor.scalar as ps
import pytensor.xtensor.math as pxm
from pytensor import function
from pytensor.scalar import ScalarOp
from pytensor.xtensor.basic import rename
from pytensor.xtensor.math import add, exp, logsumexp
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


@pytest.mark.parametrize(
    ["shape", "dims", "axis"],
    [
        ((3, 4), ("a", "b"), None),
        ((3, 4), "a", 0),
        ((3, 4), "b", 1),
    ],
)
def test_logsumexp(shape, dims, axis):
    scipy_inp = np.zeros(shape)
    scipy_out = scipy_logsumexp(scipy_inp, axis=axis)

    pytensor_inp = DataArray(scipy_inp, dims=("a", "b"))
    f = function([], logsumexp(pytensor_inp, dim=dims))
    pytensor_out = f()

    np.testing.assert_array_almost_equal(
        pytensor_out,
        scipy_out,
    )


def test_dot():
    """Test basic dot product operations."""
    # Test matrix-vector dot product (with multiple-letter dim names)
    x = xtensor("x", dims=("aa", "bb"), shape=(2, 3))
    y = xtensor("y", dims=("bb",), shape=(3,))
    z = x.dot(y)
    fn = xr_function([x, y], z)

    x_test = DataArray(np.ones((2, 3)), dims=("aa", "bb"))
    y_test = DataArray(np.ones(3), dims=("bb",))
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test)
    xr_assert_allclose(z_test, expected)

    # Test matrix-vector dot product with ellipsis
    z = x.dot(y, dim=...)
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim=...)
    xr_assert_allclose(z_test, expected)

    # Test matrix-matrix dot product
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = xtensor("y", dims=("b", "c"), shape=(3, 4))
    z = x.dot(y)
    fn = xr_function([x, y], z)

    x_test = DataArray(np.add.outer(np.arange(2.0), np.arange(3.0)), dims=("a", "b"))
    y_test = DataArray(np.add.outer(np.arange(3.0), np.arange(4.0)), dims=("b", "c"))
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test)
    xr_assert_allclose(z_test, expected)

    # Test matrix-matrix dot product with string dim
    z = x.dot(y, dim="b")
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim="b")
    xr_assert_allclose(z_test, expected)

    # Test matrix-matrix dot product with list of dims
    z = x.dot(y, dim=["b"])
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim=["b"])
    xr_assert_allclose(z_test, expected)

    # Test matrix-matrix dot product with ellipsis
    z = x.dot(y, dim=...)
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim=...)
    xr_assert_allclose(z_test, expected)

    # Test a case where there are two dimensions to sum over
    x = xtensor("x", dims=("a", "b", "c"), shape=(2, 3, 4))
    y = xtensor("y", dims=("b", "c", "d"), shape=(3, 4, 5))
    z = x.dot(y)
    fn = xr_function([x, y], z)

    x_test = DataArray(np.arange(24.0).reshape(2, 3, 4), dims=("a", "b", "c"))
    y_test = DataArray(np.arange(60.0).reshape(3, 4, 5), dims=("b", "c", "d"))
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test)
    xr_assert_allclose(z_test, expected)

    # Same but with explicit dimensions
    z = x.dot(y, dim=["b", "c"])
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim=["b", "c"])
    xr_assert_allclose(z_test, expected)

    # Same but with ellipses
    z = x.dot(y, dim=...)
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test, dim=...)
    xr_assert_allclose(z_test, expected)

    # Dot product with sum
    x_test = DataArray(np.arange(24.0).reshape(2, 3, 4), dims=("a", "b", "c"))
    y_test = DataArray(np.arange(60.0).reshape(3, 4, 5), dims=("b", "c", "d"))
    expected = x_test.dot(y_test, dim=("a", "b", "c"))

    x = xtensor("x", dims=("a", "b", "c"), shape=(2, 3, 4))
    y = xtensor("y", dims=("b", "c", "d"), shape=(3, 4, 5))
    z = x.dot(y, dim=("a", "b", "c"))
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    xr_assert_allclose(z_test, expected)

    # Dot product with sum in the middle
    x_test = DataArray(np.arange(120.0).reshape(2, 3, 4, 5), dims=("a", "b", "c", "d"))
    y_test = DataArray(np.arange(360.0).reshape(3, 4, 5, 6), dims=("b", "c", "d", "e"))
    expected = x_test.dot(y_test, dim=("b", "d"))
    x = xtensor("x", dims=("a", "b", "c", "d"), shape=(2, 3, 4, 5))
    y = xtensor("y", dims=("b", "c", "d", "e"), shape=(3, 4, 5, 6))
    z = x.dot(y, dim=("b", "d"))
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    xr_assert_allclose(z_test, expected)

    # Same but with first two dims
    expected = x_test.dot(y_test, dim=["a", "b"])
    z = x.dot(y, dim=["a", "b"])
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    xr_assert_allclose(z_test, expected)

    # Same but with last two
    expected = x_test.dot(y_test, dim=["d", "e"])
    z = x.dot(y, dim=["d", "e"])
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    xr_assert_allclose(z_test, expected)

    # Same but with every other dim
    expected = x_test.dot(y_test, dim=["a", "c", "e"])
    z = x.dot(y, dim=["a", "c", "e"])
    fn = xr_function([x, y], z)
    z_test = fn(x_test, y_test)
    xr_assert_allclose(z_test, expected)

    # Test symbolic shapes
    x = xtensor("x", dims=("a", "b"), shape=(None, 3))  # First dimension is symbolic
    y = xtensor("y", dims=("b", "c"), shape=(3, None))  # Second dimension is symbolic
    z = x.dot(y)
    fn = xr_function([x, y], z)
    x_test = DataArray(np.ones((2, 3)), dims=("a", "b"))
    y_test = DataArray(np.ones((3, 4)), dims=("b", "c"))
    z_test = fn(x_test, y_test)
    expected = x_test.dot(y_test)
    xr_assert_allclose(z_test, expected)


def test_dot_errors():
    # No matching dimensions
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = xtensor("y", dims=("b", "c"), shape=(3, 4))
    with pytest.raises(ValueError, match="Dimension e not found in either input"):
        x.dot(y, dim="e")

    # Concrete dimension size mismatches
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = xtensor("y", dims=("b", "c"), shape=(4, 5))
    with pytest.raises(
        ValueError,
        match="Size of dim 'b' does not match",
    ):
        x.dot(y)

    # Symbolic dimension size mismatches
    x = xtensor("x", dims=("a", "b"), shape=(2, None))
    y = xtensor("y", dims=("b", "c"), shape=(None, 5))
    z = x.dot(y)
    fn = xr_function([x, y], z)
    x_test = DataArray(np.ones((2, 3)), dims=("a", "b"))
    y_test = DataArray(np.ones((4, 5)), dims=("b", "c"))
    # Doesn't fail until the rewrite
    with pytest.raises(
        ValueError,
        match=r"(Input operand 1 has a mismatch in its core dimension 0|incompatible array sizes for np.dot)",
    ):
        fn(x_test, y_test)
