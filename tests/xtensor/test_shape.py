# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

import re
from itertools import chain, combinations

import numpy as np
import pytest
from xarray import DataArray
from xarray import concat as xr_concat

from pytensor.xtensor.shape import (
    concat,
    stack,
    transpose,
    unstack,
)
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import (
    xr_arange_like,
    xr_assert_allclose,
    xr_function,
    xr_random_like,
)


pytest.importorskip("xarray")


def powerset(iterable, min_group_size=0):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_group_size, len(s) + 1)
    )


def test_transpose():
    a, b, c, d, e = "abcde"

    x = xtensor("x", dims=(a, b, c, d, e), shape=(2, 3, 5, 7, 11))
    permutations = [
        (a, b, c, d, e),  # identity
        (e, d, c, b, a),  # full tranpose
        (),  # eqivalent to full transpose
        (a, b, c, e, d),  # swap last two dims
        (..., d, c),  # equivalent to (a, b, e, d, c)
        (b, a, ..., e, d),  # equivalent to (b, a, c, d, e)
        (c, a, ...),  # equivalent to (c, a, b, d, e)
    ]
    outs = [transpose(x, *perm) for perm in permutations]

    fn = xr_function([x], outs)
    x_test = xr_arange_like(x)
    res = fn(x_test)
    expected_res = [x_test.transpose(*perm) for perm in permutations]
    for outs_i, res_i, expected_res_i in zip(outs, res, expected_res):
        xr_assert_allclose(res_i, expected_res_i)


def test_xtensor_variable_transpose():
    """Test the transpose() method of XTensorVariable."""
    x = xtensor("x", dims=("a", "b", "c"), shape=(2, 3, 4))

    # Test basic transpose
    out = x.transpose()
    fn = xr_function([x], out)
    x_test = xr_arange_like(x)
    xr_assert_allclose(fn(x_test), x_test.transpose())

    # Test transpose with specific dimensions
    out = x.transpose("c", "a", "b")
    fn = xr_function([x], out)
    xr_assert_allclose(fn(x_test), x_test.transpose("c", "a", "b"))

    # Test transpose with ellipsis
    out = x.transpose("c", ...)
    fn = xr_function([x], out)
    xr_assert_allclose(fn(x_test), x_test.transpose("c", ...))

    # Test error cases
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Dimensions {'d'} do not exist. Expected one or more of: ('a', 'b', 'c')"
        ),
    ):
        x.transpose("d")

    with pytest.raises(
        ValueError,
        match=re.escape("Ellipsis (...) can only appear once in the dimensions"),
    ):
        x.transpose("a", ..., "b", ...)

    # Test missing_dims parameter
    # Test ignore
    out = x.transpose("c", ..., "d", missing_dims="ignore")
    fn = xr_function([x], out)
    xr_assert_allclose(fn(x_test), x_test.transpose("c", ...))

    # Test warn
    with pytest.warns(UserWarning, match="Dimensions {'d'} do not exist"):
        out = x.transpose("c", ..., "d", missing_dims="warn")
    fn = xr_function([x], out)
    xr_assert_allclose(fn(x_test), x_test.transpose("c", ...))


def test_xtensor_variable_T():
    """Test the T property of XTensorVariable."""
    # Test T property with 3D tensor
    x = xtensor("x", dims=("a", "b", "c"), shape=(2, 3, 4))
    out = x.T

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)
    xr_assert_allclose(fn(x_test), x_test.T)


def test_stack():
    dims = ("a", "b", "c", "d")
    x = xtensor("x", dims=dims, shape=(2, 3, 5, 7))
    outs = [
        stack(x, new_dim=dims_to_stack)
        for dims_to_stack in powerset(dims, min_group_size=2)
    ]

    fn = xr_function([x], outs)
    x_test = xr_arange_like(x)
    res = fn(x_test)

    expected_res = [
        x_test.stack(new_dim=dims_to_stack)
        for dims_to_stack in powerset(dims, min_group_size=2)
    ]
    for outs_i, res_i, expected_res_i in zip(outs, res, expected_res):
        xr_assert_allclose(res_i, expected_res_i)


def test_stack_single_dim():
    x = xtensor("x", dims=("a", "b", "c"), shape=(2, 3, 5))
    out = stack(x, {"d": ["a"]})
    assert out.type.dims == ("b", "c", "d")

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)
    res = fn(x_test)
    expected_res = x_test.stack(d=["a"])
    xr_assert_allclose(res, expected_res)


def test_multiple_stacks():
    x = xtensor("x", dims=("a", "b", "c", "d"), shape=(2, 3, 5, 7))
    out = stack(x, new_dim1=("a", "b"), new_dim2=("c", "d"))

    fn = xr_function([x], [out])
    x_test = xr_arange_like(x)
    res = fn(x_test)
    expected_res = x_test.stack(new_dim1=("a", "b"), new_dim2=("c", "d"))
    xr_assert_allclose(res[0], expected_res)


def test_unstack_constant_size():
    x = xtensor("x", dims=("a", "bc", "d"), shape=(2, 3 * 5, 7))
    y = unstack(x, bc=dict(b=3, c=5))
    assert y.type.dims == ("a", "d", "b", "c")
    assert y.type.shape == (2, 7, 3, 5)

    fn = xr_function([x], y)

    x_test = xr_arange_like(x)
    x_np = x_test.values
    res = fn(x_test)
    expected = (
        DataArray(x_np.reshape(2, 3, 5, 7), dims=("a", "b", "c", "d"))
        .stack(bc=("b", "c"))
        .unstack("bc")
    )
    xr_assert_allclose(res, expected)


def test_unstack_symbolic_size():
    x = xtensor(dims=("a", "b", "c"))
    y = stack(x, bc=("b", "c"))
    y = y / y.sum("bc")
    z = unstack(y, bc={"b": x.sizes["b"], "c": x.sizes["c"]})
    x_test = xr_arange_like(xtensor(dims=x.dims, shape=(2, 3, 5)))
    fn = xr_function([x], z)
    res = fn(x_test)
    expected_res = x_test / x_test.sum(["b", "c"])
    xr_assert_allclose(res, expected_res)


def test_stack_unstack():
    x = xtensor("x", dims=("a", "b", "c", "d"), shape=(2, 3, 5, 7))
    stack_x = stack(x, bd=("b", "d"))
    unstack_x = unstack(stack_x, bd=dict(b=3, d=7))

    x_test = xr_arange_like(x)
    fn = xr_function([x], unstack_x)
    res = fn(x_test)
    expected_res = x_test.transpose("a", "c", "b", "d")
    xr_assert_allclose(res, expected_res)


@pytest.mark.parametrize("dim", ("a", "b", "new"))
def test_concat(dim):
    rng = np.random.default_rng(sum(map(ord, dim)))

    x1 = xtensor("x1", dims=("a", "b"), shape=(2, 3))
    x2 = xtensor("x2", dims=("b", "a"), shape=(3, 2))

    x3_shape0 = 4 if dim == "a" else 2
    x3_shape1 = 5 if dim == "b" else 3
    x3 = xtensor("x3", dims=("a", "b"), shape=(x3_shape0, x3_shape1))

    out = concat([x1, x2, x3], dim=dim)

    fn = xr_function([x1, x2, x3], out)
    x1_test = xr_random_like(x1, rng)
    x2_test = xr_random_like(x2, rng)
    x3_test = xr_random_like(x3, rng)

    res = fn(x1_test, x2_test, x3_test)
    expected_res = xr_concat([x1_test, x2_test, x3_test], dim=dim)
    xr_assert_allclose(res, expected_res)


@pytest.mark.parametrize("dim", ("a", "b", "c", "d", "new"))
def test_concat_with_broadcast(dim):
    rng = np.random.default_rng(sum(map(ord, dim)) + 1)

    x1 = xtensor("x1", dims=("a", "b"), shape=(2, 3))
    x2 = xtensor("x2", dims=("b", "c"), shape=(3, 5))
    x3 = xtensor("x3", dims=("c", "d"), shape=(5, 7))
    x4 = xtensor("x4", dims=(), shape=())

    out = concat([x1, x2, x3, x4], dim=dim)

    fn = xr_function([x1, x2, x3, x4], out)

    x1_test = xr_random_like(x1, rng)
    x2_test = xr_random_like(x2, rng)
    x3_test = xr_random_like(x3, rng)
    x4_test = xr_random_like(x4, rng)
    res = fn(x1_test, x2_test, x3_test, x4_test)
    expected_res = xr_concat([x1_test, x2_test, x3_test, x4_test], dim=dim)
    xr_assert_allclose(res, expected_res)


def test_concat_scalar():
    x1 = xtensor("x1", dims=(), shape=())
    x2 = xtensor("x2", dims=(), shape=())

    out = concat([x1, x2], dim="new_dim")

    fn = xr_function([x1, x2], out)

    x1_test = xr_random_like(x1)
    x2_test = xr_random_like(x2)
    res = fn(x1_test, x2_test)
    expected_res = xr_concat([x1_test, x2_test], dim="new_dim")
    xr_assert_allclose(res, expected_res)


def test_squeeze():
    """Test squeeze."""

    # Single dimension
    x1 = xtensor("x1", dims=("city", "country"), shape=(3, 1))
    y1 = x1.squeeze("country")
    fn1 = xr_function([x1], y1)
    x1_test = xr_arange_like(x1)
    xr_assert_allclose(fn1(x1_test), x1_test.squeeze("country"))

    # Multiple dimensions and order independence
    x2 = xtensor("x2", dims=("a", "b", "c", "d"), shape=(2, 1, 1, 3))
    y2a = x2.squeeze(["b", "c"])
    y2b = x2.squeeze(["c", "b"])  # Test order independence
    y2c = x2.squeeze(["b", "b"])  # Test redundant dimensions
    y2d = x2.squeeze([])  # Test empty list (no-op)
    fn2a = xr_function([x2], y2a)
    fn2b = xr_function([x2], y2b)
    fn2c = xr_function([x2], y2c)
    fn2d = xr_function([x2], y2d)
    x2_test = xr_arange_like(x2)
    xr_assert_allclose(fn2a(x2_test), x2_test.squeeze(["b", "c"]))
    xr_assert_allclose(fn2b(x2_test), x2_test.squeeze(["c", "b"]))
    xr_assert_allclose(fn2c(x2_test), x2_test.squeeze(["b", "b"]))
    xr_assert_allclose(fn2d(x2_test), x2_test)

    # Unknown shapes
    x3 = xtensor("x3", dims=("a", "b", "c"))  # shape unknown
    y3 = x3.squeeze("b")
    x3_test = xr_arange_like(xtensor(dims=x3.dims, shape=(2, 1, 3)))
    fn3 = xr_function([x3], y3)
    xr_assert_allclose(fn3(x3_test), x3_test.squeeze("b"))

    # Mixed known + unknown shapes
    x4 = xtensor("x4", dims=("a", "b", "c"), shape=(None, 1, 3))
    y4 = x4.squeeze("b")
    x4_test = xr_arange_like(xtensor(dims=x4.dims, shape=(4, 1, 3)))
    fn4 = xr_function([x4], y4)
    xr_assert_allclose(fn4(x4_test), x4_test.squeeze("b"))

    # Test axis parameter
    x5 = xtensor("x5", dims=("a", "b", "c"), shape=(2, 1, 3))
    y5 = x5.squeeze(axis=1)  # squeeze dimension at index 1 (b)
    fn5 = xr_function([x5], y5)
    x5_test = xr_arange_like(x5)
    xr_assert_allclose(fn5(x5_test), x5_test.squeeze(axis=1))

    # Test axis parameter with negative index
    y5 = x5.squeeze(axis=-1)  # squeeze dimension at index -2 (b)
    fn5 = xr_function([x5], y5)
    x5_test = xr_arange_like(x5)
    xr_assert_allclose(fn5(x5_test), x5_test.squeeze(axis=-2))

    # Test axis parameter with sequence of ints
    y6 = x2.squeeze(axis=[1, 2])
    fn6 = xr_function([x2], y6)
    x2_test = xr_arange_like(x2)
    xr_assert_allclose(fn6(x2_test), x2_test.squeeze(axis=[1, 2]))

    # Test drop parameter warning
    x7 = xtensor("x7", dims=("a", "b"), shape=(2, 1))
    with pytest.warns(
        UserWarning, match="drop parameter has no effect in pytensor.xtensor"
    ):
        y7 = x7.squeeze("b", drop=True)  # squeeze and drop coordinate
    fn7 = xr_function([x7], y7)
    x7_test = xr_arange_like(x7)
    xr_assert_allclose(fn7(x7_test), x7_test.squeeze("b", drop=True))


def test_squeeze_errors():
    """Test error cases for squeeze."""

    # Non-existent dimension
    x1 = xtensor("x1", dims=("city", "country"), shape=(3, 1))
    with pytest.raises(ValueError, match="Dimension .* not found"):
        x1.squeeze("time")

    # Dimension size > 1
    with pytest.raises(ValueError, match="has static size .* not 1"):
        x1.squeeze("city")

    # Symbolic shape: dim is not 1 at runtime → should raise
    x2 = xtensor("x2", dims=("a", "b", "c"))  # shape unknown
    y2 = x2.squeeze("b")
    x2_test = xr_arange_like(xtensor(dims=x2.dims, shape=(2, 2, 3)))
    fn2 = xr_function([x2], y2)
    with pytest.raises(Exception):
        fn2(x2_test)
