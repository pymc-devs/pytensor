# ruff: noqa: E402
import re

import pytest


pytest.importorskip("xarray")

from itertools import chain, combinations

import numpy as np
from xarray import DataArray
from xarray import concat as xr_concat

from pytensor.xtensor.shape import (
    concat,
    expand_dims,
    squeeze,
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

    with pytest.raises(ValueError, match="an index can only have a single ellipsis"):
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
    fn.fn.dprint(print_type=True)
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
    b_idx, c_idx = np.unravel_index(np.arange(15)[::-1].reshape((3, 5)), (3, 5))
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


def test_expand_dims():
    import xarray as xr

    # 1D case
    x_xr = xr.DataArray([0, 1, 2], dims=["city"])
    y_xr = x_xr.expand_dims("country")
    x = xtensor("x", dims=("city",), shape=(3,))
    y = expand_dims(x, "country")
    assert y.type.dims == y_xr.dims
    assert y.type.shape == y_xr.shape
    fn = xr_function([x], y)
    x_test = xr_arange_like(x)
    res = fn(x_test)
    expected_res = x_test.expand_dims("country")
    xr_assert_allclose(res, expected_res)

    # 2D case
    x2d_xr = xr.DataArray([[0, 1, 2], [3, 4, 5]], dims=["row", "col"])
    y2d_xr = x2d_xr.expand_dims("batch")
    x2d = xtensor("x2d", dims=("row", "col"), shape=(2, 3))
    y2d = expand_dims(x2d, "batch")
    assert y2d.type.dims == y2d_xr.dims
    assert y2d.type.shape == y2d_xr.shape
    fn = xr_function([x2d], y2d)
    x2d_test = xr_arange_like(x2d)
    res = fn(x2d_test)
    expected_res = x2d_test.expand_dims("batch")
    xr_assert_allclose(res, expected_res)

    # Expansion with different dimension name
    z_xr = x_xr.expand_dims("time")
    z = expand_dims(x, "time")
    assert z.type.dims == z_xr.dims
    assert z.type.shape == z_xr.shape
    fn = xr_function([x], z)
    res = fn(x_test)
    expected_res = x_test.expand_dims("time")
    xr_assert_allclose(res, expected_res)

    # Expanding with an existing dimension raises an error
    with pytest.raises(ValueError):
        expand_dims(y, "city")

    # Expanding with None dimension
    z_xr = x_xr.expand_dims(None)
    z = expand_dims(x, None)
    assert z.type.dims == z_xr.dims
    assert z.type.shape == z_xr.shape
    fn = xr_function([x], z)
    res = fn(x_test)
    expected_res = x_test.expand_dims(None)
    xr_assert_allclose(res, expected_res)


def test_squeeze():
    # Test squeezing a specific dimension
    x = xtensor("x", dims=("city", "country"), shape=(3, 1))
    y = squeeze(x, "country")
    fn = xr_function([x], y)
    x_test = xr_arange_like(x)
    res = fn(x_test)
    expected_res = x_test.squeeze("country")
    xr_assert_allclose(res, expected_res)

    # Test squeezing multiple specific dimensions
    x_multi = xtensor("x_multi", dims=("a", "b", "c", "d"), shape=(2, 1, 1, 3))
    y_multi = squeeze(x_multi, ["b", "c"])
    fn = xr_function([x_multi], y_multi)
    x_multi_test = xr_arange_like(x_multi)
    res = fn(x_multi_test)
    expected_res = x_multi_test.squeeze(["b", "c"])
    xr_assert_allclose(res, expected_res)

    # Test squeezing a non-last dimension
    x_nonlast = xtensor("x_nonlast", dims=("a", "b", "c"), shape=(2, 1, 3))
    y_nonlast = squeeze(x_nonlast, "b")
    fn = xr_function([x_nonlast], y_nonlast)
    x_nonlast_test = xr_arange_like(x_nonlast)
    res = fn(x_nonlast_test)
    expected_res = x_nonlast_test.squeeze("b")
    xr_assert_allclose(res, expected_res)

    # Test squeezing in a higher-dimensional tensor
    x_high = xtensor("x_high", dims=("a", "b", "c", "d", "e"), shape=(2, 1, 3, 1, 4))
    y_high = squeeze(x_high, ["b", "d"])
    fn = xr_function([x_high], y_high)
    x_high_test = xr_arange_like(x_high)
    res = fn(x_high_test)
    expected_res = x_high_test.squeeze(["b", "d"])
    xr_assert_allclose(res, expected_res)

    # Test with symbolic shapes
    x_sym = xtensor("x_sym", dims=("a", "b", "c"))
    y_sym = squeeze(x_sym, "b")
    x_sym_test = xr_arange_like(xtensor(dims=x_sym.dims, shape=(2, 1, 3)))
    fn = xr_function([x_sym], y_sym)
    res = fn(x_sym_test)
    expected_res = x_sym_test.squeeze("b")
    xr_assert_allclose(res, expected_res)

    # Test squeezing all dimensions of size 1
    x2d = xtensor("x2d", dims=("row", "col", "batch"), shape=(2, 1, 1))
    y2d = squeeze(x2d)
    fn = xr_function([x2d], y2d)
    x2d_test = xr_arange_like(x2d)
    res = fn(x2d_test)
    expected_res = x2d_test.squeeze()
    xr_assert_allclose(res, expected_res)

    # Test that squeezing a non-existent dimension raises an error
    with pytest.raises(ValueError):
        squeeze(x, "time")

    # Test that squeezing a dimension of size > 1 raises an error
    with pytest.raises(ValueError):
        squeeze(x, "city")

    # Test that squeezing when no dimensions are of size 1 raises an error
    x3d = xtensor("x3d", dims=("row", "col", "batch"), shape=(2, 3, 4))
    with pytest.raises(ValueError):
        squeeze(x3d)
