# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

import re
from itertools import chain, combinations

import numpy as np
import pytest
import xarray as xr
from xarray import DataArray
from xarray import concat as xr_concat

from pytensor.tensor import scalar
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


def test_expand_dims_explicit():
    """Test expand_dims with explicitly named dimensions and sizes."""

    # 1D case
    x = xtensor("x", dims=("city",), shape=(3,))
    y = expand_dims(x, "country")
    fn = xr_function([x], y)
    x_xr = xr_arange_like(x)
    xr_assert_allclose(fn(x_xr), x_xr.expand_dims("country"))

    # 2D case
    x = xtensor("x", dims=("city", "year"), shape=(2, 2))
    y = expand_dims(x, "country")
    fn = xr_function([x], y)
    xr_assert_allclose(fn(xr_arange_like(x)), xr_arange_like(x).expand_dims("country"))

    # 3D case
    x = xtensor("x", dims=("city", "year", "month"), shape=(2, 2, 2))
    y = expand_dims(x, "country")
    fn = xr_function([x], y)
    xr_assert_allclose(fn(xr_arange_like(x)), xr_arange_like(x).expand_dims("country"))

    # Prepending various dims
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    for new_dim in ("x", "y", "z"):
        y = expand_dims(x, new_dim)
        assert y.type.dims == (new_dim, "a", "b")
        assert y.type.shape == (1, 2, 3)

    # Explicit size=1 behaves like default
    y1 = expand_dims(x, "batch", size=1)
    y2 = expand_dims(x, "batch")
    fn1 = xr_function([x], y1)
    fn2 = xr_function([x], y2)
    x_test = xr_arange_like(x)
    xr_assert_allclose(fn1(x_test), fn2(x_test))

    # Scalar expansion
    x = xtensor("x", dims=(), shape=())
    y = expand_dims(x, "batch")
    assert y.type.dims == ("batch",)
    assert y.type.shape == (1,)
    fn = xr_function([x], y)
    xr_assert_allclose(fn(xr_arange_like(x)), xr_arange_like(x).expand_dims("batch"))

    # Static size > 1: broadcast
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = expand_dims(x, "batch", size=4)
    fn = xr_function([x], y)
    expected = xr.DataArray(
        np.broadcast_to(xr_arange_like(x).data, (4, 2, 3)),
        dims=("batch", "a", "b"),
        coords={"a": xr_arange_like(x).coords["a"], "b": xr_arange_like(x).coords["b"]},
    )
    xr_assert_allclose(fn(xr_arange_like(x)), expected)

    # Insert new dim between existing dims
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = expand_dims(x, "new")
    # Insert new dim between a and b: ("a", "new", "b")
    y = transpose(y, "a", "new", "b")
    fn = xr_function([x], y)
    x_test = xr_arange_like(x)
    expected = x_test.expand_dims("new").transpose("a", "new", "b")
    xr_assert_allclose(fn(x_test), expected)

    # Expand with multiple dims
    x = xtensor("x", dims=(), shape=())
    y = expand_dims(expand_dims(x, "a"), "b")
    fn = xr_function([x], y)
    expected = xr_arange_like(x).expand_dims("a").expand_dims("b")
    xr_assert_allclose(fn(xr_arange_like(x)), expected)


def test_expand_dims_implicit():
    """Test expand_dims with default or symbolic sizes and dim=None."""

    # Symbolic size=1: same as default
    size_sym_1 = scalar("size_sym_1", dtype="int64")
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = expand_dims(x, "batch", size=size_sym_1)
    fn = xr_function([x, size_sym_1], y, on_unused_input="ignore")
    expected = xr_arange_like(x).expand_dims("batch")
    xr_assert_allclose(fn(xr_arange_like(x), 1), expected)

    # Symbolic size > 1 (but expand only adds dim=1)
    size_sym_4 = scalar("size_sym_4", dtype="int64")
    y = expand_dims(x, "batch", size=size_sym_4)
    fn = xr_function([x, size_sym_4], y, on_unused_input="ignore")
    xr_assert_allclose(fn(xr_arange_like(x), 4), expected)

    # Reversibility: expand then squeeze
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = expand_dims(x, "batch")
    z = squeeze(y, "batch")
    fn = xr_function([x], z)
    xr_assert_allclose(fn(xr_arange_like(x)), xr_arange_like(x))

    # expand_dims with dim=None = no-op
    x = xtensor("x", dims=("a",), shape=(3,))
    y = expand_dims(x, None)
    fn = xr_function([x], y)
    xr_assert_allclose(fn(xr_arange_like(x)), xr_arange_like(x))

    # broadcast after symbolic size
    size_sym = scalar("size_sym", dtype="int64")
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = expand_dims(x, "batch", size=size_sym)
    z = y + y  # triggers shape alignment
    fn = xr_function([x, size_sym], z, on_unused_input="ignore")
    x_test = xr_arange_like(x)
    out = fn(x_test, 1)
    expected = x_test.expand_dims("batch") + x_test.expand_dims("batch")
    xr_assert_allclose(out, expected)


def test_expand_dims_errors():
    """Test error handling in expand_dims."""

    # Expanding existing dim
    x = xtensor("x", dims=("city",), shape=(3,))
    y = expand_dims(x, "country")
    with pytest.raises(ValueError, match="already exists"):
        expand_dims(y, "city")

    # Size = 0 is invalid
    with pytest.raises(ValueError, match="size must be.*positive"):
        expand_dims(x, "batch", size=0)

    # Invalid dim type
    with pytest.raises(TypeError):
        expand_dims(x, 123)

    # Invalid size type
    with pytest.raises(TypeError):
        expand_dims(x, "new", size=[1])

    # Duplicate dimension creation
    y = expand_dims(x, "new")
    with pytest.raises(ValueError):
        expand_dims(y, "new")

    # Symbolic size with invalid runtime value
    size_sym = scalar("size_sym", dtype="int64")
    y = expand_dims(x, "batch", size=size_sym)
    fn = xr_function([x, size_sym], y, on_unused_input="ignore")
    with pytest.raises(Exception):
        fn(xr_arange_like(x), 0)


def test_squeeze_explicit_dims():
    """Test squeeze with explicit dimension(s)."""

    # Single dimension
    x1 = xtensor("x1", dims=("city", "country"), shape=(3, 1))
    y1 = squeeze(x1, "country")
    fn1 = xr_function([x1], y1)
    x1_test = xr_arange_like(x1)
    xr_assert_allclose(fn1(x1_test), x1_test.squeeze("country"))

    # Multiple dimensions
    x2 = xtensor("x2", dims=("a", "b", "c", "d"), shape=(2, 1, 1, 3))
    y2 = squeeze(x2, ["b", "c"])
    fn2 = xr_function([x2], y2)
    x2_test = xr_arange_like(x2)
    xr_assert_allclose(fn2(x2_test), x2_test.squeeze(["b", "c"]))

    # Order independence
    x3 = xtensor("x3", dims=("a", "b", "c"), shape=(2, 1, 1))
    y3a = squeeze(x3, ["b", "c"])
    y3b = squeeze(x3, ["c", "b"])
    fn3a = xr_function([x3], y3a)
    fn3b = xr_function([x3], y3b)
    x3_test = xr_arange_like(x3)
    xr_assert_allclose(fn3a(x3_test), fn3b(x3_test))

    # Redundant dimensions
    y3c = squeeze(x3, ["b", "b"])
    fn3c = xr_function([x3], y3c)
    xr_assert_allclose(fn3c(x3_test), x3_test.squeeze("b"))

    # Empty list = no-op
    y3d = squeeze(x3, [])
    fn3d = xr_function([x3], y3d)
    xr_assert_allclose(fn3d(x3_test), x3_test)


def test_squeeze_implicit_dims():
    """Test squeeze with implicit dim=None (all size-1 dimensions)."""

    # All dimensions size 1
    x1 = xtensor("x1", dims=("a", "b"), shape=(1, 1))
    y1 = squeeze(x1)
    fn1 = xr_function([x1], y1)
    x1_test = xr_arange_like(x1)
    xr_assert_allclose(fn1(x1_test), x1_test.squeeze())

    # No dimensions size 1 = no-op
    x2 = xtensor("x2", dims=("row", "col", "batch"), shape=(2, 3, 4))
    y2 = squeeze(x2)
    fn2 = xr_function([x2], y2)
    x2_test = xr_arange_like(x2)
    xr_assert_allclose(fn2(x2_test), x2_test)

    # Symbolic shape where runtime shape is 1 → should squeeze
    x3 = xtensor("x3", dims=("a", "b", "c"))  # shape unknown
    y3 = squeeze(x3, "b")
    x3_test = xr_arange_like(xtensor(dims=x3.dims, shape=(2, 1, 3)))
    fn3 = xr_function([x3], y3)
    xr_assert_allclose(fn3(x3_test), x3_test.squeeze("b"))

    # Mixed static + symbolic shapes, where symbolic shape is 1
    x4 = xtensor("x4", dims=("a", "b", "c"), shape=(None, 1, 3))
    y4 = squeeze(x4, "b")
    x4_test = xr_arange_like(xtensor(dims=x4.dims, shape=(4, 1, 3)))
    fn4 = xr_function([x4], y4)
    xr_assert_allclose(fn4(x4_test), x4_test.squeeze("b"))

    # Reversibility with expand_dims
    x5 = xtensor("x5", dims=("batch", "time", "feature"), shape=(2, 1, 3))
    y5 = squeeze(x5, "time")
    z5 = expand_dims(y5, "time")
    fn5 = xr_function([x5], z5)
    x5_test = xr_arange_like(x5)
    xr_assert_allclose(fn5(x5_test).transpose(*x5_test.dims), x5_test)


def test_squeeze_errors():
    """Test error cases for squeeze."""

    # Non-existent dimension
    x1 = xtensor("x1", dims=("city", "country"), shape=(3, 1))
    with pytest.raises(ValueError, match="Dimension .* not found"):
        squeeze(x1, "time")

    # Dimension size > 1
    with pytest.raises(ValueError, match="has static size .* not 1"):
        squeeze(x1, "city")

    # Symbolic shape: dim is not 1 at runtime → should raise
    x2 = xtensor("x2", dims=("a", "b", "c"))  # shape unknown
    y2 = squeeze(x2, "b")
    x2_test = xr_arange_like(xtensor(dims=x2.dims, shape=(2, 2, 3)))
    fn2 = xr_function([x2], y2)
    with pytest.raises(Exception):
        fn2(x2_test)
