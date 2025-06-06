# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

from itertools import chain, combinations

from pytensor.xtensor.shape import stack
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import (
    xr_arange_like,
    xr_assert_allclose,
    xr_function,
)


def powerset(iterable, min_group_size=0):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_group_size, len(s) + 1)
    )


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
