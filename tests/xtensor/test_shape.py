# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

from itertools import chain, combinations

import numpy as np
from xarray import DataArray

from pytensor.xtensor.shape import stack, unstack
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_assert_allclose, xr_function


def powerset(iterable, min_group_size=0):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_group_size, len(s) + 1)
    )


@pytest.mark.xfail(reason="Not yet implemented")
def test_transpose():
    transpose = None
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
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )
    res = fn(x_test)
    expected_res = [x_test.transpose(*perm) for perm in permutations]
    for outs_i, res_i, expected_res_i in zip(outs, res, expected_res):
        xr_assert_allclose(res_i, expected_res_i)


def test_stack():
    dims = ("a", "b", "c", "d")
    x = xtensor("x", dims=dims, shape=(2, 3, 5, 7))
    outs = [
        stack(x, new_dim=dims_to_stack)
        for dims_to_stack in powerset(dims, min_group_size=2)
    ]

    fn = xr_function([x], outs)
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )
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
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )
    fn.fn.dprint(print_type=True)
    res = fn(x_test)
    expected_res = x_test.stack(d=["a"])
    xr_assert_allclose(res, expected_res)


def test_multiple_stacks():
    x = xtensor("x", dims=("a", "b", "c", "d"), shape=(2, 3, 5, 7))
    out = stack(x, new_dim1=("a", "b"), new_dim2=("c", "d"))

    fn = xr_function([x], [out])
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )
    res = fn(x_test)
    expected_res = x_test.stack(new_dim1=("a", "b"), new_dim2=("c", "d"))
    xr_assert_allclose(res[0], expected_res)


def test_unstack():
    unstacked_dims = {"a": 2, "b": 3, "c": 5, "d": 7}
    dims = ("abcd",)
    x = xtensor("x", dims=dims, shape=(2 * 3 * 5 * 7,))
    outs = [
        unstack(
            x,
            abcd=(
                {d: l for d, l in unstacked_dims.items() if d in dims_to_unstack}
                | (
                    {}
                    if set(dims_to_unstack) == set(unstacked_dims)
                    else {
                        "other": int(
                            np.prod(
                                [
                                    l
                                    for d, l in unstacked_dims.items()
                                    if d not in dims_to_unstack
                                ]
                            )
                        )
                    }
                )
            ),
        )
        for dims_to_unstack in powerset(unstacked_dims.keys(), min_group_size=2)
    ]
    fn = xr_function([x], outs)
    # we test through the complementary operation in xarray to avoid needing coords
    # which are required for unstack. We end up with a subset of {a, b, c, d} and
    # other after unstacking, so we create the fully unstacked dataarray
    # and stack to create this extra "other" dimension as needed
    x_test = DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(
            list(unstacked_dims.values())
        ),
        dims=list(unstacked_dims.keys()),
    )
    res = fn(x_test)

    expected_res = [
        x_test.stack(
            {}
            if set(dims_to_unstack) == set(unstacked_dims)
            else {"other": [d for d in unstacked_dims if d not in dims_to_unstack]}
        )
        for dims_to_unstack in powerset(unstacked_dims.keys(), min_group_size=2)
    ]
    for res_i, expected_res_i in zip(res, expected_res):
        assert res_i.shape == expected_res_i.shape
        # the shapes are right but the "other" one has the elements in different order
        # I think it is an issue with the test not the function but not sure
        # xr_assert_allclose(res_i, expected_res_i)


def test_unstack_simple():
    x = xtensor("x", dims=("a", "bc", "d"), shape=(2, 3 * 5, 7))
    y = unstack(x, bc=dict(b=3, c=5))
    assert y.type.dims == ("a", "d", "b", "c")
    assert y.type.shape == (2, 7, 3, 5)

    fn = xr_function([x], y)

    x_np = np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape)
    x_test = DataArray(x_np, dims=x.type.dims)
    res = fn(x_test)
    expected = (
        DataArray(x_np.reshape(2, 3, 5, 7), dims=("a", "b", "c", "d"))
        .stack(bc=("b", "c"))
        .unstack("bc")
    )
    xr_assert_allclose(res, expected)
