import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")


import re

import numpy as np
from xarray import DataArray

from pytensor.tensor import tensor
from pytensor.xtensor import xtensor
from tests.unittest_tools import assert_equal_computations
from tests.xtensor.util import (
    xr_arange_like,
    xr_assert_allclose,
    xr_function,
    xr_random_like,
)


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

    # Equivalent ways of indexing a->a
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

    y = x[((("a",), idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[((("a",), idx_test),)]
    xr_assert_allclose(res, expected_res)

    y = x[xidx]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test]
    xr_assert_allclose(res, expected_res)


def test_single_vector_indexing_on_new_dim():
    x = xtensor(dims=("a", "b"), shape=(3, 5))
    idx = tensor("idx", dtype=int, shape=(4,))
    xidx = xtensor("idx", dtype=int, shape=(4,), dims=("new_a",))

    x_test = xr_arange_like(x)
    idx_test = np.array([0, 1, 0, 2], dtype=int)
    xidx_test = DataArray(idx_test, dims=("new_a",))

    # Equivalent ways of indexing a->new_a
    y = x[(("new_a", idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[(("new_a", idx_test),)]
    xr_assert_allclose(res, expected_res)

    y = x[((["new_a"], idx),)]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[((["new_a"], idx_test),)]
    xr_assert_allclose(res, expected_res)

    y = x[xidx]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test]
    xr_assert_allclose(res, expected_res)


def test_single_vector_indexing_interacting_with_existing_dim():
    x = xtensor(dims=("a", "b"), shape=(3, 5))
    idx = tensor("idx", dtype=int, shape=(4,))
    xidx = xtensor("idx", dtype=int, shape=(4,), dims=("a",))

    x_test = xr_arange_like(x)
    idx_test = np.array([0, 1, 0, 2], dtype=int)
    xidx_test = DataArray(idx_test, dims=("a",))

    # Two equivalent ways of indexing a->b
    # By labeling the index on a, as "b", we cause pointwise indexing between the two dimensions.
    y = x[("b", idx), 1:]
    fn = xr_function([x, idx], y)
    res = fn(x_test, idx_test)
    expected_res = x_test[("b", idx_test), 1:]
    xr_assert_allclose(res, expected_res)

    y = x[xidx.rename(a="b"), 1:]
    fn = xr_function([x, xidx], y)
    res = fn(x_test, xidx_test)
    expected_res = x_test[xidx_test.rename(a="b"), 1:]
    xr_assert_allclose(res, expected_res)


@pytest.mark.parametrize(
    "dims_order",
    [
        ("a", "b", "ar", "br", "o"),
        ("o", "br", "ar", "b", "a"),
        ("a", "b", "o", "ar", "br"),
        ("a", "o", "ar", "b", "br"),
    ],
)
def test_multiple_vector_indexing(dims_order):
    x = xtensor(dims=dims_order, shape=(5, 7, 11, 13, 17))
    idx_a = xtensor("idx_a", dtype=int, shape=(4,), dims=("a",))
    idx_b = xtensor("idx_b", dtype=int, shape=(3,), dims=("b",))

    idxs = [slice(None)] * 5
    idxs[x.type.dims.index("a")] = idx_a
    idxs[x.type.dims.index("b")] = idx_b
    idxs[x.type.dims.index("ar")] = idx_a[::-1]
    idxs[x.type.dims.index("br")] = idx_b[::-1]

    out = x[tuple(idxs)]
    fn = xr_function([x, idx_a, idx_b], out)

    x_test = xr_arange_like(x)
    idx_a_test = DataArray(np.array([0, 1, 0, 2], dtype=int), dims=("a",))
    idx_b_test = DataArray(np.array([1, 3, 0], dtype=int), dims=("b",))
    res = fn(x_test, idx_a_test, idx_b_test)
    idxs_test = [slice(None)] * 5
    idxs_test[x.type.dims.index("a")] = idx_a_test
    idxs_test[x.type.dims.index("b")] = idx_b_test
    idxs_test[x.type.dims.index("ar")] = idx_a_test[::-1]
    idxs_test[x.type.dims.index("br")] = idx_b_test[::-1]
    expected_res = x_test[tuple(idxs_test)]
    xr_assert_allclose(res, expected_res)


def test_matrix_indexing():
    x = xtensor(dims=("a", "b", "c"), shape=(3, 5, 7))
    idx_ab = xtensor("idx_ab", dtype=int, shape=(4, 2), dims=("a", "b"))
    idx_cd = xtensor("idx_cd", dtype=int, shape=(4, 3), dims=("c", "d"))

    out = x[idx_ab, slice(1, 3), idx_cd]
    fn = xr_function([x, idx_ab, idx_cd], out)

    x_test = xr_arange_like(x)
    idx_ab_test = DataArray(
        np.array([[0, 1], [1, 2], [0, 2], [-1, -2]], dtype=int), dims=("a", "b")
    )
    idx_cd_test = DataArray(
        np.array([[1, 2, 3], [0, 4, 5], [2, 6, -1], [3, -2, 0]], dtype=int),
        dims=("c", "d"),
    )
    res = fn(x_test, idx_ab_test, idx_cd_test)
    expected_res = x_test[idx_ab_test, slice(1, 3), idx_cd_test]
    xr_assert_allclose(res, expected_res)


def test_assign_multiple_out_dims():
    x = xtensor("x", shape=(5, 7), dims=("a", "b"))
    idx1 = tensor("idx1", dtype=int, shape=(4, 3))
    idx2 = tensor("idx2", dtype=int, shape=(3, 2))
    out = x[(("out1", "out2"), idx1), (["out2", "out3"], idx2)]

    fn = xr_function([x, idx1, idx2], out)

    rng = np.random.default_rng()
    x_test = xr_arange_like(x)
    idx1_test = rng.binomial(n=4, p=0.5, size=(4, 3))
    idx2_test = rng.binomial(n=4, p=0.5, size=(3, 2))
    res = fn(x_test, idx1_test, idx2_test)
    expected_res = x_test[(("out1", "out2"), idx1_test), (["out2", "out3"], idx2_test)]
    xr_assert_allclose(res, expected_res)


def test_assign_indexer_dims_fails():
    # Test cases where the implicit naming of the indexer dimensions is not allowed.
    x = xtensor("x", shape=(5, 7), dims=("a", "b"))
    idx1 = xtensor("idx1", dtype=int, shape=(4,), dims=("c",))

    with pytest.raises(
        IndexError,
        match=re.escape(
            "Giving a dimension name to an XTensorVariable indexer is not supported: ('d', idx1). "
            "Use .rename() instead."
        ),
    ):
        x[("d", idx1),]

    with pytest.raises(
        IndexError,
        match=re.escape(
            "Boolean indexer should be unlabeled or on the same dimension to the indexed array. "
            "Indexer is on ('c',) but the target dimension is a."
        ),
    ):
        x[idx1.astype("bool")]


class TestVectorizedIndexingNotAllowedToBroadcast:
    def test_compile_time_error(self):
        x = xtensor(dims=("a", "b"), shape=(3, 5))
        idx_a = xtensor("idx_a", dtype=int, shape=(4,), dims=("b",))
        idx_b = xtensor("idx_b", dtype=int, shape=(1,), dims=("b",))
        with pytest.raises(
            IndexError, match="Dimension of indexers mismatch for dim b"
        ):
            x[idx_a, idx_b]

    @pytest.mark.xfail(
        reason="Check that lowered indexing is not allowed to broadcast not implemented yet"
    )
    def test_runtime_error(self):
        """
        Test that, unlike in numpy, indices with different shapes cannot act on the same dimension,
        even if the shapes could broadcast as per numpy semantics.
        """
        x = xtensor(dims=("a", "b"), shape=(3, 5))
        idx_a = xtensor("idx_a", dtype=int, shape=(None,), dims=("b",))
        idx_b = xtensor("idx_b", dtype=int, shape=(None,), dims=("b",))
        out = x[idx_a, idx_b]

        fn = xr_function([x, idx_a, idx_b], out)

        x_test = xr_arange_like(x)
        valid_idx_a_test = DataArray(np.array([0], dtype=int), dims=("b",))
        idx_b_test = DataArray(np.array([1], dtype=int), dims=("b",))
        xr_assert_allclose(
            fn(x_test, valid_idx_a_test, idx_b_test),
            x_test[valid_idx_a_test, idx_b_test],
        )

        invalid_idx_a_test = DataArray(np.array([0, 1, 0, 1], dtype=int), dims=("b",))
        with pytest.raises(ValueError):
            fn(x_test, invalid_idx_a_test, idx_b_test)


@pytest.mark.parametrize(
    "dims_order",
    [
        ("a", "b", "c", "d"),
        ("d", "c", "b", "a"),
        ("c", "a", "b", "d"),
    ],
)
def test_scalar_integer_indexing(dims_order):
    x = xtensor(dims=dims_order, shape=(3, 5, 7, 11))
    scalar_idx = xtensor("scalar_idx", dtype=int, shape=(), dims=())
    vec_idx1 = xtensor("vec_idx", dtype=int, shape=(4,), dims=("a",))
    vec_idx2 = xtensor("vec_idx2", dtype=int, shape=(4,), dims=("c",))

    idxs = [None] * 4
    idxs[x.type.dims.index("a")] = scalar_idx
    idxs[x.type.dims.index("b")] = vec_idx1
    idxs[x.type.dims.index("c")] = vec_idx2
    idxs[x.type.dims.index("d")] = -scalar_idx
    out1 = x[tuple(idxs)]

    idxs[x.type.dims.index("a")] = vec_idx1.rename(a="c")
    out2 = x[tuple(idxs)]

    fn = xr_function([x, scalar_idx, vec_idx1, vec_idx2], (out1, out2))

    x_test = xr_arange_like(x)
    scalar_idx_test = DataArray(np.array(1, dtype=int), dims=())
    vec_idx_test1 = DataArray(np.array([0, 1, 0, 2], dtype=int), dims=("a",))
    vec_idx_test2 = DataArray(np.array([0, 2, 2, 1], dtype=int), dims=("c",))
    res1, res2 = fn(x_test, scalar_idx_test, vec_idx_test1, vec_idx_test2)
    idxs = [None] * 4
    idxs[x.type.dims.index("a")] = scalar_idx_test
    idxs[x.type.dims.index("b")] = vec_idx_test1
    idxs[x.type.dims.index("c")] = vec_idx_test2
    idxs[x.type.dims.index("d")] = -scalar_idx_test
    expected_res1 = x_test[tuple(idxs)]
    idxs[x.type.dims.index("a")] = vec_idx_test1.rename(a="c")
    expected_res2 = x_test[tuple(idxs)]
    xr_assert_allclose(res1, expected_res1)
    xr_assert_allclose(res2, expected_res2)


def test_unsupported_boolean_indexing():
    x = xtensor(dims=("a", "b"), shape=(3, 5))

    mat_idx = xtensor("idx", dtype=bool, shape=(4, 2), dims=("a", "b"))
    scalar_idx = mat_idx.isel(a=0, b=1)

    for idx in (mat_idx, scalar_idx, scalar_idx.values):
        with pytest.raises(
            NotImplementedError,
            match="Only 1d boolean indexing arrays are supported",
        ):
            x[idx]


def test_boolean_indexing():
    x = xtensor("x", shape=(8, 7), dims=("a", "b"))
    bool_idx = xtensor("bool_idx", dtype=bool, shape=(8,), dims=("a",))
    int_idx = xtensor("int_idx", dtype=int, shape=(4, 3), dims=("a", "new_dim"))

    out_vectorized = x[bool_idx, int_idx]
    out_orthogonal = x[bool_idx, int_idx.rename(a="b")]
    fn = xr_function([x, bool_idx, int_idx], [out_vectorized, out_orthogonal])

    x_test = xr_arange_like(x)
    bool_idx_test = DataArray(np.array([True, False] * 4, dtype=bool), dims=("a",))
    int_idx_test = DataArray(
        np.random.binomial(n=4, p=0.5, size=(4, 3)),
        dims=("a", "new_dim"),
    )
    res1, res2 = fn(x_test, bool_idx_test, int_idx_test)
    expected_res1 = x_test[bool_idx_test, int_idx_test]
    expected_res2 = x_test[bool_idx_test, int_idx_test.rename(a="b")]
    xr_assert_allclose(res1, expected_res1)
    xr_assert_allclose(res2, expected_res2)


@pytest.mark.parametrize("mode", ("set", "inc"))
def test_basic_index_update(mode):
    x = xtensor("x", shape=(11, 7), dims=("a", "b"))
    y = xtensor("y", shape=(7, 5), dims=("a", "b"))
    x_indexed = x[2:-2, 2:]
    update_method = getattr(x_indexed, mode)

    x_updated = [
        update_method(y),
        update_method(y.T),
        update_method(y.isel(a=-1)),
        update_method(y.isel(b=-1)),
        update_method(y.isel(a=-2, b=-2)),
    ]

    fn = xr_function([x, y], x_updated)
    x_test = xr_random_like(x)
    y_test = xr_random_like(y)
    results = fn(x_test, y_test)

    def update_fn(y):
        x = x_test.copy()
        if mode == "set":
            x[2:-2, 2:] = y
        elif mode == "inc":
            x[2:-2, 2:] += y
        return x

    expected_results = [
        update_fn(y_test),
        update_fn(y_test.T),
        update_fn(y_test.isel(a=-1)),
        update_fn(y_test.isel(b=-1)),
        update_fn(y_test.isel(a=-2, b=-2)),
    ]
    for result, expected_result in zip(results, expected_results):
        xr_assert_allclose(result, expected_result)


@pytest.mark.parametrize("mode", ("set", "inc"))
@pytest.mark.parametrize("idx_dtype", (int, bool))
def test_adv_index_update(mode, idx_dtype):
    x = xtensor("x", shape=(5, 5), dims=("a", "b"))
    y = xtensor("y", shape=(3,), dims=("b",))
    idx = xtensor("idx", dtype=idx_dtype, shape=(None,), dims=("a",))

    orthogonal_update1 = getattr(x[idx, -3:], mode)(y)
    orthogonal_update2 = getattr(x[idx, -3:], mode)(y.rename(b="a"))
    if idx_dtype is not bool:
        # Vectorized booling indexing/update is not allowed
        vectorized_update = getattr(x[idx.rename(a="b"), :3], mode)(y)
    else:
        with pytest.raises(
            IndexError,
            match="Boolean indexer should be unlabeled or on the same dimension to the indexed array\\.",
        ):
            getattr(x[idx.rename(a="b"), :3], mode)(y)
        vectorized_update = x

    outs = [orthogonal_update1, orthogonal_update2, vectorized_update]

    fn = xr_function([x, idx, y], outs)
    x_test = xr_random_like(x)
    y_test = xr_random_like(y)
    if idx_dtype is int:
        idx_test = DataArray([0, 1, 2], dims=("a",))
    else:
        idx_test = DataArray([True, False, True, True, False], dims=("a",))
    results = fn(x_test, idx_test, y_test)

    def update_fn(x, idx, y):
        x = x.copy()
        if mode == "set":
            x[idx] = y
        else:
            x[idx] += y
        return x

    expected_results = [
        update_fn(x_test, (idx_test, slice(-3, None)), y_test),
        update_fn(
            x_test,
            (idx_test, slice(-3, None)),
            y_test.rename(b="a"),
        ),
        update_fn(x_test, (idx_test.rename(a="b"), slice(None, 3)), y_test)
        if idx_dtype is not bool
        else x_test,
    ]
    for result, expected_result in zip(results, expected_results):
        xr_assert_allclose(result, expected_result)


@pytest.mark.parametrize("mode", ("set", "inc"))
def test_non_consecutive_idx_update(mode):
    x = xtensor("x", shape=(2, 3, 5, 7), dims=("a", "b", "c", "d"))
    y = xtensor("y", shape=(5, 4), dims=("c", "b"))
    x_indexed = x[:, [0, 1, 2, 2], :, ("b", [0, 1, 1, 2])]
    out = getattr(x_indexed, mode)(y)

    fn = xr_function([x, y], out)
    x_test = xr_random_like(x)
    y_test = xr_random_like(y)

    result = fn(x_test, y_test)
    expected_result = x_test.copy()
    # xarray fails inplace operation with the "tuple trick"
    # https://github.com/pydata/xarray/issues/10387
    d_indexer = DataArray([0, 1, 1, 2], dims=("b",))
    if mode == "set":
        expected_result[:, [0, 1, 2, 2], :, d_indexer] = y_test
    else:
        expected_result[:, [0, 1, 2, 2], :, d_indexer] += y_test
    xr_assert_allclose(result, expected_result)


def test_indexing_renames_into_update_variable():
    x = xtensor("x", shape=(5, 5), dims=("a", "b"))
    y = xtensor("y", shape=(3,), dims=("d",))
    idx = xtensor("idx", dtype=int, shape=(None,), dims=("d",))

    # define "d" dimension by slicing the "a" dimension so we can set y into x
    orthogonal_update1 = x[idx].set(y)
    fn = xr_function([x, idx, y], orthogonal_update1)

    x_test = np.abs(xr_random_like(x))
    y_test = -np.abs(xr_random_like(y))
    idx_test = DataArray([0, 2, 3], dims=("d",))

    result = fn(x_test, idx_test, y_test)
    expected_result = x_test.copy()
    expected_result[idx_test] = y_test
    xr_assert_allclose(result, expected_result)


@pytest.mark.parametrize("n", ["implicit", 1, 2])
@pytest.mark.parametrize("dim", ["a", "b"])
def test_diff(dim, n):
    x = xtensor(dims=("a", "b"), shape=(7, 11))
    if n == "implicit":
        out = x.diff(dim)
    else:
        out = x.diff(dim, n=n)

    fn = xr_function([x], out)
    x_test = xr_arange_like(x)
    res = fn(x_test)
    if n == "implicit":
        expected_res = x_test.diff(dim)
    else:
        expected_res = x_test.diff(dim, n=n)
    xr_assert_allclose(res, expected_res)


def test_empty_index():
    x = xtensor("x", shape=(5, 5), dims=("a", "b"))
    out1 = x[()]
    out2 = x[...]
    out3 = x.isel({})
    out4 = x.isel({"c": 0}, missing_dims="ignore")
    assert_equal_computations([out1], [out2])
    assert_equal_computations([out1], [out3])
    assert_equal_computations([out1], [out4])

    fn = xr_function([x], out1)
    x_test = xr_random_like(x)
    xr_assert_allclose(fn(x_test), x_test)


def test_empty_update_index():
    x = xtensor("x", shape=(5, 5), dims=("a", "b"))
    out1 = x[()].inc(1)
    out2 = x[...].inc(1)
    out3 = x.isel({}).inc(1)
    out4 = x.isel({"c": 0}, missing_dims="ignore").inc(1)
    assert_equal_computations([out1], [out2])
    assert_equal_computations([out1], [out3])
    assert_equal_computations([out1], [out4])

    fn = xr_function([x], out1)
    x_test = xr_random_like(x)
    xr_assert_allclose(fn(x_test), x_test + 1)
