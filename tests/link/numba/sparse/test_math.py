import numpy as np
import pytest
import scipy

import pytensor
import pytensor.sparse as ps
import pytensor.tensor as pt
from tests.link.numba.sparse.test_basic import (
    compare_numba_and_py_sparse,
    sparse_assert_fn,
)
from tests.link.numba.test_basic import numba_mode


pytestmark = pytest.mark.filterwarnings("error")

DOT_SHAPES = [((20, 11), (11, 4)), ((10, 3), (3, 1)), ((1, 10), (10, 5))]


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("y_ndim", [0, 1, 2])
def test_sparse_dense_multiply(y_ndim, format):
    x = ps.matrix(format, name="x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,) * y_ndim)
    z = x * y

    rng = np.random.default_rng((155, y_ndim, format == "csr"))
    x_test = scipy.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,) * y_ndim)

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_sparse_dense(op, sp_format, x_shape, y_shape):
    x = ps.matrix(format=sp_format, name="x", shape=x_shape)
    y = pt.matrix("y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=sp_format, random_state=rng
    )
    y_test = rng.normal(size=y_shape)

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_dense_sparse(op, sp_format, x_shape, y_shape):
    x = pt.matrix(name="x", shape=x_shape)
    y = ps.matrix(format=sp_format, name="y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = rng.normal(size=x_shape)
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=sp_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_sparse_dot_sparse_sparse(op, x_format, y_format, x_shape, y_shape):
    x = ps.matrix(x_format, name="x", shape=x_shape)
    y = ps.matrix(y_format, name="y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, x_format)) + sum(map(ord, y_format)))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=x_format, random_state=rng
    )
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=y_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("sp_format", ["csr", "csc"])
def test_sparse_spmv(sp_format):
    x = ps.matrix(format=sp_format, name="x", shape=(20, 6))
    y = pt.vector("y", shape=(6,))
    z = ps.dot(x, y)

    rng = np.random.default_rng(sp_format == "csr")
    x_test = scipy.sparse.random(20, 6, density=0.5, format=sp_format, random_state=rng)
    y_test = rng.normal(size=(6,))

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [
        ("int64", "complex64"),
        ("int64", "float32"),
    ],
)
def test_structured_dot_upcast(x_dtype, y_dtype):
    """Numba scalar-array mul keeps the array dtype; numpy upcasts to a wider type."""
    x = ps.matrix(format="csc", name="x", dtype=x_dtype, shape=(4, 3))
    y = pt.matrix("y", dtype=y_dtype, shape=(3, 5))
    z = ps.structured_dot(x, y)

    x_test = scipy.sparse.csc_matrix(
        np.array([[97, 0, 0], [0, 83, 0], [0, 0, 71], [42, 0, 0]], dtype=x_dtype)
    )
    y_test = np.array(
        [
            [9.12345, -3.98765, 7.55555, 1.23456, -5.67890],
            [2.34567, 8.76543, -4.32109, 6.54321, 0.98765],
            [-1.11111, 3.33333, 9.99999, -7.77777, 2.22222],
        ],
        dtype=y_dtype,
    )

    def strict_assert(a, b):
        if scipy.sparse.issparse(a):
            a = a.toarray()
        if scipy.sparse.issparse(b):
            b = b.toarray()
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=0, strict=True)

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test], assert_fn=strict_assert)


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc", "dense"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_structured_dot_grad(x_format, y_format, x_shape, y_shape):
    rng = np.random.default_rng()
    g_xy_shape = (x_shape[0], y_shape[1])

    x = ps.matrix(format=x_format, name="x", shape=x_shape)
    x_test = scipy.sparse.random(*x_shape, density=0.4, format=x_format)

    if y_format == "dense":
        y = pt.matrix("y", shape=y_shape)
        g_xy = pt.matrix(name="g_xy", shape=g_xy_shape)
        y_test = rng.normal(size=y_shape)
        g_xy_test = rng.normal(size=g_xy_shape)
    else:
        y = ps.matrix(format=y_format, name="y", shape=y_shape)
        g_xy = ps.matrix(format=x_format, name="g_xy", shape=g_xy_shape)
        y_test = scipy.sparse.random(*y_shape, density=0.5, format=y_format)
        g_xy_test = scipy.sparse.random(*g_xy_shape, density=0.3, format=x_format)

    z = ps.structured_dot_grad(x, y, g_xy)
    compare_numba_and_py_sparse([x, y, g_xy], z, [x_test, y_test, g_xy_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sparse_sum(format, axis):
    x = ps.matrix(format=format, name="x", shape=(7, 5))
    z = ps.sp_sum(x, axis=axis)
    x_test = scipy.sparse.random(7, 5, density=0.4, format=format)

    compare_numba_and_py_sparse([x], z, [x_test])


class TestSparseSparseBinary:
    @pytest.fixture(
        scope="class",
        params=[
            (x_format, y_format, x_dtype, y_dtype)
            for x_format in ("csr", "csc")
            for y_format in ("csr", "csc")
            for x_dtype, y_dtype in (
                ("float64", "float64"),
                ("int64", "float32"),
                ("float32", "complex64"),
                ("int8", "int8"),
                ("int8", "float64"),
            )
        ],
        ids=lambda params: "-".join(params),
    )
    @staticmethod
    def inputs(request):
        x_format, y_format, x_dtype, y_dtype = request.param
        return (
            ps.matrix(x_format, name="x", dtype=x_dtype),
            ps.matrix(y_format, name="y", dtype=y_dtype),
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def add_fn(inputs):
        x, y = inputs
        return pytensor.function(inputs, ps.add(x, y), mode=numba_mode)

    @pytest.fixture(scope="class")
    @staticmethod
    def multiply_fn(inputs):
        x, y = inputs
        return pytensor.function(inputs, ps.multiply(x, y), mode=numba_mode)

    @pytest.fixture
    @staticmethod
    def canonical_values(inputs):
        x, y = inputs
        rng = np.random.default_rng(47)
        x_value = scipy.sparse.random(
            31,
            47,
            density=0.3,
            format=x.type.format,
            random_state=rng,
            data_rvs=lambda size: rng.integers(1, 10, size=size),
        ).astype(x.dtype)

        y_value = scipy.sparse.random(
            31,
            47,
            density=0.5,
            format=y.type.format,
            random_state=rng,
            data_rvs=lambda size: rng.integers(1, 10, size=size),
        ).astype(y.dtype)

        if np.issubdtype(y_value.dtype, np.inexact):
            y_value.data[:] = rng.normal(size=y_value.nnz)

        if np.issubdtype(y_value.dtype, np.complexfloating):
            y_value.data.imag[:] = rng.normal(size=y_value.nnz)

        return x_value, y_value

    @pytest.fixture(
        params=[
            pytest.param((([2, 3, 4], [2, 0, 1], [0, 3, 3]), None), id="unsorted"),
            pytest.param((([2, 3, 4], [0, 0, 2], [0, 3, 3]), None), id="duplicates"),
            pytest.param((([2, 0, 3], [0, 1, 2], [0, 3, 3]), None), id="explicit-zero"),
            pytest.param((([2, 3], [0, 2], [0, 2, 2]), None), id="cancellation"),
            pytest.param((([64, 1, 0], [0, 2, 1], [0, 2, 3]), None), id="overflow"),
            pytest.param(
                (([100, 100], [0, 0], [0, 2, 2]), None), id="duplicate-overflow"
            ),
            pytest.param(
                (
                    ([2, 3, -2, 0, 4], [2, 0, 2, 1, 1], [0, 4, 5]),
                    ([1, -3, 2, 5], [2, 2, 0, 0], [0, 2, 4]),
                ),
                id="distinct-patterns",
            ),
        ]
    )
    def special_values(self, request, inputs):
        x, y = inputs
        if x.type.format == "csr":
            constructor = scipy.sparse.csr_matrix
            shape = (2, 3)
        else:
            constructor = scipy.sparse.csc_matrix
            shape = (3, 2)

        x_components, y_components = request.param
        data, indices, indptr = x_components
        x_value = constructor(
            (
                np.array(data, dtype=x.dtype),
                np.array(indices, dtype="int32"),
                np.array(indptr, dtype="int32"),
            ),
            shape=shape,
        )

        if y_components is None:
            y_value = (-x_value).astype(y.dtype).asformat(y.type.format)
        else:
            data, indices, indptr = y_components
            y_value = constructor(
                (
                    np.array(data, dtype=y.dtype),
                    np.array(indices, dtype="int32"),
                    np.array(indptr, dtype="int32"),
                ),
                shape=shape,
            ).asformat(y.type.format)

        return x_value, y_value

    @pytest.fixture
    def empty_values(self, inputs):
        x, y = inputs
        shape = (0, 3) if x.type.format == "csr" else (3, 0)
        return (
            scipy.sparse.csr_matrix(shape, dtype=x.dtype).asformat(x.type.format),
            scipy.sparse.csr_matrix(shape, dtype=y.dtype).asformat(y.type.format),
        )

    def test_add_canonical(self, add_fn, canonical_values):
        x, y = canonical_values
        result = add_fn(x, y)
        expected = x + y
        sparse_assert_fn(result, expected)
        if (x.dtype, y.dtype) == ("int64", "float32"):
            np.testing.assert_allclose(
                result.toarray(), expected.toarray(), rtol=1e-14, atol=0, strict=True
            )

    def test_multiply_canonical(self, multiply_fn, canonical_values):
        x, y = canonical_values
        result = multiply_fn(x, y)
        expected = x.multiply(y)
        sparse_assert_fn(result, expected)
        if (x.dtype, y.dtype) == ("int64", "float32"):
            np.testing.assert_allclose(
                result.toarray(), expected.toarray(), rtol=1e-14, atol=0, strict=True
            )

    def test_add_special_values(self, add_fn, special_values):
        x, y = special_values
        result = add_fn(x, y)
        # SciPy may normalize its operands in place.
        sparse_assert_fn(result, x.copy() + y.copy())

    def test_multiply_special_values(self, multiply_fn, special_values):
        x, y = special_values
        result = multiply_fn(x, y)
        sparse_assert_fn(result, x.copy().multiply(y.copy()))

    def test_add_empty(self, add_fn, empty_values):
        x, y = empty_values
        sparse_assert_fn(add_fn(x, y), x + y)

    def test_multiply_empty(self, multiply_fn, empty_values):
        x, y = empty_values
        sparse_assert_fn(multiply_fn(x, y), x.multiply(y))

    def test_add_preserves_inputs(self, add_fn, special_values):
        x, y = special_values
        x_before, y_before = x.copy(), y.copy()
        add_fn(x, y)
        for value, before in ((x, x_before), (y, y_before)):
            np.testing.assert_array_equal(value.data, before.data)
            np.testing.assert_array_equal(value.indices, before.indices)
            np.testing.assert_array_equal(value.indptr, before.indptr)

    def test_multiply_preserves_inputs(self, multiply_fn, special_values):
        x, y = special_values
        x_before, y_before = x.copy(), y.copy()
        multiply_fn(x, y)
        for value, before in ((x, x_before), (y, y_before)):
            np.testing.assert_array_equal(value.data, before.data)
            np.testing.assert_array_equal(value.indices, before.indices)
            np.testing.assert_array_equal(value.indptr, before.indptr)

    @pytest.mark.parametrize(
        "format, constructor, shape",
        [
            ("csr", scipy.sparse.csr_matrix, (2, 3)),
            ("csc", scipy.sparse.csc_matrix, (3, 2)),
        ],
    )
    @pytest.mark.parametrize("indices", [[0, 2], [2, 0]])
    @np.errstate(invalid="ignore")
    def test_multiply_nonfinite(self, format, constructor, shape, indices):
        x = ps.matrix(format, dtype="float64")
        y = ps.matrix(format, dtype="float64")
        value = constructor(
            (
                np.array([np.nan, np.inf]),
                np.array(indices, dtype="int32"),
                np.array([0, 2, 2], dtype="int32"),
            ),
            shape=shape,
        )
        fn = compare_numba_and_py_sparse(
            [x, y],
            [ps.multiply(x, y), ps.multiply(y, x)],
            [value, constructor(shape, dtype="float64")],
        )[0]
        other = constructor(
            (
                np.array([np.nan]),
                np.array([1], dtype="int32"),
                np.array([0, 1, 1], dtype="int32"),
            ),
            shape=shape,
        )
        actual = fn(value, other)
        actual_xy = actual[0]
        actual_yx = actual[1]
        sparse_assert_fn(actual_xy, value.multiply(other))
        sparse_assert_fn(actual_yx, other.multiply(value))


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "int32"])
def test_sparse_sparse_data_add(format, dtype):
    x = ps.matrix(format, name="x", dtype=dtype, shape=(11, 17))
    y = ps.matrix(format, name="y", dtype=dtype, shape=(11, 17))
    z = ps.add_s_s_data(x, y)
    rng = np.random.default_rng(51)
    integer_dtype = np.issubdtype(np.dtype(dtype), np.integer)

    def data_rvs(size):
        if integer_dtype:
            return rng.integers(-100, 101, size=size)
        return rng.normal(size=size)

    x_test = scipy.sparse.random(
        11,
        17,
        density=0.3,
        format=format,
        dtype=dtype,
        random_state=rng,
        data_rvs=data_rvs,
    )
    # x and y have to follow the same sparsity pattern
    y_test = x_test.copy()
    y_test.data = data_rvs(size=y_test.nnz).astype(dtype)

    if dtype == "int8":
        x_test.data[0] = y_test.data[0] = 100
        with np.errstate(over="ignore"):
            # The Python-only execution warns about the deliberate int8 overflow
            compare_numba_and_py_sparse([x, y], z, [x_test, y_test])
    else:
        compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize(
    "x_dtype, y_dtype", [("float64", "float64"), ("int64", "float32")]
)
def test_sparse_dense_add(format, x_dtype, y_dtype):
    x = ps.matrix(format, name="x", dtype=x_dtype, shape=(7, 9))
    y = pt.matrix("y", dtype=y_dtype, shape=(7, 9))
    outputs = [ps.add(x, y), ps.add(y, x)]
    rng = np.random.default_rng(71)
    x_test = scipy.sparse.random(
        7,
        9,
        density=0.5,
        format=format,
        random_state=rng,
        data_rvs=lambda size: rng.integers(1, 10, size=size),
    ).astype(x_dtype)

    y_test = rng.normal(size=(7, 9)).astype(y_dtype)
    compare_numba_and_py_sparse([x, y], outputs, [x_test, y_test])


@pytest.mark.parametrize(
    "format, constructor, shape",
    [
        ("csr", scipy.sparse.csr_matrix, (2, 3)),
        ("csc", scipy.sparse.csc_matrix, (3, 2)),
    ],
)
def test_structured_sparse_vector_add(format, constructor, shape):
    x = ps.matrix(format, name="x")
    y = pt.vector("y")
    z = ps.structured_add_s_v(x, y)
    rng = np.random.default_rng(87)
    x_test = scipy.sparse.random(11, 13, density=0.4, format=format, random_state=rng)
    y_test = rng.normal(size=13)
    fn = compare_numba_and_py_sparse([x, y], z, [x_test, y_test])[0]

    x_test = constructor(
        (
            np.array([2.0, 3.0, -2.0, 0.0, 4.0]),
            np.array([2, 0, 2, 1, 1]),
            np.array([0, 4, 5]),
        ),
        shape=shape,
    )
    y_test = np.array([-3.0, -4.0, 7.0][: shape[1]])
    original = x_test.copy()
    expected = constructor(x_test + (x_test.toarray() != 0) * y_test)
    actual = fn(x_test, y_test)
    sparse_assert_fn(actual, expected)
    np.testing.assert_array_equal(x_test.data, original.data)
    np.testing.assert_array_equal(x_test.indices, original.indices)
    np.testing.assert_array_equal(x_test.indptr, original.indptr)

    # Gradient graphs can embed readonly sparse indices.
    x_test.indices.flags.writeable = False
    x_test.indptr.flags.writeable = False
    sparse_assert_fn(fn(x_test, y_test), expected)
    np.testing.assert_array_equal(x_test.data, original.data)
    np.testing.assert_array_equal(x_test.indices, original.indices)
    np.testing.assert_array_equal(x_test.indptr, original.indptr)

    empty = constructor(shape, dtype=x.dtype)
    sparse_assert_fn(fn(empty, y_test), empty)


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_structured_sparse_vector_add_nonfinite(format, dtype):
    if format == "csr":
        constructor = scipy.sparse.csr_matrix
    else:
        constructor = scipy.sparse.csc_matrix

    x = ps.matrix(format, dtype=dtype)
    y = pt.vector(dtype=dtype)
    x_test = scipy.sparse.csr_matrix(
        (
            np.array([1, 0, -2, 0, 3, 4], dtype=dtype),
            np.array([0, 1, 2, 0, 1, 3], dtype="int32"),
            np.array([0, 3, 6, 6], dtype="int32"),
        ),
        shape=(3, 4),
    ).asformat(format)
    y_test = np.array([np.nan, np.inf, -np.inf, 2], dtype=dtype)
    # The structured mask deliberately evaluates 0 * inf and 0 * NaN.
    with np.errstate(invalid="ignore"):
        comparison = compare_numba_and_py_sparse(
            [x, y], ps.structured_add_s_v(x, y), [x_test, y_test]
        )
    fn = comparison[0]
    actual = comparison[1]
    assert np.isnan(actual.toarray()[:, 0]).all()
    assert np.isnan(actual.toarray()[2, :3]).all()

    for shape in [(3, 4), (0, 4), (3, 0)]:
        value = constructor(shape, dtype=dtype)
        vector = y_test[: shape[1]]
        dense_value = value.toarray()
        with np.errstate(invalid="ignore"):
            dense_expected = dense_value + (dense_value != 0) * vector

        expected = constructor(dense_expected)
        sparse_assert_fn(fn(value, vector), expected)

    x_test.data[:3] = [np.nan, np.inf, -np.inf]
    dense_x = x_test.toarray()
    with np.errstate(invalid="ignore"):
        dense_expected = dense_x + (dense_x != 0) * y_test
    expected = constructor(dense_expected)
    sparse_assert_fn(fn(x_test, y_test), expected)

    y_test = np.arange(1, 5, dtype=dtype)
    with np.errstate(invalid="ignore"):
        dense_expected = dense_x + (dense_x != 0) * y_test
    expected = constructor(dense_expected)
    sparse_assert_fn(fn(x_test, y_test), expected)
