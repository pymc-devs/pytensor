import numpy as np
import pytest
import scipy

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor import In, Out, config, function
from pytensor.compile.mode import Mode
from pytensor.link.numba.dispatch.basic import numba_funcify_and_cache_key
from pytensor.sparse.rewriting import UsmmCscDense
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse
from tests.link.numba.test_basic import numba_inplace_mode


pytestmark = pytest.mark.filterwarnings("error")

DOT_SHAPES = [((20, 11), (11, 4)), ((10, 3), (3, 1)), ((1, 10), (10, 5))]


@pytest.mark.parametrize("specialized", [False, True])
def test_usmm_cache_keys(specialized):
    keys = []
    cases = [
        ("float64", None, "csc", False),
        ("float32", None, "csc", False),
        ("float64", 1, "csc", False),
        ("float64", None, "csc", True)
        if specialized
        else ("float64", None, "csr", False),
    ]
    for dtype, n_cols, format, inplace in cases:
        alpha = pt.scalar(dtype=dtype)
        x = ps.matrix(format=format, dtype=dtype)
        y = pt.matrix(dtype=dtype, shape=(None, n_cols))
        z = pt.matrix(dtype=dtype)
        if specialized:
            values, indices, indptr, shape = ps.csm_properties(x)
            node = UsmmCscDense(inplace).make_node(
                alpha, values, indices, indptr, shape[0], y, z
            )
        else:
            node = ps.Usmm().make_node(alpha, x, y, z)
        _, key = numba_funcify_and_cache_key(node.op, node)
        clone = node.op.make_node(*(inp.clone() for inp in node.inputs))
        _, clone_key = numba_funcify_and_cache_key(clone.op, clone)
        assert key == clone_key
        keys.append(key)

    assert len(set(keys)) == len(keys)


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


@pytest.mark.parametrize(
    "x_format, y_format, x_dtype, y_dtype",
    [
        ("csr", "dense", "float32", "float32"),
        ("csc", "dense", "float32", "float64"),
        ("dense", "csr", "float64", "float32"),
        ("dense", "csc", "float64", "float64"),
        ("csr", "csr", "float64", "float64"),
        ("csr", "csc", "float32", "float64"),
        ("csc", "csr", "float64", "float32"),
        ("csc", "csc", "float32", "float32"),
    ],
    scope="class",
)
class TestUsmm:
    @pytest.fixture(scope="class")
    @staticmethod
    def usmm(x_format, y_format, x_dtype, y_dtype):
        def make_input(name, format, dtype):
            if format == "dense":
                return pt.matrix(name, dtype=dtype)
            return ps.matrix(format, name=name, dtype=dtype)

        alpha = pt.scalar("alpha", dtype=x_dtype)
        x = make_input("x", x_format, x_dtype)
        y = make_input("y", y_format, y_dtype)
        z = pt.matrix("z", dtype=x_dtype)
        return function(
            [alpha, x, y, z],
            ps.usmm(alpha, x, y, z),
            mode=Mode(linker="numba", optimizer=None),
        )

    @staticmethod
    def as_input(value, format, dtype):
        value = np.asarray(value, dtype=dtype)
        if format == "dense":
            return value
        return getattr(scipy.sparse, f"{format}_matrix")(value)

    @pytest.fixture(scope="class")
    @classmethod
    def matrices(cls, x_format, y_format, x_dtype, y_dtype):
        x = cls.as_input([[1, 0], [0, 0]], x_format, x_dtype)
        y = cls.as_input([[1, 0, 2], [0, 0, 0]], y_format, y_dtype)
        return x, y

    def test_product(self, usmm, x_format, y_format, x_dtype, y_dtype):
        rng = np.random.default_rng(95)
        x = rng.normal(size=(11, 13)).astype(x_dtype)
        y = rng.normal(size=(13, 7)).astype(y_dtype)
        x *= rng.random(x.shape) < 0.31
        y *= rng.random(y.shape) < 0.29
        z = rng.normal(size=(11, 7)).astype(x_dtype)
        alpha = np.array(1.7, dtype=x_dtype)
        result = usmm(
            alpha,
            self.as_input(x, x_format, x_dtype),
            self.as_input(y, y_format, y_dtype),
            z,
        )
        np.testing.assert_allclose(
            result, alpha * (x @ y) + z, rtol=1e-5, atol=1e-6, strict=True
        )
        assert not np.shares_memory(result, z)

    def test_broadcast(self, usmm, matrices, x_dtype, y_dtype):
        x, y = matrices
        for z_shape in ((1, 3), (2, 1), (1, 1)):
            z = np.ones(z_shape, dtype=x_dtype)
            result = usmm(np.array(0.5, dtype=x_dtype), x, y, z)
            expected = np.array(
                [[1.5, 1, 2], [1, 1, 1]], dtype=np.result_type(x_dtype, y_dtype)
            )
            np.testing.assert_allclose(result, expected, strict=True)
            np.testing.assert_array_equal(z, np.ones(z_shape, dtype=x_dtype))
            assert not np.shares_memory(result, z)

    def test_nonfinite_alpha(self, usmm, matrices, x_dtype, y_dtype):
        x, y = matrices
        product = np.array(
            [[1, 0, 2], [0, 0, 0]], dtype=np.result_type(x_dtype, y_dtype)
        )
        for alpha_value, z_shape in (
            (np.inf, (2, 3)),
            (-np.inf, (1, 3)),
            (np.nan, (1, 1)),
        ):
            alpha = np.array(alpha_value, dtype=x_dtype)
            z = np.ones(z_shape, dtype=x_dtype)
            with np.errstate(invalid="ignore"):
                expected = alpha * product + z
                result = usmm(alpha, x, y, z)
            np.testing.assert_allclose(result, expected, equal_nan=True, strict=True)
            np.testing.assert_array_equal(z, np.ones(z_shape, dtype=x_dtype))
            assert not np.shares_memory(result, z)

    def test_invalid_shape(self, usmm, matrices, x_dtype):
        x, y = matrices
        with pytest.raises(ValueError):
            usmm(np.array(0.5, dtype=x_dtype), x, y, np.ones((3, 3), dtype=x_dtype))

    def test_broadcast_product(self, usmm, matrices, x_dtype, y_dtype):
        x, y = matrices
        for n_rows, n_cols in ((1, 3), (2, 1)):
            result = usmm(
                np.array(0.5, dtype=x_dtype),
                x[:n_rows],
                y[:, :n_cols],
                np.ones((2, 3), dtype=x_dtype),
            )
            expected_product = np.array(
                [[1, 0, 2], [0, 0, 0]], dtype=np.result_type(x_dtype, y_dtype)
            )[:n_rows, :n_cols]
            expected = 0.5 * expected_product + np.ones((2, 3), dtype=x_dtype)
            np.testing.assert_allclose(result, expected, strict=True)

    def test_vectors(self, usmm, matrices, x_dtype, y_dtype):
        x, y = matrices
        for n_rows, n_cols in ((1, 3), (2, 1), (1, 0)):
            z = np.ones((n_rows, n_cols), dtype=x_dtype)
            result = usmm(np.array(0.5, dtype=x_dtype), x[:n_rows], y[:, :n_cols], z)
            expected = np.array(
                [[1.5, 1, 2], [1, 1, 1]], dtype=np.result_type(x_dtype, y_dtype)
            )[:n_rows, :n_cols]
            np.testing.assert_allclose(result, expected, strict=True)

    def test_dense_layouts(self, usmm, matrices, x_format, y_format, x_dtype, y_dtype):
        if x_format != "dense" and y_format != "dense":
            pytest.skip("Both inputs are sparse")
        x, y = matrices
        if x_format == "dense":
            y = self.as_input(
                np.random.default_rng(18).normal(size=(2, 64)), y_format, y_dtype
            )
        dense = x if x_format == "dense" else y
        for value in (np.asfortranarray(dense), dense[:, ::-1]):
            layout_x = value if x_format == "dense" else x
            layout_y = value if y_format == "dense" else y
            z = np.ones((2, layout_y.shape[1]), dtype=x_dtype)
            result = usmm(np.array(0.5, dtype=x_dtype), layout_x, layout_y, z)
            np.testing.assert_allclose(
                result, 0.5 * (layout_x @ layout_y) + z, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("mutable, direct", [(False, True), (True, False)])
def test_usmm_csc_dense_rewrite(mutable, direct):
    dtype = "float64"
    alpha = pt.scalar("alpha", dtype=dtype)
    x = ps.csc_matrix("x", dtype=dtype)
    y = pt.matrix("y", dtype=dtype)
    z = pt.matrix("z", dtype=dtype)
    out = ps.usmm(-alpha, x, y, z) if direct else z - alpha * ps.dot(x, y)
    f = function(
        [alpha, x, y, In(z, mutable=mutable, borrow=mutable)],
        Out(out, borrow=True),
        mode=numba_inplace_mode.including("specialize"),
    )
    ops = [n.op for n in f.maker.fgraph.toposort() if isinstance(n.op, UsmmCscDense)]
    assert len(ops) == 1
    assert ops[0].inplace == mutable


@pytest.mark.parametrize(
    "op, format, dtype, fastmath",
    [
        (ps.Usmm(), "csr", "float32", False),
        (ps.Usmm(), "csc", "float64", True),
        (UsmmCscDense(False), "csc", "float32", False),
        (UsmmCscDense(True), "csc", "float64", False),
        (UsmmCscDense(False), "csc", "float64", True),
        (UsmmCscDense(True), "csc", "float32", True),
    ],
)
def test_usmm_extreme_values(dtype, fastmath, op, format):
    alpha = pt.scalar("alpha", dtype=dtype)
    x = ps.matrix(format=format, name="x", dtype=dtype)
    y = pt.matrix("y", dtype=dtype)
    z = pt.matrix("z", dtype=dtype)
    inplace = isinstance(op, UsmmCscDense) and op.inplace
    if isinstance(op, UsmmCscDense):
        values, indices, indptr, shape = ps.csm_properties(x)
        out = op(alpha, values, indices, indptr, shape[0], y, z)
    else:
        out = op(alpha, x, y, z)
    with config.change_flags(numba__fastmath=fastmath):
        f = function(
            [alpha, x, y, In(z, mutable=inplace, borrow=inplace)],
            Out(out, borrow=True),
            mode=Mode(linker="numba", optimizer=None),
            accept_inplace=True,
        )
    big = np.finfo(dtype).max * np.array(0.75, dtype=dtype)
    overflow_cases = (
        (2, [[big]], [[0.25]], [[0]], [[np.inf]]),
        (2, [[big]], [[0.25, 0, -0.25]], [[0, 0, 0]], [[np.inf, np.nan, -np.inf]]),
        (0.25, [[big]], [[2]], [[0]], [[big * 0.5]]),
        # Starting from z can overflow before the opposite contributions cancel.
        (1, [[big, big]], [[1], [-1]], [[big]], [[np.inf]]),
    )
    cases = (
        (2, [[4]], [[3]], [[-24]], [[0]]),
        (2, [[4]], [[3]], [[-23]], [[1]]),
        (1, [[np.inf]], [[1]], [[-np.inf]], [[np.nan]]),
        (np.inf, [[0]], [[1]], [[0]], [[np.nan]]),
        (-np.inf, [[1]], [[1]], [[0]], [[-np.inf]]),
    )
    # Reassociation can change overflow, so compare extreme cases without it.
    for scale, x_value, y_value, z_value, expected in cases + (
        () if fastmath else overflow_cases
    ):
        z_array = np.array(z_value, dtype=dtype)[:, ::-1].copy()
        result = f(
            np.array(scale, dtype=dtype),
            getattr(scipy.sparse, f"{format}_matrix")(np.array(x_value, dtype=dtype)),
            np.array(y_value, dtype=dtype)[:, ::-1],
            z_array,
        )
        np.testing.assert_array_equal(
            result,
            np.array(expected, dtype=dtype)[:, ::-1],
            strict=True,
        )
        assert np.shares_memory(result, z_array) == inplace


@pytest.mark.parametrize(
    "dtype, inplace", [("float32", False), ("float64", True)], scope="class"
)
class TestUsmmCscDense:
    @pytest.fixture(scope="class")
    @staticmethod
    def usmm(dtype, inplace):
        alpha = pt.scalar("alpha", dtype=dtype)
        x = ps.csc_matrix("x", dtype=dtype)
        y = pt.matrix("y", dtype=dtype)
        z = pt.matrix("z", dtype=dtype)
        values, indices, indptr, shape = ps.csm_properties(x)
        out = UsmmCscDense(inplace)(alpha, values, indices, indptr, shape[0], y, z)
        return function(
            [alpha, x, y, In(z, mutable=inplace, borrow=inplace)],
            Out(out, borrow=True),
            mode=Mode(linker="numba", optimizer=None),
            accept_inplace=True,
        )

    def test_shapes_and_layouts(self, usmm, dtype, inplace):
        rng = np.random.default_rng(1211)
        # Shape, dense stride, and z shape are paired instead of taking their product.
        for shape, step, z_shape in [
            ((2, 3, 1), 2, (2, 1)),
            ((2, 3, 5), -2, (2, 5)),
            ((2, 5, 3), 2, (1, 3)),
            ((2, 3, 5), -2, (2, 1)),
            ((2, 3, 5), 2, (1, 1)),
            ((2, 0, 5), -2, (2, 5)),
            ((2, 3, 0), 2, (2, 0)),
            ((0, 3, 5), -2, (0, 5)),
        ]:
            n_rows, n_inner, n_cols = shape
            x = scipy.sparse.csc_matrix(
                rng.normal(size=(n_rows, n_inner)).astype(dtype)
            )
            y = rng.normal(size=(n_inner, 2 * n_cols)).astype(dtype)[:, ::step]
            z = rng.normal(size=z_shape).astype(dtype)[:, ::-1]
            z_before = z.copy()
            expected = z_before + 0.5 * (x @ y)
            result = usmm(np.array(0.5, dtype=dtype), x, y, z)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
            reuse_z = inplace and z_shape == (n_rows, n_cols)
            if result.size:
                assert np.shares_memory(result, z) == reuse_z
            if not reuse_z:
                np.testing.assert_array_equal(z, z_before)

    def test_nonfinite_alpha(self, usmm, dtype, inplace):
        x = scipy.sparse.csc_matrix(np.array([[1, 0], [0, 0]], dtype=dtype))
        y = np.array([[1, 0, 2], [0, 0, 0]], dtype=dtype)
        for alpha_value, z_shape in (
            (np.inf, (2, 3)),
            (-np.inf, (1, 3)),
            (np.nan, (1, 1)),
        ):
            z = np.ones(z_shape, dtype=dtype)
            with np.errstate(invalid="ignore"):
                expected = z + alpha_value * (x @ y)
                result = usmm(np.array(alpha_value, dtype=dtype), x, y, z)
            np.testing.assert_allclose(result, expected, equal_nan=True)
            assert np.shares_memory(result, z) == (inplace and z_shape == (2, 3))
            if not inplace or z_shape != (2, 3):
                np.testing.assert_array_equal(z, np.ones(z_shape, dtype=dtype))

    def test_invalid_shapes(self, usmm, dtype):
        x = scipy.sparse.csc_matrix((2, 3), dtype=dtype)
        alpha = np.array(0.5, dtype=dtype)
        with pytest.raises(ValueError, match="z must broadcast"):
            usmm(alpha, x, np.ones((3, 5), dtype=dtype), np.ones((3, 5), dtype=dtype))
        with pytest.raises(AssertionError):
            usmm(alpha, x, np.ones((4, 5), dtype=dtype), np.ones((2, 5), dtype=dtype))


@pytest.mark.parametrize("z_shape", [(1, None), (None, 1), (1, 1)])
def test_usmm_broadcast_rewrite(z_shape):
    x = ps.csc_matrix("x", dtype="float64")
    y = pt.matrix("y", dtype="float64")
    z = pt.matrix("z", shape=z_shape, dtype="float64")
    f = function(
        [x, y, z],
        z - 0.5 * ps.dot(x, y),
        mode=numba_inplace_mode.including("specialize"),
    )
    nodes = f.maker.fgraph.toposort()
    assert any(isinstance(n.op, UsmmCscDense) for n in nodes)
    assert not any(isinstance(n.op, ps.Usmm) for n in nodes)
