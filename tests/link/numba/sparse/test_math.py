import numpy as np
import pytest
import scipy

import pytensor.sparse as ps
import pytensor.tensor as pt
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse


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
